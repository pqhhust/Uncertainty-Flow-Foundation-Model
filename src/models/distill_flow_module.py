"""Distillation LightningModule: distill fine-tuned Qwen2 into a flow-based model.

Uses Flow Recompilation to learn a continuous velocity field from the teacher's
discrete layer-wise forward pass. Each transformer layer is viewed as a
discretized ODE step:
    X_{t+1} = FM_θ^(t)(X_t)
    velocity_target = (X_{t+1} - X_t) / Δt

The student u_ω(X_t, t) is distilled to match these velocity targets, defining
a Neural ODE: dX_t = u_ω(X_t, t) dt.

Training stages:
  - Stage 1: Velocity matching loss (MSE between predicted and target velocity)
  - Stage 2: Cross-entropy loss on classification labels (freeze flow, unfreeze suffix)
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MeanMetric, MaxMetric
from torchmetrics.classification import Accuracy

from src.models.components.hooked_qwen2 import (
    HookedQwen2ForSequenceClassification,
    load_pretrained_hooked_qwen2,
)
from src.models.components.flow_transformer import FlowTransformerText


class DistillFlowModule(LightningModule):
    """LightningModule for distilling a fine-tuned Qwen2 via Flow Recompilation.

    The teacher (HookedQwen2ForSequenceClassification) is treated as a discretized
    ODE: each layer l maps X_l → X_{l+1} = FM_θ^(l)(X_l). The velocity target at
    layer l is (X_{l+1} - X_l) / Δt. The student (FlowTransformerText) learns the
    continuous velocity field u_ω(X_t, t) via MSE matching.

    Training stages:
        Stage 1 (epochs < epochs_stage1): Velocity matching loss only.
            Train DiT + velocity head to predict layer-wise velocity targets.
        Stage 2 (epochs >= epochs_stage1): Cross-entropy loss only.
            Freeze flow, unfreeze suffix + classifier, fine-tune on labels.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        teacher_ckpt_path: Optional[str] = None,
        num_labels: int = 2,
        from_layer: int = 6,
        to_layer: int = 18,
        dit_depth: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        teacher_dropout: float = 0.1,
        learning_rate: float = 1e-4,
        lr_stage2: float = 1e-5,
        weight_decay: float = 0.01,
        epochs_stage1: int = 10,
        lambda_flow: float = 1.0,
        lambda_ce: float = 1.0,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # --- Load teacher model ---
        self.teacher = self._load_teacher(model_name, num_labels, teacher_ckpt_path)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # --- Build student model ---
        teacher_config = self.teacher.config
        self.student = FlowTransformerText(
            config=teacher_config,
            dit_depth=dit_depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_labels=num_labels,
            from_layer=from_layer,
            to_layer=to_layer,
        )

        # Transfer teacher weights to student
        self.student.load_teacher_weights(self.teacher)

        # Start in Stage 1: freeze non-flow params
        self.student.freeze_non_flow_params()

        # --- Losses ---
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        # --- Metrics ---
        self.train_flow_loss = MeanMetric()
        self.train_ce_loss = MeanMetric()
        self.train_acc = Accuracy(
            task="multiclass" if num_labels > 2 else "binary",
            num_classes=num_labels,
        )
        self.val_loss = MeanMetric()
        self.val_acc = Accuracy(
            task="multiclass" if num_labels > 2 else "binary",
            num_classes=num_labels,
        )
        self.test_acc = Accuracy(
            task="multiclass" if num_labels > 2 else "binary",
            num_classes=num_labels,
        )
        self.val_acc_best = MaxMetric()

        # Stage tracking
        self._current_stage = 1

        # Collect logits/labels for calibration metrics
        self._test_logits: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._val_logits: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []

    def _load_teacher(
        self,
        model_name: str,
        num_labels: int,
        ckpt_path: Optional[str],
    ) -> HookedQwen2ForSequenceClassification:
        """Load teacher model from checkpoint or pretrained."""
        if ckpt_path is not None:
            # Load from Lightning checkpoint
            from src.models.qwen2_finetune_module import Qwen2FineTuneModule

            finetune_module = Qwen2FineTuneModule.load_from_checkpoint(
                ckpt_path,
                map_location="cpu",
            )
            teacher = finetune_module.net
            del finetune_module
        else:
            # Load pretrained (no fine-tuning)
            teacher = load_pretrained_hooked_qwen2(
                model_name=model_name,
                num_labels=num_labels,
                device="cpu",
            )
        return teacher

    def on_train_epoch_start(self) -> None:
        """Switch between training stages."""
        epoch = self.current_epoch
        epochs_stage1 = self.hparams.epochs_stage1

        if epochs_stage1 is not None and epoch == epochs_stage1 and self._current_stage == 1:
            self.print(f"\n{'='*60}")
            self.print(f"Switching to Stage 2 at epoch {epoch}")
            self.print(f"Freezing flow components, unfreezing suffix + classifier")
            self.print(f"{'='*60}\n")
            self.student.freeze_flow_unfreeze_suffix()
            self._current_stage = 2

    def _get_teacher_trajectory(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """Extract teacher's hidden-state trajectory with dropout.

        Per the Flow Recompilation formulation, we run the teacher with dropout
        enabled (FM_{θ_dropout}) so velocity targets are stochastic, acting as
        data augmentation / regularization.
        """
        p = self.hparams.teacher_dropout
        with torch.no_grad():
            _, x_t = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_hidden_trajectory=True,
            )
        if p > 0.0 and self.training:
            # Apply dropout to each hidden state (stochastic velocity targets)
            x_t = [F.dropout(h, p=p, training=True) for h in x_t]
        return x_t

    def _compute_flow_loss(
        self,
        x_t_teacher: List[torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute velocity matching loss via Flow Recompilation.

        Each transformer layer is a discrete ODE step:
            X_{t+1} = FM_θ^(t)(X_t)

        The velocity target at step l is:
            v_target = (X_{l+1} - X_l) / Δt

        where Δt = 1 / num_steps, and time t_l = l / num_steps (normalized to [0,1]
        within the replaced layer range).

        The student u_ω(X_t, t) is trained to match:
            loss = MSE(u_ω(X_l, t_l), v_target)
        """
        from_l = self.hparams.from_layer
        to_l = self.hparams.to_layer
        num_steps = to_l - from_l
        delta_t = 1.0 / num_steps

        total_loss = 0.0

        for step in range(num_steps):
            layer_idx = from_l + step

            # X_t: teacher hidden state at input of this layer
            X_t = x_t_teacher[layer_idx].detach()
            # X_{t+1}: teacher hidden state at output of this layer
            X_t1 = x_t_teacher[layer_idx + 1].detach()

            # Velocity target: (X_{t+1} - X_t) / Δt
            velocity_target = (X_t1 - X_t) / delta_t

            # Normalized time for this step: t ∈ [0, 1)
            t = torch.full(
                (X_t.shape[0],),
                step * delta_t,
                device=X_t.device,
                dtype=X_t.dtype,
            )

            # Predict velocity: u_ω(X_t, t)
            v_pred = self.student.predict_velocity(X_t, t)

            # MSE loss
            total_loss += self.mse_loss(v_pred, velocity_target)

        return total_loss / num_steps

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        epoch = self.current_epoch
        epochs_stage1 = self.hparams.epochs_stage1

        if epochs_stage1 is not None and epoch < epochs_stage1:
            # Stage 1: Velocity matching loss only
            x_t_teacher = self._get_teacher_trajectory(
                input_ids, attention_mask
            )
            flow_loss = self._compute_flow_loss(x_t_teacher, input_ids, attention_mask)
            loss = self.hparams.lambda_flow * flow_loss

            self.train_flow_loss(flow_loss)
            self.log("train/flow_loss", self.train_flow_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/stage", 1.0, on_step=False, on_epoch=True)

        elif epochs_stage1 is not None:
            # Stage 2: Cross-entropy loss only
            pooled_logits, _ = self.student(input_ids, attention_mask)
            ce_loss = self.ce_loss(pooled_logits, labels)
            loss = self.hparams.lambda_ce * ce_loss

            preds = torch.argmax(pooled_logits, dim=-1)
            self.train_acc(preds, labels)
            self.train_ce_loss(ce_loss)
            self.log("train/ce_loss", self.train_ce_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/accuracy", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/stage", 2.0, on_step=False, on_epoch=True)

        else:
            # Joint: both losses
            x_t_teacher = self._get_teacher_trajectory(
                input_ids, attention_mask
            )
            flow_loss = self._compute_flow_loss(x_t_teacher, input_ids, attention_mask)
            pooled_logits, _ = self.student(input_ids, attention_mask)
            ce_loss = self.ce_loss(pooled_logits, labels)
            loss = self.hparams.lambda_flow * flow_loss + self.hparams.lambda_ce * ce_loss

            self.train_flow_loss(flow_loss)
            self.train_ce_loss(ce_loss)
            preds = torch.argmax(pooled_logits, dim=-1)
            self.train_acc(preds, labels)
            self.log("train/flow_loss", self.train_flow_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/ce_loss", self.train_ce_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/accuracy", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/stage", 0.0, on_step=False, on_epoch=True)

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        pooled_logits, _ = self.student(input_ids, attention_mask)
        loss = self.ce_loss(pooled_logits, labels)
        preds = torch.argmax(pooled_logits, dim=-1)

        self.val_loss(loss)
        self.val_acc(preds, labels)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/accuracy", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        # Collect for calibration
        self._val_logits.append(pooled_logits.detach().cpu())
        self._val_labels.append(labels.detach().cpu())

    def _gather_predictions(self, logits_list, labels_list):
        """Gather logits/labels across GPUs (if distributed), then compute on full data."""
        all_logits = torch.cat(logits_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)

        # In distributed setting, gather all predictions to rank 0
        if self.trainer and self.trainer.world_size > 1:
            gathered_logits = self.all_gather(all_logits.to(self.device))
            gathered_labels = self.all_gather(all_labels.to(self.device))
            # all_gather returns (world_size, local_N, ...) — flatten
            all_logits = gathered_logits.reshape(-1, gathered_logits.shape[-1]).cpu()
            all_labels = gathered_labels.reshape(-1).cpu()

        return all_logits, all_labels

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/accuracy_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

        # Compute calibration metrics on validation set
        if self._val_logits:
            from src.utils.metrics import compute_all_metrics

            all_logits, all_labels = self._gather_predictions(
                self._val_logits, self._val_labels
            )
            metrics = compute_all_metrics(all_logits, all_labels)
            for k, v in metrics.items():
                self.log(f"val/{k}", v, on_step=False, on_epoch=True, rank_zero_only=True)

        self._val_logits.clear()
        self._val_labels.clear()

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        pooled_logits, _ = self.student(input_ids, attention_mask)
        preds = torch.argmax(pooled_logits, dim=-1)

        self.test_acc(preds, labels)
        self.log("test/accuracy", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        # Collect for calibration
        self._test_logits.append(pooled_logits.detach().cpu())
        self._test_labels.append(labels.detach().cpu())

    def on_test_epoch_end(self) -> None:
        """Compute full calibration metrics and plot reliability diagrams."""
        if not self._test_logits:
            return

        from src.utils.metrics import (
            compute_all_metrics,
            format_metrics_table,
            plot_reliability_diagram,
        )

        all_logits, all_labels = self._gather_predictions(
            self._test_logits, self._test_labels
        )

        # Scalar metrics
        metrics = compute_all_metrics(all_logits, all_labels)
        for k, v in metrics.items():
            self.log(f"test/{k}", v, on_step=False, on_epoch=True, rank_zero_only=True)

        self.print(f"\n{'='*60}")
        self.print("Test Calibration Metrics (TransDiff-style):")
        self.print(format_metrics_table(metrics))
        self.print(f"{'='*60}\n")

        # Reliability diagram + bin-count plot
        save_dir = "."
        if self.trainer and self.trainer.log_dir:
            save_dir = self.trainer.log_dir
        plot_metrics = plot_reliability_diagram(
            all_logits, all_labels, save_dir=save_dir, split="test"
        )
        self.print(f"Reliability diagram saved to {save_dir}/calibration/")

        self._test_logits.clear()
        self._test_labels.clear()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer — only optimizes trainable params."""
        trainable_params = [p for p in self.student.parameters() if p.requires_grad]

        if self._current_stage == 1:
            lr = self.hparams.learning_rate
        else:
            lr = self.hparams.lr_stage2

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=self.hparams.weight_decay,
        )

        return {"optimizer": optimizer}
