"""Distillation LightningModule: distill fine-tuned Qwen2 into a flow-based model.

Uses continuous flow matching (CondOTProbPath) to learn a velocity field that
maps noise → teacher's intermediate hidden states. The training follows a
two-stage approach:
  - Stage 1: Velocity matching loss (MSE between predicted and target velocity)
  - Stage 2: Cross-entropy loss on classification labels (freeze flow, unfreeze suffix)
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MeanMetric, MaxMetric
from torchmetrics.classification import Accuracy

from flow_matching.path import CondOTProbPath

from src.models.components.hooked_qwen2 import (
    HookedQwen2ForSequenceClassification,
    load_pretrained_hooked_qwen2,
)
from src.models.components.flow_transformer import FlowTransformerText


class DistillFlowModule(LightningModule):
    """LightningModule for distilling a fine-tuned Qwen2 into a flow-based model.

    The teacher (HookedQwen2ForSequenceClassification) is frozen and used to
    extract hidden-state trajectories. The student (FlowTransformerText) learns
    to reproduce these trajectories via continuous OT flow matching.

    Training stages:
        Stage 1 (epochs < epochs_stage1): Velocity matching loss only.
            Train the DiT backbone + velocity head to predict the OT velocity
            that transforms noise into teacher hidden states.
        Stage 2 (epochs >= epochs_stage1): Cross-entropy loss only.
            Freeze flow components, unfreeze suffix + classifier, fine-tune on labels.
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

        # --- Flow matching path ---
        self.flow_path = CondOTProbPath()

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
        """Extract teacher's hidden-state trajectory."""
        with torch.no_grad():
            _, x_t, means = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_hidden_trajectory=True,
            )
        return x_t, means

    def _compute_flow_loss(
        self,
        x_t_teacher: List[torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute velocity matching loss using CondOTProbPath.

        For each replaced layer l (from_layer to to_layer-1):
          - x_1 = teacher hidden state at layer l+1 (target)
          - x_0 ~ N(0, I) (noise)
          - t ~ U[0, 1]
          - x_t = (1-t)*x_0 + t*x_1 (OT interpolant)
          - target velocity = x_1 - x_0
          - loss = MSE(v_θ(x_t, t), x_1 - x_0)
        """
        from_l = self.hparams.from_layer
        to_l = self.hparams.to_layer

        total_loss = 0.0
        num_steps = to_l - from_l

        for step in range(num_steps):
            layer_idx = from_l + step

            # x_1: teacher hidden state at this layer (target)
            x_1 = x_t_teacher[layer_idx + 1].detach()  # x_t[layer_idx+1] = output after layer_idx

            # x_0: noise from standard normal
            x_0 = torch.randn_like(x_1)

            # Sample t ~ U[0,1]
            t = torch.rand(x_1.shape[0], device=x_1.device, dtype=x_1.dtype)

            # CondOTProbPath: x_t = (1-t)*x_0 + t*x_1, dx_t = x_1 - x_0
            path_sample = self.flow_path.sample(t=t, x_0=x_0, x_1=x_1)

            # Predict velocity
            t_step = torch.full(
                (x_1.shape[0],),
                step / num_steps,  # Normalized timestep for this step
                device=x_1.device,
                dtype=x_1.dtype,
            )
            v_pred = self.student.predict_velocity(path_sample.x_t, t_step)

            # Velocity matching loss
            total_loss += self.mse_loss(v_pred, path_sample.dx_t)

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
            x_t_teacher, means_teacher = self._get_teacher_trajectory(
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
            x_t_teacher, means_teacher = self._get_teacher_trajectory(
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

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/accuracy_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

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
