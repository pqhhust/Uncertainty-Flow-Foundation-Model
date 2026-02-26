"""Qwen2 fine-tuning LightningModule for GLUE classification tasks.

Supports three fine-tuning modes:
  - full: fine-tune all parameters
  - lora: LoRA fine-tuning via PEFT (targets attention + MLP projections)
  - linear_probe: freeze all except the classification head
"""

from typing import Any, Dict, List, Optional

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric, MaxMetric
from torchmetrics.classification import (
    Accuracy,
    F1Score,
    MatthewsCorrCoef,
)
from torchmetrics.regression import (
    PearsonCorrCoef,
    SpearmanCorrCoef,
)

from src.models.components.hooked_qwen2 import (
    HookedQwen2ForSequenceClassification,
    load_pretrained_hooked_qwen2,
)


# Metric factories per task
TASK_METRICS = {
    "accuracy": lambda nl: Accuracy(task="multiclass" if nl > 2 else "binary", num_classes=nl),
    "f1": lambda nl: F1Score(task="binary", num_classes=nl),
    "matthews_correlation": lambda nl: MatthewsCorrCoef(task="binary", num_classes=nl),
    "spearmanr": lambda _: SpearmanCorrCoef(),
    "pearsonr": lambda _: PearsonCorrCoef(),
}


class Qwen2FineTuneModule(LightningModule):
    """LightningModule for fine-tuning Qwen2 on GLUE tasks.

    Uses HookedQwen2ForSequenceClassification as the backbone, which supports
    TransformerLens hooks for later distillation.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        num_labels: int = 2,
        task_metric: str = "accuracy",
        finetune_mode: str = "full",  # "full", "lora", "linear_probe"
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Build model
        self.net = load_pretrained_hooked_qwen2(
            model_name=model_name,
            num_labels=num_labels,
            device="cpu",  # Lightning handles device placement
        )

        # Apply fine-tuning mode
        self._apply_finetune_mode(finetune_mode)

        # Loss
        if num_labels == 1:
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        # Metrics
        metric_fn = TASK_METRICS.get(task_metric, TASK_METRICS["accuracy"])
        self.train_metric = metric_fn(num_labels)
        self.val_metric = metric_fn(num_labels)
        self.test_metric = metric_fn(num_labels)
        self.metric_name = task_metric

        # Loss averaging
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # Best validation metric
        self.val_metric_best = MaxMetric()

        # Collect logits/labels for calibration metrics
        self._test_logits: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._val_logits: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []

    def _apply_finetune_mode(self, mode: str) -> None:
        """Configure which parameters are trainable based on fine-tuning mode."""
        if mode == "linear_probe":
            # Freeze everything except the classification head
            for param in self.net.model.parameters():
                param.requires_grad = False
            for param in self.net.score.parameters():
                param.requires_grad = True

        elif mode == "lora":
            # Freeze base model, then apply LoRA
            for param in self.net.parameters():
                param.requires_grad = False

            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                bias="none",
                task_type="SEQ_CLS",
            )
            self.net = get_peft_model(self.net, lora_config)
            # Unfreeze classification head
            for param in self.net.base_model.model.score.parameters():
                param.requires_grad = True
            self.net.print_trainable_parameters()

        elif mode == "full":
            # All parameters trainable
            for param in self.net.parameters():
                param.requires_grad = True

        else:
            raise ValueError(f"Unknown finetune_mode: {mode}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        outputs = self.net(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        return outputs

    def model_step(self, batch: Dict[str, torch.Tensor]):
        """Single model step shared by train/val/test."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        logits = outputs.logits

        if self.hparams.num_labels == 1:
            preds = logits.squeeze()
        else:
            preds = torch.argmax(logits, dim=-1)

        return loss, preds, labels, logits

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_metric.reset()
        self.val_metric_best.reset()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, preds, targets, _ = self.model_step(batch)
        self.train_loss(loss)
        self.train_metric(preds, targets)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"train/{self.metric_name}", self.train_metric, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets, logits = self.model_step(batch)
        self.val_loss(loss)
        self.val_metric(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"val/{self.metric_name}", self.val_metric, on_step=False, on_epoch=True, prog_bar=True)

        # Collect logits for calibration
        self._val_logits.append(logits.detach().cpu())
        self._val_labels.append(targets.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        metric_val = self.val_metric.compute()
        self.val_metric_best(metric_val)
        self.log(f"val/{self.metric_name}_best", self.val_metric_best.compute(), sync_dist=True, prog_bar=True)

        # Compute calibration metrics on validation set
        if self._val_logits and self.hparams.num_labels > 1:
            from src.utils.metrics import compute_all_metrics

            all_logits = torch.cat(self._val_logits, dim=0)
            all_labels = torch.cat(self._val_labels, dim=0)
            metrics = compute_all_metrics(all_logits, all_labels)
            for k, v in metrics.items():
                self.log(f"val/{k}", v, on_step=False, on_epoch=True)

        self._val_logits.clear()
        self._val_labels.clear()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets, logits = self.model_step(batch)
        self.test_loss(loss)
        self.test_metric(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"test/{self.metric_name}", self.test_metric, on_step=False, on_epoch=True, prog_bar=True)

        # Collect logits for calibration
        self._test_logits.append(logits.detach().cpu())
        self._test_labels.append(targets.detach().cpu())

    def on_test_epoch_end(self) -> None:
        """Compute full calibration metrics and plot reliability diagrams."""
        if not self._test_logits or self.hparams.num_labels <= 1:
            return

        from src.utils.metrics import (
            compute_all_metrics,
            format_metrics_table,
            plot_reliability_diagram,
        )

        all_logits = torch.cat(self._test_logits, dim=0)
        all_labels = torch.cat(self._test_labels, dim=0)

        # Scalar metrics
        metrics = compute_all_metrics(all_logits, all_labels)
        for k, v in metrics.items():
            self.log(f"test/{k}", v, on_step=False, on_epoch=True)

        self.print(f"\n{'='*60}")
        self.print("Test Calibration Metrics (TransDiff-style):")
        self.print(format_metrics_table(metrics))
        self.print(f"{'='*60}\n")

        # Reliability diagram + bin-count plot
        save_dir = "."
        if self.trainer and self.trainer.log_dir:
            save_dir = self.trainer.log_dir
        plot_reliability_diagram(
            all_logits, all_labels, save_dir=save_dir, split="test"
        )
        self.print(f"Reliability diagram saved to {save_dir}/calibration/")

        self._test_logits.clear()
        self._test_labels.clear()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and optional LR scheduler."""
        # Only optimize trainable parameters
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        config = {"optimizer": optimizer}

        if self.hparams.warmup_steps > 0:
            from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=self.hparams.warmup_steps,
            )
            # Estimate total steps
            total_steps = self.trainer.estimated_stepping_batches
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps - self.hparams.warmup_steps,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.hparams.warmup_steps],
            )
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }

        return config
