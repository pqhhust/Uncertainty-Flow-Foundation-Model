"""GLUE Benchmark DataModule for fine-tuning LLMs.

Supports all GLUE tasks: SST-2, CoLA, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI.
Uses HuggingFace datasets and tokenizers.
"""

from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from datasets import load_dataset


# Task metadata: input keys, number of labels, metric
GLUE_TASK_INFO = {
    "sst2": {
        "keys": ("sentence",),
        "num_labels": 2,
        "metric": "accuracy",
    },
    "cola": {
        "keys": ("sentence",),
        "num_labels": 2,
        "metric": "matthews_correlation",
    },
    "mrpc": {
        "keys": ("sentence1", "sentence2"),
        "num_labels": 2,
        "metric": "f1",
    },
    "stsb": {
        "keys": ("sentence1", "sentence2"),
        "num_labels": 1,  # regression
        "metric": "spearmanr",
    },
    "qqp": {
        "keys": ("question1", "question2"),
        "num_labels": 2,
        "metric": "f1",
    },
    "mnli": {
        "keys": ("premise", "hypothesis"),
        "num_labels": 3,
        "metric": "accuracy",
    },
    "qnli": {
        "keys": ("question", "sentence"),
        "num_labels": 2,
        "metric": "accuracy",
    },
    "rte": {
        "keys": ("sentence1", "sentence2"),
        "num_labels": 2,
        "metric": "accuracy",
    },
    "wnli": {
        "keys": ("sentence1", "sentence2"),
        "num_labels": 2,
        "metric": "accuracy",
    },
}


class GLUEDataModule(LightningDataModule):
    """LightningDataModule for GLUE benchmark tasks.

    Handles tokenization, padding, and data loading for all GLUE tasks.
    Supports decoder-only models (Qwen2, LLaMA, GPT) with left-padding.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        task_name: str = "sst2",
        max_length: int = 128,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        padding_side: str = "left",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.task_name = task_name.lower()
        assert self.task_name in GLUE_TASK_INFO, (
            f"Unknown task: {task_name}. Choose from {list(GLUE_TASK_INFO.keys())}"
        )
        self.task_info = GLUE_TASK_INFO[self.task_name]

        self.tokenizer = None
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.batch_size_per_device = batch_size

    @property
    def num_labels(self) -> int:
        return self.task_info["num_labels"]

    @property
    def metric_name(self) -> str:
        return self.task_info["metric"]

    def _init_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.model_name,
                trust_remote_code=True,
            )
            # For decoder-only models, set padding side to left
            self.tokenizer.padding_side = self.hparams.padding_side
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _tokenize_function(self, examples):
        """Tokenize examples based on task type (single sentence or sentence pair)."""
        keys = self.task_info["keys"]
        if len(keys) == 1:
            texts = examples[keys[0]]
            return self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=self.hparams.max_length,
            )
        else:
            return self.tokenizer(
                examples[keys[0]],
                examples[keys[1]],
                truncation=True,
                padding="max_length",
                max_length=self.hparams.max_length,
            )

    def prepare_data(self) -> None:
        """Download GLUE dataset. Called only on 1 GPU."""
        load_dataset("glue", self.task_name)
        # Pre-download tokenizer
        AutoTokenizer.from_pretrained(
            self.hparams.model_name, trust_remote_code=True
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Load and tokenize data. Called on every GPU."""
        if self.data_train is not None and self.data_val is not None:
            return

        self._init_tokenizer()

        # Adjust batch size for DDP
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by "
                    f"the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        dataset = load_dataset("glue", self.task_name)

        # Tokenize
        tokenized = dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=[
                c
                for c in dataset["train"].column_names
                if c not in ("label", "idx")
            ],
        )

        # Set format
        columns = ["input_ids", "attention_mask", "label"]
        if "token_type_ids" in tokenized["train"].column_names:
            columns.append("token_type_ids")
        tokenized.set_format(type="torch", columns=columns)

        self.data_train = tokenized["train"]

        # Handle MNLI's matched/mismatched validation
        if self.task_name == "mnli":
            self.data_val = tokenized["validation_matched"]
            self.data_test = tokenized["validation_mismatched"]
        else:
            self.data_val = tokenized["validation"]
            # GLUE test sets don't have labels, use validation for test too
            self.data_test = tokenized["validation"]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    # Quick test
    dm = GLUEDataModule(task_name="sst2", batch_size=4, max_length=64)
    dm.prepare_data()
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"attention_mask shape: {batch['attention_mask'].shape}")
    print(f"labels shape: {batch['label'].shape}")
    print(f"num_labels: {dm.num_labels}")
    print(f"metric: {dm.metric_name}")
