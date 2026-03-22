from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
import torch.nn as nn


@dataclass
class TrainingConfig(ABC):
    train_split: float = 0.70
    val_split: float = 0.15
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    loss_fn: Literal["cross_entropy", "focal", "dice", "bce"] = "cross_entropy"
    num_workers: int = 4
    precision: Literal["16-mixed", "32-true", "bf16-mixed"] = "16-mixed"
    early_stopping_patience: int = 20
    early_stopping_monitor: str = "val_loss"
    early_stopping_mode: Literal["min", "max"] = "min"
    checkpoint_filename: str = "best_model"
    checkpoint_save_top_k: int = 1
    checkpoint_monitor: str = "val_f1"
    checkpoint_mode: Literal["min", "max"] = "max"
    warmup_epochs: int = 3
    cosine_t_max: int | None = None
    scheduler_min_lr: float = 1e-6
    output_dir: str = "outputs"
    compute_target: str = "cpu-cluster"
    image_size: int = 224
    num_classes: int = 10

    @property
    def test_split(self) -> float:
        return 1.0 - self.train_split - self.val_split

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.output_dir) / "checkpoints"

    @property
    def predictions_dir(self) -> Path:
        return Path(self.output_dir) / "predictions"

    @property
    def plots_dir(self) -> Path:
        return Path(self.output_dir) / "plots"

    @abstractmethod
    def build_model(self, num_classes: int | None = None) -> nn.Module:
        pass

    def get_callbacks(self) -> list:
        callbacks = self.get_checkpoint_callbacks()
        callbacks.append(self.get_early_stopping_callback())
        callbacks.append(LearningRateMonitor(logging_interval="step"))
        return callbacks

    def get_checkpoint_callbacks(self) -> list[ModelCheckpoint]:
        checkpoint_configs = [
            ("best_loss", "val_loss", "min"),
            ("best_f1", "val_f1", "max"),
        ]
        return [
            ModelCheckpoint(
                dirpath=str(self.checkpoint_dir),
                filename=filename,
                save_top_k=1,
                monitor=monitor,
                mode=mode,
            )
            for filename, monitor, mode in checkpoint_configs
        ]

    def get_early_stopping_callback(self) -> EarlyStopping:
        return EarlyStopping(
            patience=self.early_stopping_patience,
            mode=self.early_stopping_mode,
            monitor=self.early_stopping_monitor,
        )
