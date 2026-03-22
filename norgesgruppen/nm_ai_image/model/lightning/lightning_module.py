import logging

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score

from nm_ai_image.config.task import TaskConfig, TaskType
from nm_ai_image.evaluation.metrics import calculate_all_metrics

logger = logging.getLogger(__name__)


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: (B, C, H, W), targets: (B, H, W)
        num_classes = inputs.shape[1]
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets.long(), num_classes).permute(0, 3, 1, 2).float()

        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


LOSS_FNS = {
    "cross_entropy": lambda num_classes, multi_label: torch.nn.CrossEntropyLoss(),
    "focal": lambda num_classes, multi_label: FocalLoss(),
    "dice": lambda num_classes, multi_label: DiceLoss(),
    "bce": lambda num_classes, multi_label: torch.nn.BCEWithLogitsLoss(),
}


class ImageTask(L.LightningModule):
    def __init__(self, model: torch.nn.Module, config: TaskConfig):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "config"])
        self.model = model
        self.config = config

        self.task_type = config.task_type

        if self.task_type == TaskType.CLASSIFICATION:
            self.loss_fn = LOSS_FNS[config.loss_fn](config.num_classes, config.multi_label)
            task = "binary" if config.num_classes == 2 else "multiclass"
            self.train_acc = Accuracy(task=task, num_classes=config.num_classes)
            self.val_acc = Accuracy(task=task, num_classes=config.num_classes)
            self.val_f1 = F1Score(task=task, num_classes=config.num_classes, average="macro")
        elif self.task_type == TaskType.SEMANTIC_SEGMENTATION:
            self.ce_loss = torch.nn.CrossEntropyLoss()
            self.dice_loss = DiceLoss()
        # Detection models compute their own loss

        self.val_outputs = []
        self.test_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if self.task_type in (TaskType.OBJECT_DETECTION, TaskType.INSTANCE_SEGMENTATION):
            images, targets = batch
            loss_dict = self.model(images, targets)
            loss = sum(loss_dict.values())
            self.log("train_loss", loss, prog_bar=True, on_epoch=True)
            return loss

        x, y = batch

        if self.task_type == TaskType.CLASSIFICATION:
            y_hat = self(x)
            loss = self.loss_fn(y_hat, y)
            preds = torch.argmax(y_hat, dim=1)
            self.train_acc(preds, y)
            self.log("train_loss", loss, prog_bar=True, on_epoch=True)
            self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True)

        elif self.task_type == TaskType.SEMANTIC_SEGMENTATION:
            y_hat = self(x)
            ce = self.ce_loss(y_hat, y)
            dice = self.dice_loss(y_hat, y)
            loss = ce + dice
            self.log("train_loss", loss, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.task_type in (TaskType.OBJECT_DETECTION, TaskType.INSTANCE_SEGMENTATION):
            images, targets = batch
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(images)
            self.val_outputs.append({"preds": outputs, "targets": targets})
            # Log a dummy val_loss for checkpointing
            self.log("val_loss", 0.0, prog_bar=True)
            return

        x, y = batch

        if self.task_type == TaskType.CLASSIFICATION:
            y_hat = self(x)
            loss = self.loss_fn(y_hat, y)
            preds = torch.argmax(y_hat, dim=1)
            self.val_acc(preds, y)
            self.val_f1(preds, y)
            self.log("val_loss", loss, prog_bar=True, on_epoch=True)
            self.log("val_acc", self.val_acc, prog_bar=True, on_epoch=True)
            self.log("val_f1", self.val_f1, prog_bar=True, on_epoch=True)

            self.val_outputs.append({
                "y_true": y.detach().cpu().numpy(),
                "y_pred": preds.detach().cpu().numpy(),
            })

        elif self.task_type == TaskType.SEMANTIC_SEGMENTATION:
            y_hat = self(x)
            ce = self.ce_loss(y_hat, y)
            dice = self.dice_loss(y_hat, y)
            loss = ce + dice
            self.log("val_loss", loss, prog_bar=True, on_epoch=True)

            preds = torch.argmax(y_hat, dim=1)
            self.val_outputs.append({
                "y_true": y.detach().cpu().numpy(),
                "y_pred": preds.detach().cpu().numpy(),
            })

    def on_validation_epoch_end(self):
        if not self.val_outputs:
            return

        if self.task_type == TaskType.CLASSIFICATION:
            y_true = np.concatenate([x["y_true"] for x in self.val_outputs])
            y_pred = np.concatenate([x["y_pred"] for x in self.val_outputs])
            m = calculate_all_metrics("classification", y_pred, y_true, self.config.num_classes)
            self.log("val_precision", m["precision"])
            self.log("val_recall", m["recall"])

        elif self.task_type == TaskType.SEMANTIC_SEGMENTATION:
            y_true = np.concatenate([x["y_true"] for x in self.val_outputs])
            y_pred = np.concatenate([x["y_pred"] for x in self.val_outputs])
            m = calculate_all_metrics("semantic_segmentation", y_pred, y_true, self.config.num_classes)
            self.log("val_mIoU", m["mIoU"], prog_bar=True)
            self.log("val_dice", m["dice"])
            self.log("val_f1", m.get("dice", 0.0))

        self.val_outputs.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=1e-2)
        t_max = self.config.cosine_t_max or self.config.epochs
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=t_max, eta_min=self.config.scheduler_min_lr
        )
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.01, total_iters=self.config.warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[self.config.warmup_epochs]
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def test_step(self, batch, batch_idx):
        if self.task_type in (TaskType.OBJECT_DETECTION, TaskType.INSTANCE_SEGMENTATION):
            images, targets = batch
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(images)
            self.test_outputs.append({"preds": outputs, "targets": targets})
            return

        x, y = batch
        y_hat = self(x)

        if self.task_type == TaskType.CLASSIFICATION:
            loss = self.loss_fn(y_hat, y)
            preds = torch.argmax(y_hat, dim=1)
            self.log("test_loss", loss)
            self.test_outputs.append({
                "y_true": y.detach().cpu().numpy(),
                "y_pred": preds.detach().cpu().numpy(),
            })

        elif self.task_type == TaskType.SEMANTIC_SEGMENTATION:
            preds = torch.argmax(y_hat, dim=1)
            self.test_outputs.append({
                "y_true": y.detach().cpu().numpy(),
                "y_pred": preds.detach().cpu().numpy(),
            })

    def on_test_epoch_end(self):
        if not self.test_outputs:
            return

        task_str = self.config.task_type.value

        if self.task_type == TaskType.CLASSIFICATION:
            y_true = np.concatenate([x["y_true"] for x in self.test_outputs])
            y_pred = np.concatenate([x["y_pred"] for x in self.test_outputs])
            m = calculate_all_metrics(task_str, y_pred, y_true, self.config.num_classes)
            for key in ["accuracy", "precision", "recall", "f1"]:
                self.log(f"test_{key}", m.get(key, 0))
            logger.info(
                "TEST — Acc: %.4f | F1: %.4f | Prec: %.4f | Rec: %.4f",
                m["accuracy"], m["f1"], m["precision"], m["recall"],
            )

        elif self.task_type == TaskType.SEMANTIC_SEGMENTATION:
            y_true = np.concatenate([x["y_true"] for x in self.test_outputs])
            y_pred = np.concatenate([x["y_pred"] for x in self.test_outputs])
            m = calculate_all_metrics(task_str, y_pred, y_true, self.config.num_classes)
            for key in ["pixel_accuracy", "mIoU", "dice"]:
                self.log(f"test_{key}", m.get(key, 0))
            logger.info("TEST — mIoU: %.4f | Dice: %.4f", m["mIoU"], m["dice"])

        self.test_outputs.clear()
