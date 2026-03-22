import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    average = "binary" if num_classes == 2 else "macro"

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }

    if y_prob is not None and num_classes == 2:
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        except (ValueError, IndexError):
            pass
    elif y_prob is not None and num_classes > 2:
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except (ValueError, IndexError):
            pass

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    return metrics


def calculate_detection_metrics(
    preds: list[dict],
    targets: list[dict],
) -> dict[str, float]:
    try:
        from torchmetrics.detection import MeanAveragePrecision
        import torch

        metric = MeanAveragePrecision(iou_thresholds=[0.5])
        metric.update(preds, targets)
        result = metric.compute()

        return {
            "mAP_50": float(result["map_50"]),
            "mAP": float(result["map"]),
        }
    except Exception:
        return {"mAP_50": 0.0, "mAP": 0.0}


def calculate_segmentation_metrics(
    pred_masks: np.ndarray,
    true_masks: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    pixel_acc = float(np.mean(pred_masks == true_masks))

    ious = []
    dices = []
    for cls in range(num_classes):
        pred_cls = pred_masks == cls
        true_cls = true_masks == cls

        intersection = np.logical_and(pred_cls, true_cls).sum()
        union = np.logical_or(pred_cls, true_cls).sum()

        iou = float(intersection / union) if union > 0 else 0.0
        ious.append(iou)

        dice_denom = pred_cls.sum() + true_cls.sum()
        dice = float(2 * intersection / dice_denom) if dice_denom > 0 else 0.0
        dices.append(dice)

    return {
        "pixel_accuracy": pixel_acc,
        "mIoU": float(np.mean(ious)),
        "dice": float(np.mean(dices)),
    }


def calculate_all_metrics(
    task_type: str,
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 10,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    if task_type == "classification":
        return calculate_classification_metrics(targets, preds, num_classes, y_prob)
    elif task_type in ("object_detection", "instance_segmentation"):
        return calculate_detection_metrics(preds, targets)
    elif task_type == "semantic_segmentation":
        return calculate_segmentation_metrics(preds, targets, num_classes)
    else:
        return calculate_classification_metrics(targets, preds, num_classes, y_prob)
