"""Evaluation for NorgesGruppen detection task.

Computes the competition score (0.7 * detection_mAP + 0.3 * classification_mAP)
and generates diagnostic plots.
"""
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    detection_map50: float = 0.0
    classification_map50: float = 0.0
    competition_score: float = 0.0
    num_predictions: int = 0
    num_ground_truth: int = 0
    per_class_ap: dict[int, float] = field(default_factory=dict)
    per_class_det_ap: dict[int, float] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"Competition Score: {self.competition_score:.4f}\n"
            f"  Detection mAP@0.5:       {self.detection_map50:.4f} (× 0.7 = {self.detection_map50 * 0.7:.4f})\n"
            f"  Classification mAP@0.5:  {self.classification_map50:.4f} (× 0.3 = {self.classification_map50 * 0.3:.4f})\n"
            f"  Predictions: {self.num_predictions}, Ground truth: {self.num_ground_truth}"
        )


def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Average Precision using 101-point interpolation (COCO style)."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    recall_points = np.linspace(0, 1, 101)
    precision_at_recall = np.zeros_like(recall_points)
    for i, r in enumerate(recall_points):
        idx = np.where(mrec >= r)[0]
        if len(idx) > 0:
            precision_at_recall[i] = mpre[idx[0]]
    return float(precision_at_recall.mean())


def _compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """IoU between two COCO-format boxes [x, y, w, h]."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _compute_map_at_iou(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_threshold: float = 0.5,
    match_category: bool = False,
) -> tuple[float, dict[int, float]]:
    """Compute mAP@IoU threshold.

    If match_category=False, only box overlap matters (detection mAP).
    If match_category=True, box overlap AND correct category required (classification mAP).
    """
    # Group GT by image
    gt_by_image = defaultdict(list)
    for gt in ground_truths:
        gt_by_image[gt["image_id"]].append(gt)

    # All categories
    all_cats = set()
    for gt in ground_truths:
        all_cats.add(gt["category_id"])

    if not match_category:
        # Detection: treat all as single class
        all_cats = {0}

    # Sort predictions by score descending
    preds_sorted = sorted(predictions, key=lambda p: p["score"], reverse=True)

    # Per-category AP
    per_class_ap = {}

    for cat_id in all_cats:
        if match_category:
            cat_preds = [p for p in preds_sorted if p["category_id"] == cat_id]
            cat_gts_by_image = {}
            for img_id, gts in gt_by_image.items():
                cat_gts = [g for g in gts if g["category_id"] == cat_id]
                if cat_gts:
                    cat_gts_by_image[img_id] = cat_gts
        else:
            cat_preds = preds_sorted
            cat_gts_by_image = dict(gt_by_image)

        n_gt = sum(len(v) for v in cat_gts_by_image.values())
        if n_gt == 0:
            continue

        # Track which GTs are matched
        matched = {img_id: [False] * len(gts) for img_id, gts in cat_gts_by_image.items()}

        tp = np.zeros(len(cat_preds))
        fp = np.zeros(len(cat_preds))

        for i, pred in enumerate(cat_preds):
            img_gts = cat_gts_by_image.get(pred["image_id"], [])
            best_iou = 0.0
            best_j = -1
            for j, gt in enumerate(img_gts):
                iou = _compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= iou_threshold and best_j >= 0 and not matched.get(pred["image_id"], [False])[best_j]:
                tp[i] = 1
                matched[pred["image_id"]][best_j] = True
            else:
                fp[i] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / n_gt
        precisions = cum_tp / (cum_tp + cum_fp)

        ap = _compute_ap(recalls, precisions)
        per_class_ap[cat_id] = ap

    mean_ap = float(np.mean(list(per_class_ap.values()))) if per_class_ap else 0.0
    return mean_ap, per_class_ap


def evaluate_predictions(
    predictions: list[dict],
    coco_json: str | Path,
) -> EvalResult:
    """Evaluate predictions against COCO ground truth using competition scoring."""
    with open(coco_json) as f:
        coco = json.load(f)

    ground_truths = []
    images = {img["id"]: img for img in coco["images"]}
    for ann in coco["annotations"]:
        ground_truths.append({
            "image_id": ann["image_id"],
            "category_id": ann["category_id"],
            "bbox": ann["bbox"],
        })

    det_map, det_per_class = _compute_map_at_iou(predictions, ground_truths, iou_threshold=0.5, match_category=False)
    cls_map, cls_per_class = _compute_map_at_iou(predictions, ground_truths, iou_threshold=0.5, match_category=True)
    score = 0.7 * det_map + 0.3 * cls_map

    result = EvalResult(
        detection_map50=det_map,
        classification_map50=cls_map,
        competition_score=score,
        num_predictions=len(predictions),
        num_ground_truth=len(ground_truths),
        per_class_ap=cls_per_class,
        per_class_det_ap=det_per_class,
    )
    logger.info("\n%s", result.summary())
    return result


def evaluate_model(
    weights: str | Path,
    coco_json: str | Path,
    image_dir: str | Path,
    output_dir: str | Path = "outputs/eval",
    imgsz: int = 640,
) -> EvalResult:
    """Run a full model evaluation: inference + scoring + plots."""
    from nm_ai_image.detection.inference import Detector

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run inference
    detector = Detector(weights, imgsz=imgsz, conf=0.01, iou=0.6, tta=False)
    detections = detector.predict_dir(image_dir)
    predictions = [{"image_id": d.image_id, "category_id": d.category_id, "bbox": d.bbox, "score": d.score} for d in detections]

    # Save predictions
    with open(output_dir / "predictions.json", "w") as f:
        json.dump(predictions, f)

    # Evaluate
    result = evaluate_predictions(predictions, coco_json)

    # Save metrics
    metrics = {
        "competition_score": result.competition_score,
        "detection_map50": result.detection_map50,
        "classification_map50": result.classification_map50,
        "num_predictions": result.num_predictions,
        "num_ground_truth": result.num_ground_truth,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Generate plots
    _save_plots(result, coco_json, output_dir)

    return result


def _save_plots(result: EvalResult, coco_json: str | Path, output_dir: Path):
    """Generate and save evaluation plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with open(coco_json) as f:
            coco = json.load(f)
        catid_to_name = {c["id"]: c["name"] for c in coco["categories"]}

        # Per-class classification AP (top/bottom 20)
        if result.per_class_ap:
            sorted_ap = sorted(result.per_class_ap.items(), key=lambda x: x[1], reverse=True)

            # Top 20
            top = sorted_ap[:20]
            fig, ax = plt.subplots(figsize=(12, 6))
            cats = [catid_to_name.get(c, str(c))[:30] for c, _ in top]
            vals = [v for _, v in top]
            ax.barh(range(len(cats)), vals)
            ax.set_yticks(range(len(cats)))
            ax.set_yticklabels(cats, fontsize=8)
            ax.set_xlabel("AP@0.5")
            ax.set_title("Top 20 Classes by Classification AP@0.5")
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(output_dir / "top20_class_ap.png", dpi=150)
            plt.close()

            # Bottom 20
            bottom = sorted_ap[-20:]
            fig, ax = plt.subplots(figsize=(12, 6))
            cats = [catid_to_name.get(c, str(c))[:30] for c, _ in bottom]
            vals = [v for _, v in bottom]
            ax.barh(range(len(cats)), vals)
            ax.set_yticks(range(len(cats)))
            ax.set_yticklabels(cats, fontsize=8)
            ax.set_xlabel("AP@0.5")
            ax.set_title("Bottom 20 Classes by Classification AP@0.5")
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(output_dir / "bottom20_class_ap.png", dpi=150)
            plt.close()

            # AP distribution histogram
            fig, ax = plt.subplots(figsize=(10, 5))
            all_aps = list(result.per_class_ap.values())
            ax.hist(all_aps, bins=50, edgecolor="black", alpha=0.7)
            ax.axvline(np.mean(all_aps), color="red", linestyle="--", label=f"Mean: {np.mean(all_aps):.3f}")
            ax.set_xlabel("AP@0.5")
            ax.set_ylabel("Number of Classes")
            ax.set_title("Classification AP@0.5 Distribution")
            ax.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "ap_distribution.png", dpi=150)
            plt.close()

        logger.info("Plots saved to %s", output_dir)
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
