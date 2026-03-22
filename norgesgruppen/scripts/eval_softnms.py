"""Evaluate soft-NMS vs hard-NMS on cached predictions before WBF fusion.

Loads cached per-model predictions (which already have hard NMS applied),
then applies soft-NMS within each model's detections to re-weight scores
before WBF ensemble fusion. Compares multiple sigma values.

Usage:
  uv run python scripts/eval_softnms.py --cache cached_preds_arch_diverse.pkl
"""
import argparse
import copy
import pickle
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from nm_ai_image.detection.evaluate import evaluate_predictions

NC = 356


def soft_nms_gaussian(boxes, scores, sigma=0.5, score_threshold=0.001):
    """Soft-NMS with Gaussian penalty.

    For each box i (highest score first), decay overlapping boxes:
        score[j] *= exp(-iou(i,j)^2 / sigma)

    Returns indices of boxes with score > score_threshold and updated scores.
    """
    if len(boxes) == 0:
        return np.array([], dtype=int), np.array([])

    N = len(boxes)
    scores = scores.copy().astype(np.float64)
    indices = np.arange(N)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    for i in range(N):
        # Find the box with max score among remaining (i..N-1)
        max_pos = i + scores[i:].argmax()
        # Swap to position i
        scores[i], scores[max_pos] = scores[max_pos], scores[i]
        indices[i], indices[max_pos] = indices[max_pos], indices[i]
        # Swap box coordinates
        for arr in [x1, y1, x2, y2, areas]:
            arr[i], arr[max_pos] = arr[max_pos], arr[i]

        # Compute IoU of box i with all boxes j > i
        if i + 1 >= N:
            break
        xx1 = np.maximum(x1[i], x1[i + 1:])
        yy1 = np.maximum(y1[i], y1[i + 1:])
        xx2 = np.minimum(x2[i], x2[i + 1:])
        yy2 = np.minimum(y2[i], y2[i + 1:])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[i + 1:] - inter + 1e-8)

        # Gaussian decay
        scores[i + 1:] *= np.exp(-(iou ** 2) / sigma)

    # Filter by score threshold
    keep_mask = scores > score_threshold
    return indices[keep_mask], scores[keep_mask]


def soft_nms_linear(boxes, scores, iou_threshold=0.3, score_threshold=0.001):
    """Soft-NMS with linear penalty.

    For overlapping boxes: score[j] *= (1 - iou) if iou > threshold, else unchanged.
    """
    if len(boxes) == 0:
        return np.array([], dtype=int), np.array([])

    N = len(boxes)
    scores = scores.copy().astype(np.float64)
    indices = np.arange(N)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    for i in range(N):
        max_pos = i + scores[i:].argmax()
        scores[i], scores[max_pos] = scores[max_pos], scores[i]
        indices[i], indices[max_pos] = indices[max_pos], indices[i]
        for arr in [x1, y1, x2, y2, areas]:
            arr[i], arr[max_pos] = arr[max_pos], arr[i]

        if i + 1 >= N:
            break
        xx1 = np.maximum(x1[i], x1[i + 1:])
        yy1 = np.maximum(y1[i], y1[i + 1:])
        xx2 = np.minimum(x2[i], x2[i + 1:])
        yy2 = np.minimum(y2[i], y2[i + 1:])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[i + 1:] - inter + 1e-8)

        decay = np.where(iou > iou_threshold, 1 - iou, 1.0)
        scores[i + 1:] *= decay

    keep_mask = scores > score_threshold
    return indices[keep_mask], scores[keep_mask]


def apply_soft_nms_to_model_preds(preds, method="gaussian", sigma=0.5,
                                   linear_iou=0.3, score_threshold=0.001):
    """Apply per-class soft-NMS to a single model's predictions.

    The cached predictions already had hard NMS applied, so boxes within the
    same class that survived hard NMS may still have overlaps (since hard NMS
    used a high IoU threshold of 0.7). Soft-NMS re-weights their scores.
    """
    boxes, scores, probs = preds["boxes"], preds["scores"], preds["probs"]
    if len(boxes) == 0:
        return preds

    cls_ids = probs.argmax(axis=1)

    new_boxes, new_scores, new_probs = [], [], []
    for cid in np.unique(cls_ids):
        mask = cls_ids == cid
        cls_boxes = boxes[mask]
        cls_scores = scores[mask]
        cls_probs = probs[mask]

        if method == "gaussian":
            keep_idx, new_sc = soft_nms_gaussian(cls_boxes, cls_scores, sigma, score_threshold)
        else:
            keep_idx, new_sc = soft_nms_linear(cls_boxes, cls_scores, linear_iou, score_threshold)

        if len(keep_idx) > 0:
            new_boxes.append(cls_boxes[keep_idx])
            new_scores.append(new_sc)
            new_probs.append(cls_probs[keep_idx])

    if new_boxes:
        return {
            "boxes": np.concatenate(new_boxes),
            "scores": np.concatenate(new_scores).astype(np.float32),
            "probs": np.concatenate(new_probs),
        }
    return {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "probs": np.zeros((0, NC))}


def apply_soft_nms_to_cache(cache, method="gaussian", sigma=0.5,
                             linear_iou=0.3, score_threshold=0.001):
    """Apply soft-NMS to all models in the cache, returning a modified copy."""
    new_cache = {}
    for image_id, data in cache.items():
        new_models = []
        for preds in data["models"]:
            new_preds = apply_soft_nms_to_model_preds(
                preds, method, sigma, linear_iou, score_threshold
            )
            new_models.append(new_preds)
        new_cache[image_id] = {"width": data["width"], "height": data["height"], "models": new_models}
    return new_cache


def evaluate_with_wbf(cache, coco_json, conf=0.005, wbf_iou=0.6):
    """Evaluate cache using standard WBF fusion."""
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError:
        print("pip install ensemble-boxes")
        return None

    n_models = len(list(cache.values())[0]["models"])
    predictions = []

    for image_id, data in cache.items():
        w, h = data["width"], data["height"]
        boxes_list, scores_list, labels_list = [], [], []
        for preds in data["models"]:
            boxes, scores, probs = preds["boxes"], preds["scores"], preds["probs"]
            mask = scores > conf
            if mask.sum() == 0:
                boxes_list.append(np.zeros((0, 4)))
                scores_list.append(np.zeros(0))
                labels_list.append(np.zeros(0, dtype=float))
                continue
            b = boxes[mask].copy()
            b[:, [0, 2]] /= w
            b[:, [1, 3]] /= h
            b = np.clip(b, 0, 1)
            boxes_list.append(b)
            scores_list.append(scores[mask])
            labels_list.append(probs[mask].argmax(axis=1).astype(float))

        if all(len(b) == 0 for b in boxes_list):
            continue

        merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            iou_thr=wbf_iou, skip_box_thr=conf
        )
        for box, score, label in zip(merged_boxes, merged_scores, merged_labels):
            x1, y1, x2, y2 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
            bw, bh = x2 - x1, y2 - y1
            predictions.append({
                "image_id": image_id, "category_id": int(label),
                "bbox": [round(float(x1), 1), round(float(y1), 1), round(float(bw), 1), round(float(bh), 1)],
                "score": round(float(score), 4)
            })

    return evaluate_predictions(predictions, coco_json)


def count_detections(cache, conf=0.005):
    """Count total detections across all models above conf threshold."""
    total = 0
    for data in cache.values():
        for preds in data["models"]:
            total += (preds["scores"] > conf).sum()
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="cached_preds_arch_diverse.pkl")
    parser.add_argument("--coco-json", default="data/raw/coco_dataset/train/annotations.json")
    parser.add_argument("--wbf-iou", type=float, default=0.6)
    parser.add_argument("--conf", type=float, default=0.005)
    args = parser.parse_args()

    coco_json = str(Path(args.coco_json).resolve())
    print(f"Loading cache: {args.cache}")
    with open(args.cache, "rb") as f:
        cache = pickle.load(f)
    n_images = len(cache)
    n_models = len(list(cache.values())[0]["models"])
    print(f"Loaded {n_images} images x {n_models} models")
    print(f"WBF IoU: {args.wbf_iou}, conf: {args.conf}\n")

    configs = []

    # Baseline: no soft-NMS (hard NMS only, as cached)
    configs.append(("hard_NMS (baseline)", None, {}))

    # Gaussian soft-NMS with various sigma
    for sigma in [0.3, 0.5, 0.7, 1.0]:
        configs.append((f"soft_NMS_gauss_s{sigma}", "gaussian", {"sigma": sigma}))

    # Linear soft-NMS with various IoU thresholds
    for iou_thr in [0.3, 0.5]:
        configs.append((f"soft_NMS_linear_t{iou_thr}", "linear", {"linear_iou": iou_thr}))

    results = []
    print(f"{'Config':<30} {'Score':>8} {'Det mAP50':>10} {'Cls mAP50':>10} {'#Preds':>8} {'#Dets(pre-WBF)':>15}")
    print("=" * 85)

    for label, method, params in configs:
        if method is None:
            eval_cache = cache
        else:
            eval_cache = apply_soft_nms_to_cache(cache, method=method, **params)

        n_dets = count_detections(eval_cache, args.conf)
        r = evaluate_with_wbf(eval_cache, coco_json, conf=args.conf, wbf_iou=args.wbf_iou)
        if r is None:
            continue

        row = {
            "label": label,
            "score": r.competition_score,
            "det_map50": r.detection_map50,
            "cls_map50": r.classification_map50,
            "n_preds": r.num_predictions,
            "n_dets": n_dets,
        }
        results.append(row)
        print(f"{label:<30} {row['score']:>8.4f} {row['det_map50']:>10.4f} {row['cls_map50']:>10.4f} {row['n_preds']:>8} {row['n_dets']:>15}")

    # Summary
    print(f"\n{'='*85}")
    print("RANKED BY COMPETITION SCORE:")
    print(f"{'='*85}")
    baseline_score = results[0]["score"] if results else 0
    for r in sorted(results, key=lambda x: -x["score"]):
        delta = r["score"] - baseline_score
        delta_str = f"({delta:+.4f})" if r["label"] != "hard_NMS (baseline)" else "(baseline)"
        print(f"  {r['score']:.4f}  det={r['det_map50']:.4f}  cls={r['cls_map50']:.4f}  "
              f"n={r['n_preds']:>6}  dets={r['n_dets']:>8}  {delta_str:>12}  {r['label']}")


if __name__ == "__main__":
    main()
