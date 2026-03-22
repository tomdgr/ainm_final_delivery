"""Evaluate ensemble strategies using cached predictions (instant).

Run cache_predictions.py first, then sweep params here in seconds.

Usage:
  uv run python scripts/eval_cached.py --cache cached_preds.pkl
  uv run python scripts/eval_cached.py --cache cached_preds.pkl --temperature 0.5 --neighbor-voting
  uv run python scripts/eval_cached.py --cache cached_preds.pkl --sweep
"""
import argparse
import json
import pickle
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from nm_ai_image.detection.evaluate import evaluate_predictions

NC = 356


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter + 1e-8)


def soft_vote_merge(model_results, n_models, conf=0.005, wbf_iou=0.55, temperature=1.0):
    all_dets = []
    for midx, preds in enumerate(model_results):
        boxes, scores, probs = preds["boxes"], preds["scores"], preds["probs"]
        mask = scores > conf
        for i in np.where(mask)[0]:
            all_dets.append((boxes[i], scores[i], probs[i], midx))
    if not all_dets:
        return [], [], [], []

    all_dets.sort(key=lambda x: -x[1])
    clusters = []
    used = [False] * len(all_dets)
    for i in range(len(all_dets)):
        if used[i]:
            continue
        cluster = [all_dets[i]]
        used[i] = True
        models_in = {all_dets[i][3]}
        for j in range(i + 1, len(all_dets)):
            if used[j] or all_dets[j][3] in models_in:
                continue
            iou = compute_iou(all_dets[i][0], all_dets[j][0])
            if iou >= wbf_iou:
                cluster.append(all_dets[j])
                used[j] = True
                models_in.add(all_dets[j][3])
        clusters.append(cluster)

    final_boxes, final_scores, final_labels, final_probs = [], [], [], []
    for cluster in clusters:
        total_score = sum(d[1] for d in cluster)
        avg_box = np.zeros(4)
        avg_prob = np.zeros(NC)
        for b, s, p, m in cluster:
            avg_box += b * s
            if temperature != 1.0:
                p = p / temperature
                p = p - p.max()
                ep = np.exp(p)
                p = ep / ep.sum()
            avg_prob += p
        avg_box /= total_score
        avg_prob /= len(cluster)
        cls_id = int(avg_prob.argmax())
        avg_score = total_score / len(cluster)
        agreement_boost = len(cluster) / n_models
        final_score = avg_score * (0.7 + 0.3 * agreement_boost)
        final_boxes.append(avg_box)
        final_scores.append(float(final_score))
        final_labels.append(cls_id)
        final_probs.append(avg_prob)
    return final_boxes, final_scores, final_labels, final_probs


def wbf_merge(model_results, n_models, conf=0.005, wbf_iou=0.7):
    """Standard WBF using ensemble_boxes library."""
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError:
        print("pip install ensemble-boxes")
        return [], [], []

    boxes_list, scores_list, labels_list = [], [], []
    for preds in model_results:
        boxes, scores, probs = preds["boxes"], preds["scores"], preds["probs"]
        mask = scores > conf
        if mask.sum() == 0:
            boxes_list.append(np.zeros((0, 4)))
            scores_list.append(np.zeros(0))
            labels_list.append(np.zeros(0, dtype=int))
            continue
        b = boxes[mask].copy()
        # Get image dims from first call context — normalize to [0,1]
        # caller must normalize before calling
        boxes_list.append(b)
        scores_list.append(scores[mask])
        labels_list.append(probs[mask].argmax(axis=1).astype(float))

    if all(len(b) == 0 for b in boxes_list):
        return [], [], []

    merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        iou_thr=wbf_iou, skip_box_thr=conf
    )
    return merged_boxes, merged_scores, merged_labels.astype(int)


def neighbor_class_vote(predictions, y_threshold=50, score_threshold=0.8, prob_ratio=2.0):
    """Correct outlier classifications using shelf row context."""
    per_image = {}
    for i, p in enumerate(predictions):
        per_image.setdefault(p["image_id"], []).append((i, p))

    flips = 0
    for image_id, items in per_image.items():
        if len(items) < 3:
            continue
        # Group by y-center into rows
        rows = []
        used = set()
        sorted_items = sorted(items, key=lambda x: x[1]["bbox"][1])
        for idx, (i, p) in enumerate(sorted_items):
            if i in used:
                continue
            row = [(i, p)]
            used.add(i)
            y_center = p["bbox"][1] + p["bbox"][3] / 2
            for idx2, (j, q) in enumerate(sorted_items[idx + 1:]):
                if j in used:
                    continue
                qy = q["bbox"][1] + q["bbox"][3] / 2
                if abs(qy - y_center) < y_threshold:
                    row.append((j, q))
                    used.add(j)
            if len(row) >= 3:
                rows.append(row)

        for row in rows:
            label_counts = {}
            for i, p in row:
                label_counts[p["category_id"]] = label_counts.get(p["category_id"], 0) + 1
            for i, p in row:
                if p["score"] > score_threshold:
                    continue
                if label_counts.get(p["category_id"], 0) > 1:
                    continue
                if "_top2_cls" not in p:
                    continue
                alt_cls = p["_top2_cls"]
                if label_counts.get(alt_cls, 0) >= 2 and p["_top1_prob"] / max(p["_top2_prob"], 1e-8) < prob_ratio:
                    predictions[i]["category_id"] = alt_cls
                    label_counts[p["category_id"]] -= 1
                    label_counts[alt_cls] = label_counts.get(alt_cls, 0) + 1
                    flips += 1
    if flips:
        print(f"  Neighbor voting: {flips} flips")
    return predictions


def evaluate_config(cache, coco_json, method="softvote", conf=0.005, nms_iou=0.7,
                    wbf_iou=0.55, temperature=1.0, neighbor_voting=False):
    n_models = len(list(cache.values())[0]["models"])
    predictions = []

    for image_id, data in cache.items():
        w, h = data["width"], data["height"]
        model_results = data["models"]

        if method == "wbf":
            # Normalize boxes to [0,1] for WBF
            norm_results = []
            for preds in model_results:
                boxes = preds["boxes"].copy()
                if len(boxes) > 0:
                    boxes[:, [0, 2]] /= w
                    boxes[:, [1, 3]] /= h
                    boxes = np.clip(boxes, 0, 1)
                norm_results.append({"boxes": boxes, "scores": preds["scores"], "probs": preds["probs"]})
            merged_boxes, merged_scores, merged_labels = wbf_merge(norm_results, n_models, conf, wbf_iou)
            for box, score, label in zip(merged_boxes, merged_scores, merged_labels):
                x1, y1, x2, y2 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
                bw, bh = x2 - x1, y2 - y1
                predictions.append({
                    "image_id": image_id, "category_id": int(label),
                    "bbox": [round(float(x1), 1), round(float(y1), 1), round(float(bw), 1), round(float(bh), 1)],
                    "score": round(float(score), 4)
                })
        else:
            final_boxes, final_scores, final_labels, final_probs = soft_vote_merge(
                model_results, n_models, conf, wbf_iou, temperature
            )
            for box, score, label, prob in zip(final_boxes, final_scores, final_labels, final_probs):
                x1, y1, x2, y2 = box
                bw, bh = x2 - x1, y2 - y1
                p = {
                    "image_id": image_id, "category_id": label,
                    "bbox": [round(float(x1), 1), round(float(y1), 1), round(float(bw), 1), round(float(bh), 1)],
                    "score": round(float(score), 4)
                }
                if neighbor_voting:
                    sorted_cls = prob.argsort()[::-1]
                    p["_top2_cls"] = int(sorted_cls[1])
                    p["_top1_prob"] = float(prob[sorted_cls[0]])
                    p["_top2_prob"] = float(prob[sorted_cls[1]])
                predictions.append(p)

    if neighbor_voting:
        predictions = neighbor_class_vote(predictions)
        for p in predictions:
            p.pop("_top2_cls", None)
            p.pop("_top1_prob", None)
            p.pop("_top2_prob", None)

    eval_result = evaluate_predictions(predictions, coco_json)
    return {
        "competition_score": eval_result.competition_score,
        "detection_map50": eval_result.detection_map50,
        "classification_map50": eval_result.classification_map50,
        "num_predictions": eval_result.num_predictions,
    }


def run_sweep(cache, coco_json):
    """Sweep key parameters and print results."""
    configs = []

    # Temperature sweep
    for t in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        configs.append({"method": "softvote", "temperature": t, "label": f"softvote_T{t}"})

    # WBF IoU sweep
    for iou in [0.4, 0.5, 0.55, 0.6, 0.7]:
        configs.append({"method": "wbf", "wbf_iou": iou, "label": f"wbf_iou{iou}"})

    # Conf sweep
    for c in [0.001, 0.005, 0.01, 0.02]:
        configs.append({"method": "softvote", "conf": c, "label": f"softvote_conf{c}"})
        configs.append({"method": "wbf", "conf": c, "label": f"wbf_conf{c}"})

    # Neighbor voting
    configs.append({"method": "softvote", "neighbor_voting": True, "label": "softvote_nv"})
    configs.append({"method": "softvote", "temperature": 0.5, "neighbor_voting": True, "label": "softvote_T0.5_nv"})

    results = []
    for cfg in configs:
        label = cfg.pop("label")
        print(f"  {label}...", end=" ", flush=True)
        r = evaluate_config(cache, coco_json, **cfg)
        r["label"] = label
        results.append(r)
        print(f"{r['competition_score']:.4f} (det={r['detection_map50']:.4f} cls={r['classification_map50']:.4f} n={r['num_predictions']})")

    print(f"\n{'='*70}")
    print("RANKED BY SCORE:")
    print(f"{'='*70}")
    for r in sorted(results, key=lambda x: -x["competition_score"]):
        print(f"  {r['competition_score']:.4f}  det={r['detection_map50']:.4f}  cls={r['classification_map50']:.4f}  n={r['num_predictions']:>6}  {r['label']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="cached_preds.pkl")
    parser.add_argument("--coco-json", default="data/raw/coco_dataset/train/annotations.json")
    parser.add_argument("--method", default="softvote", choices=["softvote", "wbf"])
    parser.add_argument("--conf", type=float, default=0.005)
    parser.add_argument("--wbf-iou", type=float, default=0.55)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--neighbor-voting", action="store_true")
    parser.add_argument("--sweep", action="store_true", help="Run full parameter sweep")
    args = parser.parse_args()

    coco_json = str(Path(args.coco_json).resolve())
    print(f"Loading cache: {args.cache}")
    with open(args.cache, "rb") as f:
        cache = pickle.load(f)
    print(f"Loaded {len(cache)} images")

    if args.sweep:
        run_sweep(cache, coco_json)
    else:
        r = evaluate_config(
            cache, coco_json,
            method=args.method, conf=args.conf, wbf_iou=args.wbf_iou,
            temperature=args.temperature, neighbor_voting=args.neighbor_voting,
        )
        print(f"Score: {r['competition_score']:.4f} (det={r['detection_map50']:.4f}, cls={r['classification_map50']:.4f}, preds={r['num_predictions']})")
