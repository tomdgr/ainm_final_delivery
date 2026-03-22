"""Evaluate gallery-based re-ranking on cached predictions.

Phase 1: Embed all detection crops once (slow, ~10 min on CPU)
Phase 2: Sweep re-ranking parameters instantly using cached embeddings

Usage:
  uv run python scripts/eval_rerank.py --gallery gallery_convnext.npz
  uv run python scripts/eval_rerank.py --gallery gallery_convnext.npz --sweep
"""
import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
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


def fuse_all_images(cache, conf=0.005, wbf_iou=0.55, temperature=1.0):
    """Fuse all cached predictions. Returns per-image detection lists."""
    n_models = len(list(cache.values())[0]["models"])
    per_image = {}
    for image_id, data in cache.items():
        model_results = data["models"]
        boxes, scores, labels, probs = soft_vote_merge(
            model_results, n_models, conf, wbf_iou, temperature
        )
        per_image[image_id] = {
            "boxes": boxes, "scores": scores, "labels": labels, "probs": probs,
            "width": data["width"], "height": data["height"],
        }
    return per_image


def embed_all_crops(fused, gallery_path, coco_json, min_score=0.05, max_score=1.0):
    """Embed detection crops that might be reranked. Returns per-image crop embedding arrays."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(gallery_path, allow_pickle=True)
    model_name = str(data["model_name"])

    logger.info("Loading timm model: %s on %s", model_name, device)
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model = model.to(device)
    model.eval()

    config = resolve_data_config(model.pretrained_cfg)
    config["input_size"] = (3, 224, 224)
    transform = create_transform(**config, is_training=False)

    with torch.no_grad():
        feat_dim = model(torch.randn(1, 3, 224, 224, device=device)).shape[1]

    with open(coco_json) as f:
        coco = json.load(f)
    img_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    image_dir = ROOT / "data" / "raw" / "coco_dataset" / "train" / "images"

    # Collect crops only for detections in the reranking score range
    crop_list = []  # (image_id, det_idx, PIL.Image)
    skipped = 0
    for image_id, dets in fused.items():
        # Check if any detection needs embedding
        needs_embed = False
        for i, score in enumerate(dets["scores"]):
            if min_score <= score <= max_score:
                needs_embed = True
                break
        if not needs_embed:
            skipped += 1
            continue

        img_file = img_id_to_file.get(image_id)
        if not img_file:
            continue
        img_path = image_dir / img_file
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        for i, (box, score) in enumerate(zip(dets["boxes"], dets["scores"])):
            if score < min_score or score > max_score:
                continue
            x1, y1, x2, y2 = box
            pad_x = (x2 - x1) * 0.05
            pad_y = (y2 - y1) * 0.05
            crop = img.crop((
                max(0, x1 - pad_x), max(0, y1 - pad_y),
                min(img.width, x2 + pad_x), min(img.height, y2 + pad_y),
            ))
            if crop.width >= 5 and crop.height >= 5:
                crop_list.append((image_id, i, crop))
            else:
                crop_list.append((image_id, i, Image.new("RGB", (32, 32))))

    logger.info("Embedding %d detection crops (skipped %d images with no candidates)...",
                len(crop_list), skipped)
    t0 = time.time()

    # Embed in batches
    batch_size = 64
    all_embs = np.zeros((len(crop_list), feat_dim), dtype=np.float32)
    for start in range(0, len(crop_list), batch_size):
        batch = crop_list[start:start + batch_size]
        imgs = [item[2] for item in batch]
        with torch.no_grad():
            tensors = torch.stack([transform(img) for img in imgs]).to(device)
            feats = F.normalize(model(tensors), dim=1)
            all_embs[start:start + len(batch)] = feats.cpu().numpy()
        if (start // batch_size + 1) % 50 == 0:
            elapsed = time.time() - t0
            done = start + len(batch)
            rate = done / elapsed
            remaining = (len(crop_list) - done) / rate
            logger.info("  Embedded %d/%d crops (%.0f/s, ~%.0fs left)",
                        done, len(crop_list), rate, remaining)

    elapsed = time.time() - t0
    logger.info("Embedded %d crops in %.1fs (%.0f/s)", len(crop_list), elapsed, len(crop_list) / elapsed)

    # Organize by image_id
    crop_embs = {}  # image_id -> (n_dets, feat_dim) array
    for idx, (image_id, det_idx, _) in enumerate(crop_list):
        if image_id not in crop_embs:
            n_dets = len(fused[image_id]["boxes"])
            crop_embs[image_id] = np.zeros((n_dets, feat_dim), dtype=np.float32)
        crop_embs[image_id][det_idx] = all_embs[idx]

    return crop_embs


def apply_reranking(fused, crop_embs, gallery_path,
                    method="knn", rerank_threshold=0.7, prob_gap_threshold=0.3,
                    gallery_override_threshold=0.3, margin=0.05, knn_k=5):
    """Apply re-ranking using precomputed crop embeddings. Fast (no model inference)."""
    data = np.load(gallery_path, allow_pickle=True)
    gallery_mean = data["gallery_mean"]
    all_gallery_embs = data["all_embs"]
    all_gallery_labels = data["all_labels"]

    override_count = 0
    rerank_count = 0
    # Track what happens
    changes = []

    for image_id, dets in fused.items():
        if image_id not in crop_embs:
            continue
        embs = crop_embs[image_id]

        for i in range(len(dets["boxes"])):
            score = dets["scores"][i]
            prob = dets["probs"][i]
            old_cls = dets["labels"][i]

            # Decide whether to rerank
            if score >= rerank_threshold:
                continue
            sorted_probs = np.sort(prob)[::-1]
            if sorted_probs[0] - sorted_probs[1] >= prob_gap_threshold:
                continue

            rerank_count += 1
            crop_emb = embs[i:i+1]  # (1, feat_dim)

            if method == "knn":
                sims = (crop_emb @ all_gallery_embs.T)[0]  # (n_gallery,)
                top_k_idx = np.argpartition(sims, -knn_k)[-knn_k:]
                top_k_idx = top_k_idx[np.argsort(sims[top_k_idx])[::-1]]

                class_votes = {}
                for idx in top_k_idx:
                    cls = int(all_gallery_labels[idx])
                    class_votes[cls] = class_votes.get(cls, 0.0) + float(sims[idx])

                best_cls = max(class_votes, key=class_votes.get)
                best_score = class_votes[best_cls] / knn_k
                old_cls_score = class_votes.get(old_cls, 0.0) / knn_k

                if best_cls != old_cls and best_score > gallery_override_threshold and best_score - old_cls_score > margin:
                    dets["labels"][i] = best_cls
                    override_count += 1
                    changes.append((image_id, i, old_cls, best_cls, best_score, old_cls_score))

            else:  # mean
                sims = (crop_emb @ gallery_mean.T)[0]
                best_cls = int(np.argmax(sims))
                best_sim = float(sims[best_cls])
                old_sim = float(sims[old_cls])

                if best_cls != old_cls and best_sim > gallery_override_threshold and best_sim - old_sim > margin:
                    dets["labels"][i] = best_cls
                    override_count += 1
                    changes.append((image_id, i, old_cls, best_cls, best_sim, old_sim))

    return rerank_count, override_count, changes


def evaluate_fused(fused, coco_json):
    """Evaluate fused predictions."""
    from nm_ai_image.detection.evaluate import evaluate_predictions
    predictions = []
    for image_id, dets in fused.items():
        for box, score, label in zip(dets["boxes"], dets["scores"], dets["labels"]):
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            predictions.append({
                "image_id": image_id, "category_id": int(label),
                "bbox": [round(float(x1), 1), round(float(y1), 1), round(float(bw), 1), round(float(bh), 1)],
                "score": round(float(score), 4),
            })
    return evaluate_predictions(predictions, coco_json)


def run_sweep(cache, coco_json, gallery_path):
    """Run full parameter sweep."""
    import copy

    # Phase 1: Fuse predictions and embed crops (slow, one-time)
    print("Phase 1: Fusing predictions...")
    fused = fuse_all_images(cache)
    n_dets = sum(len(d["boxes"]) for d in fused.values())
    print(f"  {n_dets} total detections across {len(fused)} images")

    print("\nPhase 2: Embedding candidate crops (one-time, may take a few min on CPU)...")
    crop_embs = embed_all_crops(fused, gallery_path, coco_json, min_score=0.3)
    print(f"  Embedded crops for {len(crop_embs)} images")

    # Baseline
    print("\n" + "=" * 70)
    baseline = evaluate_fused(fused, coco_json)
    print(f"BASELINE: {baseline.competition_score:.4f} (det={baseline.detection_map50:.4f}, cls={baseline.classification_map50:.4f})")
    print("=" * 70)

    # Phase 3: Sweep parameters (fast, no model inference)
    print("\nPhase 3: Sweeping parameters...")
    configs = [
        # Method comparison
        {"method": "mean", "rerank_threshold": 0.7, "margin": 0.05, "label": "mean_default"},
        {"method": "knn", "rerank_threshold": 0.7, "margin": 0.05, "label": "knn_default"},

        # Rerank threshold sweep
        {"method": "knn", "rerank_threshold": 0.3, "margin": 0.05, "label": "knn_thresh0.3"},
        {"method": "knn", "rerank_threshold": 0.5, "margin": 0.05, "label": "knn_thresh0.5"},
        {"method": "knn", "rerank_threshold": 0.7, "margin": 0.05, "label": "knn_thresh0.7"},
        {"method": "knn", "rerank_threshold": 0.9, "margin": 0.05, "label": "knn_thresh0.9"},
        {"method": "knn", "rerank_threshold": 1.0, "margin": 0.05, "label": "knn_thresh1.0"},

        # Prob gap threshold sweep
        {"method": "knn", "rerank_threshold": 0.7, "prob_gap_threshold": 0.1, "margin": 0.05, "label": "knn_gap0.1"},
        {"method": "knn", "rerank_threshold": 0.7, "prob_gap_threshold": 0.2, "margin": 0.05, "label": "knn_gap0.2"},
        {"method": "knn", "rerank_threshold": 0.7, "prob_gap_threshold": 0.5, "margin": 0.05, "label": "knn_gap0.5"},
        {"method": "knn", "rerank_threshold": 1.0, "prob_gap_threshold": 0.5, "margin": 0.05, "label": "knn_all_gap0.5"},

        # Margin sweep
        {"method": "knn", "rerank_threshold": 0.7, "margin": 0.0, "label": "knn_margin0.0"},
        {"method": "knn", "rerank_threshold": 0.7, "margin": 0.02, "label": "knn_margin0.02"},
        {"method": "knn", "rerank_threshold": 0.7, "margin": 0.05, "label": "knn_margin0.05"},
        {"method": "knn", "rerank_threshold": 0.7, "margin": 0.1, "label": "knn_margin0.1"},
        {"method": "knn", "rerank_threshold": 0.7, "margin": 0.15, "label": "knn_margin0.15"},
        {"method": "knn", "rerank_threshold": 0.7, "margin": 0.2, "label": "knn_margin0.2"},

        # Override threshold sweep
        {"method": "knn", "rerank_threshold": 0.7, "gallery_override_threshold": 0.1, "margin": 0.05, "label": "knn_ot0.1"},
        {"method": "knn", "rerank_threshold": 0.7, "gallery_override_threshold": 0.2, "margin": 0.05, "label": "knn_ot0.2"},
        {"method": "knn", "rerank_threshold": 0.7, "gallery_override_threshold": 0.4, "margin": 0.05, "label": "knn_ot0.4"},
        {"method": "knn", "rerank_threshold": 0.7, "gallery_override_threshold": 0.5, "margin": 0.05, "label": "knn_ot0.5"},

        # k sweep
        {"method": "knn", "rerank_threshold": 0.7, "knn_k": 3, "margin": 0.05, "label": "knn_k3"},
        {"method": "knn", "rerank_threshold": 0.7, "knn_k": 5, "margin": 0.05, "label": "knn_k5"},
        {"method": "knn", "rerank_threshold": 0.7, "knn_k": 10, "margin": 0.05, "label": "knn_k10"},
        {"method": "knn", "rerank_threshold": 0.7, "knn_k": 15, "margin": 0.05, "label": "knn_k15"},
        {"method": "knn", "rerank_threshold": 0.7, "knn_k": 20, "margin": 0.05, "label": "knn_k20"},

        # Aggressive configs
        {"method": "knn", "rerank_threshold": 1.0, "gallery_override_threshold": 0.2,
         "margin": 0.0, "knn_k": 10, "label": "knn_aggressive"},
        {"method": "knn", "rerank_threshold": 1.0, "gallery_override_threshold": 0.3,
         "margin": 0.02, "knn_k": 10, "label": "knn_semi_aggressive"},

        # Conservative configs
        {"method": "knn", "rerank_threshold": 0.5, "gallery_override_threshold": 0.5,
         "margin": 0.15, "knn_k": 10, "label": "knn_conservative"},
        {"method": "knn", "rerank_threshold": 0.5, "gallery_override_threshold": 0.4,
         "margin": 0.1, "knn_k": 10, "label": "knn_moderate"},

        # Mean method variants
        {"method": "mean", "rerank_threshold": 0.7, "margin": 0.0, "label": "mean_margin0.0"},
        {"method": "mean", "rerank_threshold": 0.7, "margin": 0.1, "label": "mean_margin0.1"},
        {"method": "mean", "rerank_threshold": 1.0, "margin": 0.05, "label": "mean_all"},
    ]

    results = []
    for cfg in configs:
        label = cfg.pop("label")
        print(f"  {label}...", end=" ", flush=True)
        test_fused = copy.deepcopy(fused)
        rerank_count, override_count, changes = apply_reranking(
            test_fused, crop_embs, gallery_path, **cfg
        )
        r = evaluate_fused(test_fused, coco_json)
        delta = r.competition_score - baseline.competition_score
        print(f"{r.competition_score:.4f} (det={r.detection_map50:.4f} cls={r.classification_map50:.4f} "
              f"overrides={override_count:>4} delta={delta:+.4f})")
        results.append({
            "label": label,
            "competition_score": r.competition_score,
            "detection_map50": r.detection_map50,
            "classification_map50": r.classification_map50,
            "override_count": override_count,
            "rerank_count": rerank_count,
            "delta": delta,
        })

    print(f"\n{'='*80}")
    print(f"RANKED BY SCORE (baseline={baseline.competition_score:.4f}):")
    print(f"{'='*80}")
    for r in sorted(results, key=lambda x: -x["competition_score"]):
        print(f"  {r['competition_score']:.4f} ({r['delta']:+.4f})  "
              f"det={r['detection_map50']:.4f}  cls={r['classification_map50']:.4f}  "
              f"reranked={r['rerank_count']:>5}  overrides={r['override_count']:>4}  {r['label']}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="cached_preds.pkl")
    parser.add_argument("--gallery", default="gallery_convnext.npz")
    parser.add_argument("--coco-json", default="data/raw/coco_dataset/train/annotations.json")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--method", default="knn")
    parser.add_argument("--rerank-threshold", type=float, default=0.7)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--knn-k", type=int, default=5)
    args = parser.parse_args()

    coco_json = str(Path(args.coco_json).resolve())
    print(f"Loading cache: {args.cache}")
    with open(args.cache, "rb") as f:
        cache = pickle.load(f)
    print(f"Loaded {len(cache)} images")

    if args.sweep:
        run_sweep(cache, coco_json, args.gallery)
    else:
        import copy
        fused = fuse_all_images(cache)
        crop_embs = embed_all_crops(fused, args.gallery, coco_json, min_score=0.3)
        baseline = evaluate_fused(fused, coco_json)
        print(f"Baseline: {baseline.competition_score:.4f} (det={baseline.detection_map50:.4f}, cls={baseline.classification_map50:.4f})")

        test_fused = copy.deepcopy(fused)
        rerank_count, override_count, changes = apply_reranking(
            test_fused, crop_embs, args.gallery,
            method=args.method, rerank_threshold=args.rerank_threshold,
            margin=args.margin, knn_k=args.knn_k,
        )
        result = evaluate_fused(test_fused, coco_json)
        delta = result.competition_score - baseline.competition_score
        print(f"After reranking: {result.competition_score:.4f} ({delta:+.4f}) "
              f"(det={result.detection_map50:.4f}, cls={result.classification_map50:.4f})")
        print(f"Reranked: {rerank_count}, Overridden: {override_count}")
        if changes:
            # Load category names for reporting
            with open(coco_json) as f:
                coco = json.load(f)
            catid_to_name = {c["id"]: c["name"] for c in coco["categories"]}
            print(f"\nSample overrides (first 20):")
            for img_id, det_idx, old_cls, new_cls, new_score, old_score in changes[:20]:
                print(f"  img={img_id} det={det_idx}: {catid_to_name.get(old_cls, old_cls)} -> {catid_to_name.get(new_cls, new_cls)} "
                      f"(gallery: {new_score:.3f} vs {old_score:.3f})")
