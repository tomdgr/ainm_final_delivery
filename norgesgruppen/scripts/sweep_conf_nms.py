"""Sweep confidence threshold and NMS IoU on training set using ONNX model.

Finds optimal conf/iou settings to maximize competition score.

Usage:
  uv run python scripts/sweep_conf_nms.py --onnx weights/yolov8x_640_best.onnx
  uv run python scripts/sweep_conf_nms.py --onnx weights/yolov8x_640_best.onnx --conf-values 0.001 0.005 0.01 0.02 0.05
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort


def preprocess(img, sz=640):
    w, h = img.size
    scale = min(sz / w, sz / h)
    nw, nh = int(w * scale), int(h * scale)
    img_r = img.resize((nw, nh), Image.BILINEAR)
    pad = np.full((sz, sz, 3), 114, dtype=np.uint8)
    px, py = (sz - nw) // 2, (sz - nh) // 2
    pad[py:py + nh, px:px + nw] = np.array(img_r)
    return np.transpose(pad.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, ...], scale, px, py


def nms(boxes, scores, thr=0.5):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= thr)[0] + 1]
    return keep


def soft_nms(boxes, scores, sigma=0.5, score_thr=0.001):
    """Soft-NMS: decay overlapping scores instead of removing them."""
    if len(boxes) == 0:
        return [], []
    boxes = boxes.copy()
    scores = scores.copy()
    idxs = np.arange(len(boxes))
    keep_boxes, keep_scores = [], []

    while len(scores) > 0:
        max_idx = scores.argmax()
        keep_boxes.append(boxes[max_idx])
        keep_scores.append(scores[max_idx])

        # Compute IoU with remaining
        x1 = np.maximum(boxes[max_idx, 0], boxes[:, 0])
        y1 = np.maximum(boxes[max_idx, 1], boxes[:, 1])
        x2 = np.minimum(boxes[max_idx, 2], boxes[:, 2])
        y2 = np.minimum(boxes[max_idx, 3], boxes[:, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_a = (boxes[max_idx, 2] - boxes[max_idx, 0]) * (boxes[max_idx, 3] - boxes[max_idx, 1])
        area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iou = inter / (area_a + area_b - inter + 1e-6)

        # Gaussian decay
        scores = scores * np.exp(-iou ** 2 / sigma)

        # Remove selected and low-scoring
        mask = np.ones(len(scores), dtype=bool)
        mask[max_idx] = False
        mask &= scores > score_thr
        boxes = boxes[mask]
        scores = scores[mask]

    return keep_boxes, keep_scores


def detect_raw(sess, input_name, img, imgsz=640, nc=356, conf_thr=0.001):
    """Run ONNX detection, return raw boxes/scores/classes before NMS."""
    ow, oh = img.size
    blob, scale, px, py = preprocess(img, imgsz)
    out = sess.run(None, {input_name: blob})[0]
    if out.ndim == 3:
        out = out[0].T if out.shape[1] == (4 + nc) else out[0]
    cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    cls_scores = out[:, 4:]
    cls_ids = cls_scores.argmax(axis=1)
    max_sc = cls_scores.max(axis=1)
    mask = max_sc > conf_thr
    x1 = np.clip((cx - bw / 2 - px) / scale, 0, ow)[mask]
    y1 = np.clip((cy - bh / 2 - py) / scale, 0, oh)[mask]
    x2 = np.clip((cx + bw / 2 - px) / scale, 0, ow)[mask]
    y2 = np.clip((cy + bh / 2 - py) / scale, 0, oh)[mask]
    cls_ids = cls_ids[mask]
    max_sc = max_sc[mask]
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    return boxes, max_sc, cls_ids


def apply_nms(boxes, scores, cls_ids, nms_iou=0.5, use_soft_nms=False, soft_sigma=0.5):
    """Apply per-class NMS (hard or soft) and return filtered results."""
    final_boxes, final_scores, final_cls = [], [], []
    for cid in np.unique(cls_ids):
        m = cls_ids == cid
        if use_soft_nms:
            kept_boxes, kept_scores = soft_nms(boxes[m], scores[m], sigma=soft_sigma)
            for b, s in zip(kept_boxes, kept_scores):
                final_boxes.append(b)
                final_scores.append(s)
                final_cls.append(int(cid))
        else:
            keep = nms(boxes[m], scores[m], nms_iou)
            for k in keep:
                final_boxes.append(boxes[m][k])
                final_scores.append(float(scores[m][k]))
                final_cls.append(int(cid))
    return final_boxes, final_scores, final_cls


def make_predictions(raw_results, conf_thr, nms_iou=0.5, use_soft_nms=False, soft_sigma=0.5):
    """From cached raw detections, apply conf filter + NMS and produce predictions."""
    predictions = []
    for image_id, boxes, scores, cls_ids in raw_results:
        # Apply conf threshold
        mask = scores > conf_thr
        if not mask.any():
            continue
        b, s, c = boxes[mask], scores[mask], cls_ids[mask]
        fb, fs, fc = apply_nms(b, s, c, nms_iou, use_soft_nms, soft_sigma)
        for box, score, cid in zip(fb, fs, fc):
            x1, y1, x2, y2 = box
            predictions.append({
                "image_id": image_id,
                "category_id": int(cid),
                "bbox": [round(float(x1), 1), round(float(y1), 1), round(float(x2 - x1), 1), round(float(y2 - y1), 1)],
                "score": round(float(score), 4),
            })
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="ONNX model path")
    parser.add_argument("--coco-json", default="data/raw/coco_dataset/train/annotations.json")
    parser.add_argument("--image-dir", default="data/raw/coco_dataset/train/images")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf-values", nargs="+", type=float,
                        default=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1])
    parser.add_argument("--nms-values", nargs="+", type=float,
                        default=[0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--soft-nms", action="store_true", help="Also test soft-NMS")
    parser.add_argument("--soft-sigma-values", nargs="+", type=float,
                        default=[0.3, 0.5, 0.75, 1.0])
    parser.add_argument("--imgsz-values", nargs="+", type=int, default=None,
                        help="Also test different inference resolutions (e.g. 640 800 960)")
    parser.add_argument("--output-dir", default="outputs/sweep")
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(args.onnx, providers=providers)
    input_name = sess.get_inputs()[0].name
    print(f"Device: {device}, Model: {args.onnx}")

    image_dir = Path(args.image_dir)
    images = sorted(f for f in image_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    print(f"Processing {len(images)} images...")

    imgsz_values = args.imgsz_values or [args.imgsz]

    from nm_ai_image.detection.evaluate import evaluate_predictions

    all_results = []
    best_score = 0
    best_config = {}

    for imgsz in imgsz_values:
        # Run detection at very low conf to cache all raw detections
        print(f"\n--- Running detection at imgsz={imgsz} (conf=0.001) ---")
        min_conf = min(args.conf_values)
        raw_results = []
        t0 = time.time()
        for idx, img_path in enumerate(images):
            if idx % 50 == 0:
                print(f"  {idx}/{len(images)}", flush=True)
            image_id = int(img_path.stem.split("_")[-1])
            img = Image.open(img_path).convert("RGB")
            boxes, scores, cls_ids = detect_raw(sess, input_name, img, imgsz=imgsz, conf_thr=min_conf)
            raw_results.append((image_id, boxes, scores, cls_ids))
        det_time = time.time() - t0
        total_raw = sum(len(r[1]) for r in raw_results)
        print(f"  Detection done: {total_raw} raw detections in {det_time:.1f}s")

        # Sweep hard NMS
        for conf in args.conf_values:
            for nms_iou in args.nms_values:
                preds = make_predictions(raw_results, conf, nms_iou, use_soft_nms=False)
                result = evaluate_predictions(preds, args.coco_json)
                config = {"imgsz": imgsz, "conf": conf, "nms_iou": nms_iou, "nms_type": "hard"}
                entry = {
                    **config,
                    "score": result.competition_score,
                    "det_map": result.detection_map50,
                    "cls_map": result.classification_map50,
                    "n_preds": len(preds),
                }
                all_results.append(entry)
                if result.competition_score > best_score:
                    best_score = result.competition_score
                    best_config = config
                print(f"  conf={conf:.3f} nms={nms_iou:.1f} hard → "
                      f"score={result.competition_score:.4f} "
                      f"(det={result.detection_map50:.4f} cls={result.classification_map50:.4f}) "
                      f"n={len(preds)}")

        # Sweep soft-NMS
        if args.soft_nms:
            for conf in args.conf_values:
                for sigma in args.soft_sigma_values:
                    preds = make_predictions(raw_results, conf, use_soft_nms=True, soft_sigma=sigma)
                    result = evaluate_predictions(preds, args.coco_json)
                    config = {"imgsz": imgsz, "conf": conf, "soft_sigma": sigma, "nms_type": "soft"}
                    entry = {
                        **config,
                        "score": result.competition_score,
                        "det_map": result.detection_map50,
                        "cls_map": result.classification_map50,
                        "n_preds": len(preds),
                    }
                    all_results.append(entry)
                    if result.competition_score > best_score:
                        best_score = result.competition_score
                        best_config = config
                    print(f"  conf={conf:.3f} soft_σ={sigma:.2f} → "
                          f"score={result.competition_score:.4f} "
                          f"(det={result.detection_map50:.4f} cls={result.classification_map50:.4f}) "
                          f"n={len(preds)}")

    # Summary
    print(f"\n{'='*60}")
    print(f"BEST: score={best_score:.4f} config={best_config}")
    print(f"{'='*60}")

    # Sort and show top 10
    all_results.sort(key=lambda x: x["score"], reverse=True)
    print("\nTop 10 configurations:")
    for i, r in enumerate(all_results[:10]):
        print(f"  {i+1}. score={r['score']:.4f} det={r['det_map']:.4f} cls={r['cls_map']:.4f} "
              f"conf={r['conf']:.3f} nms={r.get('nms_iou', '-')} type={r['nms_type']} "
              f"imgsz={r['imgsz']} n={r['n_preds']}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "sweep_results.json", "w") as f:
        json.dump({"best_config": best_config, "best_score": best_score, "all_results": all_results}, f, indent=2)
    print(f"\nResults saved to {output_dir / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
