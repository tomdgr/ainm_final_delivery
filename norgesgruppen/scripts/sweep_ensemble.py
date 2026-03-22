"""Sweep WBF and conf/NMS parameters for ONNX ensemble on training set.

Usage:
  uv run python scripts/sweep_ensemble.py \
    --onnx weights/yolov8x_640_best.onnx weights/yolov8m_640_diverse_best.onnx
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


def detect_raw(sess, input_name, img, imgsz=640, nc=356, conf_thr=0.001):
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
    return np.stack([x1, y1, x2, y2], axis=1), max_sc[mask], cls_ids[mask]


def per_class_nms(boxes, scores, cls_ids, nms_iou=0.5):
    final_b, final_s, final_c = [], [], []
    for cid in np.unique(cls_ids):
        m = cls_ids == cid
        keep = nms(boxes[m], scores[m], nms_iou)
        for k in keep:
            final_b.append(boxes[m][k])
            final_s.append(float(scores[m][k]))
            final_c.append(int(cid))
    return final_b, final_s, final_c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", nargs="+", required=True, help="ONNX model paths")
    parser.add_argument("--coco-json", default="data/raw/coco_dataset/train/annotations.json")
    parser.add_argument("--image-dir", default="data/raw/coco_dataset/train/images")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output-dir", default="outputs/sweep_ensemble")
    args = parser.parse_args()

    import torch
    from ensemble_boxes import weighted_boxes_fusion

    device = "cuda" if torch.cuda.is_available() else "cpu"
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]

    sessions = []
    input_names = []
    for p in args.onnx:
        sess = ort.InferenceSession(p, providers=providers)
        sessions.append(sess)
        input_names.append(sess.get_inputs()[0].name)
    print(f"Loaded {len(sessions)} models on {device}")

    image_dir = Path(args.image_dir)
    images = sorted(f for f in image_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    print(f"Processing {len(images)} images...")

    # Run all models, cache per-model NMS'd detections (at low conf, high nms_iou)
    # For WBF we need per-model results, not merged
    all_model_results = [[] for _ in sessions]  # model_idx -> [(image_id, w, h, boxes_norm, scores, labels)]

    t0 = time.time()
    for idx, img_path in enumerate(images):
        if idx % 50 == 0:
            print(f"  {idx}/{len(images)}", flush=True)
        image_id = int(img_path.stem.split("_")[-1])
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        for mi, (sess, inp_name) in enumerate(zip(sessions, input_names)):
            boxes, scores, cls_ids = detect_raw(sess, inp_name, img, args.imgsz, conf_thr=0.001)
            fb, fs, fc = per_class_nms(boxes, scores, cls_ids, nms_iou=0.7)
            # Normalize boxes for WBF
            boxes_norm = [[b[0]/w, b[1]/h, b[2]/w, b[3]/h] for b in fb]
            all_model_results[mi].append((image_id, w, h, boxes_norm, fs, fc))

    print(f"Detection done in {time.time()-t0:.1f}s")

    from nm_ai_image.detection.evaluate import evaluate_predictions

    # Sweep WBF parameters
    conf_values = [0.001, 0.005, 0.01]
    wbf_iou_values = [0.4, 0.5, 0.55, 0.6, 0.7]

    all_results = []
    best_score = 0
    best_config = {}

    for conf in conf_values:
        for wbf_iou in wbf_iou_values:
            predictions = []
            n_images = len(all_model_results[0])
            for i in range(n_images):
                image_id = all_model_results[0][i][0]
                w = all_model_results[0][i][1]
                h = all_model_results[0][i][2]

                ens_boxes, ens_scores, ens_labels = [], [], []
                for mi in range(len(sessions)):
                    _, _, _, boxes_norm, scores, labels = all_model_results[mi][i]
                    # Filter by conf
                    filt = [(b, s, l) for b, s, l in zip(boxes_norm, scores, labels) if s > conf]
                    if filt:
                        fb, fs, fl = zip(*filt)
                        ens_boxes.append(list(fb))
                        ens_scores.append(list(fs))
                        ens_labels.append(list(fl))
                    else:
                        ens_boxes.append([])
                        ens_scores.append([])
                        ens_labels.append([])

                if any(len(b) > 0 for b in ens_boxes):
                    fb, fs, fl = weighted_boxes_fusion(
                        ens_boxes, ens_scores, ens_labels,
                        iou_thr=wbf_iou, skip_box_thr=conf
                    )
                    for box, score, label in zip(fb, fs, fl):
                        x1, y1, x2, y2 = box
                        predictions.append({
                            "image_id": image_id, "category_id": int(label),
                            "bbox": [round(x1*w, 1), round(y1*h, 1), round((x2-x1)*w, 1), round((y2-y1)*h, 1)],
                            "score": round(float(score), 4),
                        })

            result = evaluate_predictions(predictions, args.coco_json)
            config = {"conf": conf, "wbf_iou": wbf_iou}
            entry = {
                **config,
                "score": result.competition_score,
                "det_map": result.detection_map50,
                "cls_map": result.classification_map50,
                "n_preds": len(predictions),
            }
            all_results.append(entry)
            if result.competition_score > best_score:
                best_score = result.competition_score
                best_config = config
            print(f"  conf={conf:.3f} wbf_iou={wbf_iou:.2f} → "
                  f"score={result.competition_score:.4f} "
                  f"(det={result.detection_map50:.4f} cls={result.classification_map50:.4f}) "
                  f"n={len(predictions)}")

    print(f"\n{'='*60}")
    print(f"BEST ENSEMBLE: score={best_score:.4f} config={best_config}")
    print(f"BEST SINGLE:   score=0.9055 (conf=0.001, nms=0.7)")
    print(f"{'='*60}")

    all_results.sort(key=lambda x: x["score"], reverse=True)
    print("\nTop 5 ensemble configurations:")
    for i, r in enumerate(all_results[:5]):
        print(f"  {i+1}. score={r['score']:.4f} det={r['det_map']:.4f} cls={r['cls_map']:.4f} "
              f"conf={r['conf']:.3f} wbf_iou={r['wbf_iou']:.2f} n={r['n_preds']}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "sweep_results.json", "w") as f:
        json.dump({"best_config": best_config, "best_score": best_score, "all_results": all_results}, f, indent=2)


if __name__ == "__main__":
    main()
