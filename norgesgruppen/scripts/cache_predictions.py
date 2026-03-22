"""Cache raw per-model predictions for fast post-processing sweeps.

Runs each ONNX model once on all images and saves raw detections
(boxes, scores, class probabilities) per image. Then any ensemble
strategy (WBF, soft vote, temperature, neighbor voting) can be
evaluated in seconds without re-running inference.

Usage:
  uv run python scripts/cache_predictions.py \
    --onnx weights/yolov8x_640_fulldata_best_fp16.onnx \
           weights/yolov8x_640_s123_best_fp16.onnx \
           weights/yolov8x_640_seed999_best_fp16.onnx \
    --output cached_preds.pkl
"""
import argparse
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
import onnxruntime as ort

NC = 356
CONF_THR = 0.001  # very low to keep all candidates
NMS_IOU = 0.7


def preprocess(img, sz):
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
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        order = order[np.where(iou <= thr)[0] + 1]
    return keep


def detect_raw(sess, input_name, img, ow, oh, sz=640):
    blob, scale, px, py = preprocess(img, sz)
    out = sess.run(None, {input_name: blob})[0]
    if out.ndim == 3:
        out = out[0].T if out.shape[1] == (4 + NC) else out[0]
    cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    cls_scores = out[:, 4:]
    max_sc = cls_scores.max(axis=1)
    mask = max_sc > CONF_THR
    x1 = np.clip((cx - bw / 2 - px) / scale, 0, ow)[mask]
    y1 = np.clip((cy - bh / 2 - py) / scale, 0, oh)[mask]
    x2 = np.clip((cx + bw / 2 - px) / scale, 0, ow)[mask]
    y2 = np.clip((cy + bh / 2 - py) / scale, 0, oh)[mask]
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    scores = max_sc[mask]
    probs = cls_scores[mask]
    cls_ids = probs.argmax(axis=1)
    # Per-class NMS
    fb, fs, fp = [], [], []
    for cid in np.unique(cls_ids):
        m = cls_ids == cid
        keep = nms(boxes[m], scores[m], NMS_IOU)
        for k in keep:
            fb.append(boxes[m][k])
            fs.append(float(scores[m][k]))
            fp.append(probs[m][k])
    if fb:
        return np.array(fb), np.array(fs), np.array(fp)
    return np.zeros((0, 4)), np.zeros(0), np.zeros((0, NC))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", nargs="+", required=True)
    parser.add_argument("--image-dir", default="data/raw/coco_dataset/train/images")
    parser.add_argument("--output", default="cached_preds.pkl")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    sessions = []
    input_names = []
    for onnx_path in args.onnx:
        sess = ort.InferenceSession(onnx_path, providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
        sessions.append(sess)
        input_names.append(sess.get_inputs()[0].name)
    print(f"Loaded {len(sessions)} models")

    image_dir = Path(args.image_dir)
    img_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])

    from tqdm import tqdm
    cache = {}
    for img_path in tqdm(img_paths, desc="Caching predictions"):
        image_id = int(img_path.stem.split("_")[-1])
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        model_preds = []
        for sess, inp_name in zip(sessions, input_names):
            boxes, scores, probs = detect_raw(sess, inp_name, img, w, h, args.imgsz)
            model_preds.append({"boxes": boxes, "scores": scores, "probs": probs})
        cache[image_id] = {"width": w, "height": h, "models": model_preds}

    with open(args.output, "wb") as f:
        pickle.dump(cache, f)
    print(f"Cached {len(cache)} images x {len(sessions)} models -> {args.output}")
    size_mb = Path(args.output).stat().st_size / 1024 / 1024
    print(f"Cache size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
