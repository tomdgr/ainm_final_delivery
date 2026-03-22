"""Generate predictions from ONNX model on training set for visualization."""
import argparse
import json
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
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        order = order[np.where(iou <= thr)[0] + 1]
    return keep


def detect(sess, input_name, img, conf=0.005, nms_iou=0.7, nc=356):
    ow, oh = img.size
    blob, scale, px, py = preprocess(img)
    out = sess.run(None, {input_name: blob})[0]
    if out.ndim == 3:
        out = out[0].T if out.shape[1] == (4 + nc) else out[0]
    cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    cls_scores = out[:, 4:]
    cls_ids = cls_scores.argmax(axis=1)
    max_sc = cls_scores.max(axis=1)
    mask = max_sc > conf
    x1 = np.clip((cx - bw / 2 - px) / scale, 0, ow)[mask]
    y1 = np.clip((cy - bh / 2 - py) / scale, 0, oh)[mask]
    x2 = np.clip((cx + bw / 2 - px) / scale, 0, ow)[mask]
    y2 = np.clip((cy + bh / 2 - py) / scale, 0, oh)[mask]
    cls_ids = cls_ids[mask]
    max_sc = max_sc[mask]
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    fb, fs, fc = [], [], []
    for cid in np.unique(cls_ids):
        m = cls_ids == cid
        keep = nms(boxes[m], max_sc[m], nms_iou)
        for k in keep:
            fb.append(boxes[m][k].tolist())
            fs.append(float(max_sc[m][k]))
            fc.append(int(cid))
    return fb, fs, fc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--image-dir", default="data/raw/coco_dataset/train/images")
    parser.add_argument("--output", default="outputs/predictions_for_viewer.json")
    parser.add_argument("--conf", type=float, default=0.005)
    args = parser.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    images = sorted(f for f in Path(args.image_dir).iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))

    predictions = []
    for idx, img_path in enumerate(images):
        if idx % 20 == 0:
            print(f"{idx}/{len(images)}", flush=True)
        image_id = int(img_path.stem.split("_")[-1])
        img = Image.open(img_path).convert("RGB")
        fb, fs, fc = detect(sess, input_name, img, conf=args.conf)
        for b, s, c in zip(fb, fs, fc):
            x1, y1, x2, y2 = b
            predictions.append({
                "image_id": image_id,
                "category_id": c,
                "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                "score": round(s, 4),
            })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)
    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
