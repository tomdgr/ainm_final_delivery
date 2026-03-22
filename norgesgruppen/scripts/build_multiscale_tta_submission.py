"""Build multi-scale + TTA + WBF ensemble submission.

3 models x 2 scales (640, 800) x 2 flips (orig, hflip) = 12 inference passes per image.
"""
import shutil
import zipfile
from pathlib import Path

ONNX_PATHS = [
    Path("weights/yolov8x_640_fulldata_best_fp16_dynamic.onnx"),
    Path("weights/yolov8x_640_seed999_best_fp16_dyn.onnx"),
    Path("weights/yolov8l_640_highaug_fp16_dyn.onnx"),
]

SUBMISSION_NAME = "wbf_2x1l_multiscale_tta"
OUTPUT_DIR = Path("submissions")

RUN_PY = r'''import argparse
import json
# sys is BLOCKED in sandbox
import time
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort
from ensemble_boxes import weighted_boxes_fusion

# --- Config ---
ONNX_FILES = [
    "yolov8x_640_fulldata_best_fp16_dynamic.onnx",
    "yolov8x_640_seed999_best_fp16_dyn.onnx",
    "yolov8l_640_highaug_fp16_dyn.onnx",
]
SCALES = [640, 800]
NC = 356
CONF_THR = 0.005
NMS_IOU_THR = 0.7
WBF_IOU_THR = 0.6
WBF_SKIP_BOX_THR = 0.01

# --- Preprocessing ---
def preprocess(img, sz=640):
    w, h = img.size
    scale = min(sz / w, sz / h)
    nw, nh = int(w * scale), int(h * scale)
    img_r = img.resize((nw, nh), Image.BILINEAR)
    pad = np.full((sz, sz, 3), 114, dtype=np.uint8)
    px, py = (sz - nw) // 2, (sz - nh) // 2
    pad[py:py + nh, px:px + nw] = np.array(img_r)
    blob = np.transpose(pad.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, ...]
    return blob, scale, px, py


# --- NMS (per-class) ---
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
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        order = order[np.where(iou <= thr)[0] + 1]
    return keep


def apply_nms(boxes, scores, cls_ids, nms_thr=NMS_IOU_THR):
    """Per-class NMS."""
    if len(boxes) == 0:
        return [], [], []
    fb, fs, fc = [], [], []
    for cid in np.unique(cls_ids):
        m = cls_ids == cid
        keep = nms(boxes[m], scores[m], nms_thr)
        for k in keep:
            fb.append(boxes[m][k])
            fs.append(float(scores[m][k]))
            fc.append(int(cid))
    return fb, fs, fc


# --- Detection at a given scale ---
def detect_onnx(sess, input_name, img, ow, oh, sz=640):
    blob, scale, px, py = preprocess(img, sz=sz)
    out = sess.run(None, {input_name: blob})[0]
    if out.ndim == 3:
        out = out[0].T if out.shape[1] == (4 + NC) else out[0]
    cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    cls_scores = out[:, 4:]
    cls_ids = cls_scores.argmax(axis=1)
    max_sc = cls_scores.max(axis=1)
    mask = max_sc > CONF_THR
    x1 = np.clip((cx - bw / 2 - px) / scale, 0, ow)[mask]
    y1 = np.clip((cy - bh / 2 - py) / scale, 0, oh)[mask]
    x2 = np.clip((cx + bw / 2 - px) / scale, 0, ow)[mask]
    y2 = np.clip((cy + bh / 2 - py) / scale, 0, oh)[mask]
    cls_ids = cls_ids[mask]
    max_sc = max_sc[mask]
    boxes = np.stack([x1, y1, x2, y2], axis=1) if len(x1) > 0 else np.zeros((0, 4))
    fb, fs, fc = apply_nms(boxes, max_sc, cls_ids)
    return fb, fs, fc


def main():
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Determine script directory for finding ONNX files
    script_dir = Path(__file__).resolve().parent

    # Load models
    print("Loading models...")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sessions = []
    input_names = []
    for name in ONNX_FILES:
        model_path = str(script_dir / name)
        sess = ort.InferenceSession(model_path, providers=providers)
        sessions.append(sess)
        input_names.append(sess.get_inputs()[0].name)
    print(f"Loaded {len(sessions)} ONNX models in {time.time()-t0:.1f}s")

    # Collect image paths
    img_paths = sorted(
        [p for p in Path(args.input).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )
    print(f"Processing {len(img_paths)} images, {len(sessions)} models x {len(SCALES)} scales x 2 flips = {len(sessions)*len(SCALES)*2} passes per image")

    try:
        from tqdm import tqdm
        img_iter = tqdm(img_paths, desc="Inference")
    except ImportError:
        img_iter = img_paths

    predictions = []
    for img_path in img_iter:
        image_id = int(img_path.stem.split("_")[-1])
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Prepare flipped image once
        img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Collect all model+scale+flip results as separate "models" for WBF
        all_boxes, all_scores, all_labels = [], [], []

        for sess, inp_name in zip(sessions, input_names):
            for sz in SCALES:
                for use_flip in [False, True]:
                    cur_img = img_flip if use_flip else img
                    det_b, det_s, det_c = detect_onnx(sess, inp_name, cur_img, w, h, sz=sz)

                    boxes_norm, scores_list, labels_list = [], [], []
                    for b, s, c in zip(det_b, det_s, det_c):
                        bx = b.copy()
                        if use_flip:
                            # Mirror x-coordinates back
                            x1_orig = w - bx[2]
                            x2_orig = w - bx[0]
                            bx[0] = x1_orig
                            bx[2] = x2_orig
                        # Normalize to [0, 1] for WBF
                        boxes_norm.append([
                            max(0, bx[0] / w),
                            max(0, bx[1] / h),
                            min(1, bx[2] / w),
                            min(1, bx[3] / h),
                        ])
                        scores_list.append(s)
                        labels_list.append(c)

                    all_boxes.append(boxes_norm)
                    all_scores.append(scores_list)
                    all_labels.append(labels_list)

        # WBF fusion across all 12 "model" passes
        if any(len(b) > 0 for b in all_boxes):
            try:
                fb, fs, fl = weighted_boxes_fusion(
                    all_boxes, all_scores, all_labels,
                    iou_thr=WBF_IOU_THR,
                    skip_box_thr=WBF_SKIP_BOX_THR,
                    conf_type='absent_model_aware_avg',
                )
            except (TypeError, ValueError):
                fb, fs, fl = weighted_boxes_fusion(
                    all_boxes, all_scores, all_labels,
                    iou_thr=WBF_IOU_THR,
                    skip_box_thr=WBF_SKIP_BOX_THR,
                    conf_type='avg',
                )
            for box, score, label in zip(fb, fs, fl):
                x1, y1, x2, y2 = box
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [
                        round(float(x1 * w), 1),
                        round(float(y1 * h), 1),
                        round(float((x2 - x1) * w), 1),
                        round(float((y2 - y1) * h), 1),
                    ],
                    "score": round(float(score), 4),
                })

    elapsed = time.time() - t0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)
    print(f"Wrote {len(predictions)} predictions in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
'''


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build_dir = OUTPUT_DIR / f"{SUBMISSION_NAME}_build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir()

    # Copy ONNX files
    for p in ONNX_PATHS:
        print(f"Copying {p.name} ({p.stat().st_size / 1024 / 1024:.0f} MB)")
        shutil.copy2(p, build_dir / p.name)

    # Write run.py
    (build_dir / "run.py").write_text(RUN_PY)
    print("Wrote run.py")

    # Build ZIP
    zip_path = OUTPUT_DIR / f"{SUBMISSION_NAME}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(build_dir.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(build_dir))
    shutil.rmtree(build_dir)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\nBuilt: {zip_path} ({size_mb:.1f} MB)")
    if size_mb > 420:
        print("WARNING: ZIP exceeds 420 MB limit!")
    else:
        print(f"OK: {420 - size_mb:.1f} MB under limit")


if __name__ == "__main__":
    main()
