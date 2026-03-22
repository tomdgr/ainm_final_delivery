"""Build two-stage (detect + classify) ONNX submission.

Stage 1: 2x YOLOv8x ONNX models, multi-scale (640+800) + TTA hflip → WBF fusion
Stage 2: ConvNeXt-Tiny ONNX classifier re-scores uncertain detections

Classification logic:
- YOLO confidence > 0.8 → keep YOLO class (high confidence, don't override)
- YOLO confidence <= 0.8 → blend: argmax(0.5 * yolo_probs + 0.5 * classifier_probs)

Usage:
  uv run python scripts/build_twostage_submission.py \
    --yolo weights/yolov8x_model1.onnx weights/yolov8x_model2.onnx \
    --classifier weights/convnext_tiny_classifier.onnx \
    --name twostage_2x_convnext
"""
import argparse
import shutil
import zipfile
from pathlib import Path


def generate_run_py(
    yolo_names: list[str],
    classifier_name: str,
    nc: int = 356,
    conf: float = 0.005,
    nms_iou: float = 0.7,
    wbf_iou: float = 0.55,
    wbf_skip: float = 0.01,
    cls_threshold: float = 0.8,
    yolo_weight: float = 0.5,
    cls_weight: float = 0.5,
    crop_padding: float = 0.1,
) -> str:
    yolo_repr = repr(yolo_names)

    return f'''import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort

# --- Config ---
YOLO_FILES = {yolo_repr}
CLASSIFIER_FILE = "{classifier_name}"
NC = {nc}
CONF_THR = {conf}
NMS_IOU = {nms_iou}
WBF_IOU = {wbf_iou}
WBF_SKIP = {wbf_skip}
SCALES = [640, 800]
CLS_THRESHOLD = {cls_threshold}
YOLO_WEIGHT = {yolo_weight}
CLS_WEIGHT = {cls_weight}
CROP_PADDING = {crop_padding}

# ImageNet normalization for classifier
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


# --- YOLO Preprocessing ---
def preprocess(img, sz):
    w, h = img.size
    scale = min(sz / w, sz / h)
    nw, nh = int(w * scale), int(h * scale)
    img_r = img.resize((nw, nh), Image.BILINEAR)
    pad = np.full((sz, sz, 3), 114, dtype=np.uint8)
    px, py = (sz - nw) // 2, (sz - nh) // 2
    pad[py:py + nh, px:px + nw] = np.array(img_r)
    return np.transpose(pad.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, ...], scale, px, py


# --- NMS ---
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


# --- Raw YOLO detection at a given scale (returns class probs too) ---
def detect_raw(sess, input_name, img, ow, oh, sz):
    blob, scale, px, py = preprocess(img, sz)
    out = sess.run(None, {{input_name: blob}})[0]
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
    boxes = np.stack([x1, y1, x2, y2], axis=1) if len(x1) > 0 else np.zeros((0, 4))
    scores = max_sc[mask]
    probs = cls_scores[mask]
    # Per-class NMS
    cls_ids = probs.argmax(axis=1)
    fb, fs, fp = [], [], []
    for cid in np.unique(cls_ids):
        m = cls_ids == cid
        keep = nms(boxes[m], scores[m], NMS_IOU)
        for k in keep:
            fb.append(boxes[m][k])
            fs.append(float(scores[m][k]))
            fp.append(probs[m][k])
    return fb, fs, fp


# --- Multi-scale detection ---
def detect_multiscale(sess, input_name, img, ow, oh, scales):
    all_b, all_s, all_p = [], [], []
    for sz in scales:
        fb, fs, fp = detect_raw(sess, input_name, img, ow, oh, sz)
        all_b.extend(fb)
        all_s.extend(fs)
        all_p.extend(fp)
    if len(scales) > 1 and len(all_b) > 0:
        boxes_arr = np.array(all_b)
        scores_arr = np.array(all_s)
        probs_arr = np.array(all_p)
        cls_ids = probs_arr.argmax(axis=1)
        mb, ms, mp = [], [], []
        for cid in np.unique(cls_ids):
            m = cls_ids == cid
            keep = nms(boxes_arr[m], scores_arr[m], NMS_IOU)
            for k in keep:
                mb.append(boxes_arr[m][k])
                ms.append(float(scores_arr[m][k]))
                mp.append(probs_arr[m][k])
        return mb, ms, mp
    return all_b, all_s, all_p


# --- TTA: multi-scale + horizontal flip ---
def detect_with_tta(sess, input_name, img, ow, oh, scales):
    fb1, fs1, fp1 = detect_multiscale(sess, input_name, img, ow, oh, scales)
    img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
    fb2, fs2, fp2 = detect_multiscale(sess, input_name, img_flip, ow, oh, scales)
    # Mirror flipped boxes back
    for i, b in enumerate(fb2):
        x1, y1, x2, y2 = b
        fb2[i] = np.array([ow - x2, y1, ow - x1, y2])
    all_b = fb1 + fb2
    all_s = fs1 + fs2
    all_p = fp1 + fp2
    if len(all_b) == 0:
        return all_b, all_s, all_p
    # NMS to merge orig + flipped
    boxes_arr = np.array(all_b)
    scores_arr = np.array(all_s)
    probs_arr = np.array(all_p)
    cls_ids = probs_arr.argmax(axis=1)
    mb, ms, mp = [], [], []
    for cid in np.unique(cls_ids):
        m = cls_ids == cid
        keep = nms(boxes_arr[m], scores_arr[m], NMS_IOU)
        for k in keep:
            mb.append(boxes_arr[m][k])
            ms.append(float(scores_arr[m][k]))
            mp.append(probs_arr[m][k])
    return mb, ms, mp


# --- Soft-vote merge across models ---
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter + 1e-8)


def soft_vote_merge(model_results, n_models):
    """Merge detections from multiple models using soft class voting."""
    if not model_results or all(len(r[0]) == 0 for r in model_results):
        return [], [], [], []

    all_dets = []
    for midx, (boxes, scores, probs) in enumerate(model_results):
        for b, s, p in zip(boxes, scores, probs):
            all_dets.append((np.array(b), s, np.array(p), midx))

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
        models_in = {{all_dets[i][3]}}
        for j in range(i + 1, len(all_dets)):
            if used[j] or all_dets[j][3] in models_in:
                continue
            iou = compute_iou(all_dets[i][0], all_dets[j][0])
            if iou >= WBF_IOU:
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
            # Softmax the raw class logits for probability averaging
            p_shifted = p - p.max()
            p_exp = np.exp(p_shifted)
            p_soft = p_exp / p_exp.sum()
            avg_prob += p_soft
        avg_box /= total_score
        avg_prob /= len(cluster)
        cls_id = int(avg_prob.argmax())
        avg_score = total_score / len(cluster)
        # Boost score by model agreement
        agreement_boost = len(cluster) / n_models
        final_score = avg_score * (0.7 + 0.3 * agreement_boost)
        final_boxes.append(avg_box)
        final_scores.append(float(final_score))
        final_labels.append(cls_id)
        final_probs.append(avg_prob)

    return final_boxes, final_scores, final_labels, final_probs


# --- Classifier preprocessing ---
def preprocess_crop(crop_img):
    """Resize crop to 224x224, normalize with ImageNet stats. Returns (1, 3, 224, 224)."""
    crop_resized = crop_img.resize((224, 224), Image.BILINEAR)
    arr = np.array(crop_resized, dtype=np.float32) / 255.0  # (224, 224, 3)
    arr = np.transpose(arr, (2, 0, 1))  # (3, 224, 224)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr[np.newaxis, ...].astype(np.float32)  # (1, 3, 224, 224)


def softmax(logits):
    """Compute softmax over last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def crop_detection(img, box_xyxy, padding=CROP_PADDING):
    """Crop a detection box from the original image with percentage padding."""
    w, h = img.size
    x1, y1, x2, y2 = box_xyxy
    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * padding
    pad_y = bh * padding
    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(w, int(x2 + pad_x))
    cy2 = min(h, int(y2 + pad_y))
    return img.crop((cx1, cy1, cx2, cy2))


def main():
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # Load YOLO models
    print("Loading YOLO models...")
    yolo_sessions = []
    yolo_input_names = []
    for name in YOLO_FILES:
        model_path = str(script_dir / name)
        sess = ort.InferenceSession(model_path, providers=providers)
        yolo_sessions.append(sess)
        yolo_input_names.append(sess.get_inputs()[0].name)
    print(f"Loaded {{len(yolo_sessions)}} YOLO models in {{time.time() - t0:.1f}}s")

    # Load classifier model
    print("Loading classifier model...")
    cls_sess = ort.InferenceSession(
        str(script_dir / CLASSIFIER_FILE), providers=providers
    )
    cls_input_name = cls_sess.get_inputs()[0].name
    print(f"Classifier loaded in {{time.time() - t0:.1f}}s")

    # Collect image paths
    img_paths = sorted(
        p for p in Path(args.input).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    n_models = len(yolo_sessions)
    n_passes = n_models * len(SCALES) * 2  # x2 for TTA flip
    print(f"Processing {{len(img_paths)}} images, {{n_passes}} YOLO passes/image + classifier")

    try:
        from tqdm import tqdm
        img_iter = tqdm(img_paths, desc="Two-stage inference")
    except ImportError:
        img_iter = img_paths

    predictions = []
    cls_applied = 0
    cls_changed = 0

    for img_path in img_iter:
        image_id = int(img_path.stem.split("_")[-1])
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Stage 1: Run each YOLO model with multi-scale + TTA
        model_results = []
        for sess, inp_name in zip(yolo_sessions, yolo_input_names):
            fb, fs, fp = detect_with_tta(sess, inp_name, img, w, h, SCALES)
            model_results.append((fb, fs, fp))

        # Soft-vote merge across models
        final_boxes, final_scores, final_labels, final_probs = soft_vote_merge(
            model_results, n_models
        )

        # Stage 2: Classifier re-scoring for uncertain detections
        for box, score, yolo_label, yolo_prob in zip(
            final_boxes, final_scores, final_labels, final_probs
        ):
            x1, y1, x2, y2 = box

            if score <= CLS_THRESHOLD:
                # Crop from original image with padding
                crop = crop_detection(img, box, CROP_PADDING)
                cls_input = preprocess_crop(crop)
                cls_logits = cls_sess.run(None, {{cls_input_name: cls_input}})[0]  # (1, 356)
                cls_probs = softmax(cls_logits)[0]  # (356,)

                # Blend YOLO and classifier probabilities
                blended = YOLO_WEIGHT * yolo_prob + CLS_WEIGHT * cls_probs
                new_label = int(blended.argmax())

                cls_applied += 1
                if new_label != yolo_label:
                    cls_changed += 1
                    yolo_label = new_label

            bw = x2 - x1
            bh = y2 - y1
            predictions.append({{
                "image_id": image_id,
                "category_id": yolo_label,
                "bbox": [
                    round(float(x1), 1),
                    round(float(y1), 1),
                    round(float(bw), 1),
                    round(float(bh), 1),
                ],
                "score": round(float(score), 4),
            }})

    elapsed = time.time() - t0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)
    print(f"Wrote {{len(predictions)}} predictions in {{elapsed:.1f}}s")
    print(f"Classifier applied to {{cls_applied}} detections, changed {{cls_changed}} classes")


if __name__ == "__main__":
    main()
'''


def build(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    build_dir = out / f"{args.name}_build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir()

    # Copy YOLO ONNX files
    yolo_names = []
    for op in args.yolo:
        op = Path(op)
        if not op.exists():
            print(f"WARNING: YOLO model not found: {op} (will copy when available)")
            yolo_names.append(op.name)
            continue
        print(f"Copying YOLO model {op.name} ({op.stat().st_size / 1024 / 1024:.0f} MB)")
        shutil.copy2(op, build_dir / op.name)
        yolo_names.append(op.name)

    # Copy classifier ONNX file
    cls_path = Path(args.classifier)
    cls_name = cls_path.name
    if cls_path.exists():
        print(f"Copying classifier {cls_name} ({cls_path.stat().st_size / 1024 / 1024:.0f} MB)")
        shutil.copy2(cls_path, build_dir / cls_name)
    else:
        print(f"WARNING: Classifier not found: {cls_path} (will copy when available)")

    # Generate run.py
    run_py = generate_run_py(
        yolo_names=yolo_names,
        classifier_name=cls_name,
        nc=args.nc,
        conf=args.conf,
        nms_iou=args.nms_iou,
        wbf_iou=args.wbf_iou,
        wbf_skip=args.wbf_skip,
        cls_threshold=args.cls_threshold,
        yolo_weight=args.yolo_weight,
        cls_weight=args.cls_weight,
        crop_padding=args.crop_padding,
    )
    (build_dir / "run.py").write_text(run_py)
    print("Wrote run.py")

    # Check all model files exist before building ZIP
    all_models = [build_dir / n for n in yolo_names] + [build_dir / cls_name]
    missing = [p for p in all_models if not p.exists()]
    if missing:
        print(f"\nWARNING: {len(missing)} model file(s) missing — ZIP will be incomplete:")
        for m in missing:
            print(f"  - {m.name}")
        print("Re-run once all models are available.\n")

    # Build ZIP
    zip_path = out / f"{args.name}.zip"
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

    return zip_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build two-stage (YOLO detect + ConvNeXt classify) submission"
    )
    parser.add_argument(
        "--yolo", nargs="+", required=True,
        help="Paths to YOLO ONNX models (2x yolov8x recommended)",
    )
    parser.add_argument(
        "--classifier", required=True,
        help="Path to classifier ONNX model (convnext_tiny, output shape 1x356)",
    )
    parser.add_argument("--name", default="twostage_ensemble")
    parser.add_argument("--output-dir", default="submissions")
    parser.add_argument("--nc", type=int, default=356)
    parser.add_argument("--conf", type=float, default=0.005)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--wbf-iou", type=float, default=0.55)
    parser.add_argument("--wbf-skip", type=float, default=0.01)
    parser.add_argument(
        "--cls-threshold", type=float, default=0.8,
        help="YOLO score threshold above which classifier is skipped (default: 0.8)",
    )
    parser.add_argument(
        "--yolo-weight", type=float, default=0.5,
        help="Weight for YOLO class probs in blending (default: 0.5)",
    )
    parser.add_argument(
        "--cls-weight", type=float, default=0.5,
        help="Weight for classifier class probs in blending (default: 0.5)",
    )
    parser.add_argument(
        "--crop-padding", type=float, default=0.1,
        help="Fractional padding around detection crop for classifier (default: 0.1 = 10%%)",
    )
    args = parser.parse_args()
    build(args)
