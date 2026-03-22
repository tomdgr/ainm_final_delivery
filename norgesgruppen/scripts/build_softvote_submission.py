"""Build ONNX ensemble submission with SOFT CLASS VOTING.

Key improvement over standard WBF:
- WBF splits same-box-different-class predictions into separate clusters
- Soft voting matches boxes across models by IoU, then averages the full
  356-dim class probability vectors before taking argmax
- This directly addresses the classification bottleneck (730 misclassifications)

Also supports:
- Multi-scale TTA (run same model at multiple resolutions)
- Horizontal flip TTA
- FP16 ONNX models

Usage:
  uv run python scripts/build_softvote_submission.py \
    --onnx weights/yolov8x_640_fulldata_best_fp16.onnx \
           weights/yolov8x_640_s123_best_fp16.onnx \
           weights/yolov8x_640_seed999_best_fp16.onnx \
    --name softvote_3x_fp16 --tta --multiscale 640 800
"""
import argparse
import shutil
import zipfile
from pathlib import Path


def generate_softvote_run_py(
    onnx_names: list[str],
    imgsz: int = 640,
    nc: int = 356,
    conf: float = 0.005,
    nms_iou: float = 0.7,
    wbf_iou: float = 0.55,
    tta: bool = False,
    multiscale: list[int] | None = None,
    box_expand: float = 0.0,
    temperature: float = 1.0,
    neighbor_voting: bool = False,
) -> str:
    names_repr = repr(onnx_names)
    scales_repr = repr(multiscale) if multiscale else repr([imgsz])

    tta_block = ""
    if tta:
        tta_block = """
def detect_with_tta(sess, input_name, img, ow, oh, scales):
    fb1, fs1, fp1 = detect_multiscale(sess, input_name, img, ow, oh, scales)
    img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
    fb2, fs2, fp2 = detect_multiscale(sess, input_name, img_flip, ow, oh, scales)
    for i, b in enumerate(fb2):
        x1, y1, x2, y2 = b
        fb2[i] = np.array([ow - x2, y1, ow - x1, y2])
    all_b = fb1 + fb2; all_s = fs1 + fs2; all_p = fp1 + fp2
    if len(all_b) == 0:
        return all_b, all_s, all_p
    boxes_arr = np.array(all_b)
    scores_arr = np.array(all_s)
    probs_arr = np.array(all_p)
    cls_ids = probs_arr.argmax(axis=1)
    mb, ms, mp = [], [], []
    for cid in np.unique(cls_ids):
        m = cls_ids == cid
        keep = nms(boxes_arr[m], scores_arr[m], NMS_IOU)
        for k in keep:
            mb.append(boxes_arr[m][k]); ms.append(float(scores_arr[m][k])); mp.append(probs_arr[m][k])
    return mb, ms, mp
"""
    detect_fn_name = "detect_with_tta" if tta else "detect_multiscale"

    expand_block = ""
    if box_expand > 0:
        expand_block = f"""            dx, dy = bw * {box_expand}, bh * {box_expand}
            x1 = max(0, x1 - dx); y1 = max(0, y1 - dy); bw += 2*dx; bh += 2*dy"""

    neighbor_voting_block = ""
    neighbor_voting_call = ""
    neighbor_voting_extra_fields = ""
    if neighbor_voting:
        # Store top-2 class info alongside each prediction for neighbor voting
        neighbor_voting_extra_fields = """
                "_top2_cls": int(sorted_cls[-2]) if len(sorted_cls) >= 2 else label,
                "_top1_prob": float(avg_prob[sorted_cls[-1]]),
                "_top2_prob": float(avg_prob[sorted_cls[-2]]) if len(sorted_cls) >= 2 else 0.0,"""
        neighbor_voting_call = "    predictions = neighbor_class_vote(predictions)"
        neighbor_voting_block = """
def neighbor_class_vote(predictions, y_threshold=50, score_cap=0.8, prob_ratio=2.0, min_row_size=3):
    \"\"\"Correct outlier class predictions using row-based spatial neighbor voting.

    Groups detections into shelf rows by y-center proximity, then for uncertain
    outliers, flips to 2nd-choice class if it matches the row majority.

    Conservative guards - only flips when ALL conditions met:
    - Detection score <= score_cap (skip high-confidence)
    - Top-2 class probability ratio < prob_ratio (prediction is uncertain)
    - Predicted class appears only once in the row (it is the outlier)
    - 2nd-choice class appears >= 2 times in the row (strong spatial signal)
    - Row has >= min_row_size detections (enough context to vote)
    \"\"\"
    from collections import Counter
    if not predictions:
        return predictions
    by_image = {}
    for i, p in enumerate(predictions):
        by_image.setdefault(p["image_id"], []).append(i)
    flips = 0
    for image_id, indices in by_image.items():
        dets = [(idx, predictions[idx]["bbox"][1] + predictions[idx]["bbox"][3] / 2.0) for idx in indices]
        dets.sort(key=lambda x: x[1])
        rows = []
        current_row = [dets[0]]
        for d in dets[1:]:
            if abs(d[1] - current_row[-1][1]) <= y_threshold:
                current_row.append(d)
            else:
                rows.append(current_row)
                current_row = [d]
        rows.append(current_row)
        for row in rows:
            if len(row) < min_row_size:
                continue
            row_indices = [r[0] for r in row]
            row_labels = [predictions[idx]["category_id"] for idx in row_indices]
            label_counts = Counter(row_labels)
            for idx in row_indices:
                pred = predictions[idx]
                if pred["score"] > score_cap:
                    continue
                current_cls = pred["category_id"]
                if label_counts[current_cls] != 1:
                    continue
                if "_top2_cls" not in pred:
                    continue
                second_cls = pred["_top2_cls"]
                top1_prob = pred["_top1_prob"]
                top2_prob = pred["_top2_prob"]
                if top2_prob <= 0 or top1_prob / top2_prob >= prob_ratio:
                    continue
                if label_counts.get(second_cls, 0) < 2:
                    continue
                pred["category_id"] = second_cls
                label_counts[current_cls] -= 1
                label_counts[second_cls] = label_counts.get(second_cls, 0) + 1
                flips += 1
    for p in predictions:
        p.pop("_top2_cls", None)
        p.pop("_top1_prob", None)
        p.pop("_top2_prob", None)
    print(f"Neighbor voting: flipped {{flips}} predictions")
    return predictions
"""

    code = f"""import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort

NC = {nc}
CONF_THR = {conf}
NMS_IOU = {nms_iou}
WBF_IOU = {wbf_iou}
SCALES = {scales_repr}
TEMPERATURE = {temperature}

def preprocess(img, sz):
    w, h = img.size; scale = min(sz/w, sz/h)
    nw, nh = int(w*scale), int(h*scale)
    img_r = img.resize((nw, nh), Image.BILINEAR)
    pad = np.full((sz, sz, 3), 114, dtype=np.uint8)
    px, py = (sz-nw)//2, (sz-nh)//2
    pad[py:py+nh, px:px+nw] = np.array(img_r)
    return np.transpose(pad.astype(np.float32)/255.0, (2,0,1))[np.newaxis,...], scale, px, py

def nms(boxes, scores, thr=0.5):
    if len(boxes)==0: return []
    x1,y1,x2,y2=boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas=(x2-x1)*(y2-y1); order=scores.argsort()[::-1]; keep=[]
    while len(order)>0:
        i=order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        inter=np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
        iou=inter/(areas[i]+areas[order[1:]]-inter+1e-8)
        order=order[np.where(iou<=thr)[0]+1]
    return keep

def detect_raw(sess, input_name, img, ow, oh, sz):
    blob, scale, px, py = preprocess(img, sz)
    out = sess.run(None, {{input_name: blob}})[0]
    if out.ndim==3: out = out[0].T if out.shape[1]==(4+NC) else out[0]
    cx,cy,bw,bh = out[:,0],out[:,1],out[:,2],out[:,3]
    cls_scores = out[:,4:]
    max_sc = cls_scores.max(axis=1)
    mask = max_sc > CONF_THR
    x1=np.clip((cx-bw/2-px)/scale,0,ow)[mask]; y1=np.clip((cy-bh/2-py)/scale,0,oh)[mask]
    x2=np.clip((cx+bw/2-px)/scale,0,ow)[mask]; y2=np.clip((cy+bh/2-py)/scale,0,oh)[mask]
    boxes=np.stack([x1,y1,x2,y2],axis=1)
    scores=max_sc[mask]
    probs=cls_scores[mask]
    cls_ids = probs.argmax(axis=1)
    fb, fs, fp = [], [], []
    for cid in np.unique(cls_ids):
        m = cls_ids == cid
        keep = nms(boxes[m], scores[m], NMS_IOU)
        for k in keep:
            fb.append(boxes[m][k]); fs.append(float(scores[m][k])); fp.append(probs[m][k])
    return fb, fs, fp

def detect_multiscale(sess, input_name, img, ow, oh, scales):
    all_b, all_s, all_p = [], [], []
    for sz in scales:
        fb, fs, fp = detect_raw(sess, input_name, img, ow, oh, sz)
        all_b.extend(fb); all_s.extend(fs); all_p.extend(fp)
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
                mb.append(boxes_arr[m][k]); ms.append(float(scores_arr[m][k])); mp.append(probs_arr[m][k])
        return mb, ms, mp
    return all_b, all_s, all_p
{tta_block}
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (a1 + a2 - inter + 1e-8)

def soft_vote_merge(model_results, n_models):
    if not model_results or all(len(r[0])==0 for r in model_results):
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
        if used[i]: continue
        cluster = [all_dets[i]]
        used[i] = True
        models_in = {{all_dets[i][3]}}
        for j in range(i+1, len(all_dets)):
            if used[j] or all_dets[j][3] in models_in: continue
            iou = compute_iou(all_dets[i][0], all_dets[j][0])
            if iou >= WBF_IOU:
                cluster.append(all_dets[j]); used[j] = True; models_in.add(all_dets[j][3])
        clusters.append(cluster)
    final_boxes, final_scores, final_labels, final_probs = [], [], [], []
    for cluster in clusters:
        total_score = sum(d[1] for d in cluster)
        avg_box = np.zeros(4)
        avg_prob = np.zeros(NC)
        for b, s, p, m in cluster:
            avg_box += b * s
            p_scaled = p / TEMPERATURE
            p_scaled = p_scaled - p_scaled.max()
            p_exp = np.exp(p_scaled)
            p_soft = p_exp / p_exp.sum()
            avg_prob += p_soft
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
{neighbor_voting_block}
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Loading models...")
    sessions, input_names = [], []
    for name in {names_repr}:
        sess = ort.InferenceSession(name, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        sessions.append(sess)
        input_names.append(sess.get_inputs()[0].name)
    n_models = len(sessions)
    print(f"Loaded {{n_models}} ONNX models")
    predictions = []
    img_paths = [p for p in sorted(Path(args.input).iterdir()) if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    try:
        from tqdm import tqdm
        img_iter = tqdm(img_paths, desc="Inference")
    except ImportError:
        img_iter = img_paths
        print(f"Processing {{len(img_paths)}} images...")
    for img_path in img_iter:
        image_id = int(img_path.stem.split("_")[-1])
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        model_results = []
        for sess, inp_name in zip(sessions, input_names):
            fb, fs, fp = {detect_fn_name}(sess, inp_name, img, w, h, SCALES)
            model_results.append((fb, fs, fp))
        final_boxes, final_scores, final_labels, final_probs = soft_vote_merge(model_results, n_models)
        for box, score, label, avg_prob in zip(final_boxes, final_scores, final_labels, final_probs):
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
{expand_block}
            sorted_cls = np.argsort(avg_prob)
            pred = {{"image_id": image_id, "category_id": label,
                "bbox": [round(float(x1),1), round(float(y1),1), round(float(bw),1), round(float(bh),1)],
                "score": round(score, 4),{neighbor_voting_extra_fields}}}
            predictions.append(pred)
{neighbor_voting_call}
    # Clean up internal keys before writing
    for p in predictions:
        p.pop("_top2_cls", None)
        p.pop("_top1_prob", None)
        p.pop("_top2_prob", None)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)
    print(f"Wrote {{len(predictions)}} predictions")

if __name__ == "__main__":
    main()
"""
    return code


def build(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    build_dir = out / f"{args.name}_build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir()

    onnx_names = []
    for op in args.onnx:
        op = Path(op)
        shutil.copy2(op, build_dir / op.name)
        onnx_names.append(op.name)

    multiscale = [int(s) for s in args.multiscale] if args.multiscale else None

    run_py = generate_softvote_run_py(
        onnx_names=onnx_names,
        imgsz=args.imgsz,
        nc=args.nc,
        conf=args.conf,
        nms_iou=args.nms_iou,
        wbf_iou=args.wbf_iou,
        tta=args.tta,
        multiscale=multiscale,
        box_expand=args.box_expand,
        temperature=args.temperature,
        neighbor_voting=args.neighbor_voting,
    )
    (build_dir / "run.py").write_text(run_py)

    zip_path = out / f"{args.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(build_dir.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(build_dir))
    shutil.rmtree(build_dir)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Built: {zip_path} ({size_mb:.1f} MB)")
    return zip_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", nargs="+", required=True)
    parser.add_argument("--name", default="softvote_ensemble")
    parser.add_argument("--output-dir", default="submissions")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--nc", type=int, default=356)
    parser.add_argument("--conf", type=float, default=0.005)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--wbf-iou", type=float, default=0.55)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--multiscale", nargs="+", default=None, help="e.g. 640 800")
    parser.add_argument("--box-expand", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for softmax scaling of class probs before averaging (lower=sharper)")
    parser.add_argument("--neighbor-voting", action="store_true", help="Enable row-based neighbor class voting post-processing")
    args = parser.parse_args()
    build(args)
