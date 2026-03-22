"""Build advanced ONNX ensemble submission with TTA, box expansion, score calibration, two-stage WBF, and class-agnostic NMS.

Usage:
  uv run python scripts/build_advanced_submission.py \
    --onnx weights/yolov8x_640_fulldata_best.onnx weights/yolov8m_640_diverse_best.onnx weights/yolov8s_640_best.onnx \
    --name advanced_ensemble --tta --box-expand 0.1 --score-calibration --two-stage-wbf
"""
import argparse
import shutil
import zipfile
from pathlib import Path


def generate_advanced_run_py(
    onnx_names: list[str],
    imgsz: int = 640,
    nc: int = 356,
    conf: float = 0.005,
    nms_iou: float = 0.7,
    wbf_iou: float = 0.7,
    tta: bool = False,
    box_expand: float = 0.0,
    score_calibration: bool = False,
    two_stage_wbf: bool = False,
    agnostic_nms: bool = False,
) -> str:
    names_repr = repr(onnx_names)

    # Build the NMS function - either per-class or agnostic
    if agnostic_nms:
        nms_block = f"""
def apply_nms(boxes, scores, cls_ids, nms_thr={nms_iou}):
    if len(boxes) == 0:
        return [], [], []
    keep = nms(boxes, scores, nms_thr)
    return [boxes[k] for k in keep], [float(scores[k]) for k in keep], [int(cls_ids[k]) for k in keep]
"""
    else:
        nms_block = f"""
def apply_nms(boxes, scores, cls_ids, nms_thr={nms_iou}):
    if len(boxes) == 0:
        return [], [], []
    fb, fs, fc = [], [], []
    for cid in np.unique(cls_ids):
        m = cls_ids == cid
        keep = nms(boxes[m], scores[m], nms_thr)
        for k in keep:
            fb.append(boxes[m][k]); fs.append(float(scores[m][k])); fc.append(int(cid))
    return fb, fs, fc
"""

    # Score calibration function
    score_cal_block = ""
    score_cal_import = ""
    if score_calibration:
        score_cal_import = "from scipy.stats import rankdata\n"
        score_cal_block = """
def calibrate_scores(scores_list):
    \"\"\"Normalize scores via rank for each model before WBF.\"\"\"
    calibrated = []
    for scores in scores_list:
        if len(scores) == 0:
            calibrated.append(scores)
        else:
            ranked = rankdata(scores) / len(scores)
            calibrated.append((0.5 * ranked + 0.5).tolist())
    return calibrated
"""

    # Box expansion function
    box_expand_block = ""
    if box_expand > 0:
        box_expand_block = f"""
def expand_boxes(predictions, expand={box_expand}):
    \"\"\"Expand predicted boxes by a fraction to improve IoU overlap.\"\"\"
    for p in predictions:
        x, y, w, h = p['bbox']
        dx, dy = w * expand, h * expand
        p['bbox'] = [round(max(0, x - dx), 1), round(max(0, y - dy), 1), round(w + 2*dx, 1), round(h + 2*dy, 1)]
    return predictions
"""

    # TTA block
    tta_detect = ""
    if tta:
        tta_detect = """
def detect_with_tta(sess, input_name, img, ow, oh):
    \"\"\"Run detection on original + horizontally flipped image, merge results.\"\"\"
    # Original
    fb1, fs1, fc1 = detect_onnx(sess, input_name, img, ow, oh)

    # Horizontal flip
    img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
    fb2, fs2, fc2 = detect_onnx(sess, input_name, img_flip, ow, oh)

    # Mirror x-coordinates back for flipped detections
    for i, b in enumerate(fb2):
        x1, y1, x2, y2 = b
        fb2[i] = np.array([ow - x2, y1, ow - x1, y2])

    # Combine
    all_b = fb1 + fb2
    all_s = fs1 + fs2
    all_c = fc1 + fc2
    return all_b, all_s, all_c
"""

    detect_call = "detect_with_tta" if tta else "detect_onnx"

    # WBF conf_type for two-stage
    if two_stage_wbf:
        wbf_conf_type = "'max'"
    else:
        wbf_conf_type = "'avg'"

    # Score calibration call
    score_cal_call = ""
    if score_calibration:
        score_cal_call = "        all_scores = calibrate_scores(all_scores)"

    return f'''\
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort
from ensemble_boxes import weighted_boxes_fusion
{score_cal_import}
def preprocess(img, sz={imgsz}):
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

def detect_onnx(sess, input_name, img, ow, oh, nc={nc}, conf_thr={conf}):
    blob, scale, px, py = preprocess(img)
    out = sess.run(None, {{input_name: blob}})[0]
    if out.ndim==3: out = out[0].T if out.shape[1]==(4+nc) else out[0]
    cx,cy,bw,bh = out[:,0],out[:,1],out[:,2],out[:,3]
    cls_scores = out[:,4:]; cls_ids = cls_scores.argmax(axis=1); max_sc = cls_scores.max(axis=1)
    mask = max_sc > conf_thr
    x1=np.clip((cx-bw/2-px)/scale,0,ow)[mask]; y1=np.clip((cy-bh/2-py)/scale,0,oh)[mask]
    x2=np.clip((cx+bw/2-px)/scale,0,ow)[mask]; y2=np.clip((cy+bh/2-py)/scale,0,oh)[mask]
    cls_ids=cls_ids[mask]; max_sc=max_sc[mask]; boxes=np.stack([x1,y1,x2,y2],axis=1)
    fb, fs, fc = apply_nms(boxes, max_sc, cls_ids)
    return fb, fs, fc
{nms_block}
{tta_detect}
{score_cal_block}
{box_expand_block}
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
    print(f"Loaded {{len(sessions)}} ONNX models")

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

        all_boxes, all_scores, all_labels = [], [], []
        for sess, inp_name in zip(sessions, input_names):
            det_b, det_s, det_c = {detect_call}(sess, inp_name, img, w, h)
            boxes, scores, labels = [], [], []
            for b, s, c in zip(det_b, det_s, det_c):
                boxes.append([b[0]/w, b[1]/h, b[2]/w, b[3]/h])
                scores.append(s)
                labels.append(c)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

{score_cal_call}

        if any(len(b)>0 for b in all_boxes):
            fb, fs, fl = weighted_boxes_fusion(all_boxes, all_scores, all_labels, iou_thr={wbf_iou}, skip_box_thr={conf}, conf_type={wbf_conf_type})
            for box, score, label in zip(fb, fs, fl):
                x1, y1, x2, y2 = box
                predictions.append({{"image_id": image_id, "category_id": int(label), "bbox": [round(x1*w,1), round(y1*h,1), round((x2-x1)*w,1), round((y2-y1)*h,1)], "score": round(float(score),4)}})

{"    predictions = expand_boxes(predictions)" if box_expand > 0 else ""}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)
    print(f"Wrote {{len(predictions)}} predictions")

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

    onnx_names = []
    for op in args.onnx:
        op = Path(op)
        shutil.copy2(op, build_dir / op.name)
        onnx_names.append(op.name)

    run_py = generate_advanced_run_py(
        onnx_names=onnx_names,
        imgsz=args.imgsz,
        conf=args.conf,
        nms_iou=args.nms_iou,
        wbf_iou=args.wbf_iou,
        tta=args.tta,
        box_expand=args.box_expand,
        score_calibration=args.score_calibration,
        two_stage_wbf=args.two_stage_wbf,
        agnostic_nms=args.agnostic_nms,
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
    parser.add_argument("--name", default="advanced_ensemble")
    parser.add_argument("--output-dir", default="submissions")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.005)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--wbf-iou", type=float, default=0.7)
    parser.add_argument("--tta", action="store_true", help="TTA with horizontal flip")
    parser.add_argument("--box-expand", type=float, default=0.0, help="Expand boxes by fraction (e.g. 0.1 = 10%)")
    parser.add_argument("--score-calibration", action="store_true", help="Rank-based score calibration")
    parser.add_argument("--two-stage-wbf", action="store_true", help="Use conf_type=max for cross-model WBF")
    parser.add_argument("--agnostic-nms", action="store_true", help="Class-agnostic NMS")
    args = parser.parse_args()
    build(args)
