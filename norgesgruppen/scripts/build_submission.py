"""Build competition submission ZIP from trained weights.

Usage:
  # Single model, basic
  python scripts/build_submission.py best.pt --name baseline

  # Ensemble of 3 models
  python scripts/build_submission.py m1.pt m2.pt m3.pt --name ensemble --ensemble

  # Full pipeline: ensemble + SAHI + two-stage classifier
  python scripts/build_submission.py m1.pt m2.pt --name full \
    --ensemble --sahi --gallery data/reference_embeddings.pt

  # Test locally before uploading
  python scripts/build_submission.py best.pt --name test --test-locally
"""
import argparse
import json
import shutil
import zipfile
from pathlib import Path


def generate_full_run_py(
    weight_names: list[str],
    ensemble: bool = False,
    sahi: bool = False,
    gallery: bool = False,
    imgsz: int = 640,
    slice_size: int = 640,
    overlap: float = 0.25,
    conf: float = 0.01,
    iou: float = 0.6,
    wbf_iou: float = 0.55,
    cls_threshold: float = 0.65,
    onnx_names: list[str] | None = None,
) -> str:
    """Generate a run.py that handles all submission modes."""

    onnx_names = onnx_names or []
    yolo_names = [w for w in weight_names if w not in onnx_names]

    parts = []
    parts.append('''\
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
''')

    if ensemble:
        parts.append('from ensemble_boxes import weighted_boxes_fusion\n')

    if gallery:
        parts.append('''\
import torch.nn.functional as F
from torchvision import models, transforms

CLS_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
''')

    if onnx_names:
        parts.append('import onnxruntime as ort\n')

    if sahi:
        parts.append(f'''
def slice_image(w, h, sz={slice_size}, ov={overlap}):
    step = int(sz * (1 - ov))
    slices = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            x2, y2 = min(x + sz, w), min(y + sz, h)
            slices.append((max(0, x2 - sz), max(0, y2 - sz), x2, y2))
    return slices
''')

    # NMS function (needed for SAHI and ONNX)
    if sahi or onnx_names:
        parts.append('''
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
        iou_val = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou_val <= thr)[0] + 1]
    return keep
''')

    # ONNX inference function
    if onnx_names:
        parts.append(f'''
def onnx_predict(session, img_path, imgsz={imgsz}, nc=356, conf_thr={conf}):
    input_name = session.get_inputs()[0].name
    img = Image.open(img_path).convert("RGB")
    ow, oh = img.size
    scale = min(imgsz / ow, imgsz / oh)
    nw, nh = int(ow * scale), int(oh * scale)
    img_r = img.resize((nw, nh), Image.BILINEAR)
    pad = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    px, py = (imgsz - nw) // 2, (imgsz - nh) // 2
    pad[py:py + nh, px:px + nw] = np.array(img_r)
    blob = np.transpose(pad.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, ...]
    out = session.run(None, {{input_name: blob}})[0]
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
    dets = []
    for cid in np.unique(cls_ids):
        m = cls_ids == cid
        keep = nms(boxes[m], max_sc[m], 0.5)
        for k in keep:
            b = boxes[m][k]
            dets.append(([float(b[0]), float(b[1]), float(b[2]), float(b[3])], float(max_sc[m][k]), int(cid)))
    return dets
''')

    # Detect function for a single model on a single image/crop
    detect_fn = f'''
def detect_yolo(model, img_input, device, imgsz={imgsz}, conf={conf}, iou={iou}):
    """Run YOLO on an image path or PIL crop. Returns list of (xyxy, score, cls)."""
    results = model(img_input, device=device, verbose=False, imgsz=imgsz, conf=conf, iou=iou, max_det=300)
    dets = []
    for r in results:
        if r.boxes is None:
            continue
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
            dets.append(([x1, y1, x2, y2], float(r.boxes.conf[i].item()), int(r.boxes.cls[i].item())))
    return dets
'''
    parts.append(detect_fn)

    # Main function
    main_lines = ['''
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
''']

    # Load models
    for wn in yolo_names:
        main_lines.append(f'    model_{wn.replace(".", "_").replace("-", "_")} = YOLO("{wn}")')
    main_lines.append(f'    yolo_models = [{", ".join("model_" + w.replace(".", "_").replace("-", "_") for w in yolo_names)}]')

    if onnx_names:
        for on in onnx_names:
            vname = on.replace(".", "_").replace("-", "_")
            main_lines.append(f'    onnx_{vname} = ort.InferenceSession("{on}", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])')
        main_lines.append(f'    onnx_sessions = [{", ".join("onnx_" + o.replace(".", "_").replace("-", "_") for o in onnx_names)}]')

    if gallery:
        main_lines.append('''
    gallery_data = torch.load("reference_embeddings.pt", map_location=device)
    gallery = gallery_data["gallery"].to(device)
    cls_backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    cls_backbone.fc = torch.nn.Identity()
    cls_backbone = cls_backbone.to(device).eval()
    print(f"Gallery loaded: {gallery.shape[0]} classes")
''')

    main_lines.append('''
    predictions = []
    input_dir = Path(args.input)
    image_files = sorted(f for f in input_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    print(f"Processing {len(image_files)} images...")

    for img_path in image_files:
        image_id = int(img_path.stem.split("_")[-1])
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
''')

    if ensemble:
        # Multi-model with WBF
        main_lines.append('''
        all_boxes, all_scores, all_labels = [], [], []

        for model in yolo_models:''')

        if sahi:
            main_lines.append(f'''
            # SAHI tiled + full image
            raw_dets = []
            for sx1, sy1, sx2, sy2 in slice_image(w, h):
                crop = img.crop((sx1, sy1, sx2, sy2))
                for xyxy, sc, cl in detect_yolo(model, crop, device):
                    raw_dets.append(([xyxy[0]+sx1, xyxy[1]+sy1, xyxy[2]+sx1, xyxy[3]+sy1], sc, cl))
            for xyxy, sc, cl in detect_yolo(model, str(img_path), device):
                raw_dets.append((xyxy, sc, cl))''')
        else:
            main_lines.append('''
            raw_dets = detect_yolo(model, str(img_path), device)''')

        main_lines.append(f'''
            boxes, scores, labels = [], [], []
            for xyxy, sc, cl in raw_dets:
                boxes.append([xyxy[0]/w, xyxy[1]/h, xyxy[2]/w, xyxy[3]/h])
                scores.append(sc)
                labels.append(cl)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
''')

        if onnx_names:
            main_lines.append('''
        for session in onnx_sessions:
            raw_dets = onnx_predict(session, str(img_path))
            boxes, scores, labels = [], [], []
            for xyxy, sc, cl in raw_dets:
                boxes.append([xyxy[0]/w, xyxy[1]/h, xyxy[2]/w, xyxy[3]/h])
                scores.append(sc)
                labels.append(cl)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
''')

        main_lines.append(f'''
        if any(len(b) > 0 for b in all_boxes):
            fb, fs, fl = weighted_boxes_fusion(all_boxes, all_scores, all_labels, iou_thr={wbf_iou}, skip_box_thr={conf})
            img_dets = []
            for box, score, label in zip(fb, fs, fl):
                x1, y1, x2, y2 = box
                img_dets.append((x1*w, y1*h, x2*w, y2*h, float(score), int(label)))
        else:
            img_dets = []
''')
    else:
        # Single model
        if sahi:
            main_lines.append(f'''
        raw_dets = []
        for sx1, sy1, sx2, sy2 in slice_image(w, h):
            crop = img.crop((sx1, sy1, sx2, sy2))
            for xyxy, sc, cl in detect_yolo(yolo_models[0], crop, device):
                raw_dets.append(([xyxy[0]+sx1, xyxy[1]+sy1, xyxy[2]+sx1, xyxy[3]+sy1], sc, cl))
        for xyxy, sc, cl in detect_yolo(yolo_models[0], str(img_path), device):
            raw_dets.append((xyxy, sc, cl))
        # Per-class NMS
        if raw_dets:
            all_b = np.array([d[0] for d in raw_dets])
            all_s = np.array([d[1] for d in raw_dets])
            all_c = np.array([d[2] for d in raw_dets])
            img_dets = []
            for cid in np.unique(all_c):
                m = all_c == cid
                keep = nms(all_b[m], all_s[m], {iou})
                for k in keep:
                    b = all_b[m][k]
                    img_dets.append((b[0], b[1], b[2], b[3], float(all_s[m][k]), int(cid)))
        else:
            img_dets = []
''')
        else:
            main_lines.append('''
        img_dets = []
        for xyxy, sc, cl in detect_yolo(yolo_models[0], str(img_path), device):
            img_dets.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], sc, cl))
''')

    if gallery:
        main_lines.append(f'''
        # Two-stage classification
        if img_dets:
            crops = []
            for x1, y1, x2, y2, sc, cl in img_dets:
                bw, bh = x2 - x1, y2 - y1
                crop = img.crop((max(0, x1 - bw*0.1), max(0, y1 - bh*0.1), min(w, x2 + bw*0.1), min(h, y2 + bh*0.1)))
                if crop.width >= 10 and crop.height >= 10:
                    crops.append(crop)
                else:
                    crops.append(None)

            with torch.no_grad():
                valid_crops = [c for c in crops if c is not None]
                if valid_crops:
                    cls_results = []
                    for bi in range(0, len(valid_crops), 32):
                        batch = valid_crops[bi:bi+32]
                        tensors = torch.stack([CLS_TRANSFORM(c) for c in batch]).to(device)
                        feats = F.normalize(cls_backbone(tensors), dim=1)
                        sims = feats @ gallery.T
                        best_cats = sims.argmax(dim=1).tolist()
                        best_scs = sims.max(dim=1).values.tolist()
                        for cat, sim in zip(best_cats, best_scs):
                            cls_results.append((cat, max(0, (sim + 1) / 2)))

                    ci = 0
                    final_dets = []
                    for idx, (x1, y1, x2, y2, sc, cl) in enumerate(img_dets):
                        if crops[idx] is not None:
                            cls_cat, cls_conf = cls_results[ci]
                            ci += 1
                            if cls_conf > {cls_threshold}:
                                cl = cls_cat
                                sc = sc * 0.7 + cls_conf * 0.3
                        final_dets.append((x1, y1, x2, y2, sc, cl))
                    img_dets = final_dets
''')

    main_lines.append('''
        for x1, y1, x2, y2, sc, cl in img_dets:
            predictions.append({
                "image_id": image_id,
                "category_id": int(cl),
                "bbox": [round(float(x1), 1), round(float(y1), 1), round(float(x2 - x1), 1), round(float(y2 - y1), 1)],
                "score": round(float(sc), 4),
            })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)
    print(f"Wrote {len(predictions)} predictions")

if __name__ == "__main__":
    main()
''')

    return "\n".join(parts) + "\n".join(main_lines)


def build_zip(
    weight_paths: list[str],
    name: str = "submission",
    output_dir: str = "submissions",
    ensemble: bool = False,
    sahi: bool = False,
    gallery_path: str | None = None,
    onnx_paths: list[str] | None = None,
    imgsz: int = 640,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    build_dir = out / f"{name}_build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir()

    # Copy weights
    weight_names = []
    for wp in weight_paths:
        wp = Path(wp)
        shutil.copy2(wp, build_dir / wp.name)
        weight_names.append(wp.name)

    onnx_names = []
    if onnx_paths:
        for op in onnx_paths:
            op = Path(op)
            shutil.copy2(op, build_dir / op.name)
            onnx_names.append(op.name)

    if gallery_path:
        shutil.copy2(gallery_path, build_dir / "reference_embeddings.pt")

    # Generate run.py
    run_py = generate_full_run_py(
        weight_names=weight_names + onnx_names,
        ensemble=ensemble,
        sahi=sahi,
        gallery=gallery_path is not None,
        imgsz=imgsz,
        onnx_names=onnx_names,
    )
    (build_dir / "run.py").write_text(run_py)

    # Create ZIP
    zip_path = out / f"{name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(build_dir.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(build_dir))

    shutil.rmtree(build_dir)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Built: {zip_path} ({size_mb:.1f} MB)")

    # Verify structure
    with zipfile.ZipFile(zip_path, "r") as zf:
        files = zf.namelist()
        assert "run.py" in files, "run.py not at zip root!"
        print(f"Contents: {files}")

    return zip_path


def test_locally(zip_path: str, test_images: str = "data/raw/coco_dataset/train/images"):
    """Unzip and run locally to verify it works."""
    import subprocess
    import tempfile

    zip_path = Path(zip_path)
    test_images = Path(test_images)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Unzip
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)

        # Pick 3 test images
        imgs = sorted(test_images.iterdir())[:3]
        img_dir = tmpdir / "test_images"
        img_dir.mkdir()
        for img in imgs:
            shutil.copy2(img, img_dir / img.name)

        output_path = tmpdir / "predictions.json"
        print(f"Running: python {tmpdir}/run.py --input {img_dir} --output {output_path}")
        result = subprocess.run(
            ["python", str(tmpdir / "run.py"), "--input", str(img_dir), "--output", str(output_path)],
            capture_output=True, text=True, timeout=120, cwd=str(tmpdir),
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"FAILED:\n{result.stderr}")
            return False

        with open(output_path) as f:
            preds = json.load(f)
        print(f"Predictions: {len(preds)} boxes across {len(imgs)} images")
        if preds:
            print(f"Sample: {preds[0]}")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", nargs="+", help="Model weight files (.pt)")
    parser.add_argument("--name", default="submission")
    parser.add_argument("--output-dir", default="submissions")
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--sahi", action="store_true")
    parser.add_argument("--gallery", default=None, help="Path to reference_embeddings.pt")
    parser.add_argument("--onnx", nargs="*", default=None, help="ONNX model files")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--test-locally", action="store_true")
    args = parser.parse_args()

    zip_path = build_zip(
        weight_paths=args.weights,
        name=args.name,
        output_dir=args.output_dir,
        ensemble=args.ensemble,
        sahi=args.sahi,
        gallery_path=args.gallery,
        onnx_paths=args.onnx,
        imgsz=args.imgsz,
    )

    if args.test_locally:
        test_locally(str(zip_path))
