"""Evaluate two-stage pipeline (ONNX detector + ResNet50 classifier) on training set.
Sweeps classifier thresholds and compares against single-model baseline.
"""
import json
import numpy as np
from pathlib import Path
from PIL import Image
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torchvision import models, transforms

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


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


def detect_image(sess, input_name, img, nc=356, conf=0.01):
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

    # Per-class NMS
    final_boxes, final_scores, final_cls = [], [], []
    for cid in np.unique(cls_ids):
        m = cls_ids == cid
        keep = nms(boxes[m], max_sc[m], 0.5)
        for k in keep:
            final_boxes.append(boxes[m][k])
            final_scores.append(float(max_sc[m][k]))
            final_cls.append(int(cid))
    return final_boxes, final_scores, final_cls


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="ONNX model path")
    parser.add_argument("--coco-json", default="data/raw/coco_dataset/train/annotations.json")
    parser.add_argument("--image-dir", default="data/raw/coco_dataset/train/images")
    parser.add_argument("--gallery", default=None, help="Gallery .pt path (if None, builds one)")
    parser.add_argument("--product-dir", default="data/raw/product_images")
    parser.add_argument("--output-dir", default="outputs/eval_twostage")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    # Load detector
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(args.onnx, providers=providers)
    input_name = sess.get_inputs()[0].name

    # Build or load gallery
    if args.gallery and Path(args.gallery).exists():
        gallery_path = args.gallery
    else:
        from nm_ai_image.detection.classifier import GalleryBuilder
        builder = GalleryBuilder(backbone="resnet50", max_crops_per_class=20)
        gallery_path = "data/reference_embeddings.pt"
        builder.build(args.coco_json, args.image_dir, args.product_dir, gallery_path)

    gallery_data = torch.load(gallery_path, map_location=device)
    gallery = gallery_data["gallery"].to(device)

    # Load backbone
    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    backbone.fc = torch.nn.Identity()
    backbone = backbone.to(device).eval()
    print("Models loaded", flush=True)

    # Run detection on all images (once, reuse for all thresholds)
    image_dir = Path(args.image_dir)
    images = sorted(f for f in image_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    print(f"Processing {len(images)} images...", flush=True)

    all_image_results = []  # (image_id, det_preds, crops, det_info)
    single_preds = []

    for idx, img_path in enumerate(images):
        if idx % 50 == 0:
            print(f"  {idx}/{len(images)}", flush=True)
        image_id = int(img_path.stem.split("_")[-1])
        img = Image.open(img_path).convert("RGB")
        ow, oh = img.size

        det_boxes, det_scores, det_cls = detect_image(sess, input_name, img)
        if not det_boxes:
            continue

        # Single model predictions
        for box, score, cid in zip(det_boxes, det_scores, det_cls):
            x1, y1, x2, y2 = box
            single_preds.append({
                "image_id": image_id, "category_id": cid,
                "bbox": [round(float(x1), 1), round(float(y1), 1), round(float(x2 - x1), 1), round(float(y2 - y1), 1)],
                "score": round(score, 4),
            })

        # Crop for classifier
        crops, det_info = [], []
        for box, score, cid in zip(det_boxes, det_scores, det_cls):
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            crop = img.crop((max(0, x1 - bw * 0.1), max(0, y1 - bh * 0.1),
                             min(ow, x2 + bw * 0.1), min(oh, y2 + bh * 0.1)))
            if crop.width >= 10 and crop.height >= 10:
                crops.append(crop)
                det_info.append((float(x1), float(y1), float(x2), float(y2), score, cid))

        if not crops:
            continue

        # Classify all crops
        cls_results = []
        with torch.no_grad():
            for bi in range(0, len(crops), 64):
                batch = crops[bi:bi + 64]
                tensors = torch.stack([TRANSFORM(c) for c in batch]).to(device)
                feats = F.normalize(backbone(tensors), dim=1)
                sims = feats @ gallery.T
                best_cats = sims.argmax(dim=1).tolist()
                best_scores = sims.max(dim=1).values.tolist()
                for j in range(len(batch)):
                    cls_results.append((best_cats[j], max(0, (best_scores[j] + 1) / 2)))

        all_image_results.append((image_id, det_info, cls_results))

    # Evaluate single model
    from nm_ai_image.detection.evaluate import evaluate_predictions
    print(f"\n=== SINGLE MODEL (baseline) ===")
    r1 = evaluate_predictions(single_preds, args.coco_json)
    print(r1.summary())

    # Sweep classifier thresholds
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_summary = {"single": {"score": r1.competition_score, "det": r1.detection_map50, "cls": r1.classification_map50}}

    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        twostage_preds = []
        for image_id, det_info, cls_results in all_image_results:
            for (x1, y1, x2, y2, det_score, det_cat), (cls_cat, cls_conf) in zip(det_info, cls_results):
                cat = cls_cat if cls_conf > thr else det_cat
                score = det_score * 0.7 + cls_conf * 0.3 if cls_conf > thr else det_score
                twostage_preds.append({
                    "image_id": image_id, "category_id": cat,
                    "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                    "score": round(score, 4),
                })

        print(f"\n=== TWO-STAGE (threshold={thr}) ===")
        r2 = evaluate_predictions(twostage_preds, args.coco_json)
        print(r2.summary())
        results_summary[f"twostage_{thr}"] = {"score": r2.competition_score, "det": r2.detection_map50, "cls": r2.classification_map50}

    with open(output_dir / "results_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nResults saved to {output_dir / 'results_summary.json'}")


if __name__ == "__main__":
    main()
