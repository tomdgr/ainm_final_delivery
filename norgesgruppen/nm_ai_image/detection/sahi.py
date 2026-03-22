"""SAHI-style sliced inference for large images.

Images are 2000-4000px but model sees 640px. Small products get missed.
This slices images into overlapping tiles, runs detection on each,
and merges results with NMS.

Uses only packages available in the sandbox (no sahi package needed).
"""
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def _slice_image(
    img_w: int, img_h: int,
    slice_size: int = 640,
    overlap_ratio: float = 0.25,
) -> list[tuple[int, int, int, int]]:
    """Generate overlapping slice coordinates."""
    step = int(slice_size * (1 - overlap_ratio))
    slices = []
    for y in range(0, img_h, step):
        for x in range(0, img_w, step):
            x2 = min(x + slice_size, img_w)
            y2 = min(y + slice_size, img_h)
            x1 = max(0, x2 - slice_size)
            y1 = max(0, y2 - slice_size)
            slices.append((x1, y1, x2, y2))
    return slices


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> list[int]:
    """Standard NMS. boxes in [x1, y1, x2, y2] format."""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
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
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def predict_with_sahi(
    model,
    image_path: str | Path,
    slice_size: int = 640,
    overlap_ratio: float = 0.25,
    conf: float = 0.01,
    iou: float = 0.6,
    device: str = "cuda",
    also_full_image: bool = True,
) -> list[dict]:
    """Run sliced inference on a single image.

    Args:
        model: ultralytics YOLO model
        image_path: path to image
        slice_size: tile size in pixels
        overlap_ratio: overlap between tiles (0.25 = 25%)
        also_full_image: also run on full resized image and merge

    Returns:
        list of dicts with image_id, category_id, bbox [x,y,w,h], score
    """
    image_path = Path(image_path)
    image_id = int(image_path.stem.split("_")[-1])
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size

    all_boxes = []  # [x1, y1, x2, y2]
    all_scores = []
    all_classes = []

    # Sliced predictions
    slices = _slice_image(img_w, img_h, slice_size, overlap_ratio)
    for sx1, sy1, sx2, sy2 in slices:
        crop = img.crop((sx1, sy1, sx2, sy2))
        results = model(crop, device=device, verbose=False, imgsz=slice_size, conf=conf, iou=iou, max_det=300)
        for r in results:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                bx1, by1, bx2, by2 = r.boxes.xyxy[i].tolist()
                # Map back to full image coords
                all_boxes.append([bx1 + sx1, by1 + sy1, bx2 + sx1, by2 + sy1])
                all_scores.append(float(r.boxes.conf[i].item()))
                all_classes.append(int(r.boxes.cls[i].item()))

    # Full image prediction (catches large objects that span tiles)
    if also_full_image:
        results = model(str(image_path), device=device, verbose=False, imgsz=slice_size, conf=conf, iou=iou, max_det=300)
        for r in results:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                bx1, by1, bx2, by2 = r.boxes.xyxy[i].tolist()
                all_boxes.append([bx1, by1, bx2, by2])
                all_scores.append(float(r.boxes.conf[i].item()))
                all_classes.append(int(r.boxes.cls[i].item()))

    if not all_boxes:
        return []

    boxes_arr = np.array(all_boxes)
    scores_arr = np.array(all_scores)
    classes_arr = np.array(all_classes)

    # Per-class NMS
    detections = []
    for cls_id in np.unique(classes_arr):
        mask = classes_arr == cls_id
        cls_boxes = boxes_arr[mask]
        cls_scores = scores_arr[mask]
        keep = _nms(cls_boxes, cls_scores, iou_threshold=iou)
        for k in keep:
            x1, y1, x2, y2 = cls_boxes[k]
            detections.append({
                "image_id": image_id,
                "category_id": int(cls_id),
                "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                "score": round(float(cls_scores[k]), 4),
            })

    return detections


def predict_dir_with_sahi(
    model,
    image_dir: str | Path,
    slice_size: int = 640,
    overlap_ratio: float = 0.25,
    conf: float = 0.01,
    iou: float = 0.6,
    device: str = "cuda",
) -> list[dict]:
    """Run sliced inference on all images in a directory."""
    image_dir = Path(image_dir)
    image_files = sorted(
        f for f in image_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    logger.info("SAHI processing %d images (slice=%d, overlap=%.0f%%)", len(image_files), slice_size, overlap_ratio * 100)

    all_predictions = []
    for img_path in image_files:
        preds = predict_with_sahi(model, img_path, slice_size, overlap_ratio, conf, iou, device)
        all_predictions.extend(preds)
    return all_predictions
