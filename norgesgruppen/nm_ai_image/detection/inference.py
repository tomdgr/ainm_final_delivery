"""Detection inference with single-model TTA and multi-model WBF ensemble."""
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single detection result."""
    image_id: int
    category_id: int
    bbox: list[float]  # [x, y, w, h] COCO format
    score: float


class Detector:
    """Single-model detector with optional TTA."""

    def __init__(self, weights: str | Path, imgsz: int = 640, conf: float = 0.01, iou: float = 0.6, tta: bool = True):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.tta = tta
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def predict(self, image_path: str | Path) -> list[Detection]:
        """Run detection on a single image."""
        image_path = Path(image_path)
        image_id = int(image_path.stem.split("_")[-1])

        results = self.model(
            str(image_path),
            device=self.device,
            verbose=False,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            augment=self.tta,
            max_det=300,
        )

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                detections.append(Detection(
                    image_id=image_id,
                    category_id=int(r.boxes.cls[i].item()),
                    bbox=[round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                    score=round(float(r.boxes.conf[i].item()), 4),
                ))
        return detections

    def predict_dir(self, image_dir: str | Path) -> list[Detection]:
        """Run detection on all images in a directory."""
        image_dir = Path(image_dir)
        image_files = sorted(
            f for f in image_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        logger.info("Processing %d images with %s", len(image_files), type(self).__name__)

        all_detections = []
        for img_path in image_files:
            all_detections.extend(self.predict(img_path))
        return all_detections


class EnsembleDetector:
    """Multi-model ensemble with Weighted Boxes Fusion."""

    def __init__(self, detectors: list[Detector], iou_thr: float = 0.55, skip_box_thr: float = 0.01):
        self.detectors = detectors
        self.iou_thr = iou_thr
        self.skip_box_thr = skip_box_thr

    def predict(self, image_path: str | Path) -> list[Detection]:
        """Run all detectors and fuse results with WBF."""
        from ensemble_boxes import weighted_boxes_fusion

        image_path = Path(image_path)
        image_id = int(image_path.stem.split("_")[-1])
        img = Image.open(image_path)
        w, h = img.size

        all_boxes, all_scores, all_labels = [], [], []
        for detector in self.detectors:
            dets = detector.predict(image_path)
            boxes, scores, labels = [], [], []
            for d in dets:
                x, y, bw, bh = d.bbox
                # Normalize to [0, 1] for WBF (xyxy format)
                boxes.append([x / w, y / h, (x + bw) / w, (y + bh) / h])
                scores.append(d.score)
                labels.append(d.category_id)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        if not any(len(b) > 0 for b in all_boxes):
            return []

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            iou_thr=self.iou_thr,
            skip_box_thr=self.skip_box_thr,
        )

        detections = []
        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            x1, y1, x2, y2 = box
            detections.append(Detection(
                image_id=image_id,
                category_id=int(label),
                bbox=[round(x1 * w, 1), round(y1 * h, 1), round((x2 - x1) * w, 1), round((y2 - y1) * h, 1)],
                score=round(float(score), 4),
            ))
        return detections

    def predict_dir(self, image_dir: str | Path) -> list[Detection]:
        image_dir = Path(image_dir)
        image_files = sorted(
            f for f in image_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        logger.info("Ensemble processing %d images with %d models", len(image_files), len(self.detectors))

        all_detections = []
        for img_path in image_files:
            all_detections.extend(self.predict(img_path))
        return all_detections
