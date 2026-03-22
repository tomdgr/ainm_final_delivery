"""ONNX inference for models not in the sandbox (YOLO11, RF-DETR, etc).

Handles the full pipeline: load ONNX model → preprocess → inference → NMS → output.
Works with CUDAExecutionProvider on sandbox L4 GPU.
"""
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ONNXDetector:
    """Run detection using an ONNX model with onnxruntime."""

    def __init__(
        self,
        model_path: str | Path,
        imgsz: int = 640,
        conf: float = 0.01,
        iou: float = 0.6,
        nc: int = 356,
    ):
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.nc = nc

        # Check output shape to determine post-processing
        output_shape = self.session.get_outputs()[0].shape
        logger.info("ONNX model loaded: input=%s, output=%s", self.session.get_inputs()[0].shape, output_shape)

    def _preprocess(self, img: Image.Image) -> tuple[np.ndarray, float, float, int, int]:
        """Resize with letterbox padding, normalize."""
        orig_w, orig_h = img.size
        scale = min(self.imgsz / orig_w, self.imgsz / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)

        # Pad to imgsz x imgsz
        padded = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        pad_x = (self.imgsz - new_w) // 2
        pad_y = (self.imgsz - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = np.array(img_resized)

        # HWC -> CHW, normalize
        blob = padded.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
        return blob, scale, scale, pad_x, pad_y

    def _postprocess_yolo(
        self, output: np.ndarray, scale: float, pad_x: int, pad_y: int, orig_w: int, orig_h: int
    ) -> list[dict]:
        """Post-process YOLO-style output: [1, 4+nc, num_boxes] or [1, num_boxes, 4+nc]."""
        # Handle different output formats
        if output.ndim == 3:
            if output.shape[1] == (4 + self.nc):
                # [1, 4+nc, N] -> [N, 4+nc]
                output = output[0].T
            elif output.shape[2] == (4 + self.nc):
                # [1, N, 4+nc]
                output = output[0]
            else:
                logger.warning("Unexpected output shape: %s", output.shape)
                return []

        # output is [N, 4+nc]: cx, cy, w, h, class_scores...
        boxes_cxcywh = output[:, :4]
        class_scores = output[:, 4:]

        # Best class per box
        class_ids = class_scores.argmax(axis=1)
        max_scores = class_scores.max(axis=1)

        # Filter by confidence
        mask = max_scores > self.conf
        boxes_cxcywh = boxes_cxcywh[mask]
        class_ids = class_ids[mask]
        max_scores = max_scores[mask]

        if len(boxes_cxcywh) == 0:
            return []

        # Convert to xyxy
        x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

        # Remove padding and rescale to original image
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        # Clip
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        # NMS per class
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        detections = []
        for cls_id in np.unique(class_ids):
            cls_mask = class_ids == cls_id
            cls_boxes = boxes_xyxy[cls_mask]
            cls_scores = max_scores[cls_mask]
            keep = self._nms(cls_boxes, cls_scores, self.iou)
            for k in keep:
                bx1, by1, bx2, by2 = cls_boxes[k]
                detections.append({
                    "category_id": int(cls_id),
                    "bbox": [round(float(bx1), 1), round(float(by1), 1), round(float(bx2 - bx1), 1), round(float(by2 - by1), 1)],
                    "score": round(float(cls_scores[k]), 4),
                })
        return detections

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
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
            order = order[np.where(iou <= iou_threshold)[0] + 1]
        return keep

    def predict(self, image_path: str | Path) -> list[dict]:
        image_path = Path(image_path)
        image_id = int(image_path.stem.split("_")[-1])
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size

        blob, scale_x, scale_y, pad_x, pad_y = self._preprocess(img)
        outputs = self.session.run(None, {self.input_name: blob})

        detections = self._postprocess_yolo(outputs[0], scale_x, pad_x, pad_y, orig_w, orig_h)
        for d in detections:
            d["image_id"] = image_id
        return detections

    def predict_dir(self, image_dir: str | Path) -> list[dict]:
        image_dir = Path(image_dir)
        image_files = sorted(f for f in image_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
        logger.info("ONNX processing %d images", len(image_files))
        all_preds = []
        for img_path in image_files:
            all_preds.extend(self.predict(img_path))
        return all_preds


def export_yolo_to_onnx(weights: str | Path, imgsz: int = 640, opset: int = 17) -> Path:
    """Export an ultralytics model to ONNX format."""
    from ultralytics import YOLO
    model = YOLO(str(weights))
    export_path = model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=True)
    logger.info("Exported ONNX model to %s", export_path)
    return Path(export_path)
