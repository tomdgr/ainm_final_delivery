"""Build competition submission ZIP files."""
import json
import logging
import shutil
import zipfile
from pathlib import Path

from nm_ai_image.detection.inference import Detection

logger = logging.getLogger(__name__)

RUN_PY_TEMPLATE = '''\
"""Competition submission — auto-generated run.py."""
import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO
{extra_imports}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
{model_loading}
{inference_loop}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

if __name__ == "__main__":
    main()
'''


class SubmissionBuilder:
    """Builds a competition-ready ZIP file."""

    def __init__(self, output_dir: str | Path = "submissions"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_single_model(
        self,
        weights_path: str | Path,
        name: str = "submission",
        imgsz: int = 640,
        conf: float = 0.01,
        iou: float = 0.6,
        tta: bool = True,
    ) -> Path:
        """Build a single-model submission ZIP."""
        weights_path = Path(weights_path)
        build_dir = self.output_dir / f"{name}_build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)

        # Copy weights
        shutil.copy2(weights_path, build_dir / weights_path.name)

        # Generate run.py
        run_py = self._generate_single_run_py(weights_path.name, imgsz, conf, iou, tta)
        (build_dir / "run.py").write_text(run_py)

        # Create ZIP
        zip_path = self.output_dir / f"{name}.zip"
        self._create_zip(build_dir, zip_path)
        shutil.rmtree(build_dir)

        size_mb = zip_path.stat().st_size / (1024 * 1024)
        logger.info("Built submission: %s (%.1f MB)", zip_path, size_mb)
        return zip_path

    def build_ensemble(
        self,
        weight_paths: list[str | Path],
        name: str = "ensemble_submission",
        imgsz: int = 640,
        conf: float = 0.01,
        iou: float = 0.6,
        wbf_iou_thr: float = 0.55,
    ) -> Path:
        """Build a multi-model WBF ensemble submission ZIP."""
        build_dir = self.output_dir / f"{name}_build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)

        # Copy all weights
        weight_names = []
        for wp in weight_paths:
            wp = Path(wp)
            shutil.copy2(wp, build_dir / wp.name)
            weight_names.append(wp.name)

        run_py = self._generate_ensemble_run_py(weight_names, imgsz, conf, iou, wbf_iou_thr)
        (build_dir / "run.py").write_text(run_py)

        zip_path = self.output_dir / f"{name}.zip"
        self._create_zip(build_dir, zip_path)
        shutil.rmtree(build_dir)

        size_mb = zip_path.stat().st_size / (1024 * 1024)
        logger.info("Built ensemble submission: %s (%.1f MB, %d models)", zip_path, size_mb, len(weight_paths))
        return zip_path

    def build_twostage(
        self,
        detector_weights: str | Path,
        gallery_path: str | Path,
        name: str = "twostage_submission",
        imgsz: int = 640,
        conf: float = 0.01,
        cls_threshold: float = 0.7,
        backbone_weights: str | Path | None = None,
    ) -> Path:
        """Build a two-stage (detect + classify) submission ZIP."""
        build_dir = self.output_dir / f"{name}_build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)

        detector_weights = Path(detector_weights)
        gallery_path = Path(gallery_path)
        shutil.copy2(detector_weights, build_dir / detector_weights.name)
        shutil.copy2(gallery_path, build_dir / "reference_embeddings.pt")

        # Copy backbone weights if provided (for offline sandbox)
        use_local_backbone = backbone_weights is not None
        if use_local_backbone:
            backbone_weights = Path(backbone_weights)
            shutil.copy2(backbone_weights, build_dir / "resnet50_backbone.pt")

        # Write the two-stage run.py inline (to stay within 10 .py file limit)
        run_py = self._generate_twostage_run_py(detector_weights.name, imgsz, conf, cls_threshold, use_local_backbone)
        (build_dir / "run.py").write_text(run_py)

        zip_path = self.output_dir / f"{name}.zip"
        self._create_zip(build_dir, zip_path)
        shutil.rmtree(build_dir)

        size_mb = zip_path.stat().st_size / (1024 * 1024)
        logger.info("Built two-stage submission: %s (%.1f MB)", zip_path, size_mb)
        return zip_path

    def build_sahi(
        self,
        weights_path: str | Path,
        name: str = "sahi_submission",
        imgsz: int = 640,
        slice_size: int = 640,
        overlap: float = 0.25,
        conf: float = 0.01,
    ) -> Path:
        """Build a SAHI tiled-inference submission ZIP."""
        weights_path = Path(weights_path)
        build_dir = self.output_dir / f"{name}_build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)
        shutil.copy2(weights_path, build_dir / weights_path.name)
        run_py = self._generate_sahi_run_py(weights_path.name, imgsz, slice_size, overlap, conf)
        (build_dir / "run.py").write_text(run_py)
        zip_path = self.output_dir / f"{name}.zip"
        self._create_zip(build_dir, zip_path)
        shutil.rmtree(build_dir)
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        logger.info("Built SAHI submission: %s (%.1f MB)", zip_path, size_mb)
        return zip_path

    def build_onnx(
        self,
        onnx_path: str | Path,
        name: str = "onnx_submission",
        imgsz: int = 640,
        nc: int = 356,
        conf: float = 0.01,
        nms_iou: float = 0.5,
        use_soft_nms: bool = False,
        soft_sigma: float = 0.5,
    ) -> Path:
        """Build an ONNX-based submission ZIP."""
        onnx_path = Path(onnx_path)
        build_dir = self.output_dir / f"{name}_build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)
        shutil.copy2(onnx_path, build_dir / onnx_path.name)
        run_py = self._generate_onnx_run_py(onnx_path.name, imgsz, nc, conf, nms_iou, use_soft_nms, soft_sigma)
        (build_dir / "run.py").write_text(run_py)
        zip_path = self.output_dir / f"{name}.zip"
        self._create_zip(build_dir, zip_path)
        shutil.rmtree(build_dir)
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        logger.info("Built ONNX submission: %s (%.1f MB)", zip_path, size_mb)
        return zip_path

    def build_onnx_ensemble(
        self,
        onnx_paths: list[str | Path],
        name: str = "onnx_ensemble",
        imgsz: int = 640,
        nc: int = 356,
        conf: float = 0.01,
        wbf_iou_thr: float = 0.55,
        nms_iou: float = 0.5,
    ) -> Path:
        """Build a multi-ONNX WBF ensemble submission ZIP."""
        build_dir = self.output_dir / f"{name}_build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)

        onnx_names = []
        for op in onnx_paths:
            op = Path(op)
            shutil.copy2(op, build_dir / op.name)
            onnx_names.append(op.name)

        run_py = self._generate_onnx_ensemble_run_py(onnx_names, imgsz, nc, conf, wbf_iou_thr, nms_iou)
        (build_dir / "run.py").write_text(run_py)

        zip_path = self.output_dir / f"{name}.zip"
        self._create_zip(build_dir, zip_path)
        shutil.rmtree(build_dir)

        size_mb = zip_path.stat().st_size / (1024 * 1024)
        logger.info("Built ONNX ensemble submission: %s (%.1f MB, %d models)", zip_path, size_mb, len(onnx_paths))
        return zip_path

    @staticmethod
    def detections_to_json(detections: list[Detection]) -> list[dict]:
        return [
            {"image_id": d.image_id, "category_id": d.category_id, "bbox": d.bbox, "score": d.score}
            for d in detections
        ]

    def _create_zip(self, build_dir: Path, zip_path: Path):
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(build_dir.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(build_dir))

    def _generate_single_run_py(self, weights_name: str, imgsz: int, conf: float, iou: float, tta: bool) -> str:
        return f'''\
import argparse
import json
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("{weights_name}")
    predictions = []

    for img in sorted(Path(args.input).iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img.stem.split("_")[-1])
        results = model(str(img), device=device, verbose=False, imgsz={imgsz}, conf={conf}, iou={iou}, augment={tta}, max_det=300)
        for r in results:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                predictions.append({{"image_id": image_id, "category_id": int(r.boxes.cls[i].item()), "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)], "score": round(float(r.boxes.conf[i].item()), 4)}})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

if __name__ == "__main__":
    main()
'''

    def _generate_ensemble_run_py(self, weight_names: list[str], imgsz: int, conf: float, iou: float, wbf_iou_thr: float) -> str:
        weights_list = repr(weight_names)
        return f'''\
import argparse
import json
from pathlib import Path
import torch
from PIL import Image
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = [YOLO(w) for w in {weights_list}]
    predictions = []

    for img_path in sorted(Path(args.input).iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img_path.stem.split("_")[-1])
        img = Image.open(img_path)
        w, h = img.size

        all_boxes, all_scores, all_labels = [], [], []
        for model in models:
            results = model(str(img_path), device=device, verbose=False, imgsz={imgsz}, conf={conf}, iou={iou}, max_det=300)
            boxes, scores, labels = [], [], []
            for r in results:
                if r.boxes is None:
                    continue
                for i in range(len(r.boxes)):
                    x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                    boxes.append([x1/w, y1/h, x2/w, y2/h])
                    scores.append(float(r.boxes.conf[i].item()))
                    labels.append(int(r.boxes.cls[i].item()))
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        if any(len(b) > 0 for b in all_boxes):
            fb, fs, fl = weighted_boxes_fusion(all_boxes, all_scores, all_labels, iou_thr={wbf_iou_thr}, skip_box_thr={conf})
            for box, score, label in zip(fb, fs, fl):
                x1, y1, x2, y2 = box
                predictions.append({{"image_id": image_id, "category_id": int(label), "bbox": [round(x1*w, 1), round(y1*h, 1), round((x2-x1)*w, 1), round((y2-y1)*h, 1)], "score": round(float(score), 4)}})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

if __name__ == "__main__":
    main()
'''

    def _generate_twostage_run_py(self, weights_name: str, imgsz: int, conf: float, cls_threshold: float, use_local_backbone: bool = False, nc: int = 356) -> str:
        if use_local_backbone:
            backbone_loading = '''\
    backbone = models.resnet50(weights=None)
    backbone.fc = torch.nn.Identity()
    backbone.load_state_dict(torch.load("resnet50_backbone.pt", map_location=device, weights_only=True))
    backbone = backbone.to(device).eval()'''
        else:
            backbone_loading = '''\
    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    backbone.fc = torch.nn.Identity()
    backbone = backbone.to(device).eval()'''

        is_onnx = weights_name.endswith('.onnx')
        if is_onnx:
            return f'''\
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import onnxruntime as ort

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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
        iou=inter/(areas[i]+areas[order[1:]]-inter)
        order=order[np.where(iou<=thr)[0]+1]
    return keep

def detect_onnx(sess, input_name, img, ow, oh):
    blob, scale, px, py = preprocess(img)
    out = sess.run(None, {{input_name: blob}})[0]
    if out.ndim==3: out = out[0].T if out.shape[1]==(4+{nc}) else out[0]
    cx,cy,bw,bh = out[:,0],out[:,1],out[:,2],out[:,3]
    cls_scores = out[:,4:]; cls_ids = cls_scores.argmax(axis=1); max_sc = cls_scores.max(axis=1)
    mask = max_sc > {conf}
    x1=(cx-bw/2-px)/scale; y1=(cy-bh/2-py)/scale; x2=(cx+bw/2-px)/scale; y2=(cy+bh/2-py)/scale
    x1=np.clip(x1[mask],0,ow); y1=np.clip(y1[mask],0,oh); x2=np.clip(x2[mask],0,ow); y2=np.clip(y2[mask],0,oh)
    cls_ids=cls_ids[mask]; max_sc=max_sc[mask]; boxes=np.stack([x1,y1,x2,y2],axis=1)
    # Per-class NMS
    final_boxes, final_scores, final_cls = [], [], []
    for cid in np.unique(cls_ids):
        m=cls_ids==cid; keep=nms(boxes[m],max_sc[m],0.5)
        for k in keep:
            final_boxes.append(boxes[m][k])
            final_scores.append(float(max_sc[m][k]))
            final_cls.append(int(cid))
    return final_boxes, final_scores, final_cls

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sess = ort.InferenceSession("{weights_name}", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    gallery_data = torch.load("reference_embeddings.pt", map_location=device, weights_only=True)
    gallery = gallery_data["gallery"].to(device)
{backbone_loading}

    predictions = []
    for img_path in sorted(Path(args.input).iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img_path.stem.split("_")[-1])
        img = Image.open(img_path).convert("RGB")
        ow, oh = img.size

        det_boxes, det_scores, det_cls = detect_onnx(sess, input_name, img, ow, oh)
        if not det_boxes:
            continue

        crops, det_info = [], []
        for box, score, cid in zip(det_boxes, det_scores, det_cls):
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            crop = img.crop((max(0, x1 - bw*0.1), max(0, y1 - bh*0.1), min(ow, x2 + bw*0.1), min(oh, y2 + bh*0.1)))
            if crop.width >= 10 and crop.height >= 10:
                crops.append(crop)
                det_info.append((float(x1), float(y1), float(x2), float(y2), score, cid))

        if not crops:
            continue

        with torch.no_grad():
            for bi in range(0, len(crops), 32):
                batch = crops[bi:bi+32]
                tensors = torch.stack([TRANSFORM(c) for c in batch]).to(device)
                feats = F.normalize(backbone(tensors), dim=1)
                sims = feats @ gallery.T
                best_cats = sims.argmax(dim=1).tolist()
                best_scores = sims.max(dim=1).values.tolist()

                for j, (x1, y1, x2, y2, det_score, det_cat) in enumerate(det_info[bi:bi+32]):
                    cls_conf = max(0, (best_scores[j] + 1) / 2)
                    cat = best_cats[j] if cls_conf > {cls_threshold} else det_cat
                    score = det_score * 0.7 + cls_conf * 0.3 if cls_conf > {cls_threshold} else det_score
                    predictions.append({{"image_id": image_id, "category_id": cat, "bbox": [round(x1, 1), round(y1, 1), round(x2-x1, 1), round(y2-y1, 1)], "score": round(score, 4)}})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

if __name__ == "__main__":
    main()
'''

        return f'''\
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = YOLO("{weights_name}")

    gallery_data = torch.load("reference_embeddings.pt", map_location=device)
    gallery = gallery_data["gallery"].to(device)
{backbone_loading}

    predictions = []
    for img_path in sorted(Path(args.input).iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img_path.stem.split("_")[-1])
        results = detector(str(img_path), device=device, verbose=False, imgsz={imgsz}, conf={conf}, iou=0.6, augment=True, max_det=300)

        img = Image.open(img_path).convert("RGB")
        crops, det_info = [], []
        for r in results:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                bw, bh = x2 - x1, y2 - y1
                crop = img.crop((max(0, x1 - bw*0.1), max(0, y1 - bh*0.1), min(img.width, x2 + bw*0.1), min(img.height, y2 + bh*0.1)))
                if crop.width >= 10 and crop.height >= 10:
                    crops.append(crop)
                    det_info.append((x1, y1, x2, y2, float(r.boxes.conf[i].item()), int(r.boxes.cls[i].item())))

        if not crops:
            continue

        with torch.no_grad():
            for bi in range(0, len(crops), 32):
                batch = crops[bi:bi+32]
                tensors = torch.stack([TRANSFORM(c) for c in batch]).to(device)
                feats = F.normalize(backbone(tensors), dim=1)
                sims = feats @ gallery.T
                best_cats = sims.argmax(dim=1).tolist()
                best_scores = sims.max(dim=1).values.tolist()

                for j, (x1, y1, x2, y2, det_score, det_cat) in enumerate(det_info[bi:bi+32]):
                    cls_conf = max(0, (best_scores[j] + 1) / 2)
                    cat = best_cats[j] if cls_conf > {cls_threshold} else det_cat
                    score = det_score * 0.7 + cls_conf * 0.3 if cls_conf > {cls_threshold} else det_score
                    predictions.append({{"image_id": image_id, "category_id": cat, "bbox": [round(x1, 1), round(y1, 1), round(x2-x1, 1), round(y2-y1, 1)], "score": round(score, 4)}})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

if __name__ == "__main__":
    main()
'''

    def _generate_sahi_run_py(self, weights_name: str, imgsz: int, slice_size: int, overlap: float, conf: float) -> str:
        return f'''\
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

def slice_image(w, h, sz={slice_size}, ov={overlap}):
    step = int(sz * (1 - ov))
    slices = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            x2, y2 = min(x + sz, w), min(y + sz, h)
            slices.append((max(0, x2 - sz), max(0, y2 - sz), x2, y2))
    return slices

def nms(boxes, scores, thr=0.5):
    if len(boxes) == 0: return []
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas = (x2-x1)*(y2-y1); order = scores.argsort()[::-1]; keep = []
    while len(order) > 0:
        i = order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        inter = np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
        iou = inter/(areas[i]+areas[order[1:]]-inter)
        order = order[np.where(iou <= thr)[0] + 1]
    return keep

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("{weights_name}")
    predictions = []
    for img_path in sorted(Path(args.input).iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"): continue
        image_id = int(img_path.stem.split("_")[-1])
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        all_boxes, all_scores, all_cls = [], [], []
        for sx1, sy1, sx2, sy2 in slice_image(w, h):
            crop = img.crop((sx1, sy1, sx2, sy2))
            results = model(crop, device=device, verbose=False, imgsz={imgsz}, conf={conf}, iou=0.7, max_det=300)
            for r in results:
                if r.boxes is None: continue
                for i in range(len(r.boxes)):
                    bx1,by1,bx2,by2 = r.boxes.xyxy[i].tolist()
                    all_boxes.append([bx1+sx1,by1+sy1,bx2+sx1,by2+sy1])
                    all_scores.append(float(r.boxes.conf[i].item()))
                    all_cls.append(int(r.boxes.cls[i].item()))
        results = model(str(img_path), device=device, verbose=False, imgsz={imgsz}, conf={conf}, iou=0.7, max_det=300)
        for r in results:
            if r.boxes is None: continue
            for i in range(len(r.boxes)):
                bx1,by1,bx2,by2 = r.boxes.xyxy[i].tolist()
                all_boxes.append([bx1,by1,bx2,by2])
                all_scores.append(float(r.boxes.conf[i].item()))
                all_cls.append(int(r.boxes.cls[i].item()))
        if not all_boxes: continue
        boxes_arr=np.array(all_boxes); scores_arr=np.array(all_scores); cls_arr=np.array(all_cls)
        for cid in np.unique(cls_arr):
            m = cls_arr==cid; keep = nms(boxes_arr[m], scores_arr[m], 0.5)
            for k in keep:
                b = boxes_arr[m][k]
                predictions.append({{"image_id": image_id, "category_id": int(cid), "bbox": [round(b[0],1),round(b[1],1),round(b[2]-b[0],1),round(b[3]-b[1],1)], "score": round(float(scores_arr[m][k]),4)}})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

if __name__ == "__main__":
    main()
'''

    def _generate_onnx_run_py(self, onnx_name: str, imgsz: int, nc: int, conf: float,
                               nms_iou: float = 0.5, use_soft_nms: bool = False, soft_sigma: float = 0.5) -> str:
        soft_nms_fn = ""
        if use_soft_nms:
            soft_nms_fn = f"""
def soft_nms(boxes, scores, sigma={soft_sigma}, score_thr=0.001):
    boxes, scores = boxes.copy(), scores.copy()
    keep_b, keep_s = [], []
    while len(scores) > 0:
        mi = scores.argmax()
        keep_b.append(boxes[mi]); keep_s.append(scores[mi])
        x1 = np.maximum(boxes[mi, 0], boxes[:, 0]); y1 = np.maximum(boxes[mi, 1], boxes[:, 1])
        x2 = np.minimum(boxes[mi, 2], boxes[:, 2]); y2 = np.minimum(boxes[mi, 3], boxes[:, 3])
        inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
        aa = (boxes[mi,2]-boxes[mi,0])*(boxes[mi,3]-boxes[mi,1])
        ab = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
        iou = inter / (aa + ab - inter + 1e-6)
        scores = scores * np.exp(-iou**2 / sigma)
        mask = np.ones(len(scores), dtype=bool); mask[mi] = False; mask &= scores > score_thr
        boxes, scores = boxes[mask], scores[mask]
    return keep_b, keep_s
"""
        nms_call = f"keep = nms(boxes[m], max_sc[m], {nms_iou})"
        nms_append = """for k in keep:
                b = boxes[m][k]
                predictions.append({"image_id": image_id, "category_id": int(cid), "bbox": [round(float(b[0]),1),round(float(b[1]),1),round(float(b[2]-b[0]),1),round(float(b[3]-b[1]),1)], "score": round(float(max_sc[m][k]),4)})"""

        if use_soft_nms:
            nms_call = "kb, ks = soft_nms(boxes[m], max_sc[m])"
            nms_append = """for b, s in zip(kb, ks):
                predictions.append({"image_id": image_id, "category_id": int(cid), "bbox": [round(float(b[0]),1),round(float(b[1]),1),round(float(b[2]-b[0]),1),round(float(b[3]-b[1]),1)], "score": round(float(s),4)})"""

        return f'''\
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort

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
        iou=inter/(areas[i]+areas[order[1:]]-inter)
        order=order[np.where(iou<=thr)[0]+1]
    return keep
{soft_nms_fn}
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sess = ort.InferenceSession("{onnx_name}", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    predictions = []
    for img_path in sorted(Path(args.input).iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"): continue
        image_id = int(img_path.stem.split("_")[-1])
        img = Image.open(img_path).convert("RGB"); ow, oh = img.size
        blob, scale, px, py = preprocess(img)
        out = sess.run(None, {{input_name: blob}})[0]
        if out.ndim==3: out = out[0].T if out.shape[1]==(4+{nc}) else out[0]
        cx,cy,bw,bh = out[:,0],out[:,1],out[:,2],out[:,3]
        cls_scores = out[:,4:]; cls_ids = cls_scores.argmax(axis=1); max_sc = cls_scores.max(axis=1)
        mask = max_sc > {conf}
        x1=(cx-bw/2-px)/scale; y1=(cy-bh/2-py)/scale; x2=(cx+bw/2-px)/scale; y2=(cy+bh/2-py)/scale
        x1=np.clip(x1[mask],0,ow); y1=np.clip(y1[mask],0,oh); x2=np.clip(x2[mask],0,ow); y2=np.clip(y2[mask],0,oh)
        cls_ids=cls_ids[mask]; max_sc=max_sc[mask]; boxes=np.stack([x1,y1,x2,y2],axis=1)
        for cid in np.unique(cls_ids):
            m=cls_ids==cid
            {nms_call}
            {nms_append}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

if __name__ == "__main__":
    main()
'''

    def _generate_onnx_ensemble_run_py(self, onnx_names: list[str], imgsz: int, nc: int, conf: float,
                                         wbf_iou_thr: float, nms_iou: float) -> str:
        names_repr = repr(onnx_names)
        return f'''\
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort
from ensemble_boxes import weighted_boxes_fusion

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
        iou=inter/(areas[i]+areas[order[1:]]-inter)
        order=order[np.where(iou<=thr)[0]+1]
    return keep

def detect_onnx(sess, input_name, img, nc={nc}, conf_thr={conf}):
    ow, oh = img.size
    blob, scale, px, py = preprocess(img)
    out = sess.run(None, {{input_name: blob}})[0]
    if out.ndim==3: out = out[0].T if out.shape[1]==(4+nc) else out[0]
    cx,cy,bw,bh = out[:,0],out[:,1],out[:,2],out[:,3]
    cls_scores = out[:,4:]; cls_ids = cls_scores.argmax(axis=1); max_sc = cls_scores.max(axis=1)
    mask = max_sc > conf_thr
    x1=np.clip((cx-bw/2-px)/scale,0,ow)[mask]; y1=np.clip((cy-bh/2-py)/scale,0,oh)[mask]
    x2=np.clip((cx+bw/2-px)/scale,0,ow)[mask]; y2=np.clip((cy+bh/2-py)/scale,0,oh)[mask]
    cls_ids=cls_ids[mask]; max_sc=max_sc[mask]; boxes=np.stack([x1,y1,x2,y2],axis=1)
    # Per-class NMS
    final_b, final_s, final_c = [], [], []
    for cid in np.unique(cls_ids):
        m=cls_ids==cid; keep=nms(boxes[m],max_sc[m],{nms_iou})
        for k in keep:
            final_b.append(boxes[m][k]); final_s.append(float(max_sc[m][k])); final_c.append(int(cid))
    return final_b, final_s, final_c

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Loading models...")
    sessions = []
    input_names = []
    for name in {names_repr}:
        sess = ort.InferenceSession(name, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        sessions.append(sess)
        input_names.append(sess.get_inputs()[0].name)
    print(f"Loaded {{len(sessions)}} ONNX models")

    predictions = []
    for img_path in sorted(Path(args.input).iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"): continue
        image_id = int(img_path.stem.split("_")[-1])
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        all_boxes, all_scores, all_labels = [], [], []
        for sess, inp_name in zip(sessions, input_names):
            det_b, det_s, det_c = detect_onnx(sess, inp_name, img)
            boxes, scores, labels = [], [], []
            for b, s, c in zip(det_b, det_s, det_c):
                boxes.append([b[0]/w, b[1]/h, b[2]/w, b[3]/h])
                scores.append(s)
                labels.append(c)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        if any(len(b)>0 for b in all_boxes):
            fb, fs, fl = weighted_boxes_fusion(all_boxes, all_scores, all_labels, iou_thr={wbf_iou_thr}, skip_box_thr={conf})
            for box, score, label in zip(fb, fs, fl):
                x1, y1, x2, y2 = box
                predictions.append({{"image_id": image_id, "category_id": int(label), "bbox": [round(x1*w,1), round(y1*h,1), round((x2-x1)*w,1), round((y2-y1)*h,1)], "score": round(float(score),4)}})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)
    print(f"Wrote {{len(predictions)}} predictions")

if __name__ == "__main__":
    main()
'''
