# NM i AI 2026 — Team Experis

**Norwegian AI Championship 2026** (Mar 19-22, 69 hours)

Team: Tom Daniel Grande, Henrik Skulevold, Tobias Korten, Fridtjof Hoyer

---

## Task 2: NorgesGruppen Data — Object Detection

Detect and classify 356 grocery product categories on store shelf images.

**Scoring:** `0.7 * detection_mAP@0.5 + 0.3 * classification_mAP@0.5`

### Best Submission: 0.9226

**ZIP:** `sub2_c11_yolo11x_swap_verified.zip`

**Models (3-model WBF ensemble):**

| Model | Architecture | Training | ONNX File |
|-------|-------------|----------|-----------|
| fulldata_x | YOLOv8x | All data, seed 42, 150ep, 640px | `yolov8x_640_fulldata_best_fp16_dynamic.onnx` |
| nf_11x_s42 | YOLO11x | Noflip aug, seed 42, 150ep, 640px | `nf_11x_s42_fp16_dyn.onnx` |
| v8l_highaug | YOLOv8l | Heavy aug, seed 42, 150ep, 640px | `yolov8l_640_highaug_fp16_dyn.onnx` |

**Inference pipeline:**
- Multi-scale: 640px + 800px
- TTA: horizontal flip
- Weighted Boxes Fusion (WBF): iou=0.6, conf_type='absent_model_aware_avg'
- 3 models x 2 scales x 2 flips = 12 inference passes per image

### Reproduce Best Model

**1. Install dependencies:**
```bash
pip install uv
uv sync
```

**2. Train the 3 models:**
```bash
# Model 1: YOLOv8x fulldata (seed 42)
uv run python -c "
from nm_ai_image.detection.data import COCOToYOLO
from ultralytics import YOLO
c = COCOToYOLO('data/raw/coco_dataset/train', 'data/yolo', val_ratio=0.0, seed=42)
c.convert()
m = YOLO('yolov8x.pt')
m.train(data=str((c.output_dir / 'data.yaml').resolve()), imgsz=640, epochs=150, batch=8, name='fulldata_x', seed=42, patience=0)
"

# Model 2: YOLO11x noflip (seed 42)
uv run python -c "
from nm_ai_image.detection.data import COCOToYOLO
from ultralytics import YOLO
c = COCOToYOLO('data/raw/coco_dataset/train', 'data/yolo', val_ratio=0.0, seed=42)
c.convert()
m = YOLO('yolo11x.pt')
m.train(data=str((c.output_dir / 'data.yaml').resolve()), imgsz=640, epochs=150, batch=8, name='nf_11x_s42', seed=42, patience=0, flipud=0.0)
"

# Model 3: YOLOv8l heavy augmentation (seed 42)
uv run python -c "
from nm_ai_image.detection.data import COCOToYOLO
from ultralytics import YOLO
c = COCOToYOLO('data/raw/coco_dataset/train', 'data/yolo', val_ratio=0.0, seed=42)
c.convert()
m = YOLO('yolov8l.pt')
m.train(data=str((c.output_dir / 'data.yaml').resolve()), imgsz=640, epochs=150, batch=8, name='v8l_highaug', seed=42, patience=0, mixup=0.3, copy_paste=0.3, degrees=10.0, scale=0.9)
"
```

**3. Export to ONNX:**
```bash
uv run python -c "
from ultralytics import YOLO
for name in ['fulldata_x', 'nf_11x_s42', 'v8l_highaug']:
    m = YOLO(f'runs/detect/{name}/weights/last.pt')
    m.export(format='onnx', imgsz=640, opset=17, simplify=True, dynamic=True)
"
```

**4. Build submission ZIP:**
```bash
uv run python scripts/build_multiscale_tta_submission.py
```

### Key Findings

- **Architecture diversity** was the single strongest signal: mixing YOLOv8x + YOLO11x + YOLOv8l outperformed same-architecture ensembles (+0.005)
- **Multi-scale inference** (640+800) gave +0.005 on test
- **WBF ensemble** with `absent_model_aware_avg` outperformed soft voting and standard NMS
- **Noflip training** (flipud=0.0) gave small but consistent improvement
- Model soup, external classifiers, gallery re-ranking, and soft-NMS all hurt performance
- 150+ training runs on Azure ML, 17 submissions tested

### Project Structure

```
nm_ai_image/          # Python package
  detection/          # Detection pipeline (data, training, inference, evaluation)
scripts/              # Utility scripts (submission builders, evaluation, caching)
jobs/                 # Azure ML job definitions (~100 YAML files)
plan/                 # Competition strategy documents
docs/                 # LaTeX report
test/                 # Pytest test suite
experiments.csv       # All 150+ training runs and submissions tracked
```

### Azure ML

Training was done on Azure ML with 8x NVIDIA T4 GPUs. Job definitions are in `jobs/`.

## License

MIT
