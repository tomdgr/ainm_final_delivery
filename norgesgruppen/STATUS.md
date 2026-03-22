# NorgesGruppen Detection — Team Status

**Last updated**: 2026-03-19 ~21:00 CET (competition hour ~3)

---

## Competition Quick Ref
- **Score** = 0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5
- **Sandbox**: Python 3.11, L4 GPU (24GB), 300s timeout, no network
- **Submissions**: 3/day, resets midnight UTC
- **Data**: 248 images, 356 classes, 22,731 annotations, very long-tailed (74 classes <5 samples)

## What's Running on Azure ML

| Job | Model | Config | Status |
|-----|-------|--------|--------|
| `shy_grass` | YOLOv8x | 640px, 150ep, batch 8 | Running |
| `plucky_hand` | YOLOv8x | 1280px, 100ep, batch 2, multi-scale | Running |
| `modest_line` | YOLOv8l | 640px, 200ep, heavy aug (mixup=0.3) | Running |
| `placid_balloon` | YOLOv8m | 640px, 200ep, SGD, seed=123 | Running |
| `busy_window` | YOLOv8s | 640px, 300ep, batch 32 | Queued |
| `shy_yacht` | **YOLO11x** | 640px, 150ep + ONNX export | Starting |

YOLOv8 jobs use `ultralytics==8.1.0` (sandbox .pt). YOLO11 uses latest ultralytics + ONNX export.

## Architecture Variants In Progress

### 1. YOLOv8 family (running now)
- 5 models with different sizes/configs for WBF ensemble
- Native .pt works in sandbox, no export needed
- **Limitation**: pinned to 8.1.0 (older)

### 2. YOLO11 via ONNX (building)
- Train with latest ultralytics, export to ONNX opset 17
- Better architecture than v8, but needs ONNX inference path
- **Risk**: ONNX export adds complexity, NMS handling differs

### 3. SAHI Tiling (building)
- Images are 2000-4000px, model sees 640px → small products missed
- Slice image into overlapping tiles, run detector on each, merge with NMS
- Free accuracy boost, no retraining needed
- `supervision` is pre-installed in sandbox

### 4. Two-Stage: Detect → Classify (built, ready)
- YOLOv8 finds boxes → crop → ResNet50 embedding → match against gallery
- Gallery built: 356/356 classes covered (1,577 reference + 2,836 crop embeddings)
- Targets the 30% classification score, especially rare classes (<5 samples)
- **Files**: `nm_ai_image/detection/classifier.py`, `data/reference_embeddings.pt`

## Package Structure

```
nm_ai_image/detection/
├── __init__.py          # Lazy imports
├── data.py              # COCOToYOLO converter
├── train.py             # Ultralytics training wrapper
├── inference.py         # Detector + EnsembleDetector (WBF)
├── classifier.py        # GalleryBuilder + EmbeddingClassifier
├── evaluate.py          # Competition scoring (70/30 split)
└── submission.py        # ZIP builder (single/ensemble/twostage)
```

**CLI**:
```bash
python main.py detect --model yolov8x.pt --imgsz 640 --epochs 150
python main.py eval weights.pt
python main.py gallery
python main.py submission best.pt                          # single
python main.py submission m1.pt m2.pt --ensemble           # WBF
python main.py submission best.pt --gallery embeddings.pt  # two-stage
```

## Submission Strategy (planned)

**Submission 1** (conservative): Best single YOLOv8 model with TTA
**Submission 2** (ensemble): WBF of 3+ YOLOv8 models
**Submission 3** (full pipeline): Ensemble + SAHI tiling + two-stage classifier

## Key Decisions & Why

| Decision | Why |
|----------|-----|
| ultralytics over torchvision detectors | 10x faster to iterate, proven on similar tasks |
| WBF over NMS for ensemble | ensemble-boxes pre-installed in sandbox, better for multi-model |
| ResNet50 for classifier | Pre-installed via torchvision, no extra weight files needed |
| ONNX for non-8.1.0 models | Only way to use YOLO11/RF-DETR in sandbox |
| Multiple YOLOv8 sizes | Ensemble diversity > single best model (Kaggle wisdom) |

## Known Issues
- RT-DETR-l broken on ultralytics 8.1.0 (stride attribute error)
- Azure ML image has numpy/matplotlib conflicts (fixed with `pip install "numpy<2"`)
- `sys` may be blocked in sandbox — need to verify our run.py doesn't use it
- 74 classes have <5 training samples — classifier gallery is critical for these

## What's NOT Done Yet
- [ ] YOLO11 ONNX training + export
- [ ] SAHI tiling inference
- [ ] Local testing with Docker (no starter kit yet)
- [ ] Actual competition submission (waiting for first trained weights)
- [ ] Hyperparameter tuning based on eval results
