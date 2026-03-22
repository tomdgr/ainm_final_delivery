# Cutting-Edge Detection Models & Techniques (2025-2026)

## OUR DATA: 248 images, 356 classes, 22,731 annotations, ~92 objects/image

---

## Models (ranked by competition relevance)

### 1. YOLO26 (Sep 2025) — LATEST ULTRALYTICS
- NMS-free (end-to-end), ProgLoss + STAL for small objects
- mAP: YOLO26l = 53.0%, YOLO26m = 51.5%
- **43% faster on CPU**, best small-object performance
- `pip install ultralytics` — same API as YOLO11/v8
- https://arxiv.org/abs/2509.25164

### 2. RF-DETR (Mar 2025, ICLR 2026) — BEST FOR LOW DATA
- DINOv2 backbone = strongest pretrained features
- 60.5 mAP on COCO at 25 FPS on T4
- **Designed for fine-tuning** — converges in ~10 epochs
- `pip install rfdetr`
- https://github.com/roboflow/rf-detr

### 3. YOLO11 (Oct 2024)
- C2PSA spatial attention, 22% fewer params than v8
- YOLO11x = 51.2% mAP COCO
- Proven, stable, best tooling via Ultralytics

### 4. Grounding DINO — ZERO-SHOT DETECTION
- 52.5 AP zero-shot on COCO (no training!)
- Detect via text prompts: "grocery product on shelf"
- **Use to generate pseudo-labels** on unlabeled data
- https://github.com/IDEA-Research/GroundingDINO

### 5. YOLOE / YOLO-World — OPEN-VOCAB DETECTION
- Detect anything with text prompts, YOLO speed
- YOLOE: +3.5 AP over YOLO-World v2
- Use as pseudo-label generator
- Built into Ultralytics

### 6. Florence-2 (Microsoft) — FEW-SHOT VLM
- 0.23B-0.77B params, handles detection+segmentation+captioning
- Fine-tunable with few examples
- HuggingFace: `microsoft/Florence-2-large`

### 7. Co-DETR — MAX ACCURACY
- 66.0% AP COCO (ViT-L backbone)
- Heavy but highest accuracy available
- https://github.com/Sense-X/Co-DETR

---

## Critical Techniques

### SAHI (Slicing Aided Hyper Inference) — MUST USE
- **+5-7% AP on dense/small objects**
- Slices image into overlapping tiles, detects on each, merges
- `pip install sahi` — works with Ultralytics
- https://docs.ultralytics.com/guides/sahi-tiled-inference/
- **Critical for 2000x1500 shelf images with 92 products each**

### Copy-Paste Augmentation — CRITICAL FOR LOW DATA
- +3.6 mask AP on rare categories
- +20% accuracy with fewer images in some studies
- **Paste product instances onto different shelf backgrounds**
- Built into Ultralytics: `copy_paste=1.0`
- We have 345 product reference images — paste these onto training shelf images

### Weighted Boxes Fusion (WBF) — MUST USE FOR ENSEMBLE
- Averages ALL boxes instead of discarding (unlike NMS)
- Used by virtually every Kaggle detection winner
- `pip install ensemble-boxes`
- https://github.com/ZFTurbo/Weighted-Boxes-Fusion

### TTA (Test-Time Augmentation)
- +1-2% mAP, flip + multi-scale
- Built into Ultralytics: `--augment` flag
- 2-3x slower but worth it for 3 submissions/day

### Pseudo-Labeling Pipeline
1. Train initial model on 248 labeled images
2. Use Grounding DINO zero-shot on extra shelf images
3. Filter pseudo-labels with confidence > 0.7
4. Retrain with combined data
5. +2-4% mAP improvement

---

## Few-Shot Detection SOTA (for 356 classes with few examples each)

| Method | Setting | Performance | Notes |
|--------|---------|-------------|-------|
| CD-ViTO | Cross-domain few-shot | NTIRE 2025 winner | DE-ViT + learnable features |
| DE-ViT | 10-shot | +15 mAP over prior SOTA | DINOv2 backbone |
| FII-DETR | 1-shot / 3-shot | +6.5% over Meta-DETR | Extreme few-shot |

---

## Pretraining Strategy

Best approach for low data:
1. Start with Objects365-pretrained weights (365 categories, 2M images)
2. Fine-tune on COCO
3. Fine-tune on competition data
- RF-DETR and Co-DETR checkpoints already include Objects365 pretraining

---

## 69-Hour Game Plan

### Hours 0-8: Baselines
- Convert COCO annotations to YOLO format
- Train YOLO26-l on 248 images (mosaic + copy-paste enabled)
- Train RF-DETR-base (DINOv2 backbone, few-shot friendly)

### Hours 8-20: Data Expansion
- Use Grounding DINO zero-shot on any unlabeled shelf images
- Copy-paste augmentation with 345 product reference images
- Download Kassalapp product images for more copy-paste material

### Hours 20-50: Multi-Model Training
- YOLO26-l (Objects365 pretrained)
- RF-DETR (DINOv2 backbone)
- YOLO11-x (different augmentation config)
- Multi-scale training: 640, 1024, 1280

### Hours 50-65: Ensemble + Inference
- SAHI sliced inference on high-res images
- TTA: flip + multi-scale
- WBF ensemble all models

### Hours 65-69: Final Tuning
- Optimize confidence/IoU thresholds on validation
- **ASK USER before any submission**
