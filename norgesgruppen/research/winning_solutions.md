# Winning Solutions: Product Detection & Recognition Competitions

## !! OUR TASK IS OBJECT DETECTION (mAP@0.5) — NOT CLASSIFICATION !!

---

## SKU-110K Shelf Detection — Best Results

**Dataset:** 11,762 shelf images, 1.73M bounding boxes, densely packed products
**Challenge:** Products look similar, tightly packed, occluding each other

### Best Results on SKU-110K
| Model | mAP@0.5 | mAP@0.5:0.95 | Notes |
|-------|---------|---------------|-------|
| **YOLOv7** | **0.996** | **0.86** | Near-perfect at IoU 0.5 |
| DBA-YOLO | High | — | Better accuracy/efficiency than RT-DETR |
| YOLOv8 | ~0.99 | ~0.85 | Ultralytics, easiest to use |
| RT-DETR | Good | — | More params than YOLO for same accuracy |

**Verdict: YOLO dominates shelf detection.**
- Ultralytics has SKU-110K as a built-in dataset: https://docs.ultralytics.com/datasets/detect/sku-110k/

### Key Reference
- YOLOv8 on SKU-110K: F1=88.96% in 25 epochs (~4h on Kaggle GPU)
- Tutorial: https://medium.com/@rathnasekaratishan/densely-packed-product-detection-1c94a406802c

---

## GWFSS Challenge 2025 — 1st Place (Low Data Detection)

**Key:** Only 99 labeled training samples + 64,368 unlabeled
**Score:** 0.7704 (dev), 0.7468 (test) — 1st on both leaderboards

### Winning Tricks for Low Data
- **HVI color space** — learnable color transform to enhance contrast
- **Guided distillation** — teacher-student for robustness
- **Semi-supervised learning** — leverage unlabeled data with pseudo-labels

---

## Products-10K (ICPR 2020) — 1st Place: DeepBlueAI

**Score:** 0.73618 (private LB), single model 0.70918
**Metric:** Top-1 accuracy on 10,000 SKU classes

### Architecture Progression
1. **Baseline:** ResNeSt50 + horizontal flip → 61.26%
2. **+ BNNeck + Random Erasing + GeM Pooling** → ~68%
3. **+ CircleSoftmax + FocalLoss** → ~69%
4. **+ Higher resolution + deeper network** → ~71% (single)
5. **+ Model ensemble** → 73.01% (1st place)

### Key Tricks
- **CircleSoftmax** — margin-based loss for fine-grained discrimination
- **BNNeck** — BatchNorm before classification head
- **GeM Pooling** — Generalized Mean Pooling
- **Random Erasing** — occlusion augmentation
- **Resolution matters** — higher res = better

---

## Shopee Product Matching (Kaggle 2021) — Top Solutions

### Winning Pattern: EfficientNet + ArcFace + GeM
```
Backbone (EfficientNet-B3/B5) → GeM Pooling → BNNeck → ArcFace Head
```

### Code References
- https://www.kaggle.com/code/sandersli/shopee-product-matching-efficientnet-gem-arcface
- https://www.kaggle.com/code/debarshichanda/pytorch-arcface-gem-pooling-starter

---

## iMaterialist Product Recognition (FGVC6, CVPR 2019)

### 1st Place Solution
- **Hybrid Task Cascade** with ResNeXt-101-64x4d-FPN backbone
- **Multi-scale training** — short edge [600, 1200], long edge 1900
- Code: https://github.com/amirassov/kaggle-imaterialist

---

## Visual RAG Pipeline (CVPR 2025 FGVC12)

- Few-shot product classification: 86.8% accuracy without retraining
- RAG + VLM (GPT-4o / Gemini 2.0 Flash)
- Paper: https://arxiv.org/abs/2504.11838

---

## FathomNet 2025 Kaggle — Detection with Few Labeled Images

### 1st Place (Yonsei+SSL)
- Vision Transformer with **multi-scale image inputs**
- **Multi-context environmental attention** (MCEAM)
- Combined object-level and environmental context

---

## OBJECT DETECTION Winning Patterns

### Model Choice (for mAP@0.5)
1. **YOLOv8/v11 (Ultralytics)** — fastest, best tooling, proven on shelf detection
2. **Co-DETR** — SOTA detection transformer
3. **RT-DETR** — real-time DETR, good accuracy
4. **FasterRCNN + FPN** — classic, reliable
5. **YOLO-World** — open-vocab detection (could be useful if classes are unknown)

### Training Tricks for Detection
1. **Multi-scale training** — train at multiple image sizes (640, 800, 1024, 1280)
2. **Mosaic augmentation** — YOLO's signature, 4 images combined
3. **MixUp** — blend training images
4. **Copy-paste augmentation** — paste objects from one image to another
5. **Large image size** — shelf images are high-res, bigger = better for small products

### Low-Data Detection Tricks
1. **Pretrain on COCO or Objects365** — then finetune (YOLO does this by default)
2. **Pretrain on SKU-110K** — shelf-specific pretraining
3. **Pseudo-labeling** — use confident detections to expand training set
4. **Semi-supervised** — teacher-student with unlabeled shelf images
5. **Strong augmentation** — mosaic, mixup, copy-paste, color jitter
6. **Freeze backbone** — only train detection head with few samples

### Inference Tricks
1. **Test-Time Augmentation (TTA)** — flip + multi-scale
2. **Weighted Boxes Fusion (WBF)** — merge boxes from multiple models
3. **Model ensemble** — blend YOLO + DETR for diversity
4. **Confidence threshold tuning** — optimize on validation set

---

## REVISED Implementation Priority

### Must-Implement IMMEDIATELY
- [ ] **Set up Ultralytics YOLOv8/v11** — this should be our primary model
- [ ] Download training data and convert to YOLO format
- [ ] Train YOLOv8x on competition data (baseline)
- [ ] Set up validation split for mAP@0.5 evaluation

### Day 1 Priority
- [ ] Multi-scale training (640 → 1280)
- [ ] Train with mosaic + strong augmentation
- [ ] Try YOLOv11 and compare
- [ ] Pretrain on SKU-110K then finetune

### Day 2 Priority
- [ ] Add Co-DETR or RT-DETR as second model
- [ ] Weighted Boxes Fusion ensemble
- [ ] TTA at inference
- [ ] Pseudo-labeling with confident predictions
- [ ] Try augmenting with Kassalapp product images (copy-paste)

### Day 3 (Final Push)
- [ ] Optimize confidence thresholds on validation
- [ ] Final ensemble: YOLO + DETR + WBF
- [ ] Maximize image resolution within 360s time limit
