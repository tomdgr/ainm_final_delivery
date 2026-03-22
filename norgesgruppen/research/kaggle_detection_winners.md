# Kaggle Object Detection — Winning Solutions

## Great Barrier Reef (Dense Detection) — 1st Place
- **Link:** https://www.kaggle.com/competitions/tensorflow-great-barrier-reef
- **Data:** ~3,648 images, 1 class (starfish), dense
- **Metric:** F2 score (IoU 0.3-0.8)
- **Winner:** Ensemble of 6 YOLOv5 models
  - 3 on full-size (1280x720) + 3 on patches (512x320)
  - **Tiled/patched training** for small objects
  - Post-classification: cropped predicted boxes, classified into IoU bins
  - Heavy augmentation + HSV removal
  - **Weighted Boxes Fusion (WBF)** for ensemble
- **Writeup:** https://www.kaggle.com/competitions/tensorflow-great-barrier-reef/writeups/qns-trust-cv-1st-place-solution

## Global Wheat Detection — 1st Place
- **Link:** https://www.kaggle.com/competitions/global-wheat-detection
- **Data:** ~3,000 images, 1 class (wheat head), domain shift in test set
- **Metric:** mAP@[0.5:0.75]
- **Winner:** EfficientDet-D6 + Faster R-CNN FPN ensemble
  - **External data:** wheat2017 + spike wheat datasets
  - **Pseudo-labeling** on hidden test set (confidence >0.9)
  - Mosaic + Mixup augmentation
  - Heavy albumentations: RandomCrop, CLAHE, Sharpen, GaussNoise, MotionBlur
- **Code:** https://github.com/dungnb1333/global-wheat-dection-2020

## VinBigData Chest X-ray Detection (Low Data) — 1st Place
- **Link:** https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection
- **Data:** ~15,000 images (70% "No Finding"), 14 classes
- **Metric:** mAP@[0.4:0.75]
- **Winner:** YOLOv5 + EfficientDet ensemble
  - **2-class filter trick** — separate binary classifier for "No Finding" (biggest improvement)
  - Image resolution 512 → 1024
  - **WBF** for ensemble
- **Code:** https://github.com/ZFTurbo/2nd-place-solution-for-VinBigData-Chest-X-ray-Abnormalities-Detection

## Sartorius Cell Instance Segmentation (Very Low Data)
- **Link:** https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation
- **Data:** ~606 images, 3 classes, extremely dense/overlapping
- **Winner:** YOLOX detector → UperNet segmentor (Swin-T backbone)
  - **Pre-training on LIVECell** external dataset
  - Two-stage: detect then segment
- **Code:** https://github.com/tascj/kaggle-sartorius-cell-instance-segmentation-solution

## SKU-110K Shelf Detection (Retail Benchmark)
- **Data:** 11,762 images, 1.73M boxes, ~147 objects/image
- **Best results:**
  - YOLO-RECAP: mAP50=0.895
  - YOLOv7: mAP50=0.996, mAP50:95=0.86
- **Built into Ultralytics:** `yolo detect train data=SKU-110K.yaml`

---

## Model Benchmarks (COCO, 2025-2026)

| Model | mAP@.50:.95 | Speed | Best For |
|-------|-------------|-------|----------|
| Co-DETR (Swin-L) | ~65% | Slow | Max accuracy |
| **RF-DETR-L** | 60.5% | 25 FPS T4 | **Best fine-tuning from few data (DINOv2 backbone)** |
| RF-DETR-M | 54.7% | Fast | Real-time SOTA |
| YOLOv12 | ~53% | Fast | Attention-centric YOLO |
| **YOLO11-X** | 51.2% | Fast | **Proven, best tooling** |
| YOLOv8-X | ~50.7% | Fast | Mature, stable |

### YOLO11 vs YOLOv8
- YOLO11m: 50.3% mAP with **22% fewer params** than YOLOv8m
- C2PSA (spatial attention) improves small object + occlusion handling
- Use YOLO11, not YOLOv8

### RF-DETR (ICLR 2026) — Key for Low Data
- DINOv2 backbone = stronger pretrained features
- Converges faster with less data (plateaus in ~10 epochs)
- NMS-free (end-to-end)
- Code: https://github.com/roboflow/rf-detr

---

## Critical Techniques

### Weighted Boxes Fusion (WBF) — MUST USE
- Paper: https://arxiv.org/abs/1910.13302
- Code: https://github.com/ZFTurbo/Weighted-Boxes-Fusion
- Unlike NMS which discards boxes, WBF averages ALL boxes
- Used in virtually every winning Kaggle detection solution

### Test-Time Augmentation (TTA)
- Tool: https://github.com/kentaroy47/ODA-Object-Detection-ttA
- Typical improvement: +1.2 mAP points
- Horizontal flip + 3 resolutions
- Cost: 2-3x inference time

### Pseudo-Labeling
- Train → predict unlabeled → add confident (>0.9) predictions → retrain
- +2-4% mAP improvement
- Used by Global Wheat 1st place

### SAHI (Slicing Aided Hyper Inference)
- For detecting small objects in large images
- Slice image into overlapping patches, detect on each, merge with NMS
- Critical for shelf images with many small products

---

## Competition Game Plan

### Day 1: Baseline
1. Download competition data, understand format
2. Train YOLO11x baseline
3. Train RF-DETR baseline (if data is small, this may win)
4. Evaluate both on validation split

### Day 2: Optimize
1. Multi-scale training (640 → 1280)
2. Heavy augmentation (mosaic, mixup, copy-paste)
3. Pretrain on SKU-110K then finetune
4. Pseudo-labeling if unlabeled data available
5. SAHI for inference on high-res images

### Day 3: Ensemble & Submit
1. WBF ensemble: YOLO11 + RF-DETR + (optional 3rd model)
2. TTA: flip + multi-scale
3. Confidence threshold optimization on validation
4. Final submission (ASK USER BEFORE SUBMITTING)

### Pre-trained Models to Try
- https://huggingface.co/foduucom/product-detection-in-shelf-yolov8
- Ultralytics SKU-110K pretrained
