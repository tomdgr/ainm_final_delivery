# Improvement Ideas — Prioritized

## From Our Analysis

### 1. SAHI Tiled Inference (BUILT, ready to deploy)
- Images are 2000-4000px, model sees 640px — small products get crushed
- Slice into overlapping 640x640 tiles, detect on each, merge with NMS
- Free detection boost, no retraining
- **Module**: `nm_ai_image/detection/sahi.py`
- **Risk**: Timing — many tiles per image eats GPU time. Need to profile on L4.

### 2. Two-Stage Classifier (BUILT, ready to deploy)
- YOLOv8 finds boxes (70% score) → crop → ResNet50 embedding → match gallery (30% score)
- Gallery: 356/356 classes covered, 4,413 embeddings
- Critical for 74 classes with <5 training samples
- **Module**: `nm_ai_image/detection/classifier.py`
- **Risk**: ResNet50 inference adds time per crop. ~100 crops/image x 5ms = 500ms.

### 3. Confidence Threshold Tuning (TODO)
- Default conf=0.01 produces many false positives
- Tune on val set to find optimal threshold for competition mAP
- Different thresholds for detection vs classification
- **Effort**: Low — just sweep conf values and evaluate

### 4. Multi-Scale Test-Time Inference (TODO)
- Train at 640, infer at [640, 1280] and merge predictions
- Ultralytics supports this natively via `imgsz` parameter
- Catches different sized objects at different scales
- Can combine with WBF ensemble

### 5. Inference at Higher Resolution (TODO)
- Even without SAHI, just running the 640-trained model at imgsz=1280 helps
- The model learns features at 640 but can apply them at higher res
- Simple flag change in run.py

## From OpenImages 2019 Paper (WBF Inventors)

### 6. Class-Aware Sampling (HIGH PRIORITY — TODO)
- Their baseline jumped 58.88 → 64.64 mAP just from this
- Ensure every batch sees at least one sample from each class
- We have 74 classes with <5 samples that get buried in random batches
- **Implementation**: Custom sampler or ultralytics class weighting
- Need to check if ultralytics supports this natively

### 7. Adj-NMS (TODO)
- First apply NMS at threshold 0.5, then soft-NMS on remaining boxes
- Gave them +2.4 mAP over basic NMS
- Easy to add to inference post-processing
- **Implementation**: Replace standard NMS in run.py

### 8. Expert Models for Hard Classes (MEDIUM PRIORITY — TODO)
- Train a separate small detector on just the worst 50 classes
- At inference, run both global model + expert model, ensemble results
- Directly addresses long-tail problem
- **Implementation**: Filter annotations for rare classes, train YOLOv8s on subset
- **Risk**: Uses one of our 3 weight file slots

### 9. Co-occurrence Relationships (TODO)
- Products on same shelf section always co-occur
- If we detect "EVERGOOD CLASSIC" high conf → boost other coffee products
- Our data has 4 sections: Egg, Frokost, Knekkebrod, Varmedrikker
- **Implementation**: Build co-occurrence matrix from training data, apply as post-processing

### 10. Category Relationship Re-scoring (TODO)
- From the paper: re-weight box scores based on class co-occurrence probability
- p(i|j) = C_ij / C_i (conditional probability of class i given class j in same image)
- Boost predictions that are consistent with what else is detected in the image

## From Global Wheat Detection Winners

### 11. Heavy WBF + Multi-Scale TTA (PARTIALLY DONE)
- Winners used WBF across multiple models at multiple scales
- We have WBF built, need to add multi-scale TTA
- Combine: run model at 640, 960, 1280, WBF merge all

### 12. Pseudo-Labeling (TODO)
- Train model → predict on test set → add high-conf predictions as training data → retrain
- Effective when test set has similar distribution to train
- We can't see the test set, but could pseudo-label our own val split

## From Kaggle General Wisdom

### 13. Epoch Averaging (TODO)
- Average weights from last N checkpoints (e.g., epochs 100-130)
- Reduces variance, acts like an ensemble for free
- Ultralytics saves last.pt — could save multiple and average

### 14. FP16 Inference (TODO)
- Model trains in FP16 (AMP), should infer in FP16 too
- Halves GPU memory, 2x faster inference
- L4 has good FP16 throughput
- `model.half()` in run.py

## Priority Order for Next Actions
1. Submit yolov8x baseline (DOING NOW)
2. Build ensemble submission when 2+ more models finish
3. Add SAHI to submission
4. Add two-stage classifier to submission
5. Tune confidence threshold
6. FP16 inference
7. Multi-scale TTA (640+1280)
8. Class-aware sampling (new training run)
9. Expert model for rare classes
10. Co-occurrence re-scoring
