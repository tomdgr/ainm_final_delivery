# Plan: Final Day Push — Mar 21 ~10:30

**Current**: 12th place, 0.9160. Gap to 3rd: 0.0047
**Submissions**: 5 left today, competition ends Mar 22 15:00
**Key constraint**: Max 3 weight files, 420MB, 300s timeout

---

## Submissions So Far

| # | Submission | Test Score | Key Change |
|---|---|---|---|
| 1 | yolov8x_onnx_clean | 0.7802 | Single model baseline |
| 2 | yolov8l_640_s77 | 0.7700 | Single yolov8l |
| 3 | onnx_ensemble_fulldata_ms_tuned | 0.9091 | WBF x+m+s |
| 4 | softvote_3x_fp16 | 0.9142 | Soft vote 3x |
| 5 | wbf_3x_fp16_tta | **0.9160** | WBF 3x + TTA — **BEST** |
| 6 | wbf_3x_cls20_tta | 0.9159 | cls20 swap — no gain |
| 7 | wbf_3x_fulldata_seeds_tta | 0.9152 | Fulldata seeds — slightly worse |

---

## Error Analysis Summary
- Detection: 97.0% mAP — near perfect
- Classification: 87.1% mAP — **THE BOTTLENECK**
- 882 misclassifications (same brand, different size/variant)
- 28,942 false positives at conf>0.01 — too many
- Small objects (32-64px): 15% miss rate
- 23 classes with near-zero AP (all single-instance, can't improve)

---

## Priority Actions (Ranked)

### 1. Architecture-Diverse Ensemble (READY NOW)
`wbf_2x1l_tta.zip` — 2x yolov8x + 1x yolov8l + TTA
- Local score 0.9456 vs 0.9405 for 3x yolov8x
- Different architecture = different error patterns = genuine diversity
- **SUBMIT THIS FIRST**

### 2. Multi-Scale TTA (BUILD NOW)
Run same 3 models at 640px AND 800px, merge with WBF
- Currently using ~30s of 300s budget — massive room
- Higher resolution catches small text on products (100G vs 200G)
- No new weight files needed — just code change in run.py
- Need proper cross-scale NMS to avoid duplicates

### 3. WBF Parameter Tuning
- `conf_type='absent_model_aware_avg'` — penalizes single-model detections
- Higher `skip_box_thr` (0.05-0.10) — reduce 28k false positives
- These are free code changes, combinable with #1 and #2

### 4. Product Gallery Classifier (IF TIME PERMITS)
- 321/356 categories have reference images in data/raw/product_images/
- Pre-compute embeddings with timm backbone (pre-installed on sandbox)
- At inference: crop detections, match to gallery, override low-confidence classifications
- Could significantly help the 882 misclassifications

---

## Submission Plan (5 remaining)

1. **wbf_2x1l_tta** — architecture diversity (ready now)
2. **wbf_2x1l_multiscale_tta** — arch diversity + multi-scale 640+800
3. **wbf_3x_multiscale_tta** — original 3x with multi-scale (fallback)
4. **Best combo with gallery classifier** — if gallery works
5. **Final best iteration** — save for last hour
