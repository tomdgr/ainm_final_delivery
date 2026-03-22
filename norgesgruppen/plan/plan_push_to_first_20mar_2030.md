# Plan: Push from 0.9142 to 0.9199+ (1st Place)

**Date**: 2026-03-20 ~20:30
**Current rank**: Improved (0.9142, up from 0.9091)
**Target**: 1st (0.9199)
**Gap**: 0.0057
**Submissions remaining**: 3 today, ~6 tomorrow
**Competition ends**: Mar 22 (Sun)
**Hidden final test set**: Must generalize, don't overfit to public test

---

## Current Situation

### Leaderboard Submissions (actual test scores)

| Submission | Test Score | Type | Notes |
|---|---|---|---|
| **softvote_3x_fp16.zip** | **0.9142** | 3x yolov8x FP16, soft voting | **CURRENT BEST** |
| onnx_ensemble_fulldata_ms_tuned.zip | 0.9091 | WBF ensemble (x+m+s) | Previous best |
| yolov8x_onnx_clean.zip | 0.7802 | Single yolov8x baseline | First submission |
| yolov8l_640_s77.zip | 0.7700 | Single yolov8l | Worse despite higher val mAP |

### Local Train Eval Results

| Submission | Train Score | Det | Cls | Preds | Notes |
|---|---|---|---|---|---|
| baseline_fulldata_fp16 | 0.9397 | 0.9723 | 0.8638 | 39,842 | Single FP16 model |
| softvote_3x_fp16 | 0.9342 | 0.9710 | 0.8481 | 72,859 | 3x models, scored 0.9142 on test |
| softvote_3x_fp16_tta | 0.6378 | 0.6226 | 0.6733 | 140,834 | BROKEN: TTA duplicates not merged |
| softvote_ms_tta_v2 | 0.6174 | 0.6049 | 0.6465 | 105,808 | BROKEN: multi-scale still duplicating |
| wbf_3x_fp16_tta | running | | | | Evaluating now |

### Key Insight
- Local train eval does NOT predict test performance. softvote_3x_fp16 scored LOWER locally than single baseline but HIGHER on test.
- Soft voting works: +0.0051 on test vs WBF ensemble
- TTA is broken in both softvote and multi-scale variants — duplicate boxes tanking scores

---

## Current Best Model Details

**softvote_3x_fp16.zip** (362.2 MB, in tried_models/current_best/)
- 3x yolov8x FP16 ONNX models:
  - `yolov8x_640_fulldata_best_fp16.onnx` (fulldata, default seed)
  - `yolov8x_640_s123_best_fp16.onnx` (seed 123)
  - `yolov8x_640_seed999_best_fp16.onnx` (seed 999)
- Soft class voting: averages 356-dim probability vectors across models before argmax
- conf=0.005, nms_iou=0.7, wbf_iou=0.55

---

## Azure Jobs Running (8 new fulldata models)

| Job | Display Name | Purpose | Status |
|---|---|---|---|
| ashy_fish_1f7gdksg4y | fulldata_cls20_s42 | cls=2.0 (push classification) | Running |
| jovial_glove_730h6klztt | fulldata_x_s77 | seed 77 diversity | Running |
| shy_sugar_q2clmwt571 | fulldata_x_s123 | seed 123 diversity | Running |
| kind_card_ty66lqh6px | fulldata_x_800 | 800px (text readability) | Running |
| modest_spring_qs9d65pxl8 | fulldata_11x_s42 | YOLO11x architecture diversity | Running |
| calm_tray_v0f521fz4k | fulldata_cls15_combo | cls=1.5 + augmentations | Running |
| joyful_puppy_wt500h4tb7 | fulldata_x_300ep | 300 epochs | Running |
| cyan_picture_13ygln8m3g | fulldata_l_s42 | yolov8l fulldata | Running |

ETA: ~2-3 hours from ~20:00

---

## Strategies to Close the 0.0057 Gap

### 1. Fix TTA Merging (HIGH PRIORITY)
**Problem**: Soft vote TTA produces 140k predictions (should be ~40-70k). Flipped detections aren't being merged with originals.
**Fix**: After TTA, run NMS/soft-merge on combined original+flipped detections BEFORE passing to cross-model soft voting.
**Expected impact**: +0.002-0.005 (TTA generally helps, just needs proper dedup)
**Time**: 1 hour

### 2. Weight Averaging / SWA (HIGH)
**Why**: Average weights from multiple training runs into a single model. Creates smoother, better-calibrated predictions. Zero extra ZIP space.
**How**: Load last.pt from multiple fulldata runs, average state_dicts, export to ONNX.
**Expected impact**: +0.001-0.003 per model in ensemble
**Time**: 30 min (after Azure jobs finish)

### 3. Squeeze 4th Model into ZIP (HIGH)
**Why**: More voters = better classification majority vote.
**Size math**: 3x yolov8x FP16 (393MB) + 1x yolov8s FP16 (~22MB) = 415MB < 420MB
**Or**: 3x yolov8x FP16 (393MB) + 1x yolov8m FP16 (~25MB) = 418MB (tight)
**Expected impact**: +0.001-0.003
**Time**: 30 min

### 4. INT8 Quantization (MEDIUM)
**Why**: Halve FP16 model size again → fit 5-6 yolov8x models
**Risk**: Accuracy drop per model, but more diversity might compensate
**Expected impact**: Unknown — needs testing
**Time**: 1 hour

### 5. Swap in Better Models from Azure (MEDIUM — time dependent)
**Why**: cls=2.0 and cls=1.5 models may classify better. 800px may capture text differences.
**When**: After Azure jobs finish (~22:00-23:00)
**Expected impact**: +0.002-0.005 if cls-focused models are better
**Time**: 30 min per model swap

### 6. Confusion-Aware Post-Processing (LOW-MEDIUM)
**Why**: We know exact confusion pairs. Can apply correction rules.
**Risk**: May overfit to training confusion patterns, hurt hidden test
**Time**: 1 hour

---

## Submission Plan

### Today (3 remaining):
1. **wbf_3x_fp16_tta** — if eval looks reasonable, submit to compare WBF+TTA vs soft vote
2. **Fixed soft vote TTA** — after fixing the merge bug
3. Reserve for Azure model swap if timing works

### Tomorrow (~6 remaining):
1. SWA model in soft vote ensemble
2. 4-model ensemble (add yolov8s/m)
3. Best Azure models (cls=2.0, 800px) in ensemble
4. INT8 6-model ensemble if promising
5-6. Final tuning based on signal

---

## Files Reference
- Current best ZIP: `tried_models/current_best/softvote_3x_fp16.zip`
- Soft vote builder: `scripts/build_softvote_submission.py`
- WBF builder: `scripts/build_advanced_submission.py`
- Eval script: `scripts/eval_submissions.py`
- Eval results: `eval_results.json`
- Experiment tracker: `experiments.csv`
