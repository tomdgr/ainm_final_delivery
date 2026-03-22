# Plan: Close the Gap from 0.9091 to 0.9199+ (1st Place)

**Date**: 2026-03-20 ~12:45
**Current rank**: 6th (0.9091)
**Target**: 1st (0.9199)
**Gap**: 0.0108
**Submissions**: 4 left today, ~6 tomorrow
**Competition ends**: Mar 22

---

## The Problem: Classification, Not Detection

Error analysis on training set (fulldata_x model):
- **Detection recall: 98.3%** — near-perfect, NOT the bottleneck
- **Classification: 96.7%** — 730 misclassifications = the main error source
- Score = 0.7 * det + 0.3 * cls → need ~0.036 cls_mAP improvement → fix ~250-350 of 730 misclassifications
- Errors are between visually near-identical products (same brand different size/variant)
- 11,328 false positives but these DON'T hurt mAP

Top confusion pairs:
- KNEKKEBRØD URTER → GODT FOR DEG (29x)
- NESCAFE GULL 100G → 200G (18x)
- YELLOW LABEL TEA 50POS → 25POS (16x)
- HAVRE KNEKKEBRØD 600G → 300G (11x)

---

## Strategies (Ranked by Expected Impact)

### 1. FP16 ONNX Export → Fit More Models in Ensemble (HIGHEST PRIORITY)
**Why**: WBF only merges same-class boxes. More models = implicit majority vote on class. Currently limited to 1 yolov8x (261MB) by ZIP size. FP16 halves to ~131MB → fit 2-3 x models.

**Size math**:
- 2x yolov8x_fp16 (262MB) + m_fp16 (25-50MB) + s_fp16 (22MB) = ~334-384MB ✓
- 3x yolov8x_fp16 (393MB) = 393MB ✓

**Pick most diverse models**: fulldata_x + x_s123/seed999 (different seed) + m_s77 + s

**Implementation**: Export with `model.export(format='onnx', half=True, imgsz=640)`
**Risk**: FP16 may slightly degrade per-model accuracy, but extra diversity compensates
**Time**: 30 min
**Who**: Claude exports + builds, Tom submits

### 2. Soft Class Voting (Replace argmax with probability averaging) (HIGH)
**Why**: Current ONNX inference does `argmax` on class scores → loses info. For confused classes (NESCAFE 100G vs 200G), model might output [0.35, 0.33] — argmax picks one, but uncertainty is lost. Averaging full 356-dim probability vectors across models before argmax = better classification.

**How**: Custom box-matching + probability vector averaging instead of standard WBF for class assignment.
1. Run each model, keep full `cls_scores[N, 356]` per box (not just argmax)
2. Match boxes across models by IoU > 0.5
3. Average class probability vectors for matched boxes
4. Take argmax of averaged vector

**Why better than WBF for classification**: WBF splits same-box-different-class predictions into separate clusters. Soft voting averages the probability distributions directly.

**Time**: 1-2 hours to implement
**Who**: Claude implements
**Files**: `scripts/build_advanced_submission.py`

### 3. Multi-Scale TTA (640 + 800px) (MEDIUM-HIGH)
**Why**: Higher resolution captures text on packaging better — the exact cue needed for "100G vs 200G". Run same model at 2 scales, merge with WBF.

**How**: Re-export fulldata_x with `dynamic=True`, run at both 640 and 800 per image.
**Timing**: 50 test images × 6 passes × 25ms = 7.5s total. Well within 300s.
**No extra ZIP size** — same model file, just run twice at different scales.

**Time**: 30 min
**Who**: Claude implements
**Files**: `scripts/build_advanced_submission.py`

### 4. Azure Combo Model (cls=1.0) (MEDIUM — time-dependent)
**Why**: `yolov8x_640_combo_fulldata` trains with cls=1.0 (2x default). Higher cls loss = better classification directly.
**Status**: Running on Azure now.
**When ready**: Export to FP16 ONNX, swap into ensemble.
**Who**: Tom monitors, Claude exports + builds

### 5. Confusion-Aware Post-Processing (MEDIUM)
**Why**: We know exact confusion pairs. Can apply static correction rules.
**How**: For predictions in a known confusion group, use second-choice class score to flip if uncertain.
**Risk**: Overfitting to training confusion patterns.
**Time**: 1 hour
**Who**: Claude implements

### 6. Weight Averaging (SWA) (LOW-MEDIUM)
**Why**: Average best.pt + last.pt weights → smoother model, better calibration. Free — no extra ZIP size.
**How**: `avg = {k: (best[k] + last[k]) / 2 for k in best}`, save, export to ONNX.
**Time**: 15 min per model
**Who**: Claude implements

---

## What Tom Should Do NOW

1. **Submit `onnx_fulldata_ms77_s.zip`** or `adv_tta.zip` — get signal
2. **Monitor Azure combo jobs** — report when they finish
3. **Use Streamlit viewer** (`uv run streamlit run scripts/viewer.py`) to visually inspect worst predictions
4. **Check competition Discord** for hints

## What Claude Should Do NOW

1. **Export FP16 ONNX models** — test if FP16 works, measure size savings
2. **Implement soft class voting** — the biggest potential classification improvement
3. **Add multi-scale TTA** — re-export with dynamic axes
4. **Docker test everything**

---

## Submission Plan

### Today (4 remaining):
1. `adv_tta` or `onnx_fulldata_ms77_s` — get baseline signal
2. FP16 4-model ensemble — test if more models helps
3. Soft voting ensemble — test if probability averaging helps
4. Reserve for Azure combo model OR best of above + multi-scale TTA

### Tomorrow (~6 remaining):
- Best approach from today + refinements
- Azure combo model integration
- WBF param tuning on test signal
- Final ensemble optimization

---

## Verification
- Docker test EVERY submission: `./docker/test_submission.sh submissions/X.zip`
- ZIP size < 420MB, predictions count reasonable
- **NEVER submit without asking Tom first**
