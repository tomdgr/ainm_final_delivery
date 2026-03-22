# Plan: Diversity Over Specialization — Push Past 0.9160

**Date**: 2026-03-20 ~22:00
**Current rank**: 8th (0.9160)
**Target**: Top 3 (0.9199+)
**Gap**: 0.0039 to 3rd, 0.0061 to 1st
**Submissions used**: 7 total (2 left today, ~6 tomorrow)
**Competition ends**: Mar 22 (Sun)

---

## Critical Lessons Learned

### Lesson 1: Local Train Eval is MISLEADING for Model Selection
- `wbf_3x_cls20_tta` scored **0.9553 locally** vs 0.9405 for the baseline → predicted +0.015 improvement
- Actual test score: **0.9159** vs 0.9160 → NO improvement, arguably worse
- cls=2.0 model memorized training classes better but didn't generalize
- **RULE: Never trust local train eval for absolute model quality. Use it only for post-processing parameter comparison (same models, different params).**

### Lesson 2: Diversity > Specialization
- Our best submission uses 3 models with **different random seeds** (default, 123, 999)
- Swapping seed999 for cls=2.0 (specialized for classification) = no improvement
- The seeds disagree on different images → ensemble corrects errors
- Specialized models agree with the majority on easy cases and are WRONG on hard cases (overfitting)

### Lesson 3: WBF > Soft Voting, TTA Helps
- WBF consistently beats soft class voting on real test (+0.002)
- TTA (horizontal flip) adds +0.002 on test
- Post-processing params (IoU, conf, temperature) are already optimal per sweep

### Lesson 4: Small Test Set = High Variance
- Only ~50 test images. Random seed in data split matters hugely.
- Submissions within 0.002 of each other are essentially tied
- Need to focus on robust improvements, not marginal tuning

---

## What We Have

### Models Available
| Model | Source | Seed | Notes |
|---|---|---|---|
| yolov8x_640_fulldata | Azure (maroon_okra) | default | In current best |
| yolov8x_640_s123 | Azure (cool_potato) | 123 | In current best |
| yolov8x_640_seed999 | Azure (musing_toe) | 999 | In current best |
| fulldata_cls20_s42 | Azure (ashy_fish) | 42 | cls=2.0, doesn't generalize |
| fulldata_x_s77 | Azure (jovial_glove) | 77 | NEW — fulldata, different seed |
| fulldata_x_s123 | Azure (shy_sugar) | 123 | NEW — fulldata seed123 |
| fulldata_cls15_combo | Azure (calm_tray) | 42 | NEW — cls=1.5 + augmentations |
| yolov8s_640 | local | default | Small model for 4th slot |

### Models Still Training/Pending
- fulldata_x_800 (800px)
- fulldata_x_300ep (300 epochs)
- fulldata_l_s42 (yolov8l)
- fulldata_yolo26x_s42 & s77 (may have failed)
- fulldata_rtdetr_l_s42 (may have failed)
- fulldata_11x_s42_v2 (YOLO11x)
- fulldata_x_labelsmooth

### Current Best: `wbf_3x_fp16_tta.zip` (0.9160)
- 3x yolov8x FP16: fulldata + s123 + seed999
- WBF ensemble with IoU=0.7
- TTA horizontal flip
- 362MB

---

## Strategy: Maximize Diversity

### Priority 1: Test New Seed Combos (TONIGHT)
The 6-model cache is building. Once done, test all C(6,3)=20 three-model combinations.
Key combos to test:
- fulldata + s77 + s123_new (all fulldata, max seed diversity)
- fulldata + s77 + seed999 (replace s123 with s77)
- fulldata + s123_new + seed999 (replace old s123 with fulldata s123)

### Priority 2: SWA Weight Averaging (TONIGHT)
Average weights from multiple fulldata runs (s77, s123, seed999, default).
Creates a single smoother model. Use as one member of ensemble.
Script ready: `scripts/build_swa_model.py`

### Priority 3: 4-Model Ensemble with yolov8s (TOMORROW)
3x yolov8x (393MB) + yolov8s (22MB) = 415MB < 420MB
The 4th model adds diversity at minimal size cost.

### Priority 4: Architecture Diversity (IF YOLO11/26 WORKED)
Different architecture = maximally different errors.
Check if YOLO11x or YOLO26x jobs produced usable models.
If so, swap one yolov8x for yolo11x in ensemble.

### Priority 5: More Data (CREATIVE)
- Take photos at NorgesGruppen stores for augmentation?
- Web scrape product images from their online store?
- Must check competition rules on external data first!

---

## Submission Plan

### Tonight (1 remaining):
- **SAVE IT** unless the 6-model combo sweep shows a clear winner (>0.9405 local AND uses fundamentally different/more diverse models)

### Tomorrow (~6 remaining):
1. Best 3-model combo from sweep (new seeds)
2. SWA model in ensemble
3. 4-model with yolov8s
4. Architecture-diverse ensemble (if YOLO11/26 available)
5-6. Reserve for iterations

---

## Anti-Patterns to Avoid
- Don't trust local train eval for model selection
- Don't submit cls-tuned models expecting classification improvement
- Don't submit marginal parameter changes (temperature, conf tuning)
- Don't submit without a clear hypothesis for WHY it would be different on test
