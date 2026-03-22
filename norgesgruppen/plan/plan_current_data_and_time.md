# Plan — NM i AI 2026 Task 2: Closing the Gap to 1st Place

**Date**: 2026-03-20, ~12:15
**Current rank**: 6th (0.9091)
**Target**: 1st place (0.9199)
**Gap to close**: 0.0108
**Submissions left today**: 4
**Competition ends**: Mar 22 (2 more days, likely ~10 submissions remaining total)

---

## What We Know (Facts)

### Test Results (actual leaderboard scores)
| Submission | Score | Config |
|---|---|---|
| **fulldata_x + m_diverse + s** | **0.9091** | WBF conf=0.005, wbf_iou=0.7, nms_iou=0.7 |
| single yolov8x baseline | 0.7802 | conf=0.01, nms=0.5 |
| single yolov8l s77 | 0.7700 | conf=0.01, nms=0.5 |

### Key Insights
- Ensemble was the single biggest lever (+0.1289 over single model)
- Fulldata training (all 248 images, no val split) works well
- Val mAP does NOT predict test performance (yolov8l 0.804 val → 0.770 test)
- Competition metric: 0.7 × det_mAP + 0.3 × cls_mAP. Classification is harder.
- "No penalty for adding more boxes" — low conf threshold is better

### Available Models (downloaded, can export to ONNX)
**yolov8x variants (261MB ONNX each — can only fit ONE in 420MB ZIP):**
- yolov8x_640_fulldata (all data, no val) — **currently in our best submission**
- yolov8x_640_baseline (seed42) — test-proven at 0.7802
- yolov8x_640_seed77, s123, s314, seed999 — different seeds
- yolov8x_640_noflipud, cls10, cls15_box10, copypaste03, heavyaug, etc — different configs
- yolov8x_640_500ep — longer training
- yolov8x_800_s42 — trained at 800px

**yolov8m variants (50-100MB ONNX):**
- yolov8m_640_diverse (seed42) — **currently in our best submission** (50MB)
- yolov8m_640_s77 (val mAP 0.8069, highest overall) — exported (100MB)
- yolov8m_640_adamw_heavy, yolov8m_800_s42

**yolov8s (43MB ONNX):**
- yolov8s_640_best — **currently in our best submission**

**yolov8l (168MB ONNX):**
- yolov8l_640_heavyaug, yolov8l_640_s77, yolov8l_800_s42, etc

**Size budget (420MB):**
- Current: fulldata_x(261) + m_diverse(50) + s(43) = 354MB → 66MB headroom
- Max with current x: 261 + 159MB of smaller models

### Azure Jobs Still Running
- `yolov8x_640_combo` — noflipud + cls=1.0 + copy_paste=0.2 + mixup=0.2
- `yolov8x_640_combo_fulldata` — same but trained on ALL data (no val split)
- These are our most promising new models

---

## Levers to Pull (Ranked by Expected Impact)

### 1. Better Model Selection in Ensemble (HIGH — what to submit next)
**Goal**: Find the optimal 3-model combination within 420MB

Options ready to test:
- **A**: fulldata_x + m_s77(100MB) + s = 404MB ← replaces m with higher-val model
- **B**: fulldata_x + m_diverse + m_s77 = 411MB ← two m models, drops s
- **C**: fulldata_x + l_heavyaug(168MB) = 429MB ← TOO BIG by 9MB

**Who**: Claude builds, Tom submits
**Time**: 5 min each (already built A and B)

### 2. WBF Parameter Tuning on Test Set (MEDIUM)
**Goal**: Optimal WBF params may differ from training-set optimal
We used conf=0.005, wbf_iou=0.7 based on training sweep. But:
- VinBigData 2nd place used NMS IoU 0.25-0.3 (opposite to our finding)
- Small param changes can shift score by 0.005+

Options to test:
- wbf_iou=0.5 (less merging, more boxes)
- wbf_iou=0.6 (middle ground)
- conf=0.001 (even more boxes)

**Who**: Claude builds variants, Tom submits
**Time**: 2 min per variant

### 3. Wait for Combo Azure Jobs (HIGH — but time-dependent)
**Goal**: Get optimized-hyperparam fulldata model into ensemble
The combo_fulldata job combines best training insights on all data.
If it's better than current fulldata_x, swap it in.

**Who**: Tom monitors Azure, Claude exports ONNX + builds submission
**Time**: ~1-2 hours for jobs to finish

### 4. Export ONNX with Dynamic Input Shape (MEDIUM)
**Goal**: Enable higher-res inference (800px) at test time
Current ONNX models are fixed at 640px. Re-export with `dynamic=True` or `imgsz=800`.
Higher res catches small products better.

**Risk**: May exceed 300s timeout. Need Docker timing test.
**Who**: Claude exports + builds, Tom Docker tests
**Time**: 30 min

### 5. Add More Models via Smaller Architectures (MEDIUM)
**Goal**: Maximize diversity within 420MB
Export additional small models:
- yolov8s_800_s42 (different resolution training)
- yolov8m_800_s42 (different resolution)

These add diversity without huge size cost.
**Who**: Claude exports ONNX
**Time**: 15 min

### 6. YOLO11 Models (LOW-MEDIUM)
We have yolo11x_640_heavyaug completed on Azure. Different architecture = more diversity.
But yolo11x ONNX is ~same size as yolov8x (261MB), so can't add alongside.
Could replace the x model if it's better.

**Who**: Tom downloads from Azure, Claude exports + evaluates
**Time**: 30 min

---

## What Tom Should Do

1. **Submit `onnx_fulldata_ms77_s.zip`** — next submission, tests stronger m model
2. **Monitor Azure combo jobs** — tell Claude when they complete
3. **Check if yolo11x_640_heavyaug is worth downloading**
4. **Consider: which WBF params to try next** based on submission results
5. **Visual inspection**: look at a few test predictions to spot patterns

## What Claude Should Do

1. **Build WBF param variants** of the best ensemble (wbf=0.5, wbf=0.6)
2. **Export more models to ONNX** for potential ensemble diversity
3. **Re-export yolov8x_fulldata with imgsz=800** and dynamic axes for higher-res inference
4. **Prepare combo model integration** — have the build pipeline ready for when Azure jobs finish
5. **Docker test everything** before Tom submits

---

## Submission Strategy (4 remaining today + ~6 tomorrow)

### Today's 4:
1. `onnx_fulldata_ms77_s` — test stronger m model (READY)
2. Based on #1 result: either WBF param variant OR fulldata_mm
3. Based on #2 result: iterate
4. Save 1 for combo model if Azure finishes

### Tomorrow (~6 submissions):
- Best combo from today + new Azure models
- Higher-res inference if it helps
- Final tuned ensemble

---

## Things NOT Worth Trying (from competition research)
- ❌ Bbox crop classifier — VinBigData teams found it hurts
- ❌ External data pretraining — multiple teams found no improvement
- ❌ SAHI tiled inference — timeout risk too high for uncertain gain
- ❌ Co-occurrence re-scoring — complex to implement, marginal gain
- ❌ Pseudo-labeling — no test set access in sandbox
