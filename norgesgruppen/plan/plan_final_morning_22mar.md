# Final Morning Plan — Mar 22, 2026

**Current best**: 0.9226 (sub2_c11_yolo11x_swap) — NEW BEST!
**Previous best**: 0.9215 (candidate11)
**Submissions remaining**: 3
**Competition ends**: 15:00 CET
**All submissions must be reproducible** — code repo must be public before deadline

---

## Overnight Results

### Completed Azure Jobs:
- **YOLO11x noflip s42** — COMPLETED, downloaded, 110MB ONNX ✓
- **ConvNeXt classifier** — COMPLETED, two-stage scored 0.9109 (WORSE, classifier overfits)
- **EfficientNet-B3 classifier** — COMPLETED/recovered
- **YOLOv8x 800px trained** — COMPLETED/recovered
- **YOLOv8l 1280px** — COMPLETED/recovered
- **Distilled two-phase** — COMPLETED/recovered
- **SGD optimizer** — COMPLETED/recovered
- **No-mosaic** — COMPLETED/recovered
- **Expert 800px cls1.5** — COMPLETED/recovered
- **Noflip soup 3x** — COMPLETED/recovered
- **RT-DETR** — FAILED (architecture incompatibility)

### Key Learning from Tonight:
Two-stage classifier scored 0.9109 — **-0.0106 from best**. Trained classifier overfits to training crops, same pattern as cls=2.0.
**Conclusion: The YOLO ensemble's own classification cannot be improved by external classifiers.**

### Key Learning from Today's Submissions:
- Sub 1 (0.9190) — WORSE than candidate11 (0.9215). nf_l_s1 is a weaker third model.
- Sub 2 (0.9226) — NEW BEST! Swapping to yolov8l_highaug as third model works better.
- YOLO11x architecture diversity helps, but the third model quality matters too.
- FP16 ONNX conversion via onnxconverter_common was broken for locally-exported models; fixed with INT8 quantization.

---

## 5 Submissions Tomorrow (in order)

### Sub 1 (09:00): YOLO11x Architecture Diversity — RESULT: 0.9190 (WORSE than candidate11)
**`fulldata_x + nf_11x_s42 (YOLO11x) + nf_l_s1`** multi-scale 640+800, TTA hflip, WBF iou=0.6
- YOLO11x is a genuinely different architecture (C3k2 modules)
- 131 + 131 + 110 = 372MB ✓
- nf_l_s1 was weaker than yolov8l_highaug in this combination

### Sub 2 (10:00): YOLO11x + yolov8l_highaug swap — RESULT: 0.9226 (NEW BEST!)
**`fulldata_x + nf_11x_s42 (YOLO11x) + yolov8l_highaug`** multi-scale 640+800, TTA hflip, WBF iou=0.6
- Swapped nf_l_s1 for yolov8l_highaug — better third model
- +0.0011 over candidate11 (0.9215), +0.0036 over sub1

### Sub 3 (11:00): 800px Trained Model
**`fulldata_x + noflip_s42 + x_800_noflip`** multi-scale 640+800
- One model TRAINED at 800px (not just inferring at 800)
- Could have better feature representation for small products
- Depends on Azure job completing

### Sub 4 (12:00): Distilled + Noflip + yolov8l
**`distilled_2phase + noflip_s42 + yolov8l`** multi-scale 640+800
- Distilled model = two-phase training (heavy→light aug)
- Tests: curriculum learning + realistic aug + arch diversity

### Sub 5 (13:00): Best Variant / Safe Pick
- If any of Sub 1-4 beats 0.9215: iterate on the winner
- If none beats: re-confirm candidate11 as final pick
- **Must submit by 14:00 latest**

---

## Morning Checklist

### 08:00 — Download & Export
- [ ] Download ALL completed Azure models
- [ ] Export new .pt files to FP16 dynamic ONNX
- [ ] Verify sizes fit in 420MB

### 09:00 — Build Sub 1 (YOLO11x)
- [ ] Build ZIP: fulldata_x + noflip_s42 + YOLO11x
- [ ] validate_submission.py → PASS
- [ ] pytest → PASS
- [ ] Submit

### 10:00-13:00 — Build & Submit 2-5
- [ ] Each: build → validate → pytest → submit
- [ ] Log scores in experiments.csv

### 14:00 — Final Selection
- [ ] Review all scores
- [ ] Select final submission for private test
- [ ] Must be the most DIVERSE and ROBUST, not just highest public score

### 14:30 — Code Submission
- [ ] Make repo public (MIT license)
- [ ] Submit repo URL through platform
- [ ] Verify all code is committed and pushed

---

## Reproducibility Checklist
- [ ] All training scripts in repo (scripts/, jobs/)
- [ ] All model weights documented in experiments.csv
- [ ] Submission builder scripts work end-to-end
- [ ] Docker test passes for final submission
- [ ] LaTeX report up to date
