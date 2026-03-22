# Final Strategy Plan — Mar 21 ~12:00

**Current**: 4th place, 0.9210. Gap to 1st: 0.0045
**Submissions**: 3 left today, ~5 tomorrow. Competition ends Mar 22 15:00
**Hidden private test determines winner**

---

## What Got Us to 0.9210

The winning formula (each element's contribution):
- **Multi-scale inference** (640+800px): +0.005 on test — THE biggest single improvement
- **Architecture diversity** (2x yolov8x + 1x yolov8l): genuine error pattern diversity
- **WBF with `absent_model_aware_avg`**: aggressive FP suppression
- **TTA horizontal flip**: +0.002 on test
- **FP16 dynamic ONNX**: fits 3 models in 319MB

---

## Research Synthesis (from 5 research agents)

### Confirmed — Keep as-is:
- WBF iou_thr=0.6 is optimal for our data (flat region 0.55-0.65)
- `absent_model_aware_avg` is the right conf_type
- 2x yolov8x + 1x yolov8l is a strong model mix
- Horizontal flip TTA helps, vertical flip would NOT help
- DO NOT add confidence filtering (hurts mAP)

### High-value changes to try:
1. **960px scale** — 73% of images are >2000px wide, 960px catches text on products
2. **Soft-NMS** within each model (before WBF) — +1-2% on dense scenes per literature
3. **Model soup from noflip models** — best generalization technique per research
4. **Noflip (flipud=0.0) training** — realistic augmentation, no upside-down products

### What NOT to do:
- Don't lower WBF iou below 0.55 (would merge different products on dense shelves)
- Don't raise skip_box_thr above 0.02 (kills recall)
- Don't add 480px scale (26% of boxes become too small)
- Don't add gallery classifier (previous attempts scored worse)
- Don't add vertical flip TTA
- Don't chase local eval metrics

---

## 74% of Remaining Errors = Same-Brand Text Confusion

The detection is near-perfect. The remaining errors are:
- NESCAFE GULL 100G vs 200G
- HUSMAN KNEKKEBRØD 260G vs 520G
- YELLOW LABEL TEA 25POS vs 50POS

These require the model to READ text on packaging — 960px scale may help marginally, but fundamentally these are classification errors no amount of post-processing can fix.

---

## Candidates Ready (validated & Docker-tested or building)

| # | Name | What's different | Status |
|---|---|---|---|
| 6 | candidate6_3scale_weighted | 640+800+960, weighted WBF | Docker testing |
| 3 | candidate3_3x_multiscale | 3x yolov8x multi-scale (no yolov8l) | Validated |
| 4 | candidate4_soup_multiscale | Soup + yolov8l + multi-scale | Validated |
| - | TBD: soft-NMS variant | Soft-NMS instead of hard NMS | To build |
| - | TBD: noflip models | When Azure completes (~1-2 hours) | Training |

---

## Submission Strategy (3 today, ~5 tomorrow)

### Today — Test different hypotheses:
1. **candidate6** (3 scales + weighted WBF) — tests 960px and model weighting
2. **Soft-NMS variant** — tests if soft-NMS helps dense shelf detection
3. Save 1 for end of day if noflip models arrive

### Tomorrow — Maximize with best models:
1. **Noflip soup + noflip yolov8l + multi-scale** — most theoretically grounded for generalization
2. Best combo from Azure batch sweep
3. Final iteration on best approach
4-5. Reserve for last-hour submissions

### Final Submission Selection:
- Choose the submission for private test that is most **theoretically justified**, not the highest public score
- The multi-scale + arch diversity approach (0.9210) is principled — it should survive shake-up
- Noflip models trained with realistic augmentation should generalize even better

---

## Azure Jobs Status
- **4 clean noflip yolov8x** (s42, s77, s123, s999): Running, ~1-2 hours
- **1 clean noflip yolov8l** (s42): Running
- **85 batch jobs** (diverse seeds, augmentations, architectures): Submitting/running
- **2 transfer learning** (SKU-110K, two-phase): Running
- Total: 90+ models training overnight

---

## Key Files
- Best submission: `submissions/wbf_2x1l_multiscale_tta_tested.zip` (0.9210)
- LaTeX report: `docs/competition_report.tex`
- Error analysis: `plan/error_analysis_multiscale_21mar.md`
- Research: `plan/research_comprehensive_21mar.md`
- Azure monitor: `scripts/monitor_azure_jobs.py` (running every 15 min)
- Submission validator: `scripts/validate_submission.py`
