# Generalization Strategy: Final Day — Mar 22, 2026

**Current best**: 0.9215 (candidate11: fulldata_x + noflip_s42 + yolov8l, multi-scale 640+800)
**Submissions remaining**: 6
**Competition ends**: Mar 22 15:00 CET
**Scoring**: 0.7 * det_mAP@0.5 + 0.3 * cls_mAP@0.5
**CRITICAL**: Hidden private test determines winner, NOT public leaderboard

---

## 1. ANALYSIS: WHAT THE 11 SUBMISSIONS TELL US

### 1.1 Shake-Up Risk Assessment

The public leaderboard scores a SUBSET of the private test set. Our 11 submissions
cluster tightly (0.9091-0.9215), meaning the ranking among our top entries could
flip entirely on the private test. Key observations:

| Submission | Public Score | Delta from Best | Risk Level |
|---|---|---|---|
| candidate11 (fulldata_x + noflip_s42 + yolov8l) | 0.9215 | baseline | LOW — max diversity |
| wbf_2x1l_multiscale_tta | 0.9210 | -0.0005 | LOW — proven architecture mix |
| candidate6 (3-scale weighted) | 0.9205 | -0.0010 | MEDIUM — 960px adds noise risk |
| candidate7 (noflip + seed999 + l) | 0.9200 | -0.0015 | LOW — clean, diverse |
| wbf_3x_fp16_tta (3x yolov8x) | 0.9160 | -0.0055 | MEDIUM — no arch diversity |
| wbf_3x_cls20_tta | 0.9159 | -0.0056 | HIGH — cls overfitted |
| wbf_3x_fulldata_seeds_tta | 0.9152 | -0.0063 | MEDIUM — correlated models |
| softvote_3x_fp16 | 0.9142 | -0.0073 | MEDIUM — inferior fusion |
| onnx_ensemble_ms_tuned | 0.9091 | -0.0124 | HIGH — weak model mix |

### 1.2 What Consistently Helps on Real Test (Robust Signals)

These effects were confirmed across multiple submissions, making them ROBUST:

1. **WBF ensemble over single model**: +0.129 (massive, consistent)
2. **Multi-scale 640+800**: +0.005 (confirmed in C11, C6, wbf_2x1l)
3. **Architecture diversity (yolov8x + yolov8l)**: +0.005 (C11 vs 3x yolov8x)
4. **TTA hflip**: +0.002 (confirmed across multiple submissions)
5. **absent_model_aware_avg**: +0.002 (better FP suppression)
6. **Noflip training**: +0.0005 (small but consistent positive signal)

### 1.3 What Is Noise (Not Robust)

These showed inconsistent or negative effects — DO NOT chase:

- 960px third scale: -0.001 (adds noise, marginal text benefit)
- cls=2.0 tuning: 0.000 on test despite +0.015 local (classic overfitting)
- Fulldata-only seeds: -0.001 vs mixed training
- Soft voting: -0.002 vs WBF
- Gallery re-ranking: -0.003 to -0.005 across 33 configs
- Model soup as ensemble member: -0.008 to -0.015
- WBF param tuning beyond iou=0.6: flat region, already optimal
- Temperature/calibration: no effect

---

## 2. PRIVATE TEST: WHAT COULD GO WRONG

### 2.1 Likely Distribution Shifts

| Shift Type | Likelihood | Impact | Our Robustness |
|---|---|---|---|
| Different stores/layouts | HIGH | MEDIUM | Multi-scale helps; diverse training seeds |
| Different lighting | MEDIUM | LOW | YOLO is lighting-robust; noflip avoids unrealistic aug |
| Different product density | MEDIUM | MEDIUM | WBF handles this; absent_model_aware_avg |
| Unseen products (new SKUs) | LOW | HIGH | Cannot prepare; only robust detection matters |
| Different image resolution | MEDIUM | MEDIUM | Multi-scale inference handles this well |
| Different shelf angles | LOW | MEDIUM | Noflip models = more realistic angle assumptions |
| More/fewer products per image | MEDIUM | LOW | WBF scales naturally |

### 2.2 Which Submissions Survive Each Shift?

- **Store layout shift**: Favors diverse seeds + architecture diversity (C11, wbf_2x1l)
- **Lighting shift**: All models roughly equal (YOLO backbone handles this)
- **Resolution shift**: Favors multi-scale inference (C11, C6, wbf_2x1l)
- **Density shift**: Favors WBF with proper IoU (all our top submissions)
- **Unseen products**: Favors models trained on ALL data (fulldata models)

### 2.3 The Kaggle Grandmaster Principles

From research on competition shake-ups and private test strategies:

1. **Trust your CV, not the public LB** — But our CV is unreliable (37 val images).
   This means we should trust THEORETICAL PRINCIPLES over ANY score.

2. **Select diverse final submissions** — Don't pick 6 variants of the same thing.
   Each submission should test a different generalization hypothesis.

3. **Prefer stability over peak performance** — A model scoring 0.920 consistently
   is safer than one scoring 0.925 on public but potentially 0.910 on private.

4. **Ensemble diversity beats ensemble size** — 2 diverse models > 3 similar ones.
   Architecture diversity (yolov8x + yolov8l) is our strongest signal.

5. **Avoid overfitting to public feedback** — We've already made 11 submissions.
   Each submission gives us signal, but also temptation to overfit to the public set.
   The marginal differences (0.0005-0.0015) are within noise range.

6. **Safe + risky portfolio** — Allocate submissions: 2 safe (protect current rank),
   2 moderate (improve position), 2 aggressive (swing for the top).

---

## 3. MOST LIKELY PRIVATE-TEST-ROBUST SUBMISSIONS

### 3.1 Ranking Our Existing Submissions by Generalization Confidence

**Tier 1 — Most robust (would select for private test)**:
1. **candidate11** (0.9215) — Maximum diversity: fulldata_x (seen all data) + noflip_s42 (realistic aug) + yolov8l (different architecture). Multi-scale covers resolution shifts.
2. **wbf_2x1l_multiscale_tta** (0.9210) — Proven architecture diversity formula. Similar to C11 but with seed999 instead of noflip_s42.

**Tier 2 — Good robustness**:
3. **candidate7** (0.9200) — Noflip model + diversity. Slightly less optimized but theoretically sound.

**Tier 3 — Riskier**:
4. **candidate6** (0.9205) — 3 scales adds complexity; 960px could help OR hurt.
5. **wbf_3x_fp16_tta** (0.9160) — No arch diversity, but 3 strong models.

### 3.2 Key Insight: The Gap is Tiny

The difference between our #1 (0.9215) and #4 (0.9200) is 0.0015.
On a hidden test set of ~100-200 images, this is approximately 1-2 correct
classifications. This is statistical noise. ANY of our top 4 could win.

**Implication**: We should NOT waste submissions trying to micro-optimize.
Instead, we should test genuinely different approaches that could either:
(a) Jump significantly (+0.005+), or
(b) Confirm robustness of our current best.

---

## 4. THE 6 SUBMISSIONS FOR MARCH 22

### Submission 1: SAFE BASELINE — "Defend Our Position"
**What**: Re-submit candidate11 (or exact equivalent) with no changes
**Why**: If candidate11 was our best on public, it's our safest private bet.
If we already have it selected as final, skip this and use the slot elsewhere.
**Hypothesis**: Our best public = our best private (true ~60% of the time in competitions)
**Risk**: LOW — we know exactly what this does

### Submission 2: YOLO11x DIVERSITY — "Architecture Moonshot"
**What**: YOLO11x_noflip_s42 + fulldata_x + yolov8l, multi-scale 640+800, TTA hflip
**Why**: YOLO11x uses C3k2 modules and different attention — genuinely different error patterns.
This is the highest-upside change available. A fundamentally different architecture
in the ensemble could break through the 0.922 ceiling.
**Hypothesis**: Architecture diversity is our strongest signal; YOLO11x maximizes it
**Risk**: MEDIUM — depends on YOLO11x training quality from overnight Azure job
**Prerequisite**: Azure YOLO11x job must have completed successfully. Check first.
If not available, substitute with noflip_300ep_s42 (longer-trained model).

### Submission 3: ALL-NOFLIP ENSEMBLE — "Realistic Augmentation"
**What**: noflip_s42 + noflip_soup_4seed + nf_l_s1, multi-scale 640+800, TTA hflip
**Why**: Every model in this ensemble was trained WITHOUT vertical flip augmentation.
If the private test has products in normal orientation (which grocery shelves do),
noflip models should generalize better. The soup provides smooth decision boundaries.
**Hypothesis**: Noflip training = better generalization for realistic test images
**Risk**: MEDIUM — soup has scored poorly before, but this is all-noflip which is new
**Models needed**: noflip_soup_4seed.pt → export to ONNX; nf_l_s1_last.pt → export to ONNX

### Submission 4: LONGER TRAINING + DIVERSE SEEDS — "Better Individual Models"
**What**: nf_x_300ep_s42 + nf_x_300ep_s77 + nf_l_s1, multi-scale 640+800, TTA hflip
**Why**: 300-epoch models may have learned better representations than 150-epoch.
If the private test rewards better per-model accuracy (not just ensemble diversity),
stronger individual models should win. Different seeds ensure diversity.
**Hypothesis**: Longer training → better feature learning → better generalization
**Risk**: MEDIUM — could also mean more overfitting. But noflip training mitigates this.
**Note**: Uses 3 slots for different-seed models instead of reusing fulldata_x.

### Submission 5: TWO-STAGE WITH CLASSIFIER — "Classification Fix"
**What**: fulldata_x + noflip_s42 (detection) + ConvNeXt-Tiny classifier (post-detection re-rank)
**Why**: 74% of remaining errors are classification errors (same-brand text confusion).
A dedicated classifier trained on cropped product images could fix exactly these errors.
The ConvNeXt classifier was submitted as overnight Azure job #1.
**Hypothesis**: Two-stage detection + classification > end-to-end YOLO classification
**Risk**: HIGH — classifier must be good enough; adds complexity; timing constraints
**Prerequisite**: ConvNeXt Azure job must have completed. If not, substitute with
EfficientNet-B3 (Azure job #10) or skip this submission entirely.
**If classifier unavailable**: Replace with Submission 6B (below).

### Submission 6: MAXIMUM ENSEMBLE DIVERSITY — "Kitchen Sink"
**What**: fulldata_x + noflip_s42 + yolov8l, multi-scale 640+800+1024, TTA hflip,
WBF iou=0.55 (slightly more aggressive merging)
**Why**: This tests whether pushing to a higher third scale (1024 vs 960) AND
slightly more aggressive WBF merging helps. 1024px may read text better than 960px
without introducing as much noise. Lower IoU threshold merges more overlapping boxes.
**Hypothesis**: 1024px is the sweet spot for text reading; WBF at 0.55 better handles
dense shelves where products of the same brand overlap
**Risk**: MEDIUM-HIGH — we know 960px was -0.001; 1024px is a new gamble

**Submission 6B (fallback if classifier unavailable)**:
**What**: nf_x_300ep_s42 + fulldata_x + yolov8l_highaug, multi-scale 640+800, TTA hflip
**Why**: Mixes training strategies: long-trained noflip + fulldata + high-augmentation.
Maximum training diversity in the ensemble.

---

## 5. SUBMISSION ORDER STRATEGY

### Priority Order (submit in this sequence):

1. **Sub 2: YOLO11x** (morning, ~09:00) — Highest upside. If it works, it changes everything.
   Check Azure job status FIRST. If YOLO11x failed, pivot to Sub 4.

2. **Sub 3: All-noflip** (morning, ~10:00) — Tests noflip hypothesis cleanly.
   Quick to build since we have all models locally.

3. **Sub 5: Two-stage classifier** (midday, ~11:00) — IF classifier job completed.
   Otherwise skip and use Sub 6B.

4. **Sub 4: Longer training** (early afternoon, ~12:00) — Tests 300ep hypothesis.

5. **Sub 6: Max diversity** (afternoon, ~13:00) — Final aggressive attempt.

6. **Sub 1: Safe re-submit** (afternoon, ~14:00) — Only if we haven't improved.
   If any earlier submission beat 0.9215, use this slot for a variant of the winner instead.

### Decision Tree for Final Selection:

```
After all 6 submissions scored:
├── New best found (> 0.9215)?
│   ├── YES: Select new best as final submission
│   │   └── But also keep candidate11 as backup (if competition allows 2 final selections)
│   └── NO: Keep candidate11 as final submission
│       └── Select second-best from {candidate11, wbf_2x1l} pair
└── If two submissions tied within 0.0005:
    └── Pick the one with MORE model diversity (theoretical robustness)
```

---

## 6. AZURE JOBS: STATUS CHECK AND GAPS

### Jobs to Check (from overnight batch):

| # | Job | What It Gives Us | Priority |
|---|---|---|---|
| 1 | ConvNeXt-Tiny classifier | Two-stage pipeline (Sub 5) | CRITICAL |
| 2 | YOLO11x noflip s42 | Architecture diversity (Sub 2) | CRITICAL |
| 3 | Distilled two-phase | Better single model | LOW |
| 4 | Noflip soup | Already have noflip_soup_4seed locally | DONE |
| 5 | Grocery pretrain | Transfer learning model | LOW |
| 6 | YOLOv8l 1280px noflip | High-res YOLOv8l | MEDIUM |
| 7 | YOLOv8x no-mosaic | Different aug strategy | LOW |
| 8 | Expert 800px cls=1.5 | cls-tuned (historically bad) | SKIP |
| 9 | YOLOv8x SGD noflip | Different optimizer | LOW |
| 10 | EfficientNet-B3 classifier | Backup for ConvNeXt (Sub 5) | MEDIUM |

### Missing Jobs That Could Help:

1. **RT-DETR or DINO** — Transformer-based detector. Fundamentally different from YOLO.
   Would provide maximum architecture diversity. Too late to train now (needs 4+ hours).

2. **YOLOv8x trained at 800px** — We only have 640px-trained models running at 800px
   during inference. A model TRAINED at 800px could be significantly better at that scale.
   Probably too late to start (~3 hours training).

3. **Cross-validation ensemble selection** — Instead of picking models by intuition,
   run proper K-fold CV to identify which 3-model combo has lowest variance.
   Doable locally but our dataset is too small (248 images) for reliable CV.

4. **Stochastic Weight Averaging (SWA)** — Like soup but from checkpoints of a SINGLE run.
   Average the last 10-20 checkpoints of a training run. Flatter minima = better generalization.
   Can be done locally from existing training runs if we have checkpoint histories.

---

## 7. PRACTICAL EXECUTION CHECKLIST

### Morning (08:00-10:00):
- [ ] Check ALL Azure job statuses: `az ml job list --subscription "0a2942e9-..." -w nmai-experis -g rg-nmai-workspace`
- [ ] Download completed models: YOLO11x, ConvNeXt, EfficientNet-B3, YOLOv8l-1280
- [ ] Export new .pt models to FP16 dynamic ONNX
- [ ] Docker-test Submission 2 (YOLO11x) → Submit if passes
- [ ] Docker-test Submission 3 (All-noflip) → Submit if passes

### Midday (10:00-12:00):
- [ ] Evaluate Submission 2 and 3 scores on public leaderboard
- [ ] If classifier available: build two-stage pipeline, Docker-test Sub 5
- [ ] Build Submission 4 (300ep models), Docker-test → Submit

### Afternoon (12:00-14:30):
- [ ] Submit remaining candidates based on morning results
- [ ] Analyze all scores: which hypothesis worked?
- [ ] Final submission selection by 14:00 at latest

### Last Hour (14:00-15:00):
- [ ] Confirm final submission selection
- [ ] NO new submissions after 14:30 (risk of bugs)
- [ ] Document everything for the report

---

## 8. FINAL SUBMISSION SELECTION PHILOSOPHY

### The Anti-Overfitting Principle:

With 11+6 = 17 total submissions against a small public test set, we are at HIGH risk
of having overfit our selection to the public leaderboard. The Kaggle literature is
clear: **the submission you should select for the private test is the one that is
most theoretically justified, NOT the one with the highest public score.**

### Our Selection Criteria (in order of importance):

1. **Maximum model diversity** (architecture + seed + training strategy)
2. **Proven robust components** (WBF, multi-scale 640+800, TTA hflip)
3. **Realistic training** (noflip over heavy augmentation)
4. **Public score** (tiebreaker only, NOT primary criterion)

### If Forced to Pick ONE Submission Right Now:

**candidate11** (0.9215) — fulldata_x + noflip_s42 + yolov8l, multi-scale 640+800

Reasons:
- Maximum diversity: 3 different training strategies (fulldata, noflip, high-aug)
- Architecture diversity: yolov8x + yolov8x + yolov8l
- Multi-scale: handles resolution uncertainty
- TTA: covers horizontal orientation variation
- It IS our current best on public (not contradicting theoretical preference)
- The noflip component adds a genuine generalization advantage

**Do NOT switch to a new submission just because it scores 0.001 higher on public.**
Only switch if: (a) the new submission has a clear theoretical advantage, AND
(b) the public score improvement is > 0.003 (outside noise range).

---

## 9. RISK REGISTER

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Azure jobs failed overnight | 30% | HIGH | Have local fallback models ready (Sub 3, 4, 6B) |
| ONNX export fails for new models | 10% | MEDIUM | Always test export before building submission |
| Docker test fails | 15% | MEDIUM | Debug immediately; have backup ZIPs ready |
| Submission server down | 5% | CRITICAL | Submit early; don't wait until last hour |
| New submission scores LOWER | 40% | LOW | Keep candidate11 as final; new subs are exploration |
| Private test is very different from public | 30% | HIGH | Diversity-first strategy already accounts for this |
| Timeout (300s) with new models | 10% | HIGH | YOLO11x may be slower; test timing in Docker |

---

## 10. WHAT WINNING LOOKS LIKE

### Gap analysis:
- Current: 4th place, 0.9215
- 1st place: ~0.9260 (estimated)
- Gap: ~0.0045

### To close the gap we need ONE of:
- A genuinely better architecture in the ensemble (YOLO11x could do this)
- A working classifier for the 74% classification error tail (ConvNeXt two-stage)
- Lucky private test alignment (our diverse models happen to match their test better)

### What we should NOT do:
- Make 6 minor variations of candidate11 (wastes submissions, overfits to public)
- Trust local eval numbers (37-image val set is not representative)
- Submit untested ZIPs (always Docker-test first)
- Submit after 14:30 (risk of bugs with no time to fix)
- Panic if a submission scores lower (it tells us something about the test distribution)

---

*Plan created: Mar 22, 2026 — For NM i AI 2026 Task 3*
*Team Experis: Tom Daniel Grande, Henrik Skulevold, Tobias Korten, Fridtjof Hoyer*
