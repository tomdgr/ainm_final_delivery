# Plan: Final Push — Generalization for Hidden Test Set

**Date**: 2026-03-20 ~22:30
**Current rank**: 8th (0.9160 public test)
**Target**: Top 3 (0.9199+)
**Submissions**: 1 left today, ~6 tomorrow
**Competition ends**: Mar 22 15:00 CET (~40 hours left)
**CRITICAL**: Final ranking uses PRIVATE test set (never seen). Must generalize.

---

## Hard Constraints Discovered
- **Max 3 weight files** per ZIP (.onnx/.pt/.pth/.safetensors/.npy)
- **Max 420MB** uncompressed ZIP
- **`os` module blocked** in sandbox — use pathlib
- **Max 10 Python files**
- External data (public datasets, own photos) is ALLOWED

---

## Critical Lessons (Don't Repeat)
1. **Local train eval is USELESS for model selection** — cls=2.0 scored +0.015 locally, 0.000 on test
2. **Diversity > Specialization** — different seeds beat cls-tuned models
3. **WBF > Soft voting** on real test (+0.002)
4. **TTA horizontal flip helps** (+0.002)
5. **Temperature, neighbor voting, conf tuning** — no effect on test
6. **4-model ensembles are INVALID** — max 3 weight files rule
7. **Public test ≠ Private test** — don't overfit to leaderboard

---

## Strategy: Model Soup + Maximum Diversity

### Tier 1: Model Soup (TONIGHT — highest priority)
Average weights from multiple fulldata yolov8x runs into a single "soup" model.
- Proven to find wider optima = better generalization (Wortsman et al. 2022)
- Zero inference cost (still 1 model)
- Frees up weight file slots for architecture diversity

**Soup ingredients** (all fulldata, different seeds/configs):
- fulldata_x (default seed) — maroon_okra
- fulldata_x_s77 — jovial_glove (downloaded)
- fulldata_x_s123 — shy_sugar (downloaded)
- fulldata_x_seed999 — musing_toe
- fulldata_cls15_combo — calm_tray (downloaded)

**Plan**: Average top 3-4 of these, export FP16 ONNX, use as 1 of 3 models in ensemble.

### Tier 2: Realistic Augmentation (TONIGHT — Azure jobs)
Current training uses flipud=0.5 which is unrealistic for grocery shelves.
- Train new models with flipud=0.0 (products don't appear upside down)
- Also try reduced mosaic/mixup for more realistic augmentation
- These models should generalize better to real shelf images

### Tier 3: Architecture Diversity in Ensemble (TOMORROW)
3 weight file slots. Best use:
1. **Soup model** (yolov8x averaged) — strong generalized detector
2. **Different architecture** (yolov8l or YOLO11x) — different error patterns
3. **Different training regime** (no flipud, or different augmentation) — different biases

### Tier 4: K-Fold Cross-Validation Signal (TONIGHT — Azure)
Train 5 models on 5 different 80/20 splits. Average their val scores = reliable performance estimate.
Use this to VALIDATE which ensemble combo is actually best, instead of trusting train eval.

### Tier 5: External Data (IF TIME PERMITS)
- SKU-110K dataset (11,762 store shelf images) for pretraining
- NorgesGruppen product images from their website
- Would need annotation — probably too time-consuming

---

## Azure Jobs for Tonight

### Soup Ingredients (diverse fulldata models)
1. fulldata_x_noflip — yolov8x, flipud=0.0, seed 42
2. fulldata_x_noflip_s77 — yolov8x, flipud=0.0, seed 77
3. fulldata_x_noflip_s123 — yolov8x, flipud=0.0, seed 123
4. fulldata_x_lightaug — yolov8x, flipud=0.0, scale=0.3, mosaic=0.5
5. fulldata_x_lightaug_s77 — same but seed 77

### Architecture Diversity
6. fulldata_l_noflip — yolov8l, flipud=0.0 (different arch for ensemble)
7. fulldata_11x_noflip — yolo11x, flipud=0.0 (if ultralytics supports it)

### K-Fold Validation
8. kfold_1 through kfold_5 — 5x yolov8x on different 80/20 splits

---

## Submission Plan

### Tonight (1 remaining): SAVE IT
Unless something clearly better emerges from the 6-model combo sweep.

### Tomorrow (~6 remaining):
1. **Soup model + 2 diverse models** with WBF+TTA — primary bet
2. **Soup + yolov8l + noflip variant** — architecture diversity
3. **Best combo from K-fold validated models** — if K-fold gives reliable signal
4. **WBF param tweak** (iou=0.50, skip=0.05) — free to try
5-6. Iterations based on signal

### Last Submission (save for final hour):
- Pick the submission that maximizes DIVERSITY across:
  - Architecture diversity (different model types)
  - Training diversity (different seeds, augmentations)
  - NOT local train score — ignore that metric

---

## Files & Tools
- SWA script: `scripts/build_swa_model.py`
- Cache predictions: `scripts/cache_predictions.py`
- Instant eval sweep: `scripts/eval_cached.py --sweep`
- Build WBF submission: `scripts/build_advanced_submission.py`
- All experiments: `experiments.csv`
- Azure jobs: `jobs/`
