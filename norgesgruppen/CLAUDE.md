# CLAUDE.md — NM i AI 2026 Competition Repo (Image/CV)

## CRITICAL RULE
**NEVER submit to the competition server without asking the user first.**
No ZIP uploads, no API submissions, no predictions sent to ainm.no — always ask and get explicit approval before any external submission.

## Project
- NM i AI 2026 competition (Mar 19-22, 69 hours)
- This repo: Task 2 — NorgesGruppen Data (Object Detection, mAP@0.5)
- Team: Experis (Tom Daniel Grande, Henrik Skulevold, Tobias Korten, Fridtjof Hoyer)

## Package
- `nm_ai_image/` — Python package with config, model, training, evaluation, ensemble, competition, detection modules
- `nm_ai_image/detection/` — Detection pipeline: data conversion, training (ultralytics), inference (single/ensemble/SAHI/ONNX), classifier (embedding gallery), evaluation, submission ZIP builder
- Run tests: `uv run python -m pytest test/ -v`
- Detection training: `uv run python main.py detect --model yolov8x.pt --imgsz 640 --epochs 150`
- Build submission: `uv run python main.py submission best.pt --name my_sub`
- Evaluate: `uv run python main.py eval best.pt`

## Azure ML
- Workspace: nmai-experis (rg-nmai-workspace)
- Subscription: 0a2942e9-987d-4858-a1e9-a46350d0c669
- Compute: ainmxperis (Standard_NC4as_T4_v3, 16GB VRAM, max 8 nodes)
- Always use `--subscription "0a2942e9-987d-4858-a1e9-a46350d0c669"` with az ml commands
- Job YMLs are in `jobs/` directory
- **IMPORTANT**: Azure ML caches `code: .` snapshots. Ensure all code changes are saved before `az ml job create`. Old cached code will silently run stale files.
- The base image (`acpt-pytorch-2.2-cuda12.1:31`) has numpy 1.23.5 + matplotlib 3.5.3. Installing ultralytics upgrades numpy to 2.x which breaks matplotlib. Fix: `pip install ultralytics==8.1.0 && pip install "numpy<2"`
- T4 GPU (16GB) cannot run YOLOv8x at 1280px — use YOLOv8l or smaller for high-res
- YOLO11 and YOLO26 require `ultralytics>=8.3` (not 8.1.0). Use `uv pip install --system "ultralytics>=8.3"` in job commands
- `weights/` is excluded from `.amlignore` — to use ONNX weights in Azure jobs, reference previous job outputs as inputs or export within the job

## Submission
- **ALWAYS test in Docker before submitting**: `./docker/test_submission.sh submissions/my_sub.zip`
- **NEVER submit .pt files** — torch 2.6.0 on sandbox breaks ultralytics .pt loading. Always export to ONNX.
- Max 3 submissions/day (resets midnight UTC). Don't waste them without Docker validation.
- Sandbox: Python 3.11, torch 2.6.0, ultralytics 8.1.0, NVIDIA L4, 300s timeout, no network
- Max ZIP size: 420MB (uncompressed)
- **Max 3 weight files** (.pt, .pth, .onnx, .safetensors, .npy) — 4-model ensembles are INVALID
- Max 10 Python files, max 1000 files total
- `os`, `sys`, `subprocess`, `pickle`, `yaml`, `socket`, `ctypes`, `marshal`, `requests`, `urllib`, `multiprocessing`, `threading` are ALL BLOCKED in sandbox — auto-ban if detected
- **ALWAYS run `uv run python scripts/validate_submission.py submission.zip` before submitting**
- Use `pathlib` instead of `os`, `json` instead of `yaml`
- External data (public datasets, own photos) is explicitly allowed
- Final ranking uses PRIVATE test set (different from public leaderboard)

## Leaderboard Results (ranked)
- **Best: 0.9226** — `sub2_c11_yolo11x_swap` (fulldata_x + YOLO11x + yolov8l_highaug, multi-scale 640+800, TTA hflip, WBF iou=0.6) — NEW BEST
- **0.9215** — `candidate11_fulldata_noflip_l` (fulldata_x + noflip_s42 + yolov8l, multi-scale 640+800)
- **0.9210** — `wbf_2x1l_multiscale_tta` (2x_x + 1l, multi-scale 640+800 + TTA)
- **0.9190** — `sub1_triple_arch` (fulldata_x + YOLO11x + nf_l_s1, multi-scale 640+800, TTA hflip, WBF iou=0.6)
- **0.9160** — `wbf_3x_fp16_tta.zip` (3x yolov8x FP16 WBF ensemble + TTA hflip)
- **0.9142** — `softvote_3x_fp16.zip` (3x yolov8x FP16 soft class voting)
- **0.9091** — `onnx_ensemble_fulldata_ms_tuned.zip` (WBF ensemble fulldata x+m+s)
- **0.7802** — `yolov8x_onnx_clean.zip` (single yolov8x baseline, seed 42)
- **0.7700** — `yolov8l_640_s77.zip` (single yolov8l, seed 77)

### Key Insights
- WBF ensemble > soft voting on real test set (both better than single model)
- TTA (horizontal flip) helps: +0.0018 on test
- Val/train mAP does NOT reliably predict test performance. Rank submissions by test score only.
- Classification is the bottleneck (96.7%), not detection (98.3%). Score = 0.7*det + 0.3*cls.
- Top confusion pairs: same brand different size/variant (NESCAFE 100G vs 200G, etc.)
- YOLO11x architecture diversity helps when paired with proven models (+0.0011 over candidate11)

## Submission Builders
- WBF ensemble: `uv run python scripts/build_advanced_submission.py --onnx model1.onnx model2.onnx --name X --tta`
- Soft vote ensemble: `uv run python scripts/build_softvote_submission.py --onnx model1.onnx model2.onnx --name X --temperature 1.0 --neighbor-voting`
- SWA weight averaging: `uv run python scripts/build_swa_model.py model1.pt model2.pt --output avg.pt`
- Eval submissions: `uv run python scripts/eval_submissions.py --quick --parallel 2 submissions/*.zip`
- Optuna sweep: `uv run python scripts/sweep_postprocess.py --n-trials 30`

## Best Model
- Current best: `sub2_c11_yolo11x_swap` (test=0.9226)
- Built with: fulldata_x + YOLO11x + yolov8l_highaug, multi-scale 640+800, TTA hflip, WBF iou=0.6
- 3 ONNX models: fulldata_x + nf_11x_s42 (YOLO11x) + yolov8l_highaug
- FP16 ONNX ~131MB each, 3 fit in 420MB ZIP limit
- 4th model (yolov8s FP16 ~22MB) can be squeezed in: 382MB total
- All submitted ZIPs stored in `submissions/submitted_models/`, best in `submissions/best_model/`

## Key Files
- `experiments.csv` — all training runs, submissions, and scores
- `eval_results.json` — local evaluation results with timestamps
- `plan/` — strategy documents with timestamps
- `scripts/` — all build, eval, sweep, and utility scripts
- `weights/` — ONNX and .pt model weights (excluded from Azure uploads)

## Fast Eval with Cached Predictions
Instead of re-running ONNX inference for every submission variant (60+ min each), use the cache approach:
1. **Cache once** (~10 min): `uv run python scripts/cache_predictions.py --onnx weights/model1_fp16.onnx weights/model2_fp16.onnx weights/model3_fp16.onnx`
2. **Sweep instantly** (<1 min for 20+ configs): `uv run python scripts/eval_cached.py --cache cached_preds.pkl --sweep`
3. **Test single config**: `uv run python scripts/eval_cached.py --cache cached_preds.pkl --method wbf --temperature 0.5 --neighbor-voting`

Re-cache only when models change. Post-processing params (temperature, WBF IoU, conf, neighbor voting) can be swept without re-running inference.

## Tools
- Use `uv` for package management (not pip)
- Python 3.10
