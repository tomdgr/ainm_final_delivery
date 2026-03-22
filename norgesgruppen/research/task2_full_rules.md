# NorgesGruppen Data — Full Competition Rules

## Scoring
```
Score = 0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5
```
- **Detection (70%)**: IoU ≥ 0.5, category ignored
- **Classification (30%)**: IoU ≥ 0.5 AND correct category_id
- Detection-only (all category_id=0) → max 0.70
- Score range: 0.0 to 1.0

## Submission
- ZIP with `run.py` at root
- `python run.py --input /data/images --output /output/predictions.json`
- Output: JSON array of `{image_id, category_id, bbox: [x,y,w,h], score}`
- bbox in COCO format [x, y, width, height]
- image_id from filename: img_00042.jpg → 42
- category_id: 0-355 (356 = unknown_product, 357 total)

## Limits
- Max zip (uncompressed): 420 MB
- Max files: 1000, max .py files: 10
- Max weight files (.pt/.pth/.onnx/.safetensors/.npy): 3, total 420 MB
- Allowed types: .py, .json, .yaml, .yml, .cfg, .pt, .pth, .onnx, .safetensors, .npy
- 3 submissions/day, 2 in-flight, resets midnight UTC
- 2 infrastructure-failure freebies/day

## Sandbox
- Python 3.11, 4 vCPU, 8 GB RAM
- **NVIDIA L4 (24 GB VRAM)**, CUDA 12.4
- **No network access**
- **Timeout: 300 seconds**
- Pre-installed: PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, ultralytics 8.1.0, onnxruntime-gpu 1.20.0, opencv-python-headless 4.9.0.80, albumentations 1.3.1, Pillow 10.2.0, numpy 1.26.4, scipy 1.12.0, scikit-learn 1.4.0, pycocotools 2.0.7, ensemble-boxes 1.0.9, timm 0.9.12, supervision 0.18.0, safetensors 0.4.2

## Security Restrictions
- Blocked: import os, subprocess, socket, ctypes, builtins
- Blocked: eval(), exec(), compile(), __import__()
- No ELF binaries, symlinks, path traversal
- Use pathlib instead of os

## Training Data
- 248 shelf images (2000×1500), ~22,700 annotations, 356 categories + unknown
- 4 store sections: Egg, Frokost, Knekkebrod, Varmedrikker
- 327 product reference images (multi-angle: main, front, back, left, right, top, bottom)

## Key Constraints for Strategy
1. **ultralytics pinned to 8.1.0** — no YOLO11, YOLOv9, YOLOv10 native .pt
2. Models NOT in sandbox: YOLOv9/10/11, RF-DETR, Detectron2, MMDet, HF Transformers
3. Can use ONNX export (opset ≤ 20) for any model
4. Can include custom PyTorch model code + state_dict .pt
5. FP16 recommended for L4
6. 300s timeout with L4 GPU — larger models feasible
7. ensemble-boxes is pre-installed → WBF available at inference
