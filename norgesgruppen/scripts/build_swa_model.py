"""Stochastic Weight Averaging (SWA) for YOLO models.

Averages the state_dicts of multiple trained YOLO .pt checkpoints into a single
model, then exports to FP16 ONNX. This produces a smoother, better-calibrated
model with zero extra ZIP space cost.

Usage:
  uv run python scripts/build_swa_model.py \
    weights/yolov8x_run1_best.pt \
    weights/yolov8x_run2_best.pt \
    weights/yolov8x_run3_best.pt \
    --output weights/yolov8x_swa.pt \
    --imgsz 640

All input models must share the same architecture (e.g., all yolov8x).
"""
import argparse
import copy
import sys
from pathlib import Path

import torch


def load_state_dicts(pt_paths: list[str]) -> tuple:
    """Load YOLO models and return (base_model, list_of_state_dicts).

    The base model (first path) is used as the scaffold to load the
    averaged weights back into.
    """
    from ultralytics import YOLO

    state_dicts = []
    base_model = None

    for i, path in enumerate(pt_paths):
        print(f"  Loading [{i+1}/{len(pt_paths)}] {path}")
        model = YOLO(path)
        sd = copy.deepcopy(model.model.state_dict())
        state_dicts.append(sd)
        if i == 0:
            base_model = model

    return base_model, state_dicts


def average_state_dicts(state_dicts: list[dict]) -> dict:
    """Element-wise average of all state_dict parameter tensors."""
    n = len(state_dicts)
    assert n >= 2, f"Need at least 2 models to average, got {n}"

    avg_sd = copy.deepcopy(state_dicts[0])

    # Verify all state_dicts have the same keys
    keys = set(avg_sd.keys())
    for i, sd in enumerate(state_dicts[1:], 1):
        if set(sd.keys()) != keys:
            missing = keys - set(sd.keys())
            extra = set(sd.keys()) - keys
            raise ValueError(
                f"Model {i} has different keys. "
                f"Missing: {missing}, Extra: {extra}"
            )

    # Sum all state_dicts into avg_sd, then divide by n
    for key in avg_sd:
        # Convert to float for accurate averaging, then cast back
        avg_sd[key] = avg_sd[key].float()
        for sd in state_dicts[1:]:
            avg_sd[key] += sd[key].float()
        avg_sd[key] /= n
        # Cast back to original dtype
        avg_sd[key] = avg_sd[key].to(state_dicts[0][key].dtype)

    return avg_sd


def main():
    parser = argparse.ArgumentParser(
        description="Average YOLO model weights (SWA) and export to FP16 ONNX"
    )
    parser.add_argument(
        "models",
        nargs="+",
        help="Paths to .pt checkpoint files (must be same architecture)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for averaged .pt file (default: auto-generated)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for ONNX export (default: 640)",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip ONNX export, only save averaged .pt",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Export ONNX in FP32 instead of FP16",
    )
    args = parser.parse_args()

    # Validate inputs
    for p in args.models:
        if not Path(p).exists():
            print(f"ERROR: File not found: {p}")
            sys.exit(1)

    n = len(args.models)
    if n < 2:
        print("ERROR: Need at least 2 models to average")
        sys.exit(1)

    # Default output path
    if args.output is None:
        base_name = Path(args.models[0]).stem.rsplit("_", 1)[0]
        args.output = f"weights/{base_name}_swa{n}.pt"

    print(f"SWA: Averaging {n} models")
    print(f"  Inputs: {args.models}")
    print(f"  Output: {args.output}")

    # Load all state_dicts
    print("\nLoading models...")
    base_model, state_dicts = load_state_dicts(args.models)

    # Average weights
    print(f"\nAveraging {n} state_dicts...")
    avg_sd = average_state_dicts(state_dicts)

    # Count parameters
    total_params = sum(p.numel() for p in avg_sd.values())
    print(f"  Total parameters averaged: {total_params:,}")

    # Load averaged weights back into the base model
    print("Loading averaged weights into base model...")
    base_model.model.load_state_dict(avg_sd)

    # Save averaged .pt
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use ultralytics save method via torch
    # We need to save in ultralytics format so YOLO() can reload it
    ckpt = torch.load(args.models[0], map_location="cpu", weights_only=False)
    ckpt["model"] = base_model.model
    if "ema" in ckpt and ckpt["ema"] is not None:
        # Update EMA with averaged weights too
        ckpt["ema"] = base_model.model
    ckpt["train_args"] = ckpt.get("train_args", {})
    torch.save(ckpt, str(output_path))
    print(f"\nSaved averaged model: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1e6:.1f} MB")

    # Export to ONNX
    if not args.no_export:
        print(f"\nExporting to ONNX (half={not args.fp32}, imgsz={args.imgsz})...")
        from ultralytics import YOLO

        # Reload the saved model to ensure clean state
        swa_model = YOLO(str(output_path))
        onnx_path = swa_model.export(
            format="onnx",
            half=not args.fp32,
            imgsz=args.imgsz,
        )
        print(f"Exported ONNX: {onnx_path}")
        print(f"  Size: {Path(onnx_path).stat().st_size / 1e6:.1f} MB")

    print("\nDone!")


if __name__ == "__main__":
    main()
