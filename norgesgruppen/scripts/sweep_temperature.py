"""Sweep temperature parameter for soft-vote ensemble submissions.

Builds submissions with different temperature values for the softmax
scaling applied to class probability vectors before averaging, then
evaluates each and prints a comparison table.

Usage:
  uv run python scripts/sweep_temperature.py
"""
import subprocess
import sys
from pathlib import Path

TEMPERATURES = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

MODELS = [
    "weights/yolov8x_640_fulldata_best_fp16.onnx",
    "weights/yolov8x_640_s123_best_fp16.onnx",
    "weights/yolov8x_640_seed999_best_fp16.onnx",
]

OUTPUT_DIR = "submissions"


def main():
    zip_paths = []

    # Build submissions for each temperature
    for temp in TEMPERATURES:
        name = f"softvote_temp_{str(temp).replace('.', 'p')}"
        print(f"\n{'='*60}")
        print(f"Building submission: T={temp} -> {name}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, "scripts/build_softvote_submission.py",
            "--onnx", *MODELS,
            "--name", name,
            "--output-dir", OUTPUT_DIR,
            "--temperature", str(temp),
        ]
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
        if result.returncode != 0:
            print(f"FAILED to build T={temp}")
            continue

        zip_path = Path(OUTPUT_DIR) / f"{name}.zip"
        if zip_path.exists():
            zip_paths.append(str(zip_path))
        else:
            print(f"WARNING: Expected {zip_path} not found")

    if not zip_paths:
        print("No submissions built successfully.")
        return

    # Evaluate all submissions
    print(f"\n{'='*60}")
    print("Evaluating all temperature variants...")
    print(f"{'='*60}")

    cmd = [sys.executable, "scripts/eval_submissions.py", *zip_paths]
    subprocess.run(cmd, cwd=Path(__file__).parent.parent)


if __name__ == "__main__":
    main()
