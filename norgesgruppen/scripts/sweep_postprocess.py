"""Sweep post-processing parameters using Optuna on GPU.

Sweeps: temperature, conf threshold, NMS IoU, WBF IoU, neighbor voting.
Builds and evaluates each config against training annotations.

Usage:
  uv run python scripts/sweep_postprocess.py --n-trials 30
"""
import argparse
import json
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from nm_ai_image.detection.evaluate import evaluate_predictions

MODELS = [
    "weights/yolov8x_640_fulldata_best_fp16.onnx",
    "weights/yolov8x_640_s123_best_fp16.onnx",
    "weights/yolov8x_640_s999_best_fp16.onnx",
]
IMAGE_DIR = "data/raw/coco_dataset/train/images"
COCO_JSON = "data/raw/coco_dataset/train/annotations.json"


def build_and_eval(temperature, conf, nms_iou, wbf_iou, neighbor_voting):
    """Build a submission with given params and evaluate it."""
    name = f"sweep_t{temperature}_c{conf}_n{nms_iou}_w{wbf_iou}_nv{int(neighbor_voting)}"

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable, "scripts/build_softvote_submission.py",
            "--onnx", *MODELS,
            "--name", name,
            "--output-dir", tmpdir,
            "--temperature", str(temperature),
            "--conf", str(conf),
            "--nms-iou", str(nms_iou),
            "--wbf-iou", str(wbf_iou),
        ]
        if neighbor_voting:
            cmd.append("--neighbor-voting")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  BUILD FAILED: {result.stderr[-200:]}")
            return None

        zip_path = Path(tmpdir) / f"{name}.zip"
        if not zip_path.exists():
            print(f"  ZIP not found")
            return None

        # Extract and run inference
        extract_dir = Path(tmpdir) / "submission"
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)

        out_json = Path(tmpdir) / "predictions.json"
        image_dir = str(Path(IMAGE_DIR).resolve())
        result = subprocess.run(
            [sys.executable, "run.py", "--input", image_dir, "--output", str(out_json)],
            cwd=extract_dir,
            timeout=7200,
        )
        if result.returncode != 0:
            print(f"  INFERENCE FAILED")
            return None

        with open(out_json) as f:
            predictions = json.load(f)

        coco_json = str(Path(COCO_JSON).resolve())
        eval_result = evaluate_predictions(predictions, coco_json)
        return {
            "competition_score": eval_result.competition_score,
            "detection_map50": eval_result.detection_map50,
            "classification_map50": eval_result.classification_map50,
            "num_predictions": eval_result.num_predictions,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()

    try:
        import optuna
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "optuna"], check=True)
        import optuna

    results_file = Path("sweep_results.json")
    all_results = json.loads(results_file.read_text()) if results_file.exists() else []

    def objective(trial):
        temperature = trial.suggest_float("temperature", 0.1, 3.0, log=True)
        conf = trial.suggest_float("conf", 0.001, 0.05, log=True)
        nms_iou = trial.suggest_float("nms_iou", 0.4, 0.9)
        wbf_iou = trial.suggest_float("wbf_iou", 0.3, 0.8)
        neighbor_voting = trial.suggest_categorical("neighbor_voting", [True, False])

        print(f"\nTrial {trial.number}: T={temperature:.3f} conf={conf:.4f} nms={nms_iou:.2f} wbf={wbf_iou:.2f} nv={neighbor_voting}")

        result = build_and_eval(temperature, conf, nms_iou, wbf_iou, neighbor_voting)
        if result is None:
            return 0.0

        score = result["competition_score"]
        print(f"  -> Score: {score:.4f} (det={result['detection_map50']:.4f}, cls={result['classification_map50']:.4f}, preds={result['num_predictions']})")

        record = {
            "trial": trial.number,
            "temperature": temperature,
            "conf": conf,
            "nms_iou": nms_iou,
            "wbf_iou": wbf_iou,
            "neighbor_voting": neighbor_voting,
            "timestamp": datetime.now().isoformat(),
            **result,
        }
        all_results.append(record)
        results_file.write_text(json.dumps(all_results, indent=2))

        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    print(f"\n{'='*60}")
    print(f"BEST TRIAL: {study.best_trial.number}")
    print(f"Score: {study.best_value:.4f}")
    print(f"Params: {study.best_params}")
    print(f"{'='*60}")

    print("\nTop 5 trials:")
    for trial in sorted(study.trials, key=lambda t: t.value or 0, reverse=True)[:5]:
        print(f"  {trial.value:.4f}  {trial.params}")


if __name__ == "__main__":
    main()
