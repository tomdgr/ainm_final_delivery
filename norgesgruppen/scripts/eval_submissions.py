"""Evaluate submission ZIP files against training COCO annotations.

Extracts run.py from each ZIP, runs inference on training images,
then evaluates using our evaluation module.

Usage:
  uv run python scripts/eval_submissions.py submissions/softvote_3x_fp16.zip submissions/wbf_3x_fp16_tta.zip
  uv run python scripts/eval_submissions.py --quick submissions/*.zip   # fast ~50 image subset
  uv run python scripts/eval_submissions.py --parallel 4 submissions/*.zip  # run 4 in parallel
"""
import json
import subprocess
import sys
import tempfile
import zipfile
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from nm_ai_image.detection.evaluate import evaluate_predictions


def prepare_quick_subset(image_dir: str, n: int = 50) -> str:
    """Create a temp dir with a subset of images for quick eval."""
    images = sorted(Path(image_dir).iterdir())
    images = [p for p in images if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    # Take evenly spaced subset
    step = max(1, len(images) // n)
    subset = images[::step][:n]
    tmpdir = tempfile.mkdtemp(prefix="quick_imgs_")
    for img in subset:
        shutil.copy2(img, tmpdir)
    return tmpdir


def patch_run_py_coreml(submission_dir: Path):
    """Patch run.py to prefer CoreML over CUDA for faster Mac inference."""
    run_py = submission_dir / "run.py"
    code = run_py.read_text()
    code = code.replace(
        '["CUDAExecutionProvider", "CPUExecutionProvider"]',
        '["CoreMLExecutionProvider", "CPUExecutionProvider"]'
    )
    run_py.write_text(code)


def eval_submission(zip_path: str, image_dir: str, coco_json: str, use_coreml: bool = False) -> dict:
    zip_path = Path(zip_path)
    image_dir = str(Path(image_dir).resolve())
    coco_json = str(Path(coco_json).resolve())
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmpdir / "submission")

        if use_coreml:
            patch_run_py_coreml(tmpdir / "submission")

        out_json = tmpdir / "predictions.json"
        result = subprocess.run(
            [sys.executable, "run.py", "--input", str(image_dir), "--output", str(out_json)],
            cwd=tmpdir / "submission",
            timeout=7200,
        )
        if result.returncode != 0:
            print(f"FAILED: {zip_path.name}")
            return None

        with open(out_json) as f:
            predictions = json.load(f)

        eval_result = evaluate_predictions(predictions, coco_json)
        return {
            "name": zip_path.name,
            "competition_score": eval_result.competition_score,
            "detection_map50": eval_result.detection_map50,
            "classification_map50": eval_result.classification_map50,
            "num_predictions": eval_result.num_predictions,
        }


def eval_one(args):
    """Wrapper for parallel execution."""
    zip_path, image_dir, coco_json, use_coreml = args
    name = Path(zip_path).name
    print(f"  Starting: {name}")
    r = eval_submission(zip_path, image_dir, coco_json, use_coreml)
    if r:
        print(f"  Done: {name} -> {r['competition_score']:.4f}")
    else:
        print(f"  FAILED: {name}")
    return r


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("zips", nargs="+")
    parser.add_argument("--image-dir", default="data/raw/coco_dataset/train/images")
    parser.add_argument("--coco-json", default="data/raw/coco_dataset/train/annotations.json")
    parser.add_argument("--quick", action="store_true", help="Use ~50 image subset for fast ranking")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel evaluations")
    parser.add_argument("--coreml", action="store_true", help="Use CoreML instead of CPU (Mac only)")
    args = parser.parse_args()

    image_dir = args.image_dir
    quick_dir = None
    if args.quick:
        quick_dir = prepare_quick_subset(image_dir, n=50)
        image_dir = quick_dir
        print(f"Quick mode: using {len(list(Path(quick_dir).iterdir()))} images")

    try:
        results = []
        if args.parallel > 1:
            print(f"Running {len(args.zips)} evals with {args.parallel} workers...")
            tasks = [(z, image_dir, args.coco_json, args.coreml) for z in args.zips]
            with ProcessPoolExecutor(max_workers=args.parallel) as pool:
                futures = {pool.submit(eval_one, t): t[0] for t in tasks}
                for future in as_completed(futures):
                    r = future.result()
                    if r:
                        results.append(r)
        else:
            for z in args.zips:
                print(f"\n{'='*60}")
                print(f"Evaluating: {z}")
                print(f"{'='*60}")
                r = eval_submission(z, image_dir, args.coco_json, args.coreml)
                if r:
                    results.append(r)
                    print(f"  Score: {r['competition_score']:.4f} (det={r['detection_map50']:.4f}, cls={r['classification_map50']:.4f}, preds={r['num_predictions']})")

        if results:
            print(f"\n{'='*60}")
            print("COMPARISON (sorted by score):")
            print(f"{'='*60}")
            for r in sorted(results, key=lambda x: -x["competition_score"]):
                print(f"  {r['competition_score']:.4f}  det={r['detection_map50']:.4f}  cls={r['classification_map50']:.4f}  {r['name']}")

            # Append results to eval_results.json
            results_file = Path("eval_results.json")
            existing = json.loads(results_file.read_text()) if results_file.exists() else []
            from datetime import datetime
            mode = "quick" if args.quick else "full"
            for r in results:
                r["timestamp"] = datetime.now().isoformat()
                r["mode"] = mode
            existing.extend(results)
            results_file.write_text(json.dumps(existing, indent=2))
            print(f"\nResults saved to {results_file}")
    finally:
        if quick_dir:
            shutil.rmtree(quick_dir, ignore_errors=True)
