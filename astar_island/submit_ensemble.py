"""Reproduce and submit ensemble predictions.

Usage:
  # R16 style (3 models):
  uv run python submit_ensemble.py --round 16 --models rf cnp anp --weights 0.50 0.35 0.15

  # R17 style (4 models):
  uv run python submit_ensemble.py --round 17 --models rf cnp anp cnp_v5 --weights 0.45 0.25 0.15 0.15

  # Dry run (no submission):
  uv run python submit_ensemble.py --round 17 --dry-run
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

MODEL_DIRS = {
    "rf": "data/model_predictions/rf",
    "cnp": "data/model_predictions/cnp",
    "anp": "data/model_predictions/anp",
    "cnp_v5": "data/model_predictions_v5/cnp_v5",
    "xgb": "data/model_predictions_v5/xgb",
}


def geo_blend(preds, weights):
    eps = 1e-8
    log_sum = sum(w * np.log(np.maximum(p, eps)) for w, p in zip(weights, preds))
    b = np.exp(log_sum)
    b = np.maximum(b, 0.005)
    b /= b.sum(axis=2, keepdims=True)
    return b


def score_prediction(pred, gt):
    eps = 1e-10
    pred = np.maximum(pred, 0.005)
    pred = pred / pred.sum(axis=2, keepdims=True)
    kl = np.sum(gt * np.log((gt + eps) / (pred + eps)), axis=2)
    entropy = -np.sum(gt * np.log(gt + eps), axis=2)
    total_ent = entropy.sum()
    weighted_kl = (entropy * kl).sum() / total_ent if total_ent > eps else kl.mean()
    return max(0, min(100, 100 * np.exp(-3 * weighted_kl)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--models", nargs="+", default=["rf", "cnp", "anp"])
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.weights is None:
        n = len(args.models)
        args.weights = [1.0 / n] * n
    assert len(args.models) == len(args.weights), "Models and weights must match"

    # Normalize weights
    total = sum(args.weights)
    args.weights = [w / total for w in args.weights]

    rd = Path(f"data/rounds/round_{args.round}")
    with open(rd / "round_detail.json") as f:
        detail = json.load(f)

    logger.info("Round %d — Ensemble: %s", args.round,
                " + ".join(f"{m}({w:.0%})" for m, w in zip(args.models, args.weights)))

    # Check availability
    for m in args.models:
        f = Path(MODEL_DIRS[m]) / f"round_{args.round}_seed_0.npy"
        if not f.exists():
            logger.error("MISSING: %s — run the pipeline first", f)
            exit(1)
    logger.info("All model predictions available ✓")

    # Build ensemble
    predictions = []
    for si in range(detail["seeds_count"]):
        preds = []
        for m in args.models:
            p = np.load(Path(MODEL_DIRS[m]) / f"round_{args.round}_seed_{si}.npy")
            p = np.maximum(p, 0.005)
            p /= p.sum(axis=2, keepdims=True)
            preds.append(p)

        blend = geo_blend(preds, args.weights)
        predictions.append(blend)

        # Check agreement
        eps = 1e-8
        for m, p in zip(args.models, preds):
            kl = np.sum(preds[0] * np.log((preds[0] + eps) / (p + eps)), axis=2).mean()
            if kl > 0.3:
                logger.warning("Seed %d: %s diverges from %s (KL=%.3f)", si, m, args.models[0], kl)

    logger.info("Ensemble built: shape=%s, min=%.4f", predictions[0].shape, min(p.min() for p in predictions))

    # Score against GT if available
    gt_file = rd / "analysis" / "seed_0.json"
    if gt_file.exists():
        scores = []
        for si in range(detail["seeds_count"]):
            gt = np.array(json.load(open(rd / "analysis" / f"seed_{si}.json"))["ground_truth"])
            scores.append(score_prediction(predictions[si], gt))
        logger.info("Score vs GT: avg=%.1f [%s]", np.mean(scores), ", ".join(f"{s:.1f}" for s in scores))

    # Save
    out_dir = rd / "predictions"
    out_dir.mkdir(exist_ok=True)
    for si, pred in enumerate(predictions):
        np.save(out_dir / f"seed_{si}_final_ensemble.npy", pred)

    # Submit
    if not args.dry_run:
        from nm_ai_ml.astar.client import AstarClient
        client = AstarClient()
        round_id = detail["id"]
        for si, pred in enumerate(predictions):
            resp = client.submit(round_id, si, pred.tolist())
            logger.info("Seed %d: %s", si, resp)
        client.close()
        logger.info("ALL SUBMITTED")
    else:
        logger.info("Dry run — predictions saved to %s", out_dir)
