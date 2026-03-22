"""Offline ensemble optimizer — runs locally after GCE produces all predictions.

Loads predictions from data/model_predictions/, scores against GT,
finds optimal weights, and builds final submission.

Usage:
  uv run python ensemble_offline.py --round 17
  uv run python ensemble_offline.py --round 17 --weights 0.5 0.3 0.2
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def score_prediction(pred, gt):
    eps = 1e-10
    pred = np.maximum(pred, 0.005)
    pred = pred / pred.sum(axis=2, keepdims=True)
    kl = np.sum(gt * np.log((gt + eps) / (pred + eps)), axis=2)
    entropy = -np.sum(gt * np.log(gt + eps), axis=2)
    total_ent = entropy.sum()
    weighted_kl = (entropy * kl).sum() / total_ent if total_ent > eps else kl.mean()
    return max(0, min(100, 100 * np.exp(-3 * weighted_kl)))


def geo_blend(preds_list, weights):
    """Geometric mean (log-linear) ensemble."""
    eps = 1e-8
    log_sum = sum(w * np.log(np.maximum(p, eps)) for w, p in zip(weights, preds_list))
    blend = np.exp(log_sum)
    blend = np.maximum(blend, 0.005)
    blend /= blend.sum(axis=2, keepdims=True)
    return blend


def load_predictions(pred_dir, model, round_num, num_seeds=5, suffix=""):
    """Load predictions for a model/round. Returns list of arrays or None."""
    preds = []
    for si in range(num_seeds):
        f = pred_dir / model / f"round_{round_num}_seed_{si}{suffix}.npy"
        if f.exists():
            preds.append(np.load(f))
        else:
            return None
    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--pred-dir", default="data/model_predictions")
    parser.add_argument("--weights", nargs=3, type=float, default=None,
                        help="Manual weights: rf cnp anp")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    rounds_dir = Path("data/rounds")
    models = ["rf", "cnp", "anp"]

    # Find GT rounds
    gt_rounds = []
    for d in sorted(rounds_dir.iterdir()):
        if not d.is_dir() or d.name == "rounds":
            continue
        r = int(d.name.split("_")[1])
        if r != args.round and (d / "analysis" / "seed_0.json").exists():
            gt_rounds.append(r)

    # ============================================================
    # Phase 1: Score each model individually on GT rounds
    # ============================================================
    logger.info("=" * 70)
    logger.info("PHASE 1: Individual model scores (full-trained)")
    logger.info("=" * 70)

    model_scores = {m: {} for m in models}

    for r in gt_rounds:
        rd = rounds_dir / f"round_{r}"
        with open(rd / "round_detail.json") as f:
            detail = json.load(f)

        for m in models:
            preds = load_predictions(pred_dir, m, r, detail["seeds_count"])
            if preds is None:
                continue

            scores = []
            for si in range(detail["seeds_count"]):
                gt_file = rd / "analysis" / f"seed_{si}.json"
                if not gt_file.exists():
                    continue
                with open(gt_file) as f:
                    gt = np.array(json.load(f)["ground_truth"])
                scores.append(score_prediction(preds[si], gt))
            if scores:
                model_scores[m][r] = np.mean(scores)

    print(f"\n{'Round':<8}", end="")
    for m in models:
        print(f"{m.upper():>8}", end="")
    print()
    print("-" * 32)

    for r in gt_rounds:
        print(f"R{r:<6} ", end="")
        for m in models:
            s = model_scores[m].get(r, 0)
            print(f"{s:>7.1f} ", end="")
        print()

    print("-" * 32)
    for m in models:
        vals = list(model_scores[m].values())
        print(f"{'AVG ' + m.upper():<8} {np.mean(vals):>7.1f}  "
              f"(std={np.std(vals):.1f}, min={min(vals):.1f})" if vals else "")

    # ============================================================
    # Phase 2: Score leave-one-out predictions
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: Leave-one-out scores (validation)")
    logger.info("=" * 70)

    loo_scores = {m: {} for m in ["cnp", "anp"]}

    for r in gt_rounds:
        rd = rounds_dir / f"round_{r}"
        with open(rd / "round_detail.json") as f:
            detail = json.load(f)

        for m in ["cnp", "anp"]:
            preds = load_predictions(pred_dir, m, r, detail["seeds_count"], suffix="_loo")
            if preds is None:
                continue

            scores = []
            for si in range(detail["seeds_count"]):
                gt_file = rd / "analysis" / f"seed_{si}.json"
                if not gt_file.exists():
                    continue
                with open(gt_file) as f:
                    gt = np.array(json.load(f)["ground_truth"])
                scores.append(score_prediction(preds[si], gt))
            if scores:
                loo_scores[m][r] = np.mean(scores)

    if any(loo_scores[m] for m in loo_scores):
        print(f"\n{'Round':<8} {'RF':>8} {'CNP_loo':>8} {'ANP_loo':>8}")
        print("-" * 36)
        for r in gt_rounds:
            rf_s = model_scores["rf"].get(r, 0)
            cnp_s = loo_scores["cnp"].get(r, 0)
            anp_s = loo_scores["anp"].get(r, 0)
            print(f"R{r:<6}  {rf_s:>7.1f}  {cnp_s:>7.1f}  {anp_s:>7.1f}")

    # ============================================================
    # Phase 3: Optimize ensemble weights
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: Ensemble weight optimization")
    logger.info("=" * 70)

    # Collect rounds where all models have predictions
    valid_rounds = [r for r in gt_rounds
                    if all(r in model_scores[m] for m in models)]

    if not valid_rounds:
        logger.warning("No rounds with all 3 models — using default weights")
        best_weights = [0.5, 0.3, 0.2]
    else:
        # Grid search over weights
        best_score = 0
        best_weights = [0.5, 0.3, 0.2]

        print(f"\nGrid search on {len(valid_rounds)} rounds...")
        print(f"{'w_rf':>6} {'w_cnp':>6} {'w_anp':>6} {'avg':>8} {'min':>8}")
        print("-" * 40)

        for w_rf in np.arange(0.3, 0.8, 0.05):
            for w_cnp in np.arange(0.1, 0.6, 0.05):
                w_anp = 1.0 - w_rf - w_cnp
                if w_anp < 0.05 or w_anp > 0.5:
                    continue

                all_scores = []
                for r in valid_rounds:
                    rd = rounds_dir / f"round_{r}"
                    with open(rd / "round_detail.json") as f:
                        detail = json.load(f)

                    for si in range(detail["seeds_count"]):
                        gt_file = rd / "analysis" / f"seed_{si}.json"
                        if not gt_file.exists():
                            continue
                        with open(gt_file) as f:
                            gt = np.array(json.load(f)["ground_truth"])

                        preds = []
                        for m in models:
                            f = pred_dir / m / f"round_{r}_seed_{si}.npy"
                            if f.exists():
                                preds.append(np.load(f))
                        if len(preds) == 3:
                            blend = geo_blend(preds, [w_rf, w_cnp, w_anp])
                            all_scores.append(score_prediction(blend, gt))

                if all_scores:
                    avg = np.mean(all_scores)
                    mn = min(all_scores)
                    if avg > best_score:
                        best_score = avg
                        best_weights = [w_rf, w_cnp, w_anp]
                        print(f"{w_rf:>5.2f}  {w_cnp:>5.2f}  {w_anp:>5.2f}  {avg:>7.1f}  {mn:>7.1f}  <-- BEST")

        print(f"\nBest weights: RF={best_weights[0]:.2f} CNP={best_weights[1]:.2f} ANP={best_weights[2]:.2f}")
        print(f"Best avg score: {best_score:.1f}")

    if args.weights:
        best_weights = args.weights
        logger.info("Using manual weights: %s", best_weights)

    # ============================================================
    # Phase 4: Build final submission
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4: Build submission for round %d", args.round)
    logger.info("Weights: RF=%.2f CNP=%.2f ANP=%.2f", *best_weights)
    logger.info("=" * 70)

    target_dir = rounds_dir / f"round_{args.round}"
    with open(target_dir / "round_detail.json") as f:
        detail = json.load(f)

    sub_dir = target_dir / "predictions"
    sub_dir.mkdir(exist_ok=True)

    for si in range(detail["seeds_count"]):
        preds = []
        for m in models:
            f = pred_dir / m / f"round_{args.round}_seed_{si}.npy"
            if f.exists():
                p = np.load(f)
                p = np.maximum(p, 0.005)
                p /= p.sum(axis=2, keepdims=True)
                preds.append(p)
            else:
                logger.warning("Missing %s prediction for seed %d — using RF", m, si)
                rf_f = pred_dir / "rf" / f"round_{args.round}_seed_{si}.npy"
                p = np.load(rf_f)
                p = np.maximum(p, 0.005)
                p /= p.sum(axis=2, keepdims=True)
                preds.append(p)

        blend = geo_blend(preds, best_weights)
        np.save(sub_dir / f"seed_{si}_final_ensemble.npy", blend)
        logger.info("Seed %d: saved (shape=%s, min=%.4f, sum=%.6f)",
                    si, blend.shape, blend.min(), blend.sum(axis=2).mean())

    # Score if GT available
    has_gt = (target_dir / "analysis" / "seed_0.json").exists()
    if has_gt:
        logger.info("\n--- Scores vs GT ---")
        for name, ws in [("RF only", [1, 0, 0]), ("Ensemble", best_weights)]:
            scores = []
            for si in range(detail["seeds_count"]):
                with open(target_dir / "analysis" / f"seed_{si}.json") as f:
                    gt = np.array(json.load(f)["ground_truth"])
                preds = []
                for m in models:
                    f = pred_dir / m / f"round_{args.round}_seed_{si}.npy"
                    if f.exists():
                        p = np.load(f)
                        p = np.maximum(p, 0.005)
                        p /= p.sum(axis=2, keepdims=True)
                        preds.append(p)
                    else:
                        preds.append(preds[0])
                blend = geo_blend(preds, ws)
                scores.append(score_prediction(blend, gt))
            logger.info("  %-12s avg=%.1f [%s]", name, np.mean(scores),
                        ", ".join(f"{s:.1f}" for s in scores))

    logger.info("\nReady: %s/seed_*_final_ensemble.npy", sub_dir)
