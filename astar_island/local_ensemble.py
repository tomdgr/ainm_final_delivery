"""Local ensemble — blend RF + ConvCNP predictions for round 16.
Downloads predictions from GCE, blends locally, saves for submission."""
import json
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ROUND_DIR = Path("data/rounds/round_16")


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
    pred_dir = ROUND_DIR / "predictions"
    with open(ROUND_DIR / "round_detail.json") as f:
        detail = json.load(f)
    num_seeds = detail["seeds_count"]

    # Load RF predictions (from Fridtjof's spatial_predictor_rf)
    logger.info("Building RF predictions...")
    from nm_ai_ml.astar.spatial_predictor_rf import predict_round as rf_predict
    rf_preds = rf_predict(str(ROUND_DIR))

    # Load ConvCNP predictions (downloaded from GCE)
    cnp_preds = []
    for si in range(num_seeds):
        f = pred_dir / f"seed_{si}_cnp_quick.npy"
        if not f.exists():
            f = pred_dir / f"seed_{si}_cnp.npy"
        if f.exists():
            cnp_preds.append(np.load(f))
            logger.info("  Loaded ConvCNP seed %d", si)
        else:
            logger.warning("  No ConvCNP for seed %d, using RF", si)
            cnp_preds.append(rf_preds[si])

    # Load ANP predictions if available
    anp_preds = []
    for si in range(num_seeds):
        f = pred_dir / f"seed_{si}_anp.npy"
        if f.exists():
            anp_preds.append(np.load(f))
            logger.info("  Loaded ANP seed %d", si)
        else:
            anp_preds.append(None)

    has_anp = any(p is not None for p in anp_preds)

    # Ensemble with fallback
    logger.info("\nEnsembling with KL-divergence fallback...")
    final_preds = []
    for si in range(num_seeds):
        rf = np.maximum(rf_preds[si], 0.005)
        rf /= rf.sum(axis=2, keepdims=True)
        cnp = np.maximum(cnp_preds[si], 0.005)
        cnp /= cnp.sum(axis=2, keepdims=True)

        # Check ConvCNP divergence from RF
        eps = 1e-8
        kl_cnp = np.sum(rf * np.log((rf + eps) / (cnp + eps)), axis=2).mean()

        if kl_cnp > 0.5:
            # ConvCNP diverges too much — mostly RF
            w_cnp = 0.1
            logger.info("  Seed %d: CNP diverges (KL=%.3f), using 90%% RF", si, kl_cnp)
        else:
            w_cnp = 0.35
            logger.info("  Seed %d: CNP agrees (KL=%.3f), using 65%% RF + 35%% CNP", si, kl_cnp)

        if has_anp and anp_preds[si] is not None:
            anp = np.maximum(anp_preds[si], 0.005)
            anp /= anp.sum(axis=2, keepdims=True)
            kl_anp = np.sum(rf * np.log((rf + eps) / (anp + eps)), axis=2).mean()

            if kl_anp > 0.5:
                w_anp = 0.05
            else:
                w_anp = 0.15
            w_rf = 1.0 - w_cnp - w_anp

            log_blend = (
                w_rf * np.log(rf + eps) +
                w_cnp * np.log(cnp + eps) +
                w_anp * np.log(anp + eps)
            )
        else:
            w_rf = 1.0 - w_cnp
            log_blend = w_rf * np.log(rf + eps) + w_cnp * np.log(cnp + eps)

        blend = np.exp(log_blend)
        blend = np.maximum(blend, 0.005)
        blend /= blend.sum(axis=2, keepdims=True)
        final_preds.append(blend)

    # Save
    pred_dir.mkdir(exist_ok=True)
    for si, pred in enumerate(final_preds):
        np.save(pred_dir / f"seed_{si}_final_ensemble.npy", pred)
    logger.info("\nFinal ensemble predictions saved to %s", pred_dir)

    # Score if GT available
    has_gt = (ROUND_DIR / "analysis" / "seed_0.json").exists()
    if has_gt:
        logger.info("\n--- Scores ---")
        for name, preds in [("RF", rf_preds), ("ConvCNP", cnp_preds), ("Ensemble", final_preds)]:
            scores = []
            for si in range(num_seeds):
                with open(ROUND_DIR / "analysis" / f"seed_{si}.json") as f:
                    gt = np.array(json.load(f)["ground_truth"])
                scores.append(score_prediction(preds[si], gt))
            logger.info("  %-10s avg=%.1f [%s]", name, np.mean(scores),
                        ", ".join(f"{s:.1f}" for s in scores))

    logger.info("\nReady for submission. Files in %s/seed_*_final_ensemble.npy", pred_dir)
