"""Best prediction pipeline — ensemble RF + ConvCNP with adaptive weighting.

Combines RF (stable 86-88) with ConvCNP (potential 91+) using adaptive
geometric blending. Falls back to RF if ConvCNP collapses.

Usage:
    from nm_ai_ml.astar.best_predict import best_predict_round
    predictions = best_predict_round("data/rounds/round_17")
"""
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

GRID_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}
EPS = 1e-7
MIN_PROB = 0.005


# ============================================================
# Expansion rate estimation (Steg 3)
# ============================================================

def estimate_expansion_rate(observations, initial_map):
    """Estimate settlement dynamics from observations."""
    H, W = initial_map.shape
    obs_class = np.full((H, W), -1, dtype=int)
    obs_mask = np.zeros((H, W), dtype=bool)

    for o in observations:
        vp = o["viewport"]
        for dy in range(len(o["grid"])):
            for dx in range(len(o["grid"][0])):
                gy, gx = vp["y"] + dy, vp["x"] + dx
                if gy >= H or gx >= W:
                    continue
                obs_class[gy, gx] = GRID_TO_CLASS.get(o["grid"][dy][dx], 0)
                obs_mask[gy, gx] = True

    was_empty = np.isin(initial_map, [0, 11]) & obs_mask
    now_sett = (obs_class == 1) & was_empty
    expansion = now_sett.sum() / max(was_empty.sum(), 1)

    was_sett = np.isin(initial_map, [1, 2]) & obs_mask
    still_alive = np.isin(obs_class, [1, 2]) & was_sett
    survival = still_alive.sum() / max(was_sett.sum(), 1)

    density = (obs_class == 1).sum() / max(obs_mask.sum(), 1)

    return {
        "expansion_rate": float(expansion),
        "survival_rate": float(survival),
        "settlement_density": float(density),
    }


# ============================================================
# Adaptive geometric blending (Steg 2)
# ============================================================

def _build_obs_grids(observations, grid_size=40):
    """Build observation class and mask grids."""
    obs_class = np.full((grid_size, grid_size), -1, dtype=int)
    obs_mask = np.zeros((grid_size, grid_size), dtype=bool)

    for o in observations:
        vp = o["viewport"]
        for dy in range(len(o["grid"])):
            for dx in range(len(o["grid"][0])):
                gy, gx = vp["y"] + dy, vp["x"] + dx
                if gy >= grid_size or gx >= grid_size:
                    continue
                obs_class[gy, gx] = GRID_TO_CLASS.get(o["grid"][dy][dx], 0)
                obs_mask[gy, gx] = True

    return obs_class, obs_mask


def adaptive_geo_blend(rf_pred, cnn_pred, obs_class, obs_mask,
                       beta=3.0, min_weight=0.15):
    """Blend RF and CNN predictions using adaptive weights.

    Measures which model agrees more with observed cells,
    then blends geometrically with those weights.
    """
    kl_rf = kl_cnn = 0.0
    n = 0

    for r in range(40):
        for c in range(40):
            if not obs_mask[r, c] or obs_class[r, c] < 0:
                continue
            # Pseudo ground truth from observation (one-hot with floor)
            p = np.full(6, MIN_PROB)
            p[obs_class[r, c]] = 1.0 - 5 * MIN_PROB
            p /= p.sum()

            kl_rf += np.sum(p * np.log(p / np.clip(rf_pred[r, c], EPS, 1)))
            kl_cnn += np.sum(p * np.log(p / np.clip(cnn_pred[r, c], EPS, 1)))
            n += 1

    if n > 0:
        kl_rf /= n
        kl_cnn /= n

    # Softmax weighting
    logits = -beta * np.array([kl_rf, kl_cnn])
    w = np.exp(logits - logits.max())
    w /= w.sum()
    w = np.clip(w, min_weight, 1 - min_weight)
    w /= w.sum()
    w_rf, w_cnn = w[0], w[1]

    # Geometric blend
    log_p = w_rf * np.log(np.clip(rf_pred, EPS, 1)) + \
            w_cnn * np.log(np.clip(cnn_pred, EPS, 1))
    out = np.exp(log_p)
    out = np.maximum(out, MIN_PROB)
    out /= out.sum(axis=2, keepdims=True)

    return out, {"kl_rf": kl_rf, "kl_cnn": kl_cnn,
                 "w_rf": w_rf, "w_cnn": w_cnn, "n_obs": n}


def safe_predict(rf_pred, cnn_pred, obs_class, obs_mask):
    """Ensemble with safety fallback."""
    pred, info = adaptive_geo_blend(rf_pred, cnn_pred, obs_class, obs_mask)

    logger.info("  KL_RF=%.4f KL_CNN=%.4f → w_RF=%.2f w_CNN=%.2f (n=%d obs)",
                info["kl_rf"], info["kl_cnn"], info["w_rf"], info["w_cnn"], info["n_obs"])

    # CNN is more than 3x worse than RF → fallback
    if info["kl_rf"] > 0 and info["kl_cnn"] > 3 * info["kl_rf"]:
        logger.warning("  FALLBACK: CNN collapsed, using pure RF")
        return rf_pred.copy(), "rf_only"

    return pred, "ensemble"


# ============================================================
# Main pipeline
# ============================================================

def best_predict_round(
    round_dir: str | Path,
    cnn_epochs: int = 100,
    cnn_episodes: int = 10,
) -> list[np.ndarray]:
    """Full best-predict pipeline for one round.

    1. Run RF (spatial_predictor_rf)
    2. Run ConvCNP (with context-range fix)
    3. Adaptive ensemble with safety fallback

    Returns list of (40,40,6) predictions per seed.
    """
    from nm_ai_ml.astar.spatial_predictor_rf import predict_round as predict_rf
    from nm_ai_ml.astar.convcnp import run_convcnp_pipeline

    round_dir = Path(round_dir)

    with open(round_dir / "round_detail.json") as f:
        detail = json.load(f)

    # Load observations
    all_obs_per_seed = {}
    for si in range(detail["seeds_count"]):
        obs_file = round_dir / "observations" / f"seed_{si}.json"
        if obs_file.exists():
            with open(obs_file) as f:
                all_obs_per_seed[si] = json.load(f)
        else:
            all_obs_per_seed[si] = []

    # Step 1: RF predictions
    logger.info("Step 1: RF predictions...")
    rf_preds = predict_rf(str(round_dir))

    # Step 2: ConvCNP predictions
    logger.info("Step 2: ConvCNP predictions (%d epochs)...", cnn_epochs)
    cnn_preds = run_convcnp_pipeline(
        str(round_dir),
        epochs=cnn_epochs,
        episodes_per_round=cnn_episodes,
    )

    # Step 3: Adaptive ensemble per seed
    logger.info("Step 3: Adaptive ensemble...")
    predictions = []
    for si in range(detail["seeds_count"]):
        obs = all_obs_per_seed.get(si, [])
        if not obs:
            # No observations — collect from all seeds
            for sj in range(detail["seeds_count"]):
                obs.extend(all_obs_per_seed.get(sj, []))

        initial_map = np.array(detail["initial_states"][si]["grid"])

        # Expansion rate info
        rates = estimate_expansion_rate(obs, initial_map)
        logger.info("  Seed %d: expansion=%.3f survival=%.3f density=%.3f",
                     si, rates["expansion_rate"], rates["survival_rate"],
                     rates["settlement_density"])

        obs_class, obs_mask = _build_obs_grids(obs)

        rf_pred = rf_preds[si]
        cnn_pred = cnn_preds[si] if si < len(cnn_preds) else rf_pred

        pred, mode = safe_predict(rf_pred, cnn_pred, obs_class, obs_mask)

        # Final floor + normalize
        pred = np.maximum(pred, MIN_PROB)
        pred /= pred.sum(axis=2, keepdims=True)

        predictions.append(pred)
        logger.info("  Seed %d: %s", si, mode)

    return predictions


def best_predict_and_save(round_dir: str | Path, **kwargs) -> list[np.ndarray]:
    """Run pipeline and save predictions."""
    round_dir = Path(round_dir)
    pred_dir = round_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    predictions = best_predict_round(round_dir, **kwargs)

    for si, pred in enumerate(predictions):
        np.save(pred_dir / f"seed_{si}_best.npy", pred)
        logger.info("Saved seed %d", si)

    return predictions


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    round_dir = sys.argv[1] if len(sys.argv) > 1 else "data/rounds/round_13"
    predictions = best_predict_and_save(round_dir)

    # Score if GT available
    def score_prediction(pred, gt):
        eps = 1e-10
        pred = np.maximum(pred, MIN_PROB)
        pred = pred / pred.sum(axis=2, keepdims=True)
        kl = np.sum(gt * np.log((gt + eps) / (pred + eps)), axis=2)
        entropy = -np.sum(gt * np.log(gt + eps), axis=2)
        te = entropy.sum()
        return max(0, min(100, 100 * np.exp(-3 * (entropy * kl).sum() / te))) if te > 1e-10 else 100

    for si, pred in enumerate(predictions):
        try:
            with open(f"{round_dir}/analysis/seed_{si}.json") as f:
                a = json.load(f)
            if a.get("ground_truth"):
                gt = np.array(a["ground_truth"])
                print(f"  Seed {si}: {score_prediction(pred, gt):.1f}")
        except FileNotFoundError:
            pass
