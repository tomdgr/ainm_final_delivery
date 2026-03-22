"""Calibrate simulator parameters against ground truth from completed rounds.

Uses scipy.optimize to minimize KL divergence between simulator predictions
and competition ground truth.
"""
import json
import logging
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from nm_ai_ml.astar.simulator import monte_carlo_predict, DEFAULT_PARAMS

logger = logging.getLogger(__name__)

# Parameters to calibrate — ordered by expected impact
CALIBRATION_PARAMS = [
    ("growth_rate", 0.01, 0.5),
    ("winter_severity", 0.1, 1.0),
    ("collapse_threshold", -2.0, 0.0),
    ("founding_threshold", 2.0, 8.0),
    ("food_per_forest", 0.05, 1.0),
    ("rebuild_prob", 0.01, 0.4),
    ("raid_range", 1, 8),
    ("raid_strength", 0.05, 0.8),
    ("longship_raid_bonus", 1, 8),
    ("trade_range", 2, 12),
    ("trade_food", 0.05, 0.8),
    ("port_threshold", 0.5, 5.0),
    ("ruin_reclaim_prob", 0.01, 0.3),
    ("food_per_plains", 0.01, 0.5),
]

CALIBRATED_PARAMS_FILE = Path("data/calibrated_params.json")

# Use up to 25 samples (stratified across rounds) for robust calibration
MAX_CALIBRATION_SAMPLES = 25


def load_ground_truth() -> list[tuple[np.ndarray, np.ndarray, list[dict]]]:
    """Load ground truth data from completed rounds.

    Returns list of (ground_truth, initial_grid, settlements) tuples.
    """
    samples = []
    rounds_dir = Path("data/rounds")
    if not rounds_dir.exists():
        return samples

    for round_dir in sorted(rounds_dir.iterdir()):
        analysis_dir = round_dir / "analysis"
        detail_file = round_dir / "round_detail.json"
        if not analysis_dir.exists() or not detail_file.exists():
            continue

        with open(detail_file) as f:
            detail = json.load(f)

        for seed_file in sorted(analysis_dir.glob("seed_*.json")):
            seed_idx = int(seed_file.stem.split("_")[1])
            with open(seed_file) as f:
                analysis = json.load(f)

            gt = np.array(analysis["ground_truth"])
            state = detail["initial_states"][seed_idx]
            grid = np.array(state["grid"])
            settlements = state.get("settlements", [])
            samples.append((gt, grid, settlements))

    return samples


def compute_kl_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute entropy-weighted KL divergence, matching competition scoring.

    Score = sum_cells entropy(cell) * KL(truth[cell], pred[cell]) / sum_cells entropy(cell)
    """
    eps = 1e-10
    pred = np.maximum(pred, 0.01)
    pred = pred / pred.sum(axis=2, keepdims=True)

    # Per-cell KL divergence: KL(gt || pred)
    kl = np.sum(gt * np.log((gt + eps) / (pred + eps)), axis=2)

    # Per-cell entropy of ground truth (weight)
    entropy = -np.sum(gt * np.log(gt + eps), axis=2)

    # Entropy-weighted mean KL
    total_entropy = entropy.sum()
    if total_entropy < eps:
        return float(kl.mean())
    return float((entropy * kl).sum() / total_entropy)


def evaluate_params(param_values: np.ndarray, samples: list, n_sims: int = 100) -> float:
    """Evaluate a parameter set against ground truth samples."""
    params = dict(DEFAULT_PARAMS)
    for i, (name, _, _) in enumerate(CALIBRATION_PARAMS):
        params[name] = float(param_values[i])

    total_kl = 0.0
    for gt, grid, settlements in samples:
        pred = monte_carlo_predict(grid, settlements, n_sims=n_sims, years=50, params=params)
        total_kl += compute_kl_score(pred, gt)

    return total_kl / len(samples)


def calibrate(n_sims: int = 100) -> dict:
    """Run calibration and return optimized parameters.

    Args:
        n_sims: Number of simulations per evaluation (lower = faster but noisier).

    Returns:
        Full parameter dict with optimized values.
    """
    samples = load_ground_truth()
    if not samples:
        logger.warning("No ground truth data found — using default parameters")
        return dict(DEFAULT_PARAMS)

    # Limit samples for speed
    if len(samples) > MAX_CALIBRATION_SAMPLES:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(samples), MAX_CALIBRATION_SAMPLES, replace=False)
        samples = [samples[i] for i in indices]

    logger.info("Calibrating %d params against %d ground truth samples (%d sims each)",
                len(CALIBRATION_PARAMS), len(samples), n_sims)

    # Compute baseline score
    default_values = np.array([DEFAULT_PARAMS[name] for name, _, _ in CALIBRATION_PARAMS])
    baseline_kl = evaluate_params(default_values, samples, n_sims=n_sims)
    logger.info("Baseline KL (default params): %.4f", baseline_kl)

    bounds = [(low, high) for _, low, high in CALIBRATION_PARAMS]

    best_result = {"kl": baseline_kl, "values": default_values.copy()}
    eval_count = [0]

    def objective(x):
        kl = evaluate_params(x, samples, n_sims=n_sims)
        eval_count[0] += 1
        if kl < best_result["kl"]:
            best_result["kl"] = kl
            best_result["values"] = x.copy()
            param_str = ", ".join(f"{name}={x[i]:.3f}" for i, (name, _, _) in enumerate(CALIBRATION_PARAMS))
            logger.info("Eval %d: new best KL=%.4f (%s)", eval_count[0], kl, param_str)
        elif eval_count[0] % 20 == 0:
            logger.info("Eval %d: KL=%.4f (best=%.4f)", eval_count[0], kl, best_result["kl"])
        return kl

    # 14 params, popsize=10 → 140 initial evals, then iterations
    result = differential_evolution(
        objective,
        bounds=bounds,
        x0=default_values,
        maxiter=40,
        popsize=10,
        tol=0.001,
        seed=42,
        init="sobol",
    )

    logger.info("Calibration complete after %d evaluations", eval_count[0])
    logger.info("Best KL: %.4f (baseline: %.4f, improvement: %.1f%%)",
                best_result["kl"], baseline_kl,
                100 * (baseline_kl - best_result["kl"]) / baseline_kl)

    # Build full params dict
    optimized = dict(DEFAULT_PARAMS)
    for i, (name, _, _) in enumerate(CALIBRATION_PARAMS):
        optimized[name] = float(best_result["values"][i])

    return optimized


def save_params(params: dict) -> None:
    """Save calibrated parameters to disk."""
    CALIBRATED_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATED_PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=2)
    logger.info("Calibrated params saved to %s", CALIBRATED_PARAMS_FILE)


def load_params() -> dict | None:
    """Load calibrated parameters from disk, if available."""
    if CALIBRATED_PARAMS_FILE.exists():
        with open(CALIBRATED_PARAMS_FILE) as f:
            return json.load(f)
    return None


def sensitivity_analysis(n_sims: int = 100, n_points: int = 7) -> dict:
    """One-at-a-time sensitivity sweep for each calibration parameter.

    Returns dict mapping param name to {"range": float, "values": list[float]}.
    """
    samples = load_ground_truth()
    if not samples:
        logger.warning("No ground truth data for sensitivity analysis")
        return {}

    if len(samples) > MAX_CALIBRATION_SAMPLES:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(samples), MAX_CALIBRATION_SAMPLES, replace=False)
        samples = [samples[i] for i in indices]

    baseline = np.array([DEFAULT_PARAMS[name] for name, _, _ in CALIBRATION_PARAMS])
    baseline_kl = evaluate_params(baseline, samples, n_sims=n_sims)
    logger.info("Sensitivity baseline KL: %.4f", baseline_kl)

    results = {}
    for i, (name, low, high) in enumerate(CALIBRATION_PARAMS):
        kls = []
        for val in np.linspace(low, high, n_points):
            x = baseline.copy()
            x[i] = val
            kl = evaluate_params(x, samples, n_sims=n_sims)
            kls.append(kl)
        impact = max(kls) - min(kls)
        results[name] = {"range": impact, "values": kls}
        logger.info("  %s: impact=%.4f (min=%.4f, max=%.4f)", name, impact, min(kls), max(kls))

    ranked = sorted(results.items(), key=lambda x: x[1]["range"], reverse=True)
    logger.info("\nParameter ranking by impact:")
    for rank, (name, info) in enumerate(ranked, 1):
        logger.info("  %d. %s: %.4f", rank, name, info["range"])

    return results


def adapt_params(observations: list[dict], grid: np.ndarray,
                 settlements: list[dict], base_params: dict | None = None,
                 n_sims: int = 50, reg_lambda: float = 0.1) -> dict:
    """Adapt simulator params to current round using observed viewports.

    Runs a quick Nelder-Mead optimization starting from base_params,
    regularized to stay close to the prior.
    """
    from scipy.optimize import minimize

    if base_params is None:
        base_params = load_params() or dict(DEFAULT_PARAMS)

    # Build partial ground truth from observations
    h, w = grid.shape
    obs_mask = np.zeros((h, w), dtype=bool)
    obs_outcomes = np.zeros((h, w, NUM_CLASSES), dtype=np.float64)
    obs_counts = np.zeros((h, w), dtype=np.float64)

    for ob in observations:
        vp = ob["viewport"]
        for dy in range(len(ob["grid"])):
            for dx in range(len(ob["grid"][0])):
                gy, gx = vp["y"] + dy, vp["x"] + dx
                if gy < h and gx < w:
                    cls = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}.get(
                        int(ob["grid"][dy][dx]), 0)
                    obs_outcomes[gy, gx, cls] += 1
                    obs_counts[gy, gx] += 1
                    obs_mask[gy, gx] = True

    if obs_mask.sum() == 0:
        logger.info("No observed cells for adaptation, using base params")
        return base_params

    # Normalize to distributions
    total = obs_counts[obs_mask][:, np.newaxis]
    obs_dist = np.zeros((h, w, NUM_CLASSES))
    obs_dist[obs_mask] = obs_outcomes[obs_mask] / total

    base_values = np.array([base_params.get(name, DEFAULT_PARAMS[name])
                            for name, _, _ in CALIBRATION_PARAMS])

    def objective(x):
        params = dict(base_params)
        for i, (name, _, _) in enumerate(CALIBRATION_PARAMS):
            params[name] = float(x[i])

        pred = monte_carlo_predict(grid, settlements, n_sims=n_sims, years=50, params=params)
        pred = np.maximum(pred, 0.01)
        pred /= pred.sum(axis=2, keepdims=True)

        # KL only on observed cells
        eps = 1e-10
        kl_cells = np.sum(obs_dist[obs_mask] * np.log(
            (obs_dist[obs_mask] + eps) / (pred[obs_mask] + eps)), axis=1)
        fit_loss = kl_cells.mean()

        # Regularization toward base params
        reg_loss = reg_lambda * np.sum((x - base_values) ** 2)
        return fit_loss + reg_loss

    bounds = [(low, high) for _, low, high in CALIBRATION_PARAMS]
    result = minimize(objective, base_values, method="Nelder-Mead",
                      options={"maxiter": 150, "xatol": 0.01, "fatol": 0.005})

    adapted = dict(base_params)
    for i, (name, _, _) in enumerate(CALIBRATION_PARAMS):
        adapted[name] = float(np.clip(result.x[i], *bounds[i]))

    logger.info("Adapted params (loss %.4f → %.4f)", objective(base_values), result.fun)
    return adapted


NUM_CLASSES = 6


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    if "--sensitivity" in sys.argv:
        results = sensitivity_analysis(n_sims=100)
        print("\nSensitivity analysis complete.")
    else:
        params = calibrate(n_sims=100)
        save_params(params)
        print("\nOptimized parameters:")
        for k, v in sorted(params.items()):
            default = DEFAULT_PARAMS.get(k, "?")
            changed = " *" if v != default else ""
            print(f"  {k}: {v:.4f} (default: {default}){changed}")
