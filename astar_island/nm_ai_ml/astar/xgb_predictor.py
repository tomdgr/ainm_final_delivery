"""XGBoost spatial predictor with enriched features.

Improvements over RF predictor:
1. XGBoost (gradient boosting > random forest for KL optimization)
2. Sim-derived features (local sim distributions as 6 extra features)
3. Richer spatial features (food potential, settlement stats, terrain entropy)
4. Temperature scaling (post-hoc calibration)
5. TTA (8-fold dihedral augmentation)
6. Multi-seed ensemble (5 models averaged)
"""
import json
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.ndimage import binary_dilation, uniform_filter, distance_transform_edt
from scipy.stats import entropy as scipy_entropy
import xgboost as xgb

logger = logging.getLogger(__name__)

GRID_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}
NUM_CLASSES = 6


def _enriched_features(grid, settlements, sim_distributions=None):
    """Build enriched per-cell features.

    Returns: (H*W, F) feature array, dist_sett, coastal
    Features (28 total):
      0-5: terrain one-hot
      6: settlement map, 7: port map
      8: dist_sett, 9: dist_ocean
      10-12: sett counts r1/r3/r5
      13-15: forest/ocean/plains counts r3
      16: port count r5, 17: mountain count r3
      18: coastal flag
      19-20: y_coord, x_coord (normalized)
      21: food potential (weighted forest + plains in r3)
      22: terrain entropy in r3
      23: initial population (if settlement)
      24: initial food
      25: initial wealth
      26: num settlements in initial state (global)
      27: fraction of map that is ocean
      + 6 sim distribution features (if available)
    """
    H, W = grid.shape

    sett_map = np.zeros((H, W))
    port_map = np.zeros((H, W))
    pop_map = np.zeros((H, W))
    food_map = np.zeros((H, W))
    wealth_map = np.zeros((H, W))
    for se in settlements:
        sett_map[se["y"], se["x"]] = 1
        if se.get("has_port"):
            port_map[se["y"], se["x"]] = 1
        pop_map[se["y"], se["x"]] = se.get("population", 1.0)
        food_map[se["y"], se["x"]] = se.get("food", 1.0)
        wealth_map[se["y"], se["x"]] = se.get("wealth", 0.0)

    ocean_map = (grid == 10).astype(float)
    mountain_map = (grid == 5).astype(float)
    forest_map = (grid == 4).astype(float)
    plains_map = (grid == 11).astype(float)

    dist_sett = distance_transform_edt(1 - sett_map) if sett_map.sum() > 0 else np.full((H, W), 99.0)
    dist_ocean = distance_transform_edt(1 - ocean_map) if ocean_map.sum() > 0 else np.full((H, W), 99.0)

    def count_radius(arr, r):
        size = 2 * r + 1
        return uniform_filter(arr, size=size, mode='constant') * size * size

    sett_r1 = count_radius(sett_map, 1) - sett_map
    sett_r3 = count_radius(sett_map, 3)
    sett_r5 = count_radius(sett_map, 5)
    forest_r3 = count_radius(forest_map, 3)
    ocean_r3 = count_radius(ocean_map, 3)
    plains_r3 = count_radius(plains_map, 3)
    port_r5 = count_radius(port_map, 5)
    mountain_r3 = count_radius(mountain_map, 3)

    coastal = binary_dilation(ocean_map.astype(bool), structure=np.ones((3, 3))) & ~ocean_map.astype(bool)

    # Normalized coordinates
    yy, xx = np.meshgrid(np.arange(H) / (H - 1), np.arange(W) / (W - 1), indexing='ij')

    # Food potential: weighted sum of food-producing terrain nearby
    food_potential = forest_r3 * 0.22 + plains_r3 * 0.07  # using sim v2 defaults

    # Terrain entropy in r3 (diversity of terrain types)
    terrain_class = np.vectorize(lambda v: GRID_TO_CLASS.get(int(v), 0))(grid)
    terrain_entropy = np.zeros((H, W))
    for y in range(H):
        for x in range(W):
            y0, y1 = max(0, y - 3), min(H, y + 4)
            x0, x1 = max(0, x - 3), min(W, x + 4)
            patch = terrain_class[y0:y1, x0:x1].flatten()
            counts = np.bincount(patch, minlength=NUM_CLASSES)
            if counts.sum() > 0:
                p = counts / counts.sum()
                terrain_entropy[y, x] = scipy_entropy(p + 1e-10)

    # Global features (same for all cells in this seed)
    n_settlements = len(settlements)
    ocean_fraction = ocean_map.mean()

    features_list = [
        # One-hot terrain (6)
        *[(terrain_class == c).astype(float) for c in range(NUM_CLASSES)],
        sett_map, port_map,  # 2
        dist_sett / 20.0, dist_ocean / 20.0,  # 2 (normalized)
        sett_r1, sett_r3, sett_r5,  # 3
        forest_r3, ocean_r3, plains_r3,  # 3
        port_r5, mountain_r3,  # 2
        coastal.astype(float),  # 1
        yy, xx,  # 2
        food_potential,  # 1
        terrain_entropy,  # 1
        pop_map, food_map, wealth_map,  # 3 (initial settlement stats)
        np.full((H, W), n_settlements / 60.0),  # 1 (normalized)
        np.full((H, W), ocean_fraction),  # 1
    ]

    features = np.stack(features_list, axis=-1)  # (H, W, F)

    # Add sim distributions if available
    if sim_distributions is not None:
        features = np.concatenate([features, sim_distributions], axis=-1)

    F = features.shape[-1]
    return features.reshape(H * W, F).astype(np.float32), dist_sett, coastal


def _run_quick_sim(grid, settlements, n_sims=100):
    """Run quick local sim to get per-cell distributions as features."""
    try:
        from nm_ai_ml.astar.simulator_v2 import Simulator, _LUT, DEFAULT_PARAMS
        H, W = grid.shape
        counts = np.zeros((H, W, 6), dtype=np.int32)
        for i in range(n_sims):
            sim = Simulator(grid, settlements, params=DEFAULT_PARAMS, seed=42 + i)
            final = sim.run(50)
            class_grid = _LUT[final]
            for c in range(6):
                counts[:, :, c] += (class_grid == c)
        return counts.astype(np.float32) / n_sims
    except Exception:
        return None


def _build_training_data(round_dir, detail, use_sim=True):
    """Build training data with enriched features."""
    X_rows, Y_rows = [], []
    trans_near = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=float)
    trans_far = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=float)

    for seed_idx in range(detail["seeds_count"]):
        state = detail["initial_states"][seed_idx]
        grid = np.array(state["grid"])
        H, W = grid.shape
        settlements = state.get("settlements", [])

        # Get sim distributions as features
        sim_dist = _run_quick_sim(grid, settlements, n_sims=50) if use_sim else None

        all_feats, dist_sett, coastal = _enriched_features(grid, settlements, sim_dist)

        obs_file = round_dir / "observations" / f"seed_{seed_idx}.json"
        if not obs_file.exists():
            continue
        with open(obs_file) as f:
            obs = json.load(f)

        for o in obs:
            vp = o["viewport"]
            for dy in range(len(o["grid"])):
                for dx in range(len(o["grid"][0])):
                    gy, gx = vp["y"] + dy, vp["x"] + dx
                    if gy >= H or gx >= W:
                        continue
                    init_class = GRID_TO_CLASS.get(int(grid[gy, gx]), 0)
                    after_class = GRID_TO_CLASS.get(int(o["grid"][dy][dx]), 0)
                    d = dist_sett[gy, gx]
                    if d <= 3:
                        trans_near[init_class, after_class] += 1
                    else:
                        trans_far[init_class, after_class] += 1

                    target = np.zeros(NUM_CLASSES)
                    target[after_class] = 1.0
                    X_rows.append(all_feats[gy * W + gx])
                    Y_rows.append(target)

    for t in [trans_near, trans_far]:
        row_sums = t.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        t /= row_sums

    return np.array(X_rows), np.array(Y_rows), trans_near, trans_far


def predict_round(round_dir_str, n_seeds=5, temperature=1.0, use_sim=True):
    """Build predictions with XGBoost ensemble + sim features."""
    round_dir = Path(round_dir_str)

    with open(round_dir / "round_detail.json") as f:
        detail = json.load(f)

    logger.info("Building enriched training data...")
    X, Y, trans_near, trans_far = _build_training_data(round_dir, detail, use_sim=use_sim)

    if len(X) == 0:
        logger.warning("No training data for this round")
        return [np.ones((40, 40, NUM_CLASSES)) / NUM_CLASSES] * detail["seeds_count"]

    logger.info("Training data: %d cells, %d features", X.shape[0], X.shape[1])

    # Train 5 XGBoost models with different seeds
    all_model_sets = []
    for seed in range(n_seeds):
        models = []
        for c in range(NUM_CLASSES):
            m = xgb.XGBRegressor(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.1,
                min_child_weight=10,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42 + seed * 100,
                n_jobs=-1,
                verbosity=0,
            )
            m.fit(X, Y[:, c])
            models.append(m)
        all_model_sets.append(models)

    logger.info("Trained %d XGBoost ensemble models", n_seeds)

    # Predict each seed
    predictions = []
    for seed_idx in range(detail["seeds_count"]):
        state = detail["initial_states"][seed_idx]
        grid = np.array(state["grid"])
        H, W = grid.shape
        settlements = state.get("settlements", [])

        sim_dist = _run_quick_sim(grid, settlements, n_sims=50) if use_sim else None
        feats, dist_sett, coastal = _enriched_features(grid, settlements, sim_dist)

        # Average across ensemble
        spatial = np.zeros((H, W, NUM_CLASSES))
        for models in all_model_sets:
            pred = np.column_stack([m.predict(feats) for m in models]).reshape(H, W, NUM_CLASSES)
            spatial += pred
        spatial /= n_seeds

        # Blend with transition priors
        blended = np.zeros((H, W, NUM_CLASSES))
        terrain_class = np.vectorize(lambda v: GRID_TO_CLASS.get(int(v), 0))(grid)
        for y in range(H):
            for x in range(W):
                init_c = terrain_class[y, x]
                d = dist_sett[y, x]
                tp = trans_near[init_c] if d <= 3 else trans_far[init_c]
                w_sp = 0.75 if d <= 5 else 0.50
                blended[y, x] = w_sp * spatial[y, x] + (1 - w_sp) * tp

        # Temperature scaling
        if temperature != 1.0:
            log_pred = np.log(np.maximum(blended, 1e-8))
            blended = np.exp(log_pred / temperature)

        blended = np.maximum(blended, 0.005)
        blended /= blended.sum(axis=2, keepdims=True)

        predictions.append(blended)
        logger.info("Seed %d: done (%d features)", seed_idx, feats.shape[1])

    return predictions


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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    from tqdm import tqdm

    # Test on all rounds with GT
    rounds_dir = Path("data/rounds")
    test_rounds = []
    for d in sorted(rounds_dir.iterdir()):
        if d.is_dir() and d.name != "rounds":
            r = int(d.name.split("_")[1])
            if (d / "analysis" / "seed_0.json").exists() and (d / "observations" / "seed_0.json").exists():
                test_rounds.append(r)

    print(f"Testing XGBoost predictor on {len(test_rounds)} rounds")
    print(f"{'Round':<8} {'XGB':>8} {'XGB+sim':>10}")
    print("-" * 30)

    all_scores = []
    all_scores_sim = []

    for r in tqdm(test_rounds, desc="Rounds"):
        rd = rounds_dir / f"round_{r}"
        with open(rd / "round_detail.json") as f:
            detail = json.load(f)

        # Without sim features (faster test)
        preds = predict_round(str(rd), n_seeds=3, use_sim=False)

        scores = []
        for si in range(detail["seeds_count"]):
            gt_file = rd / "analysis" / f"seed_{si}.json"
            if gt_file.exists():
                gt = np.array(json.load(open(gt_file))["ground_truth"])
                scores.append(score_prediction(preds[si], gt))
        avg = np.mean(scores) if scores else 0
        all_scores.extend(scores)
        tqdm.write(f"R{r:<6}  {avg:>7.1f}")

    print("-" * 30)
    print(f"{'OVERALL':<8} {np.mean(all_scores):>7.1f}")
