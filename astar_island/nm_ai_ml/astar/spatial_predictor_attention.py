"""Ensemble spatial prediction for Astar Island.

Uses an ensemble of RF models with different random seeds to get
better-calibrated probability estimates. Instead of one RF with 400 trees,
train 5 independent RFs with 200 trees each and average their predictions.

This gives better uncertainty estimates because each RF sees a different
bootstrap sample, producing diverse predictions that average to better
calibrated probabilities.

Also uses vectorized feature extraction for speed.
"""
import json
import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, uniform_filter, distance_transform_edt
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

GRID_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}
NUM_CLASSES = 6
W_NEAR = 0.75
W_FAR = 0.50
N_ENSEMBLES = 5


def _cell_features(grid, settlements):
    """Same 16 features as the original RF (proven to work)."""
    H, W = grid.shape

    sett_map = np.zeros((H, W))
    port_map = np.zeros((H, W))
    for se in settlements:
        sett_map[se["y"], se["x"]] = 1
        if se.get("has_port"):
            port_map[se["y"], se["x"]] = 1

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
    terrain = np.vectorize(lambda v: GRID_TO_CLASS.get(int(v), 0))(grid).astype(float)

    features = np.stack([
        terrain, sett_map, port_map,
        ocean_map, mountain_map, forest_map, plains_map,
        dist_sett, dist_ocean,
        sett_r1, sett_r3, sett_r5,
        forest_r3, ocean_r3, plains_r3, port_r5,
    ], axis=-1)

    return features.reshape(H * W, -1).astype(np.float32), dist_sett, coastal


def _build_training_data(round_dir, detail):
    """Same training data as original RF."""
    X_rows, Y_rows = [], []
    trans_near = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=float)
    trans_far = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=float)
    trans_coastal = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=float)

    for seed_idx in range(detail["seeds_count"]):
        state = detail["initial_states"][seed_idx]
        grid = np.array(state["grid"])
        H, W = grid.shape
        settlements = state.get("settlements", [])
        all_feats, dist_sett, coastal = _cell_features(grid, settlements)

        sett_map = np.zeros((H, W))
        for se in settlements:
            sett_map[se["y"], se["x"]] = 1
        sett_coords = np.column_stack(np.where(sett_map)) if sett_map.sum() > 0 else None
        ocean_map = grid == 10
        coastal_mask = binary_dilation(ocean_map, structure=np.ones((3, 3))) & ~ocean_map

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

                    dist = dist_sett[gy, gx]
                    if dist <= 3:
                        trans_near[init_class, after_class] += 1
                    else:
                        trans_far[init_class, after_class] += 1
                    if coastal_mask[gy, gx]:
                        trans_coastal[init_class, after_class] += 1

                    target = np.zeros(NUM_CLASSES)
                    target[after_class] = 1.0
                    X_rows.append(all_feats[gy * W + gx])
                    Y_rows.append(target)

    for t in [trans_near, trans_far, trans_coastal]:
        row_sums = t.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        t /= row_sums

    return np.array(X_rows), np.array(Y_rows), trans_near, trans_far, trans_coastal


def predict_round(round_dir):
    """Build predictions with ensemble of RF models."""
    round_dir = Path(round_dir)

    with open(round_dir / "round_detail.json") as f:
        detail = json.load(f)

    logger.info("Building training data...")
    X, Y, trans_near, trans_far, trans_coastal = _build_training_data(round_dir, detail)
    logger.info("Training data: %d cells, %d features", X.shape[0], X.shape[1])

    # Train ensemble of RF models with different random seeds
    all_model_sets = []
    for ens_idx in range(N_ENSEMBLES):
        logger.info("Training ensemble %d/%d...", ens_idx + 1, N_ENSEMBLES)
        models = []
        for c in range(NUM_CLASSES):
            m = RandomForestRegressor(
                n_estimators=200, max_depth=6, min_samples_leaf=10,
                random_state=42 + ens_idx * 100, n_jobs=-1,
            )
            m.fit(X, Y[:, c])
            models.append(m)
        all_model_sets.append(models)

    predictions = []
    for seed_idx in range(detail["seeds_count"]):
        state = detail["initial_states"][seed_idx]
        grid = np.array(state["grid"])
        H, W = grid.shape
        settlements = state.get("settlements", [])

        feats, dist_sett, coastal = _cell_features(grid, settlements)

        # Average predictions from all ensemble members
        spatial = np.zeros((H, W, NUM_CLASSES))
        for models in all_model_sets:
            pred = np.column_stack([m.predict(feats) for m in models]).reshape(H, W, NUM_CLASSES)
            spatial += pred
        spatial /= N_ENSEMBLES

        # Context-specific blend (same as original)
        sett_map = np.zeros((H, W))
        for se in settlements:
            sett_map[se["y"], se["x"]] = 1
        sett_coords = np.column_stack(np.where(sett_map)) if sett_map.sum() > 0 else None

        blended = np.zeros((H, W, NUM_CLASSES))
        for y in range(H):
            for x in range(W):
                init_class = GRID_TO_CLASS.get(int(grid[y, x]), 0)
                dist = dist_sett[y, x]

                if coastal[y, x]:
                    tp = 0.5 * trans_coastal[init_class] + 0.5 * (
                        trans_near[init_class] if dist <= 3 else trans_far[init_class])
                elif dist <= 3:
                    tp = trans_near[init_class]
                else:
                    tp = trans_far[init_class]

                w_sp = W_NEAR if dist <= 5 else W_FAR
                blended[y, x] = w_sp * spatial[y, x] + (1 - w_sp) * tp

        blended = np.maximum(blended, 0.01)
        blended /= blended.sum(axis=2, keepdims=True)

        predictions.append(blended)
        logger.info("Seed %d: done", seed_idx)

    return predictions
