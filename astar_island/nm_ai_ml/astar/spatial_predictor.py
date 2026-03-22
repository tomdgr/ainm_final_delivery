"""Spatial prediction model for Astar Island.

Trains a gradient boosting model on the current round's observations
to predict cell outcomes based on spatial context (neighbors, distance
to settlements, nearby terrain). Blends with transition rates for
robust predictions.

Usage:
    from nm_ai_ml.astar.spatial_predictor import predict_round
    predictions = predict_round("data/rounds/round_8")

Cross-validated scores (leave-one-round-out, rounds 2-5):
    w_spatial=0.6: avg=82.8, min=73.7, max=87.4
"""
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

logger = logging.getLogger(__name__)

GRID_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}
NUM_CLASSES = 6
W_SPATIAL = 0.6  # blend weight: 60% spatial model, 40% transition rates


def _cell_features(grid: np.ndarray, settlements: list[dict]) -> np.ndarray:
    """Extract per-cell spatial features from initial state.

    Features (16):
        terrain, is_settlement, is_port,
        is_ocean, is_mountain, is_forest, is_plains,
        dist_to_nearest_settlement, dist_to_nearest_ocean,
        n_settlement_neighbors (adj), n_settlements_r3, n_settlements_r5,
        n_forest_r3, n_ocean_r3, n_plains_r3, n_ports_r5
    """
    H, W = grid.shape

    sett_map = np.zeros((H, W))
    port_map = np.zeros((H, W))
    for se in settlements:
        sett_map[se["y"], se["x"]] = 1
        if se.get("has_port"):
            port_map[se["y"], se["x"]] = 1

    ocean_map = (grid == 10).astype(float)
    forest_map = (grid == 4).astype(float)
    plains_map = (grid == 11).astype(float)

    # Precompute coordinate arrays for distance calculations
    sy, sx = (np.where(sett_map) if sett_map.sum() > 0
              else (np.array([]), np.array([])))
    sett_coords = np.column_stack([sy, sx]) if len(sy) > 0 else None

    oy, ox = np.where(grid == 10)
    ocean_coords = np.column_stack([oy, ox]) if len(oy) > 0 else None

    def count_in_radius(arr, cy, cx, r):
        return arr[max(0, cy - r):min(H, cy + r + 1),
                   max(0, cx - r):min(W, cx + r + 1)].sum()

    rows = []
    for y in range(H):
        for x in range(W):
            terrain = GRID_TO_CLASS.get(int(grid[y, x]), 0)
            is_sett = sett_map[y, x]
            is_port = port_map[y, x]

            dist_sett = (np.sqrt((sett_coords[:, 0] - y) ** 2 +
                                 (sett_coords[:, 1] - x) ** 2).min()
                         if sett_coords is not None else 99)
            dist_ocean = (np.sqrt((ocean_coords[:, 0] - y) ** 2 +
                                  (ocean_coords[:, 1] - x) ** 2).min()
                          if ocean_coords is not None else 99)

            rows.append([
                terrain, is_sett, is_port,
                int(grid[y, x] == 10), int(grid[y, x] == 5),
                int(grid[y, x] == 4), int(grid[y, x] == 11),
                dist_sett, dist_ocean,
                count_in_radius(sett_map, y, x, 1) - is_sett,
                count_in_radius(sett_map, y, x, 3),
                count_in_radius(sett_map, y, x, 5),
                count_in_radius(forest_map, y, x, 3),
                count_in_radius(ocean_map, y, x, 3),
                count_in_radius(plains_map, y, x, 3),
                count_in_radius(port_map, y, x, 5),
            ])
    return np.array(rows, dtype=np.float32)


def _build_training_data(round_dir: Path, detail: dict):
    """Build training data from observations: cell features → observed outcome.

    Returns (X, Y, transition_matrix).
    """
    X_rows, Y_rows = [], []
    trans = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=float)

    for seed_idx in range(detail["seeds_count"]):
        state = detail["initial_states"][seed_idx]
        grid = np.array(state["grid"])
        H, W = grid.shape
        settlements = state.get("settlements", [])
        all_feats = _cell_features(grid, settlements)

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

                    trans[init_class, after_class] += 1

                    target = np.zeros(NUM_CLASSES)
                    target[after_class] = 1.0
                    X_rows.append(all_feats[gy * W + gx])
                    Y_rows.append(target)

    # Normalize transition matrix
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans /= row_sums

    return np.array(X_rows), np.array(Y_rows), trans


def _train_models(X: np.ndarray, Y: np.ndarray) -> list:
    """Train one GBR model per class."""
    models = []
    for c in range(NUM_CLASSES):
        m = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42,
        )
        m.fit(X, Y[:, c])
        models.append(m)
    return models


def predict_round(round_dir: str | Path, w_spatial: float = W_SPATIAL) -> list[np.ndarray]:
    """Build predictions for all seeds in a round.

    Uses this round's observations to:
    1. Train a spatial GBR model (cell features → outcome)
    2. Compute transition rates (initial terrain → outcome)
    3. Blend both: w_spatial * spatial + (1-w_spatial) * transition

    Args:
        round_dir: Path to round directory (e.g. "data/rounds/round_8")
        w_spatial: Blend weight for spatial model (0-1)

    Returns:
        List of H×W×6 prediction arrays, one per seed.
    """
    round_dir = Path(round_dir)

    with open(round_dir / "round_detail.json") as f:
        detail = json.load(f)

    logger.info("Building training data from observations...")
    X, Y, trans = _build_training_data(round_dir, detail)
    logger.info("Training data: %d cells, %d features", X.shape[0], X.shape[1])

    logger.info("Transition matrix:")
    class_names = ["Empty", "Sett", "Port", "Ruin", "Forest", "Mtn"]
    for i, name in enumerate(class_names):
        row = " ".join(f"{trans[i, j]:.3f}" for j in range(NUM_CLASSES))
        logger.info("  %s → %s", name, row)

    logger.info("Training spatial models...")
    models = _train_models(X, Y)

    logger.info("Predicting (w_spatial=%.2f)...", w_spatial)
    predictions = []
    for seed_idx in range(detail["seeds_count"]):
        state = detail["initial_states"][seed_idx]
        grid = np.array(state["grid"])
        H, W = grid.shape
        settlements = state.get("settlements", [])

        feats = _cell_features(grid, settlements)

        # Spatial prediction
        spatial = np.column_stack(
            [m.predict(feats) for m in models]
        ).reshape(H, W, NUM_CLASSES)

        # Transition prediction
        trans_pred = np.zeros((H, W, NUM_CLASSES))
        for y in range(H):
            for x in range(W):
                init_class = GRID_TO_CLASS.get(int(grid[y, x]), 0)
                trans_pred[y, x] = trans[init_class]

        # Blend
        blended = w_spatial * spatial + (1 - w_spatial) * trans_pred
        blended = np.maximum(blended, 0.01)
        blended /= blended.sum(axis=2, keepdims=True)

        predictions.append(blended)
        logger.info("Seed %d: done", seed_idx)

    return predictions


def predict_and_save(round_dir: str | Path, w_spatial: float = W_SPATIAL) -> list[np.ndarray]:
    """Predict and save to predictions/ directory."""
    round_dir = Path(round_dir)
    pred_dir = round_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    predictions = predict_round(round_dir, w_spatial)

    for seed_idx, pred in enumerate(predictions):
        np.save(pred_dir / f"seed_{seed_idx}_spatial_blend.npy", pred)
        logger.info("Saved seed %d prediction", seed_idx)

    return predictions


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    round_dir = sys.argv[1] if len(sys.argv) > 1 else "data/rounds/round_8"
    predictions = predict_and_save(round_dir)
    print(f"\nPredictions saved for {len(predictions)} seeds")
