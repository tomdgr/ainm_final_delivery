"""Regime detection and RF calibration for Astar Island.

Detects the round's "regime" from observations (expansion, conflict,
collapse, stable) and applies per-regime adjustments to RF predictions.

Usage:
    from nm_ai_ml.astar.regime_detector import detect_regime, regime_adjusted_prediction
    regime = detect_regime(observations, grid)
    adjusted = regime_adjusted_prediction(rf_pred, regime, grid)
"""
import logging

import numpy as np
from scipy.ndimage import distance_transform_edt

logger = logging.getLogger(__name__)

GRID_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}

REGIMES = ["high_expansion", "high_conflict", "stable", "collapse"]


def detect_regime(observations: list[dict], grid: np.ndarray) -> str:
    """Classify the round into a regime based on observed rates.

    Args:
        observations: List of observation dicts from API.
        grid: Initial terrain grid.

    Returns:
        One of: "high_expansion", "high_conflict", "collapse", "stable"
    """
    all_classes = []
    for o in observations:
        for row in o["grid"]:
            for c in row:
                all_classes.append(GRID_TO_CLASS.get(c, 0))

    all_classes = np.array(all_classes)
    total = len(all_classes)

    settlement_rate = np.mean(all_classes == 1)
    port_rate = np.mean(all_classes == 2)
    ruin_rate = np.mean(all_classes == 3)

    logger.info("Regime detection: sett=%.3f port=%.3f ruin=%.3f",
                settlement_rate, port_rate, ruin_rate)

    if settlement_rate > 0.15:
        regime = "high_expansion"
    elif ruin_rate > 0.03 and settlement_rate < 0.05:
        regime = "high_conflict"
    elif settlement_rate < 0.03 and ruin_rate < 0.01:
        regime = "collapse"
    else:
        regime = "stable"

    logger.info("Detected regime: %s", regime)
    return regime


def distance_map_to_settlements(initial_map: np.ndarray) -> np.ndarray:
    """Compute distance from each cell to nearest initial settlement."""
    settlement_mask = (initial_map == 1) | (initial_map == 2)
    if not settlement_mask.any():
        return np.full(initial_map.shape, 99.0)
    return distance_transform_edt(~settlement_mask)


def regime_adjusted_prediction(
    rf_pred: np.ndarray,
    regime: str,
    initial_map: np.ndarray,
    min_prob: float = 0.005,
) -> np.ndarray:
    """Apply regime-specific adjustments to RF predictions.

    Args:
        rf_pred: (H, W, 6) RF prediction array.
        regime: Detected regime string.
        initial_map: Initial terrain grid (raw codes).
        min_prob: Probability floor.

    Returns:
        (H, W, 6) adjusted prediction array.
    """
    pred = rf_pred.copy().astype(np.float64)
    dist = distance_map_to_settlements(initial_map)

    if regime == "high_expansion":
        # More settlements near existing ones, less empty
        near = dist < 7
        pred[near, 1] *= 1.5   # settlement up
        pred[near, 2] *= 1.3   # port up
        pred[near, 0] *= 0.6   # empty down
        far = dist > 12
        pred[far, 0] *= 1.3    # far stays empty
        pred[far, 1] *= 0.7    # far less settlement

    elif regime == "high_conflict":
        # More ruins near settlements, fewer settlements
        near = dist < 8
        pred[near, 3] *= 2.2   # ruin up
        pred[near, 1] *= 0.55  # settlement down
        pred[near, 2] *= 0.7   # port down

    elif regime == "collapse":
        # Settlements die, forest reclaims
        pred[:, :, 3] *= 1.8   # ruin up everywhere
        pred[:, :, 1] *= 0.4   # settlement way down
        pred[:, :, 4] *= 1.3   # forest up

    elif regime == "stable":
        pass  # no adjustment

    # Preserve static terrain predictions
    mountain_mask = initial_map == 5
    ocean_mask = initial_map == 10
    pred[mountain_mask] = rf_pred[mountain_mask]
    pred[ocean_mask] = rf_pred[ocean_mask]

    # Renormalize
    pred = np.maximum(pred, min_prob)
    pred /= pred.sum(axis=2, keepdims=True)

    return pred
