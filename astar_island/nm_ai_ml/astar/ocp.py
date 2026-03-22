"""Observation-Conditioned Prediction (OCP) for Astar Island.

Blends RF predictions with spatial interpolation from direct observations.
Works best with 10+ observations per cell (e.g., diagnostic viewport strategy
with repeated viewports).

With 1-4 obs per cell: hurts score (observations are too noisy).
With 10+ obs per cell: should improve score (empirical distributions are reliable).

Usage:
    from nm_ai_ml.astar.ocp import build_observation_map, observation_conditioned_prediction
    obs_class, obs_mask, obs_probs, obs_total = build_observation_map(observations)
    adjusted = observation_conditioned_prediction(rf_pred, obs_mask, obs_probs, obs_total)
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

GRID_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}

# Tunable constants
INFLUENCE_RADIUS = 8.0
OBS_WEIGHT = 0.35
MIN_PROB = 0.005
MIN_OBS_FOR_EMPIRICAL = 4  # need this many obs before trusting empirical


def build_observation_map(observations: list[dict], grid_size: int = 40):
    """Build per-cell observation maps from API observations.

    Returns:
        obs_class: (H, W) argmax class, -1 where unobserved
        obs_mask: (H, W) bool, True where observed
        obs_probs: (H, W, 6) empirical class probabilities
        obs_total: (H, W) observation count per cell
    """
    obs_counts = np.zeros((grid_size, grid_size, 6), dtype=np.float64)
    obs_total = np.zeros((grid_size, grid_size), dtype=np.float64)

    for o in observations:
        vp = o["viewport"]
        for dy in range(len(o["grid"])):
            for dx in range(len(o["grid"][0])):
                gy, gx = vp["y"] + dy, vp["x"] + dx
                if gy >= grid_size or gx >= grid_size:
                    continue
                cls = GRID_TO_CLASS.get(o["grid"][dy][dx], 0)
                obs_counts[gy, gx, cls] += 1
                obs_total[gy, gx] += 1

    obs_mask = obs_total > 0
    obs_probs = np.zeros((grid_size, grid_size, 6))
    obs_probs[obs_mask] = obs_counts[obs_mask] / obs_total[obs_mask, np.newaxis]

    obs_class = np.argmax(obs_counts, axis=2)
    obs_class[~obs_mask] = -1

    return obs_class, obs_mask, obs_probs, obs_total


def observation_conditioned_prediction(
    rf_pred: np.ndarray,
    obs_mask: np.ndarray,
    obs_probs: np.ndarray,
    obs_total: np.ndarray,
    influence_radius: float = INFLUENCE_RADIUS,
    obs_weight: float = OBS_WEIGHT,
    min_prob: float = MIN_PROB,
    min_obs: int = MIN_OBS_FOR_EMPIRICAL,
) -> np.ndarray:
    """Blend RF predictions with observation-based spatial interpolation.

    For observed cells with enough observations: use empirical distribution.
    For nearby unobserved cells: blend RF with smoothed observation field.
    For far unobserved cells: pure RF.

    Args:
        rf_pred: (H, W, 6) RF prediction.
        obs_mask: (H, W) bool observation mask.
        obs_probs: (H, W, 6) empirical probabilities from observations.
        obs_total: (H, W) observation count per cell.
        influence_radius: How far observations influence nearby cells.
        obs_weight: Max weight for observation-based prediction.
        min_prob: Probability floor.
        min_obs: Minimum observations per cell to trust empirical.

    Returns:
        (H, W, 6) adjusted prediction.
    """
    H, W = rf_pred.shape[:2]
    pred = rf_pred.copy().astype(np.float64)

    # Distance from each cell to nearest observed cell
    dist_to_obs = distance_transform_edt(~obs_mask)

    # Build smoothed observation field per class
    obs_soft = np.zeros((H, W, 6))
    for cls in range(6):
        cls_map = obs_probs[:, :, cls] * obs_mask
        count_map = obs_mask.astype(float)
        smooth_cls = gaussian_filter(cls_map, sigma=influence_radius / 2)
        smooth_count = gaussian_filter(count_map, sigma=influence_radius / 2)
        valid = smooth_count > 0.01
        obs_soft[valid, cls] = smooth_cls[valid] / smooth_count[valid]
        obs_soft[~valid, cls] = rf_pred[~valid, cls]

    # Blend weight for unobserved cells
    blend_w = np.clip(1 - dist_to_obs / influence_radius, 0, 1) * obs_weight
    blend_w = blend_w[..., np.newaxis]

    # For observed cells: weight empirical by observation count
    # More observations → trust empirical more
    enough_obs = obs_total >= min_obs
    emp_weight = np.clip((obs_total - min_obs + 1) / (obs_total + 2.0), 0, 0.7)
    emp_weight = emp_weight[..., np.newaxis]

    # Apply
    # Cells with enough observations: empirical + RF blend
    pred[enough_obs] = (emp_weight[enough_obs] * obs_probs[enough_obs]
                        + (1 - emp_weight[enough_obs]) * rf_pred[enough_obs])

    # Unobserved cells: blend RF with smoothed observation field
    unobs = ~obs_mask
    pred[unobs] = (blend_w[unobs] * obs_soft[unobs]
                   + (1 - blend_w[unobs]) * rf_pred[unobs])

    pred = np.maximum(pred, min_prob)
    pred /= pred.sum(axis=2, keepdims=True)

    return pred
