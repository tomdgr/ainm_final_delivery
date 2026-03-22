"""Improved prediction strategy using Dirichlet-Multinomial with digamma estimator."""
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import digamma

logger = logging.getLogger(__name__)

NUM_CLASSES = 6

# Grid terrain codes (from initial_states) -> prediction class mapping
# Initial grid codes: 1=Settlement, 2=Port, 4=Forest, 5=Mountain, 10=Ocean, 11=Plains
# Prediction classes: 0=Empty, 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain
GRID_TO_CLASS = {
    10: 0,   # Ocean -> Empty
    11: 0,   # Plains -> Empty
    0: 0,    # Generic empty -> Empty
    1: 1,    # Settlement
    2: 2,    # Port
    3: 3,    # Ruin
    4: 4,    # Forest
    5: 5,    # Mountain
}

OCEAN_CODE = 10
PLAINS_CODE = 11


def flood_fill_ocean(grid: np.ndarray) -> np.ndarray:
    """Identify ocean cells via flood-fill from borders.

    Returns boolean mask where True = ocean.
    """
    h, w = grid.shape
    ocean = np.zeros((h, w), dtype=bool)
    stack = []

    # Start from all border cells that are ocean (code 10)
    for y in range(h):
        for x in range(w):
            if (y == 0 or y == h - 1 or x == 0 or x == w - 1) and grid[y, x] == OCEAN_CODE:
                if not ocean[y, x]:
                    ocean[y, x] = True
                    stack.append((y, x))

    # Flood fill through connected ocean cells
    while stack:
        cy, cx = stack.pop()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and not ocean[ny, nx] and grid[ny, nx] == OCEAN_CODE:
                ocean[ny, nx] = True
                stack.append((ny, nx))

    return ocean


def get_terrain_type(grid: np.ndarray, ocean_mask: np.ndarray,
                     settlements: list[dict], y: int, x: int) -> str:
    """Classify a cell's terrain type for prior selection."""
    terrain = int(grid[y, x])
    if terrain == 5:
        return "mountain"
    if ocean_mask[y, x]:
        return "ocean"
    if terrain == 4:
        return "forest"
    if terrain == OCEAN_CODE:
        return "ocean"  # non-flood-filled ocean (isolated)

    # Check if it's a settlement position
    for s in settlements:
        if s["x"] == x and s["y"] == y:
            if s.get("has_port"):
                return "port"
            return "settlement"

    if terrain == 1:
        return "settlement"
    if terrain == 2:
        return "port"
    if terrain == 3:
        return "ruin"
    # Plains (code 11 or 0) or anything else
    return "plains"


LEARNED_PRIORS_FILE = Path("data/learned_priors.json")

# Cache for learned priors (loaded once per process)
_learned_priors_cache: dict | None = None


def _load_learned_priors() -> dict[str, np.ndarray] | None:
    """Load learned priors from file if available."""
    global _learned_priors_cache
    if _learned_priors_cache is not None:
        return _learned_priors_cache
    if LEARNED_PRIORS_FILE.exists():
        with open(LEARNED_PRIORS_FILE) as f:
            raw = json.load(f)
        _learned_priors_cache = {k: np.array(v) for k, v in raw.items()}
        return _learned_priors_cache
    return None


def learn_terrain_priors(prior_strength: float = 5.0) -> dict[str, list[float]]:
    """Learn Dirichlet priors per terrain type from all ground truth data.

    Collects GT class distributions grouped by initial terrain type,
    then fits scaled Dirichlet alphas.
    """
    counts_by_type: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(NUM_CLASSES))
    n_by_type: dict[str, int] = defaultdict(int)

    rounds_dir = Path("data/rounds")
    if not rounds_dir.exists():
        logger.warning("No rounds directory found")
        return {}

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
            if not analysis.get("ground_truth"):
                continue

            gt = np.array(analysis["ground_truth"])
            state = detail["initial_states"][seed_idx]
            grid = np.array(state["grid"])
            settlements = state.get("settlements", [])
            ocean_mask = flood_fill_ocean(grid)
            h, w = grid.shape

            for y in range(h):
                for x in range(w):
                    ttype = get_terrain_type(grid, ocean_mask, settlements, y, x)
                    gt_class = gt[y, x]
                    if isinstance(gt_class, (list, np.ndarray)):
                        # GT is a probability distribution
                        counts_by_type[ttype] += np.array(gt_class)
                    else:
                        # GT is a class index
                        counts_by_type[ttype][int(gt_class)] += 1
                    n_by_type[ttype] += 1

    learned = {}
    for ttype in ["mountain", "ocean", "forest", "settlement", "port", "ruin", "plains"]:
        if n_by_type[ttype] == 0:
            continue
        # Convert counts to proportions, then scale by prior_strength
        proportions = counts_by_type[ttype] / counts_by_type[ttype].sum()
        # Static terrain gets much stronger priors
        strength = 50.0 if ttype in ("mountain", "ocean") else prior_strength
        alphas = proportions * strength
        alphas = np.maximum(alphas, 0.1)  # floor to avoid zero alphas
        learned[ttype] = alphas.tolist()
        logger.info("  %s (n=%d): %s", ttype, n_by_type[ttype],
                     " ".join(f"{a:.2f}" for a in alphas))

    # Save
    LEARNED_PRIORS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LEARNED_PRIORS_FILE, "w") as f:
        json.dump(learned, f, indent=2)
    logger.info("Learned priors saved to %s", LEARNED_PRIORS_FILE)

    # Clear cache so next call picks up new file
    global _learned_priors_cache
    _learned_priors_cache = None

    return learned


# Hand-tuned fallback priors
_FALLBACK_PRIORS = {
    "mountain": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 50.0]),
    "ocean": np.array([50.0, 0.1, 0.1, 0.1, 0.1, 0.1]),
    "forest": np.array([1.0, 0.5, 0.2, 0.5, 8.0, 0.1]),
    "settlement": np.array([1.0, 4.0, 1.5, 3.0, 0.5, 0.1]),
    "port": np.array([0.5, 1.5, 4.0, 2.0, 0.3, 0.1]),
    "ruin": np.array([2.0, 0.5, 0.3, 3.0, 2.5, 0.1]),
    "plains": np.array([4.0, 1.0, 0.5, 1.0, 2.0, 0.1]),
}


def get_dirichlet_prior(terrain_type: str) -> np.ndarray:
    """Get Dirichlet alpha prior based on terrain type.

    Uses learned priors from ground truth if available, otherwise falls back
    to hand-tuned priors.

    Returns alpha vector of shape (6,).
    """
    learned = _load_learned_priors()
    if learned is not None and terrain_type in learned:
        return learned[terrain_type]
    return _FALLBACK_PRIORS.get(terrain_type, _FALLBACK_PRIORS["plains"])


def digamma_estimator(alpha_posterior: np.ndarray) -> np.ndarray:
    """KL-optimal Bayes estimator using digamma function.

    This minimizes expected KL(p || q) under Dirichlet posterior.
    """
    log_q = digamma(alpha_posterior) - digamma(alpha_posterior.sum())
    q = np.exp(log_q)
    q = np.maximum(q, 1e-6)  # safety floor
    q /= q.sum()
    return q


def build_improved_predictions(
    grid: np.ndarray,
    settlements: list[dict],
    observations: list[dict],
) -> np.ndarray:
    """Build predictions using Dirichlet-Multinomial with digamma estimator.

    Args:
        grid: Initial terrain grid (H, W)
        settlements: List of settlement dicts from initial_states
        observations: List of simulation results from API queries

    Returns:
        (H, W, 6) probability tensor
    """
    h, w = grid.shape
    ocean_mask = flood_fill_ocean(grid)

    # Initialize Dirichlet alphas per cell
    alphas = np.zeros((h, w, NUM_CLASSES))
    terrain_types = np.empty((h, w), dtype=object)

    for y in range(h):
        for x in range(w):
            tt = get_terrain_type(grid, ocean_mask, settlements, y, x)
            terrain_types[y, x] = tt
            alphas[y, x] = get_dirichlet_prior(tt)

    # Accumulate observation counts
    obs_counts = np.zeros((h, w), dtype=int)
    for obs in observations:
        viewport = obs["viewport"]
        vx, vy = viewport["x"], viewport["y"]
        vw, vh = viewport["w"], viewport["h"]
        obs_grid = np.array(obs["grid"])

        for dy in range(vh):
            for dx in range(vw):
                y, x = vy + dy, vx + dx
                if y < h and x < w:
                    raw_code = int(obs_grid[dy][dx])
                    # Map grid code to prediction class
                    pred_class = GRID_TO_CLASS.get(raw_code, raw_code)
                    if 0 <= pred_class < NUM_CLASSES:
                        alphas[y, x, pred_class] += 1.0
                        obs_counts[y, x] += 1

    # Empirical Bayes: update priors for unobserved dynamic cells
    # using observed transition statistics per terrain type
    alphas = _empirical_bayes_update(alphas, obs_counts, terrain_types, grid, ocean_mask)

    # Compute predictions using digamma estimator
    pred = np.zeros((h, w, NUM_CLASSES))
    for y in range(h):
        for x in range(w):
            pred[y, x] = digamma_estimator(alphas[y, x])

    # Spatial smoothing for dynamic cells
    pred = _spatial_smooth(pred, terrain_types, sigma=1.0)

    # Final floor and normalize
    pred = np.maximum(pred, 0.01)
    pred /= pred.sum(axis=2, keepdims=True)

    logger.info("Ocean cells: %d, Mountain cells: %d, Dynamic cells: %d",
                ocean_mask.sum(),
                (grid == 5).sum(),
                h * w - ocean_mask.sum() - (grid == 5).sum())
    logger.info("Observed cells: %d/%d", (obs_counts > 0).sum(), h * w)

    return pred


def _empirical_bayes_update(
    alphas: np.ndarray,
    obs_counts: np.ndarray,
    terrain_types: np.ndarray,
    grid: np.ndarray,
    ocean_mask: np.ndarray,
) -> np.ndarray:
    """Update priors using empirical Bayes from observed cells."""
    h, w = alphas.shape[:2]

    # Collect observed outcome distributions per terrain type
    terrain_obs = defaultdict(lambda: np.zeros(NUM_CLASSES))
    terrain_n = defaultdict(int)

    for y in range(h):
        for x in range(w):
            if obs_counts[y, x] > 0:
                tt = terrain_types[y, x]
                if tt not in ("mountain", "ocean"):  # skip static
                    # Extract the observation counts (alpha - prior)
                    prior = get_dirichlet_prior(tt)
                    counts = alphas[y, x] - prior
                    terrain_obs[tt] += counts
                    terrain_n[tt] += obs_counts[y, x]

    # For each terrain type with observations, compute empirical distribution
    # and use it to update priors for ALL cells of that type
    for tt in terrain_obs:
        if terrain_n[tt] < 5:  # need minimum observations
            continue
        total = terrain_obs[tt].sum()
        if total <= 0:
            continue
        empirical_dist = terrain_obs[tt] / total

        # Create empirical Bayes prior: blend with original prior
        original_prior = get_dirichlet_prior(tt)
        original_strength = original_prior.sum()

        # Empirical prior with same strength as original
        eb_prior = empirical_dist * original_strength
        eb_prior = np.maximum(eb_prior, 0.1)  # minimum alpha

        # Update unobserved cells of this terrain type
        for y in range(h):
            for x in range(w):
                if terrain_types[y, x] == tt and obs_counts[y, x] == 0:
                    alphas[y, x] = eb_prior

        logger.info("Empirical Bayes for '%s': n=%d, dist=%s",
                     tt, terrain_n[tt],
                     np.array2string(empirical_dist, precision=3))

    return alphas


def _spatial_smooth(pred: np.ndarray, terrain_types: np.ndarray,
                    sigma: float = 1.0) -> np.ndarray:
    """Apply spatial smoothing to dynamic cells only."""
    h, w, k = pred.shape
    smoothed = pred.copy()

    # Simple 3x3 averaging for dynamic cells
    dynamic_types = {"settlement", "port", "ruin", "plains", "forest"}

    for y in range(h):
        for x in range(w):
            if terrain_types[y, x] not in dynamic_types:
                continue

            neighbors = []
            weights = []
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if terrain_types[ny, nx] in dynamic_types:
                            w_val = 1.0 if (dy == 0 and dx == 0) else 0.3 * sigma
                            neighbors.append(pred[ny, nx])
                            weights.append(w_val)

            if neighbors:
                weights = np.array(weights)
                weights /= weights.sum()
                smoothed[y, x] = sum(w * n for w, n in zip(weights, neighbors))

    return smoothed
