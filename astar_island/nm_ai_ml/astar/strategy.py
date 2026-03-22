"""Query strategy and prediction engine for Astar Island."""
import logging

import numpy as np

logger = logging.getLogger(__name__)

NUM_CLASSES = 6
# Terrain codes from initial_states -> prediction classes
# Class 0: Empty (ocean, plains)
# Class 1: Settlement
# Class 2: Port
# Class 3: Ruin
# Class 4: Forest
# Class 5: Mountain


def parse_initial_state(state: dict, height: int, width: int) -> np.ndarray:
    """Parse initial_states grid into a terrain map.

    Returns array of shape (H, W) with terrain codes.
    """
    grid = np.array(state["grid"])
    if grid.shape != (height, width):
        logger.warning("Grid shape %s != expected (%d, %d)", grid.shape, height, width)
    return grid


def build_static_prior(initial_grid: np.ndarray, settlements: list[dict]) -> np.ndarray:
    """Build prior probability tensor from initial terrain.

    Returns (H, W, 6) probability tensor.
    Mountains stay mountains. Ocean stays empty. Other cells get informed priors.
    """
    h, w = initial_grid.shape
    pred = np.full((h, w, NUM_CLASSES), 0.01)  # minimum floor

    for y in range(h):
        for x in range(w):
            terrain = int(initial_grid[y, x])
            if terrain == 5:  # Mountain - static
                pred[y, x] = [0.01, 0.01, 0.01, 0.01, 0.01, 0.95]
            elif terrain == 0:  # Check if ocean (border/fjord) vs plains
                # Ocean cells are typically on borders or fjords - stay empty
                if _is_likely_ocean(initial_grid, x, y):
                    pred[y, x] = [0.95, 0.01, 0.01, 0.01, 0.01, 0.01]
                else:
                    # Plains - could become anything dynamic
                    pred[y, x] = [0.50, 0.10, 0.05, 0.10, 0.20, 0.01]
            elif terrain == 4:  # Forest
                pred[y, x] = [0.10, 0.05, 0.01, 0.05, 0.75, 0.01]
            elif terrain == 1:  # Settlement
                pred[y, x] = [0.10, 0.40, 0.15, 0.25, 0.05, 0.01]
            elif terrain == 2:  # Port
                pred[y, x] = [0.10, 0.15, 0.40, 0.25, 0.05, 0.01]
            elif terrain == 3:  # Ruin
                pred[y, x] = [0.20, 0.10, 0.05, 0.30, 0.30, 0.01]

    # Mark settlement positions from initial data
    for s in settlements:
        sx, sy = s["x"], s["y"]
        if s.get("has_port"):
            pred[sy, sx] = [0.05, 0.15, 0.45, 0.25, 0.05, 0.01]
        elif s.get("alive", True):
            pred[sy, sx] = [0.05, 0.45, 0.15, 0.25, 0.05, 0.01]

    # Normalize
    pred = _normalize(pred)
    return pred


def _is_likely_ocean(grid: np.ndarray, x: int, y: int) -> bool:
    """Heuristic: ocean cells are at borders or connected to border empty cells."""
    h, w = grid.shape
    # Border cells that are empty are ocean
    if x == 0 or x == w - 1 or y == 0 or y == h - 1:
        return True
    # Adjacent to border empty cell
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            if (nx == 0 or nx == w - 1 or ny == 0 or ny == h - 1) and grid[ny, nx] == 0:
                return True
    return False


def update_predictions_from_observation(
    pred: np.ndarray,
    observation: dict,
    observation_count: np.ndarray,
) -> np.ndarray:
    """Update prediction tensor with observed simulation result.

    Uses incremental averaging: new_pred = (old_pred * n + obs) / (n + 1)
    """
    viewport = observation["viewport"]
    vx, vy = viewport["x"], viewport["y"]
    vw, vh = viewport["w"], viewport["h"]
    grid = np.array(observation["grid"])

    for dy in range(vh):
        for dx in range(vw):
            y, x = vy + dy, vx + dx
            if y >= pred.shape[0] or x >= pred.shape[1]:
                continue

            terrain_code = int(grid[dy][dx])
            n = observation_count[y, x]

            # One-hot for this observation (with floor)
            obs = np.full(NUM_CLASSES, 0.01)
            if 0 <= terrain_code < NUM_CLASSES:
                obs[terrain_code] = 0.95
            obs /= obs.sum()

            if n == 0:
                # First observation replaces prior with strong signal
                pred[y, x] = obs
            else:
                # Incremental average
                pred[y, x] = (pred[y, x] * n + obs) / (n + 1)

            observation_count[y, x] += 1

    # Re-normalize
    pred = _normalize(pred)
    return pred


def plan_queries(
    initial_grid: np.ndarray,
    settlements: list[dict],
    budget: int,
    num_seeds: int,
    map_width: int = 40,
    map_height: int = 40,
) -> list[list[dict]]:
    """Plan viewport queries to maximize information.

    Strategy: Focus queries on dynamic regions (near settlements).
    Repeat queries on same viewports to build empirical distributions.

    Returns list of query plans per seed, each a list of {viewport_x, viewport_y, viewport_w, viewport_h}.
    """
    # Find dynamic regions (areas with settlements)
    settlement_positions = [(s["x"], s["y"]) for s in settlements]

    if not settlement_positions:
        # Fallback: tile the map
        return _tile_queries(budget, num_seeds, map_width, map_height)

    # Cluster settlements into viewport regions
    viewports = _cluster_into_viewports(settlement_positions, map_width, map_height)

    # Allocate budget across seeds
    queries_per_seed = budget // num_seeds
    remainder = budget % num_seeds

    plans = []
    for seed_idx in range(num_seeds):
        seed_budget = queries_per_seed + (1 if seed_idx < remainder else 0)
        seed_plan = []

        # Distribute queries across viewports, repeating for empirical distribution
        for i in range(seed_budget):
            vp = viewports[i % len(viewports)]
            seed_plan.append(vp)

        plans.append(seed_plan)

    return plans


def _cluster_into_viewports(
    positions: list[tuple[int, int]],
    map_w: int, map_h: int,
    vp_size: int = 15,
) -> list[dict]:
    """Group settlement positions into 15x15 viewports that cover them."""
    if not positions:
        return [{"viewport_x": 0, "viewport_y": 0, "viewport_w": vp_size, "viewport_h": vp_size}]

    covered = set()
    viewports = []

    # Greedy: pick viewport covering most uncovered settlements
    remaining = list(positions)
    while remaining:
        best_vp = None
        best_count = 0

        # Try centering viewport on each uncovered settlement
        for sx, sy in remaining:
            vx = max(0, min(sx - vp_size // 2, map_w - vp_size))
            vy = max(0, min(sy - vp_size // 2, map_h - vp_size))

            count = sum(
                1 for px, py in remaining
                if vx <= px < vx + vp_size and vy <= py < vy + vp_size
                and (px, py) not in covered
            )
            if count > best_count:
                best_count = count
                best_vp = {"viewport_x": vx, "viewport_y": vy,
                           "viewport_w": vp_size, "viewport_h": vp_size}

        if best_vp is None:
            break

        viewports.append(best_vp)
        # Mark covered
        vx, vy = best_vp["viewport_x"], best_vp["viewport_y"]
        for px, py in remaining:
            if vx <= px < vx + vp_size and vy <= py < vy + vp_size:
                covered.add((px, py))
        remaining = [(px, py) for px, py in remaining if (px, py) not in covered]

    # Also add viewports for uncovered map areas (non-settlement dynamic regions)
    # Tile remaining uncovered areas
    covered_cells = set()
    for vp in viewports:
        for dy in range(vp["viewport_h"]):
            for dx in range(vp["viewport_w"]):
                covered_cells.add((vp["viewport_x"] + dx, vp["viewport_y"] + dy))

    # Add viewports for large uncovered land areas
    for tile_y in range(0, map_h, vp_size):
        for tile_x in range(0, map_w, vp_size):
            vy = min(tile_y, map_h - vp_size)
            vx = min(tile_x, map_w - vp_size)
            vp = {"viewport_x": vx, "viewport_y": vy,
                   "viewport_w": vp_size, "viewport_h": vp_size}
            if vp not in viewports:
                viewports.append(vp)

    return viewports


def _tile_queries(budget: int, num_seeds: int, map_w: int, map_h: int) -> list[list[dict]]:
    """Fallback: tile the map with 15x15 viewports."""
    vp_size = 15
    viewports = []
    for y in range(0, map_h, vp_size):
        for x in range(0, map_w, vp_size):
            vy = min(y, map_h - vp_size)
            vx = min(x, map_w - vp_size)
            viewports.append({"viewport_x": vx, "viewport_y": vy,
                              "viewport_w": vp_size, "viewport_h": vp_size})

    per_seed = budget // num_seeds
    plans = []
    for _ in range(num_seeds):
        plans.append(viewports[:per_seed])
    return plans


def finalize_prediction(pred: np.ndarray) -> list:
    """Apply floor, normalize, and convert to list for submission."""
    pred = np.maximum(pred, 0.01)
    pred = pred / pred.sum(axis=2, keepdims=True)
    return pred.tolist()


def _normalize(pred: np.ndarray) -> np.ndarray:
    """Normalize prediction tensor so each cell sums to 1."""
    pred = np.maximum(pred, 0.01)
    pred = pred / pred.sum(axis=2, keepdims=True)
    return pred
