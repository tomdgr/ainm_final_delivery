"""Main Astar Island competition runner.

Saves ALL raw data to data/rounds/<round_number>/ so nothing is ever lost.

Usage:
    uv run python main.py --fetch       # Fetch round data + observations only (run first!)
    uv run python main.py               # Build predictions (no submit)
    uv run python main.py --submit      # Build predictions and submit
    uv run python main.py --analysis    # Fetch ground truth for completed rounds
"""
import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from nm_ai_ml.astar.client import AstarClient
from nm_ai_ml.astar.improved_strategy import (
    GRID_TO_CLASS,
    build_improved_predictions,
    flood_fill_ocean,
)
from nm_ai_ml.astar.calibrate import calibrate, load_params, save_params
from nm_ai_ml.astar.simulator import monte_carlo_predict


class DiagType(str, Enum):
    ISOLATED = "isolated"    # nearest neighbor > 8
    COASTAL = "coastal"      # within 2 cells of ocean
    PAIRED = "paired"        # exactly 1 neighbor within 5
    CLUSTERED = "clustered"  # 3+ neighbors within 5
    FORESTED = "forested"    # forest count in radius 3 > 5

ENTROPY_SIM_RUNS = 200  # Quick sim just for viewport planning

# Viewport planning strategy: "smart" (greedy set-cover on dynamic cells)
#                              "entropy" (sim-based entropy maximization)
VIEWPORT_STRATEGY = "entropy"

# Seed budget strategy: "single_seed" (all queries to seed 0)
#                        "multi_seed" (allocate proportional to per-seed entropy)
#                        "diagnostic" (classify settlements, target pure archetypes with repeats)
SEED_STRATEGY = "multi_seed"

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

NUM_CLASSES = 6
NUM_SIM_RUNS = 2000


def _get_round_dirs(round_num) -> tuple[Path, Path, Path, Path]:
    """Create and return standard directory paths for a round."""
    save_dir = Path(f"data/rounds/round_{round_num}")
    save_dir.mkdir(parents=True, exist_ok=True)
    obs_dir = save_dir / "observations"
    obs_dir.mkdir(exist_ok=True)
    pred_dir = save_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    init_dir = save_dir / "initial_states"
    init_dir.mkdir(exist_ok=True)
    return save_dir, obs_dir, pred_dir, init_dir


def _save_observation(obs_dir: Path, seed_idx: int, obs: dict) -> None:
    """Append a single observation to the seed's observation file immediately."""
    obs_file = obs_dir / f"seed_{seed_idx}.json"
    existing = []
    if obs_file.exists():
        with open(obs_file) as f:
            existing = json.load(f)
    existing.append(obs)
    with open(obs_file, "w") as f:
        json.dump(existing, f)


def _load_observations(obs_dir: Path, seed_idx: int) -> list[dict]:
    """Load saved observations for a seed."""
    obs_file = obs_dir / f"seed_{seed_idx}.json"
    if obs_file.exists():
        with open(obs_file) as f:
            return json.load(f)
    return []


def _plan_smart_viewports(initial_state: dict, budget: int, map_width: int = 40, map_height: int = 40) -> list[tuple[int, int]]:
    """Compute optimal viewport positions targeting dynamic cells.

    Uses greedy set-cover to find viewports that cover all settlements/ports/ruins,
    then fills remaining budget by repeating the densest viewports.

    Args:
        initial_state: Seed initial state with 'grid' and 'settlements'.
        budget: Number of queries available.
        map_width: Map width (default 40).
        map_height: Map height (default 40).

    Returns:
        Ordered list of (vx, vy) viewport positions.
    """
    grid = initial_state["grid"]
    dynamic_codes = {1, 2, 3}  # Settlement, Port, Ruin

    # Find all dynamic cells, excluding outermost 1-cell border (always ocean)
    dynamic_cells = set()
    for y in range(1, map_height - 1):
        for x in range(1, map_width - 1):
            if grid[y][x] in dynamic_codes:
                dynamic_cells.add((x, y))

    if not dynamic_cells:
        # Fallback: standard 3x3 tiling + repeat center
        tile_positions = [(vx, vy) for vy in [0, 13, 25] for vx in [0, 13, 25]]
        viewports = list(tile_positions[:min(budget, 9)])
        while len(viewports) < budget:
            viewports.append(tile_positions[4])  # center tile
        return viewports

    logger.info("Smart viewport planning: %d dynamic cells to cover", len(dynamic_cells))

    # Valid viewport origins: skip 1-cell border, viewport is 15x15
    max_vx = map_width - 15   # 25 for 40-wide map
    max_vy = map_height - 15  # 25 for 40-tall map
    min_v = 1  # skip outermost border

    # Greedy set-cover: pick viewport covering most uncovered dynamic cells
    uncovered = set(dynamic_cells)
    coverage_viewports = []  # (vx, vy, count_covered)

    while uncovered:
        best_pos = None
        best_count = 0

        for vy in range(min_v, max_vy + 1):
            for vx in range(min_v, max_vx + 1):
                count = sum(
                    1 for (cx, cy) in uncovered
                    if vx <= cx < vx + 15 and vy <= cy < vy + 15
                )
                if count > best_count:
                    best_count = count
                    best_pos = (vx, vy)

        if best_pos is None or best_count == 0:
            break

        # Remove covered cells
        vx, vy = best_pos
        newly_covered = {
            (cx, cy) for (cx, cy) in uncovered
            if vx <= cx < vx + 15 and vy <= cy < vy + 15
        }
        uncovered -= newly_covered
        coverage_viewports.append((vx, vy, best_count))
        logger.info("  Viewport (%d,%d): covers %d dynamic cells, %d remaining",
                     vx, vy, best_count, len(uncovered))

    # Build viewport list: coverage viewports first
    viewports = [(vx, vy) for vx, vy, _ in coverage_viewports]

    if len(viewports) >= budget:
        return viewports[:budget]

    # Fill remaining budget: round-robin repeat by density (most dynamic cells first)
    coverage_viewports.sort(key=lambda t: t[2], reverse=True)
    ranked = [(vx, vy) for vx, vy, _ in coverage_viewports]
    remaining = budget - len(viewports)
    for i in range(remaining):
        viewports.append(ranked[i % len(ranked)])

    logger.info("Smart viewport plan: %d coverage + %d repeats = %d total",
                 len(coverage_viewports), remaining, len(viewports))
    return viewports


def _compute_cell_entropy(predictions: np.ndarray) -> np.ndarray:
    """Compute per-cell entropy from an (H, W, C) probability tensor.

    Returns (H, W) array of entropy values. Higher = more uncertain.
    """
    # Clip to avoid log(0)
    p = np.clip(predictions, 1e-10, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def _plan_entropy_viewports(
    entropy_map: np.ndarray,
    budget: int,
    map_width: int = 40,
    map_height: int = 40,
) -> list[tuple[int, int]]:
    """Pick viewport positions that maximize total entropy covered.

    Uses a greedy approach: each step picks the 15x15 viewport with the
    highest sum of remaining entropy, then zeros out covered cells to
    avoid double-counting.

    Args:
        entropy_map: (H, W) array of per-cell entropy from sim predictions.
        budget: Number of viewports to select.
        map_width: Map width.
        map_height: Map height.

    Returns:
        Ordered list of (vx, vy) viewport positions.
    """
    remaining = entropy_map.copy()
    max_vx = map_width - 15
    max_vy = map_height - 15
    viewports = []

    for qi in range(budget):
        best_pos = None
        best_entropy = -1.0

        for vy in range(max_vy + 1):
            for vx in range(max_vx + 1):
                total = remaining[vy:vy + 15, vx:vx + 15].sum()
                if total > best_entropy:
                    best_entropy = total
                    best_pos = (vx, vy)

        if best_pos is None or best_entropy <= 0:
            break

        vx, vy = best_pos
        viewports.append(best_pos)
        logger.info("  Viewport %d: (%d,%d) entropy=%.2f", qi + 1, vx, vy, best_entropy)

        # Zero out covered cells so next viewport targets fresh area
        # Use decay instead of full zero — allows repeat queries on very high-entropy areas
        remaining[vy:vy + 15, vx:vx + 15] *= 0.3

    return viewports


def _allocate_budget_by_entropy(
    initial_states: list[dict],
    total_budget: int,
    sim_runs: int = 200,
    min_per_seed: int = 5,
    params: dict | None = None,
) -> tuple[dict[int, int], dict[int, np.ndarray]]:
    """Allocate observation budget across seeds proportional to per-seed entropy.

    Runs a quick sim for each seed, computes total entropy, then allocates
    budget proportionally with a minimum floor per seed.

    Returns:
        (budget_dict, sim_predictions_dict) — budget per seed and sim predictions
        for reuse in viewport planning.
    """
    num_seeds = len(initial_states)
    entropies = []
    sim_preds = {}

    logger.info("Computing per-seed entropy (%d sims each) for budget allocation...", sim_runs)
    for seed_idx, state in enumerate(initial_states):
        grid = np.array(state["grid"])
        settlements = state.get("settlements", [])
        pred = monte_carlo_predict(grid, settlements, n_sims=sim_runs, years=50, params=params)
        sim_preds[seed_idx] = pred
        entropy_map = _compute_cell_entropy(pred)
        total_entropy = float(entropy_map.sum())
        entropies.append(total_entropy)
        logger.info("  Seed %d: total entropy=%.1f", seed_idx, total_entropy)

    # Allocate: floor of min_per_seed, then distribute remainder by entropy proportion
    total_entropy = sum(entropies)
    budget_dict = {}

    if total_entropy <= 0:
        # Fallback: equal split
        per_seed = total_budget // num_seeds
        for i in range(num_seeds):
            budget_dict[i] = per_seed
        budget_dict[0] += total_budget - per_seed * num_seeds
    else:
        # Ensure we have enough budget for minimums
        floor_total = min_per_seed * num_seeds
        if floor_total >= total_budget:
            # Not enough for minimums — distribute evenly
            per_seed = total_budget // num_seeds
            for i in range(num_seeds):
                budget_dict[i] = per_seed
            # Give remainder to highest entropy seeds
            remainder = total_budget - per_seed * num_seeds
            ranked = sorted(range(num_seeds), key=lambda i: entropies[i], reverse=True)
            for i in range(remainder):
                budget_dict[ranked[i]] += 1
        else:
            # Give each seed the minimum, then allocate remainder by entropy
            remainder = total_budget - floor_total
            for i in range(num_seeds):
                proportion = entropies[i] / total_entropy
                budget_dict[i] = min_per_seed + int(remainder * proportion)

            # Distribute any rounding leftovers to highest entropy seeds
            allocated = sum(budget_dict.values())
            leftover = total_budget - allocated
            ranked = sorted(range(num_seeds), key=lambda i: entropies[i], reverse=True)
            for i in range(leftover):
                budget_dict[ranked[i % num_seeds]] += 1

    for i in range(num_seeds):
        logger.info("  Seed %d: allocated %d queries (entropy=%.1f)", i, budget_dict[i], entropies[i])

    return budget_dict, sim_preds


def _classify_settlements(
    initial_state: dict, map_width: int = 40, map_height: int = 40,
) -> dict[tuple[int, int], set[DiagType]]:
    """Classify each settlement into diagnostic archetypes based on initial state.

    Returns dict mapping (x, y) -> set of DiagType labels.
    """
    grid = np.array(initial_state["grid"])
    settlements = initial_state.get("settlements", [])

    # Get alive settlement positions
    positions = []
    for s in settlements:
        if s.get("alive", True):
            positions.append((s["x"], s["y"]))

    if not positions:
        return {}

    # Ocean mask for coastal detection
    ocean_mask = flood_fill_ocean(grid)

    # Forest code = 4
    FOREST_CODE = 4

    classifications: dict[tuple[int, int], set[DiagType]] = {}

    for x, y in positions:
        labels: set[DiagType] = set()

        # Compute Chebyshev distances to all other settlements
        neighbors_within_5 = 0
        min_dist = float("inf")
        for ox, oy in positions:
            if (ox, oy) == (x, y):
                continue
            dist = max(abs(ox - x), abs(oy - y))
            min_dist = min(min_dist, dist)
            if dist <= 5:
                neighbors_within_5 += 1

        # ISOLATED: nearest neighbor > 8
        if min_dist > 8:
            labels.add(DiagType.ISOLATED)

        # PAIRED: exactly 1 neighbor within 5
        if neighbors_within_5 == 1:
            labels.add(DiagType.PAIRED)

        # CLUSTERED: 3+ neighbors within 5
        if neighbors_within_5 >= 3:
            labels.add(DiagType.CLUSTERED)

        # COASTAL: within 2 cells of ocean
        is_coastal = False
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny, nx = y + dy, x + dx
                if 0 <= ny < map_height and 0 <= nx < map_width:
                    if ocean_mask[ny, nx]:
                        is_coastal = True
                        break
            if is_coastal:
                break
        if is_coastal:
            labels.add(DiagType.COASTAL)

        # FORESTED: forest count in radius 3 > 5
        forest_count = 0
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                ny, nx = y + dy, x + dx
                if 0 <= ny < map_height and 0 <= nx < map_width:
                    if grid[ny, nx] == FOREST_CODE:
                        forest_count += 1
        if forest_count > 5:
            labels.add(DiagType.FORESTED)

        classifications[(x, y)] = labels

    return classifications


def _score_viewport_for_diagnostic(
    classifications: dict[tuple[int, int], set[DiagType]],
    target_type: DiagType,
    map_width: int = 40,
    map_height: int = 40,
) -> tuple[int, int] | None:
    """Find the best 15x15 viewport for observing a specific diagnostic type.

    Score = count of target_type settlements - 0.5 * count of non-target settlements.
    """
    max_vx = map_width - 15
    max_vy = map_height - 15

    best_pos = None
    best_score = -float("inf")

    for vy in range(max_vy + 1):
        for vx in range(max_vx + 1):
            target_count = 0
            confound_count = 0
            for (sx, sy), labels in classifications.items():
                if vx <= sx < vx + 15 and vy <= sy < vy + 15:
                    if target_type in labels:
                        target_count += 1
                    else:
                        confound_count += 1
            score = target_count - 0.5 * confound_count
            if score > best_score and target_count > 0:
                best_score = score
                best_pos = (vx, vy)

    return best_pos


def _plan_diagnostic(
    initial_states: list[dict],
    total_budget: int,
    map_width: int = 40,
    map_height: int = 40,
) -> dict[int, list[tuple[int, int]]]:
    """Plan diagnostic viewport allocation across seeds.

    Phase 1: Pick best_seed (most distinct diagnostic categories).
    Phase 2: On best_seed, allocate 60% of budget to diagnostic viewports with repeats.
    Phase 3: Remaining 40% across other seeds (equal split, densest area).

    Returns dict mapping seed_index -> list of (vx, vy) viewports (with repeats).
    """
    num_seeds = len(initial_states)

    # Classify settlements on each seed
    all_classifications = {}
    for seed_idx, state in enumerate(initial_states):
        classifications = _classify_settlements(state, map_width, map_height)
        all_classifications[seed_idx] = classifications
        types_present = set()
        for labels in classifications.values():
            types_present.update(labels)
        logger.info("Seed %d: %d settlements, types: %s",
                     seed_idx, len(classifications),
                     ", ".join(sorted(t.value for t in types_present)) if types_present else "none")

    # Phase 1: Pick best_seed = most distinct diagnostic categories (excluding FORESTED)
    ACTIVE_TYPES = [DiagType.ISOLATED, DiagType.PAIRED, DiagType.COASTAL, DiagType.CLUSTERED]
    best_seed = 0
    best_type_count = 0
    for seed_idx, classifications in all_classifications.items():
        types_present = set()
        for labels in classifications.values():
            types_present.update(labels & set(ACTIVE_TYPES))
        if len(types_present) > best_type_count:
            best_type_count = len(types_present)
            best_seed = seed_idx
    logger.info("Diagnostic: best_seed=%d with %d active types", best_seed, best_type_count)

    # Phase 2: Budget split
    phase2_budget = int(total_budget * 0.6)
    phase3_budget = total_budget - phase2_budget

    # Find which active types exist on best_seed
    best_classifications = all_classifications[best_seed]
    available_types = set()
    for labels in best_classifications.values():
        available_types.update(labels & set(ACTIVE_TYPES))
    available_types = sorted(available_types, key=lambda t: ACTIVE_TYPES.index(t))

    # Default repeats per type
    default_repeats = {
        DiagType.ISOLATED: 8,
        DiagType.PAIRED: 8,
        DiagType.COASTAL: 7,
        DiagType.CLUSTERED: 7,
    }

    # Filter to available and redistribute budget
    if not available_types:
        # No diagnostic types found — fall back to entropy-based viewport at center
        plan: dict[int, list[tuple[int, int]]] = {}
        per_seed = total_budget // num_seeds
        for seed_idx in range(num_seeds):
            center_vx = max(0, map_width // 2 - 7)
            center_vy = max(0, map_height // 2 - 7)
            plan[seed_idx] = [(center_vx, center_vy)] * per_seed
        return plan

    # Compute actual repeats: distribute phase2_budget among available types
    total_default = sum(default_repeats[t] for t in available_types)
    type_repeats = {}
    allocated = 0
    for t in available_types:
        repeats = int(phase2_budget * default_repeats[t] / total_default)
        type_repeats[t] = max(1, repeats)
        allocated += type_repeats[t]
    # Distribute remainder
    remainder = phase2_budget - allocated
    for i in range(remainder):
        t = available_types[i % len(available_types)]
        type_repeats[t] += 1

    # Find best viewport for each type on best_seed
    plan = {seed_idx: [] for seed_idx in range(num_seeds)}
    for diag_type in available_types:
        vp = _score_viewport_for_diagnostic(
            best_classifications, diag_type, map_width, map_height,
        )
        if vp is not None:
            repeats = type_repeats[diag_type]
            plan[best_seed].extend([vp] * repeats)
            logger.info("  %s: viewport (%d,%d) x%d repeats",
                         diag_type.value, vp[0], vp[1], repeats)
        else:
            # Redistribute to other types
            logger.info("  %s: no suitable viewport found, skipping", diag_type.value)

    # Phase 3: Remaining seeds get equal budget, 1 viewport each (densest area), repeated
    other_seeds = [i for i in range(num_seeds) if i != best_seed]
    if other_seeds and phase3_budget > 0:
        per_other = phase3_budget // len(other_seeds)
        leftover = phase3_budget - per_other * len(other_seeds)

        for idx, seed_idx in enumerate(other_seeds):
            seed_budget = per_other + (1 if idx < leftover else 0)
            if seed_budget <= 0:
                continue

            # Find densest settlement area
            classifications = all_classifications[seed_idx]
            if classifications:
                # Score viewports by total settlement count
                best_vp = None
                best_count = 0
                max_vx = map_width - 15
                max_vy = map_height - 15
                for vy in range(max_vy + 1):
                    for vx in range(max_vx + 1):
                        count = sum(
                            1 for (sx, sy) in classifications
                            if vx <= sx < vx + 15 and vy <= sy < vy + 15
                        )
                        if count > best_count:
                            best_count = count
                            best_vp = (vx, vy)
                if best_vp:
                    plan[seed_idx] = [best_vp] * seed_budget
                    logger.info("  Seed %d: densest viewport (%d,%d) x%d repeats (%d settlements)",
                                 seed_idx, best_vp[0], best_vp[1], seed_budget, best_count)
            else:
                # No settlements — center viewport
                center_vx = max(0, map_width // 2 - 7)
                center_vy = max(0, map_height // 2 - 7)
                plan[seed_idx] = [(center_vx, center_vy)] * seed_budget

    return plan


def fetch(client: AstarClient | None = None):
    """Fetch round data and observations. Run this FIRST when a round starts.

    Concentrates all observation budget on seed 0 using smart viewport placement
    that targets dynamic cells (settlements, ports, ruins).
    """
    own_client = client is None
    if own_client:
        client = AstarClient()

    try:
        active = client.get_active_round()
        if not active:
            logger.error("No active round found!")
            rounds = client.get_rounds()
            for r in rounds:
                logger.info("  Round %s: %s", r.get("round_number"), r.get("status"))
            return None
        round_id = active["id"]
        round_num = active.get("round_number", "unknown")
        logger.info("Active round: %s (round %s)", round_id, round_num)

        # Get round details
        detail = client.get_round_detail(round_id)
        width = detail["map_width"]
        height = detail["map_height"]
        num_seeds = detail["seeds_count"]
        initial_states = detail["initial_states"]
        logger.info("Map: %dx%d, %d seeds, closes at %s", width, height, num_seeds, active["closes_at"])

        # Create directories
        save_dir, obs_dir, pred_dir, init_dir = _get_round_dirs(round_num)

        # Save round detail
        with open(save_dir / "round_detail.json", "w") as f:
            json.dump(detail, f, indent=2)
        logger.info("Round detail saved")

        # Save initial states per seed
        for seed_idx, state in enumerate(initial_states):
            with open(init_dir / f"seed_{seed_idx}.json", "w") as f:
                json.dump(state, f, indent=2)
        logger.info("Initial states saved for %d seeds", num_seeds)

        # Check budget
        budget_info = client.get_budget()
        budget = budget_info.get("queries_max", 50) - budget_info.get("queries_used", 0)
        logger.info("Query budget remaining: %d / %d", budget, budget_info.get("queries_max", 50))

        if budget <= 0:
            # Load existing observations
            for seed_idx in range(num_seeds):
                obs = _load_observations(obs_dir, seed_idx)
                logger.info("Seed %d: %d saved observations loaded", seed_idx, len(obs))
            logger.info("No budget left — data fetch complete")
            return detail

        sim_params = load_params()

        if SEED_STRATEGY == "multi_seed":
            # Allocate budget across all seeds proportional to entropy
            budget_dict, sim_preds = _allocate_budget_by_entropy(
                initial_states, budget, sim_runs=ENTROPY_SIM_RUNS,
                min_per_seed=5, params=sim_params,
            )

            for seed_idx in range(num_seeds):
                seed_budget = budget_dict[seed_idx]
                if seed_budget <= 0:
                    continue

                existing_obs = _load_observations(obs_dir, seed_idx)
                if len(existing_obs) >= seed_budget:
                    logger.info("Seed %d: already have %d observations (budget=%d), skipping",
                                seed_idx, len(existing_obs), seed_budget)
                    continue

                # Plan viewports — reuse sim predictions from allocation step
                if VIEWPORT_STRATEGY == "entropy":
                    entropy_map = _compute_cell_entropy(sim_preds[seed_idx])
                    viewports = _plan_entropy_viewports(entropy_map, seed_budget, width, height)
                else:
                    viewports = _plan_smart_viewports(initial_states[seed_idx], seed_budget, width, height)

                # Skip viewports we've already queried (resume support)
                start_idx = len(existing_obs)
                for qi, (vx, vy) in enumerate(viewports[start_idx:], start=start_idx):
                    try:
                        result = client.simulate(
                            round_id=round_id,
                            seed_index=seed_idx,
                            viewport_x=vx, viewport_y=vy,
                            viewport_w=15, viewport_h=15,
                        )
                        _save_observation(obs_dir, seed_idx, result)
                        logger.info("Seed %d query %d/%d: viewport (%d,%d) OK — saved",
                                    seed_idx, qi + 1, seed_budget, vx, vy)
                    except Exception as e:
                        logger.error("Seed %d query %d failed: %s", seed_idx, qi + 1, e)
                        break

        elif SEED_STRATEGY == "diagnostic":
            # Classify settlements and target pure diagnostic archetypes with repeats
            diagnostic_plan = _plan_diagnostic(initial_states, budget, width, height)

            for seed_idx in range(num_seeds):
                viewports = diagnostic_plan.get(seed_idx, [])
                if not viewports:
                    continue

                existing_obs = _load_observations(obs_dir, seed_idx)
                if len(existing_obs) >= len(viewports):
                    logger.info("Seed %d: already have %d observations (planned=%d), skipping",
                                seed_idx, len(existing_obs), len(viewports))
                    continue

                # Resume support: skip already-queried viewports
                start_idx = len(existing_obs)
                for qi, (vx, vy) in enumerate(viewports[start_idx:], start=start_idx):
                    try:
                        result = client.simulate(
                            round_id=round_id,
                            seed_index=seed_idx,
                            viewport_x=vx, viewport_y=vy,
                            viewport_w=15, viewport_h=15,
                        )
                        _save_observation(obs_dir, seed_idx, result)
                        logger.info("Seed %d query %d/%d: viewport (%d,%d) OK — saved",
                                    seed_idx, qi + 1, len(viewports), vx, vy)
                    except Exception as e:
                        logger.error("Seed %d query %d failed: %s", seed_idx, qi + 1, e)
                        break

        else:
            # "single_seed" — all queries go to seed 0
            target_seed = 0
            seed_budget = budget

            existing_obs = _load_observations(obs_dir, target_seed)
            if len(existing_obs) >= seed_budget:
                logger.info("Seed %d: already have %d observations (budget=%d), skipping",
                            target_seed, len(existing_obs), seed_budget)
            else:
                if VIEWPORT_STRATEGY == "entropy":
                    state = initial_states[target_seed]
                    grid = np.array(state["grid"])
                    settlements = state.get("settlements", [])
                    logger.info("Running quick sim (%d runs) for entropy-guided viewport planning...",
                                ENTROPY_SIM_RUNS)
                    sim_pred = monte_carlo_predict(grid, settlements, n_sims=ENTROPY_SIM_RUNS,
                                                   years=50, params=sim_params)
                    entropy_map = _compute_cell_entropy(sim_pred)
                    logger.info("Entropy map: min=%.3f, max=%.3f, mean=%.3f",
                                entropy_map.min(), entropy_map.max(), entropy_map.mean())
                    viewports = _plan_entropy_viewports(entropy_map, seed_budget, width, height)
                else:
                    viewports = _plan_smart_viewports(initial_states[target_seed], seed_budget, width, height)

                start_idx = len(existing_obs)
                for qi, (vx, vy) in enumerate(viewports[start_idx:], start=start_idx):
                    try:
                        result = client.simulate(
                            round_id=round_id,
                            seed_index=target_seed,
                            viewport_x=vx, viewport_y=vy,
                            viewport_w=15, viewport_h=15,
                        )
                        _save_observation(obs_dir, target_seed, result)
                        logger.info("Seed %d query %d/%d: viewport (%d,%d) OK — saved",
                                    target_seed, qi + 1, seed_budget, vx, vy)
                    except Exception as e:
                        logger.error("Seed %d query %d failed: %s", target_seed, qi + 1, e)
                        break

        # Save budget info
        budget_after = client.get_budget()
        with open(save_dir / "budget.json", "w") as f:
            json.dump(budget_after, f, indent=2)

        logger.info("Fetch complete! Observations saved to %s", obs_dir)
        return detail

    finally:
        if own_client:
            client.close()


def fetch_targeted(targets: list[tuple[int, int, int, int]]):
    """Fetch observations for specific seed/viewport combos.

    Args:
        targets: list of (seed_index, viewport_x, viewport_y, num_queries) tuples.
                 Each tuple fires num_queries queries at the given viewport for the given seed.
    """
    client = AstarClient()
    try:
        active = client.get_active_round()
        if not active:
            logger.error("No active round found!")
            return
        round_id = active["id"]
        round_num = active.get("round_number", "unknown")
        logger.info("Active round: %s (round %s)", round_id, round_num)

        # Check budget
        budget_info = client.get_budget()
        budget = budget_info.get("queries_max", 50) - budget_info.get("queries_used", 0)
        total_requested = sum(n for _, _, _, n in targets)
        logger.info("Budget remaining: %d, queries requested: %d", budget, total_requested)

        if total_requested > budget:
            logger.error("Not enough budget! Need %d but only %d remaining.", total_requested, budget)
            return

        # Ensure round data is saved
        detail = client.get_round_detail(round_id)
        save_dir, obs_dir, pred_dir, init_dir = _get_round_dirs(round_num)
        with open(save_dir / "round_detail.json", "w") as f:
            json.dump(detail, f, indent=2)
        for seed_idx, state in enumerate(detail["initial_states"]):
            with open(init_dir / f"seed_{seed_idx}.json", "w") as f:
                json.dump(state, f, indent=2)

        # Run targeted queries
        total_done = 0
        for seed_idx, vx, vy, num_queries in targets:
            logger.info("Querying seed %d viewport (%d,%d) x%d ...", seed_idx, vx, vy, num_queries)
            for qi in range(num_queries):
                try:
                    result = client.simulate(
                        round_id=round_id,
                        seed_index=seed_idx,
                        viewport_x=vx, viewport_y=vy,
                        viewport_w=15, viewport_h=15,
                    )
                    _save_observation(obs_dir, seed_idx, result)
                    total_done += 1
                    logger.info("  Query %d/%d OK (total %d/%d)",
                                qi + 1, num_queries, total_done, total_requested)
                except Exception as e:
                    logger.error("  Query %d/%d FAILED: %s", qi + 1, num_queries, e)
                    break

        # Save budget after
        budget_after = client.get_budget()
        with open(save_dir / "budget.json", "w") as f:
            json.dump(budget_after, f, indent=2)

        logger.info("Targeted fetch complete! %d/%d queries executed.", total_done, total_requested)
    finally:
        client.close()


def _load_latest_round_detail() -> dict | None:
    """Load round detail from the most recent round directory."""
    rounds_dir = Path("data/rounds")
    if not rounds_dir.exists():
        return None
    round_dirs = sorted(rounds_dir.glob("round_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for rd in round_dirs:
        detail_file = rd / "round_detail.json"
        if detail_file.exists():
            with open(detail_file) as f:
                return json.load(f)
    return None


def _get_git_commit() -> str:
    """Get current git commit hash, or 'unknown' if not in a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _hash_file(path: Path) -> str:
    """Compute SHA-256 hash of a file, or 'none' if missing."""
    if not path.exists():
        return "none"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _save_submission(save_dir: Path, blend_predictions: list[np.ndarray],
                     num_seeds: int, all_observations: dict,
                     sim_params: dict | None) -> Path:
    """Archive a submission with predictions and metadata."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    sub_dir = save_dir / "submissions" / timestamp
    sub_dir.mkdir(parents=True, exist_ok=True)

    for seed_idx in range(num_seeds):
        np.save(sub_dir / f"seed_{seed_idx}_blend.npy", blend_predictions[seed_idx])

    manifest = {
        "timestamp": timestamp,
        "git_commit": _get_git_commit(),
        "calibrated_params_hash": _hash_file(Path("data/calibrated_params.json")),
        "num_sim_runs": NUM_SIM_RUNS,
        "observations_per_seed": [
            len(all_observations.get(i, [])) for i in range(num_seeds)
        ],
    }
    if sim_params:
        manifest["calibrated_params"] = sim_params

    with open(sub_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Submission archived to %s", sub_dir)
    return sub_dir


def run(submit: bool = False, do_calibrate: bool = False, skip_fetch: bool = False):
    """Build predictions and optionally submit."""
    client = AstarClient()

    if skip_fetch:
        # Load round detail from saved file instead of calling the API
        detail = _load_latest_round_detail()
        if detail is None:
            logger.error("No saved round detail found! Run --fetch first.")
            client.close()
            return
        logger.info("Loaded round detail from saved file (skip_fetch=True)")
    else:
        # Fetch data (will skip queries if budget is 0)
        detail = fetch(client)
        if detail is None:
            client.close()
            return

    round_id = detail["id"]
    round_num = detail.get("round_number", "unknown")
    num_seeds = detail["seeds_count"]
    initial_states = detail["initial_states"]

    save_dir, obs_dir, pred_dir, init_dir = _get_round_dirs(round_num)

    # 2. Calibrate or load calibrated params
    sim_params = None
    if do_calibrate:
        logger.info("Calibrating simulator parameters...")
        sim_params = calibrate(n_sims=100)
        save_params(sim_params)
    else:
        sim_params = load_params()
        if sim_params:
            logger.info("Using calibrated parameters from %s", "data/calibrated_params.json")
        else:
            logger.info("No calibrated parameters found — using defaults")

    # 3. Load observations
    all_observations = {}
    for seed_idx in range(num_seeds):
        all_observations[seed_idx] = _load_observations(obs_dir, seed_idx)
        logger.info("Seed %d: %d observations", seed_idx, len(all_observations[seed_idx]))

    # 4. Build predictions using improved strategy (Dirichlet + observations)
    api_predictions = []
    for seed_idx in range(num_seeds):
        state = initial_states[seed_idx]
        grid = np.array(state["grid"])
        settlements = state.get("settlements", [])
        obs = all_observations[seed_idx]

        pred = build_improved_predictions(grid, settlements, obs)
        pred = np.maximum(pred, 0.01)
        pred /= pred.sum(axis=2, keepdims=True)
        api_predictions.append(pred)
        np.save(pred_dir / f"seed_{seed_idx}_api.npy", pred)

    # 5. Build predictions using local simulator
    logger.info("Running local simulator (%d runs per seed)...", NUM_SIM_RUNS)
    sim_predictions = []
    for seed_idx in range(num_seeds):
        state = initial_states[seed_idx]
        grid = np.array(state["grid"])
        settlements = state.get("settlements", [])

        pred = monte_carlo_predict(grid, settlements, n_sims=NUM_SIM_RUNS, years=50,
                                   params=sim_params)
        pred = np.maximum(pred, 0.01).astype(np.float64)
        pred /= pred.sum(axis=2, keepdims=True)
        sim_predictions.append(pred)
        np.save(pred_dir / f"seed_{seed_idx}_sim.npy", pred)
        logger.info("Seed %d: local sim done", seed_idx)

    # 6. Blend API + sim predictions with per-cell adaptive weighting
    blend_predictions = []
    for seed_idx in range(num_seeds):
        obs = all_observations[seed_idx]
        if not obs:
            blended = sim_predictions[seed_idx].copy()
        else:
            # Build per-cell observation count map
            state = initial_states[seed_idx]
            grid = np.array(state["grid"])
            h, w = grid.shape
            obs_count = np.zeros((h, w), dtype=np.float64)
            for ob in obs:
                vp = ob["viewport"]
                vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
                for dy in range(vh):
                    for dx in range(vw):
                        y, x = vy + dy, vx + dx
                        if y < h and x < w:
                            obs_count[y, x] += 1

            # Adaptive weight: more observations -> trust API more
            # w_api = obs_count / (obs_count + k), k controls transition speed
            k = 3.0
            w_api = (obs_count / (obs_count + k))[:, :, np.newaxis]
            w_sim = 1.0 - w_api

            # Even observed cells benefit from sim (structural knowledge)
            # so cap API weight at 0.5
            w_api = np.minimum(w_api, 0.5)
            w_sim = 1.0 - w_api

            blended = w_api * api_predictions[seed_idx] + w_sim * sim_predictions[seed_idx]

        blended = np.maximum(blended, 0.01)
        blended /= blended.sum(axis=2, keepdims=True)
        blend_predictions.append(blended)
        np.save(pred_dir / f"seed_{seed_idx}_blend.npy", blended)

    logger.info("All predictions saved to %s", pred_dir)

    # 7. Submit if requested
    if submit:
        # Re-check that round is still active before submitting
        active = client.get_active_round()
        if not active or active["id"] != round_id:
            logger.warning("Round %s is no longer active — skipping submission", round_num)
        else:
            for seed_idx in range(num_seeds):
                pred = blend_predictions[seed_idx]
                logger.info("Submitting seed %d (blend)...", seed_idx)
                result = client.submit(round_id, seed_idx, pred.tolist())
                logger.info("  Seed %d: %s", seed_idx, result)

            # Archive the submission
            _save_submission(save_dir, blend_predictions, num_seeds,
                             all_observations, sim_params)
            logger.info("All seeds submitted!")
    else:
        logger.info("Predictions ready but NOT submitted. Run with --submit to submit.")

    client.close()


def fetch_analysis():
    """Fetch and save analysis (ground truth) for all completed rounds."""
    client = AstarClient()
    rounds = client.get_rounds()

    for r in rounds:
        status = r.get("status", "")
        if status not in ("completed", "scoring"):
            continue

        round_id = r["id"]
        round_num = r.get("round_number", "unknown")
        detail = client.get_round_detail(round_id)
        num_seeds = detail["seeds_count"]

        save_dir, _, _, _ = _get_round_dirs(round_num)
        analysis_dir = save_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)

        # Save round detail if not already saved
        detail_file = save_dir / "round_detail.json"
        if not detail_file.exists():
            with open(detail_file, "w") as f:
                json.dump(detail, f, indent=2)

        for seed_idx in range(num_seeds):
            out_file = analysis_dir / f"seed_{seed_idx}.json"
            if out_file.exists():
                logger.info("Round %s seed %d: analysis already saved", round_num, seed_idx)
                continue
            try:
                analysis = client.get_analysis(round_id, seed_idx)
                with open(out_file, "w") as f:
                    json.dump(analysis, f)
                score = analysis.get("score")
                logger.info("Round %s seed %d: score=%.2f, saved to %s",
                            round_num, seed_idx, score if score else 0, out_file)
            except Exception as e:
                logger.warning("Round %s seed %d: analysis not available (%s)",
                               round_num, seed_idx, e)

    client.close()
    logger.info("Analysis fetch complete.")


if __name__ == "__main__":
    if "--analysis" in sys.argv:
        fetch_analysis()
    elif "--targeted" in sys.argv:
        # Round 8 strategy: all 39 queries on seed 0, 3 viewports x 13 queries
        ROUND_8_TARGETS = [
            (0, 19, 6, 13),   # VP1: settlement cluster at (19,6)
            (0, 11, 22, 13),  # VP2: settlement cluster at (11,22)
            (0, 0, 1, 13),    # VP3: settlement cluster at (0,1)
        ]
        print("Targeted fetch plan:")
        for seed, vx, vy, n in ROUND_8_TARGETS:
            print(f"  Seed {seed}, viewport ({vx},{vy}) 15x15, {n} queries")
        print(f"  Total: {sum(n for _, _, _, n in ROUND_8_TARGETS)} queries")
        confirm = input("Proceed? [y/N] ")
        if confirm.strip().lower() == "y":
            fetch_targeted(ROUND_8_TARGETS)
        else:
            print("Aborted.")
    elif "--fetch" in sys.argv:
        fetch()
    else:
        do_submit = "--submit" in sys.argv
        do_calibrate = "--calibrate" in sys.argv
        skip_fetch = "--no-fetch" in sys.argv
        run(submit=do_submit, do_calibrate=do_calibrate, skip_fetch=skip_fetch)
