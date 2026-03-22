"""Infer hidden simulation parameters from diagnostic observations.

Uses settlement internal stats (population, food, wealth, defense, owner_id)
returned by the simulate endpoint to fit our local simulator's parameters
to match the real simulator's behavior.

The key insight: repeated observations of the same viewport give us
distributions over settlement outcomes. We extract summary statistics
(survival rate, avg population, port rate, expansion rate, etc.) and
optimize our simulator's parameters to match these statistics.

This is system identification: known model structure + observed outputs → parameters.

Usage:
    from nm_ai_ml.astar.param_inference import infer_params
    params = infer_params(initial_states, observations)
"""
import json
import logging
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from nm_ai_ml.astar.simulator_v2 import (
    Simulator, DEFAULT_PARAMS, OCEAN, MOUNTAIN, SETTLEMENT, PORT, RUIN, FOREST,
    PLAINS, EMPTY, _LUT,
)

logger = logging.getLogger(__name__)

# Parameters we can infer and their bounds
INFERENCE_PARAMS = [
    ("growth_rate", 0.01, 0.5),
    ("winter_severity", 0.05, 1.0),
    ("collapse_threshold", -2.0, 0.0),
    ("founding_threshold", 2.0, 8.0),
    ("food_per_forest", 0.05, 1.0),
    ("food_per_plains", 0.01, 0.5),
    ("rebuild_prob", 0.01, 0.4),
    ("raid_range", 1, 8),
    ("raid_strength", 0.05, 0.8),
    ("longship_raid_bonus", 1, 8),
    ("trade_range", 2, 12),
    ("trade_food", 0.05, 0.8),
    ("port_threshold", 0.5, 5.0),
    ("ruin_reclaim_prob", 0.01, 0.3),
]


# ---------------------------------------------------------------------------
# Extract observed statistics from API observations
# ---------------------------------------------------------------------------

def extract_obs_stats(initial_state: dict, observations: list[dict]) -> dict:
    """Extract summary statistics from repeated observations of a seed.

    Each observation contains the grid + settlement internal stats after 50 years.
    We compute distributions over these outcomes.

    Args:
        initial_state: Seed initial state with 'grid' and 'settlements'.
        observations: List of observation dicts from the simulate endpoint.

    Returns:
        Dict of summary statistics for parameter inference.
    """
    grid = np.array(initial_state["grid"])
    H, W = grid.shape
    init_settlements = initial_state.get("settlements", [])
    init_sett_positions = set((s["x"], s["y"]) for s in init_settlements)
    n_init = len(init_settlements)

    if not observations:
        return {}

    # Per-observation stats
    survival_counts = []         # fraction of initial settlements still alive
    expansion_counts = []        # number of NEW settlements (not in initial state)
    port_counts = []             # number of ports
    ruin_counts = []             # number of ruins on grid
    populations = []             # avg population of alive settlements
    foods = []                   # avg food of alive settlements
    wealths = []                 # avg wealth of alive settlements
    defenses = []                # avg defense
    allegiance_changes = []      # fraction of settlements with changed owner_id
    port_formation_counts = []   # initial non-port settlements that became ports

    # Per-cell class counts (for cells covered by viewports)
    cell_class_counts = {}  # (y, x) -> [count_class_0, ..., count_class_5]

    for obs in observations:
        vp = obs["viewport"]
        setts = obs.get("settlements", [])

        # Grid-level stats
        obs_grid = obs["grid"]
        ruin_count = 0
        for row in obs_grid:
            for cell in row:
                if cell == 3:  # Ruin
                    ruin_count += 1

        # Track per-cell outcomes
        for dy, row in enumerate(obs_grid):
            for dx, cell in enumerate(row):
                gy = vp["y"] + dy
                gx = vp["x"] + dx
                if gy >= H or gx >= W:
                    continue
                key = (gy, gx)
                if key not in cell_class_counts:
                    cell_class_counts[key] = np.zeros(6, dtype=int)
                cls = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}.get(cell, 0)
                cell_class_counts[key][cls] += 1

        ruin_counts.append(ruin_count)

        # Settlement stats
        alive_setts = [s for s in setts if s.get("alive", True)]
        alive_positions = set((s["x"], s["y"]) for s in alive_setts)

        # How many initial settlements survived (within viewport)?
        init_in_vp = [
            s for s in init_settlements
            if vp["x"] <= s["x"] < vp["x"] + vp["w"]
            and vp["y"] <= s["y"] < vp["y"] + vp["h"]
        ]
        if init_in_vp:
            survived = sum(1 for s in init_in_vp if (s["x"], s["y"]) in alive_positions)
            survival_counts.append(survived / len(init_in_vp))

        # New settlements (in viewport but not in initial state)
        new_setts = [s for s in alive_setts if (s["x"], s["y"]) not in init_sett_positions]
        expansion_counts.append(len(new_setts))

        # Port formation: initial non-port settlements that are now ports
        init_non_ports = [s for s in init_in_vp if not s.get("has_port", False)]
        if init_non_ports:
            became_port = sum(
                1 for s in init_non_ports
                if any(a["x"] == s["x"] and a["y"] == s["y"] and a.get("has_port", False)
                       for a in alive_setts)
            )
            port_formation_counts.append(became_port / len(init_non_ports))

        # Port count
        ports = [s for s in alive_setts if s.get("has_port", False)]
        port_counts.append(len(ports))

        # Internal stats
        if alive_setts:
            populations.append(np.mean([s["population"] for s in alive_setts]))
            foods.append(np.mean([s["food"] for s in alive_setts]))
            wealths.append(np.mean([s["wealth"] for s in alive_setts]))
            defenses.append(np.mean([s["defense"] for s in alive_setts]))

        # Allegiance changes
        if init_in_vp and alive_setts:
            # Match by position
            changes = 0
            matched = 0
            for init_s in init_in_vp:
                for alive_s in alive_setts:
                    if alive_s["x"] == init_s["x"] and alive_s["y"] == init_s["y"]:
                        matched += 1
                        # We don't know initial owner_id, but we can check
                        # if surviving settlements have diverse owner_ids
                        break
            # Count distinct owner_ids among surviving initial settlements
            surviving_owners = set()
            for init_s in init_in_vp:
                for alive_s in alive_setts:
                    if alive_s["x"] == init_s["x"] and alive_s["y"] == init_s["y"]:
                        surviving_owners.add(alive_s.get("owner_id", -1))
            if len(init_in_vp) > 1:
                # Faction consolidation: 1.0 = all same owner, 0.0 = all different
                faction_consolidation = 1.0 - (len(surviving_owners) - 1) / (len(init_in_vp) - 1)
                allegiance_changes.append(max(0, faction_consolidation))

    stats = {
        "n_observations": len(observations),
        "n_initial_settlements": n_init,
        "cell_class_counts": cell_class_counts,
    }

    if survival_counts:
        stats["survival_rate"] = float(np.mean(survival_counts))
        stats["survival_std"] = float(np.std(survival_counts))
    if expansion_counts:
        stats["avg_expansion"] = float(np.mean(expansion_counts))
    if port_counts:
        stats["avg_ports"] = float(np.mean(port_counts))
    if ruin_counts:
        stats["avg_ruins"] = float(np.mean(ruin_counts))
    if populations:
        stats["avg_population"] = float(np.mean(populations))
    if foods:
        stats["avg_food"] = float(np.mean(foods))
    if wealths:
        stats["avg_wealth"] = float(np.mean(wealths))
    if defenses:
        stats["avg_defense"] = float(np.mean(defenses))
    if allegiance_changes:
        stats["faction_consolidation"] = float(np.mean(allegiance_changes))
    if port_formation_counts:
        stats["port_formation_rate"] = float(np.mean(port_formation_counts))

    return stats


def extract_all_stats(initial_states: list[dict], all_observations: dict[int, list[dict]]) -> dict:
    """Extract stats across all seeds, return combined stats dict."""
    combined = {
        "per_seed": {},
        "n_total_observations": 0,
    }

    all_survival = []
    all_expansion = []
    all_population = []
    all_food = []
    all_wealth = []
    all_ports = []
    all_ruins = []
    all_port_formation = []
    all_cell_counts = {}

    for seed_idx, obs_list in all_observations.items():
        if not obs_list:
            continue
        state = initial_states[seed_idx]
        stats = extract_obs_stats(state, obs_list)
        combined["per_seed"][seed_idx] = stats
        combined["n_total_observations"] += len(obs_list)

        if "survival_rate" in stats:
            all_survival.append(stats["survival_rate"])
        if "avg_expansion" in stats:
            all_expansion.append(stats["avg_expansion"])
        if "avg_population" in stats:
            all_population.append(stats["avg_population"])
        if "avg_food" in stats:
            all_food.append(stats["avg_food"])
        if "avg_wealth" in stats:
            all_wealth.append(stats["avg_wealth"])
        if "avg_ports" in stats:
            all_ports.append(stats["avg_ports"])
        if "avg_ruins" in stats:
            all_ruins.append(stats["avg_ruins"])
        if "port_formation_rate" in stats:
            all_port_formation.append(stats["port_formation_rate"])

        for k, v in stats.get("cell_class_counts", {}).items():
            if k not in all_cell_counts:
                all_cell_counts[k] = np.zeros(6, dtype=int)
            all_cell_counts[k] += v

    combined["survival_rate"] = float(np.mean(all_survival)) if all_survival else None
    combined["avg_expansion"] = float(np.mean(all_expansion)) if all_expansion else None
    combined["avg_population"] = float(np.mean(all_population)) if all_population else None
    combined["avg_food"] = float(np.mean(all_food)) if all_food else None
    combined["avg_wealth"] = float(np.mean(all_wealth)) if all_wealth else None
    combined["avg_ports"] = float(np.mean(all_ports)) if all_ports else None
    combined["avg_ruins"] = float(np.mean(all_ruins)) if all_ruins else None
    combined["port_formation_rate"] = float(np.mean(all_port_formation)) if all_port_formation else None
    combined["cell_class_counts"] = all_cell_counts

    return combined


# ---------------------------------------------------------------------------
# Simulate with candidate params and extract same stats
# ---------------------------------------------------------------------------

def simulate_stats(grid: np.ndarray, settlements: list[dict], params: dict,
                   n_sims: int = 50, viewport: dict | None = None) -> dict:
    """Run local simulator and extract the same summary stats.

    If viewport is given, only extract stats within that viewport.
    Otherwise uses full grid.
    """
    H, W = grid.shape
    init_sett_positions = set((s["x"], s["y"]) for s in settlements)

    survival_counts = []
    expansion_counts = []
    port_counts = []
    ruin_counts = []
    populations = []
    foods = []
    wealths = []
    defenses = []
    port_formation_counts = []
    cell_class_counts = {}

    for i in range(n_sims):
        sim = Simulator(grid, settlements, params=params, seed=1000 + i)
        final_grid = sim.run(50)
        alive_setts = sim.get_settlements()
        alive_positions = set((s["x"], s["y"]) for s in alive_setts)

        # Determine viewport bounds
        if viewport:
            vx, vy = viewport["x"], viewport["y"]
            vw, vh = viewport["w"], viewport["h"]
        else:
            vx, vy, vw, vh = 0, 0, W, H

        # Grid stats within viewport
        ruin_count = 0
        for dy in range(vh):
            for dx in range(vw):
                gy, gx = vy + dy, vx + dx
                if gy >= H or gx >= W:
                    continue
                cell = final_grid[gy, gx]
                if cell == RUIN:
                    ruin_count += 1
                key = (gy, gx)
                if key not in cell_class_counts:
                    cell_class_counts[key] = np.zeros(6, dtype=int)
                cell_class_counts[key][_LUT[cell]] += 1

        ruin_counts.append(ruin_count)

        # Settlement stats within viewport
        setts_in_vp = [
            s for s in alive_setts
            if vx <= s["x"] < vx + vw and vy <= s["y"] < vy + vh
        ]
        init_in_vp = [
            s for s in settlements
            if vx <= s["x"] < vx + vw and vy <= s["y"] < vy + vh
        ]

        if init_in_vp:
            survived = sum(1 for s in init_in_vp if (s["x"], s["y"]) in alive_positions)
            survival_counts.append(survived / len(init_in_vp))

        new_setts = [s for s in setts_in_vp if (s["x"], s["y"]) not in init_sett_positions]
        expansion_counts.append(len(new_setts))

        ports = [s for s in setts_in_vp if s.get("has_port", False)]
        port_counts.append(len(ports))

        init_non_ports = [s for s in init_in_vp if not s.get("has_port", False)]
        if init_non_ports:
            became_port = sum(
                1 for s in init_non_ports
                if any(a["x"] == s["x"] and a["y"] == s["y"] and a.get("has_port", False)
                       for a in setts_in_vp)
            )
            port_formation_counts.append(became_port / len(init_non_ports))

        if setts_in_vp:
            populations.append(np.mean([s["population"] for s in setts_in_vp]))
            foods.append(np.mean([s["food"] for s in setts_in_vp]))
            wealths.append(np.mean([s["wealth"] for s in setts_in_vp]))
            defenses.append(np.mean([s["defense"] for s in setts_in_vp]))

    stats = {
        "survival_rate": float(np.mean(survival_counts)) if survival_counts else None,
        "avg_expansion": float(np.mean(expansion_counts)) if expansion_counts else None,
        "avg_ports": float(np.mean(port_counts)) if port_counts else None,
        "avg_ruins": float(np.mean(ruin_counts)) if ruin_counts else None,
        "avg_population": float(np.mean(populations)) if populations else None,
        "avg_food": float(np.mean(foods)) if foods else None,
        "avg_wealth": float(np.mean(wealths)) if wealths else None,
        "avg_defense": float(np.mean(defenses)) if defenses else None,
        "port_formation_rate": float(np.mean(port_formation_counts)) if port_formation_counts else None,
        "cell_class_counts": cell_class_counts,
    }
    return stats


# ---------------------------------------------------------------------------
# Objective function: match observed stats
# ---------------------------------------------------------------------------

def _stats_distance(observed: dict, simulated: dict) -> float:
    """Compute weighted distance between observed and simulated statistics.

    Each statistic is weighted by its informativeness for parameter inference.
    """
    distance = 0.0
    n_terms = 0

    # Survival rate: most informative (constrains growth × winter)
    if observed.get("survival_rate") is not None and simulated.get("survival_rate") is not None:
        diff = observed["survival_rate"] - simulated["survival_rate"]
        distance += 10.0 * diff ** 2
        n_terms += 1

    # Expansion: constrains founding_threshold
    if observed.get("avg_expansion") is not None and simulated.get("avg_expansion") is not None:
        diff = observed["avg_expansion"] - simulated["avg_expansion"]
        # Normalize by scale
        scale = max(observed["avg_expansion"], 1.0)
        distance += 5.0 * (diff / scale) ** 2
        n_terms += 1

    # Port formation: constrains port_threshold
    if observed.get("port_formation_rate") is not None and simulated.get("port_formation_rate") is not None:
        diff = observed["port_formation_rate"] - simulated["port_formation_rate"]
        distance += 5.0 * diff ** 2
        n_terms += 1

    # Average population: constrains growth_rate
    if observed.get("avg_population") is not None and simulated.get("avg_population") is not None:
        diff = observed["avg_population"] - simulated["avg_population"]
        scale = max(observed["avg_population"], 0.5)
        distance += 3.0 * (diff / scale) ** 2
        n_terms += 1

    # Average food: constrains food_per_forest, food_per_plains
    if observed.get("avg_food") is not None and simulated.get("avg_food") is not None:
        diff = observed["avg_food"] - simulated["avg_food"]
        scale = max(abs(observed["avg_food"]), 0.5)
        distance += 3.0 * (diff / scale) ** 2
        n_terms += 1

    # Average wealth: constrains trade params
    if observed.get("avg_wealth") is not None and simulated.get("avg_wealth") is not None:
        diff = observed["avg_wealth"] - simulated["avg_wealth"]
        scale = max(observed["avg_wealth"], 0.3)
        distance += 2.0 * (diff / scale) ** 2
        n_terms += 1

    # Ruins: constrains winter + collapse
    if observed.get("avg_ruins") is not None and simulated.get("avg_ruins") is not None:
        diff = observed["avg_ruins"] - simulated["avg_ruins"]
        scale = max(observed["avg_ruins"], 1.0)
        distance += 3.0 * (diff / scale) ** 2
        n_terms += 1

    # Per-cell class distribution match (most directly informative)
    obs_cells = observed.get("cell_class_counts", {})
    sim_cells = simulated.get("cell_class_counts", {})
    if obs_cells and sim_cells:
        cell_kl_sum = 0.0
        cell_count = 0
        for key in obs_cells:
            if key not in sim_cells:
                continue
            obs_dist = obs_cells[key].astype(float)
            sim_dist = sim_cells[key].astype(float)
            obs_total = obs_dist.sum()
            sim_total = sim_dist.sum()
            if obs_total < 2 or sim_total < 2:
                continue
            # Normalize to probabilities
            p = obs_dist / obs_total
            q = sim_dist / sim_total
            # KL(p || q) with smoothing
            p = np.maximum(p, 0.01)
            q = np.maximum(q, 0.01)
            p /= p.sum()
            q /= q.sum()
            kl = np.sum(p * np.log(p / q))
            cell_kl_sum += kl
            cell_count += 1
        if cell_count > 0:
            distance += 15.0 * (cell_kl_sum / cell_count)
            n_terms += 1

    return distance / max(n_terms, 1)


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def infer_params(
    initial_states: list[dict],
    all_observations: dict[int, list[dict]],
    n_sims_per_eval: int = 30,
    maxiter: int = 20,
    popsize: int = 8,
) -> dict:
    """Infer hidden parameters from diagnostic observations.

    Args:
        initial_states: List of seed initial states.
        all_observations: Dict mapping seed_idx → list of observations.
        n_sims_per_eval: Sims per parameter evaluation (higher = slower but more accurate).
        maxiter: Max iterations for differential evolution.
        popsize: Population size for DE (× num params).

    Returns:
        Full parameter dict with inferred values.
    """
    logger.info("Extracting observed statistics from %d seeds...",
                sum(1 for v in all_observations.values() if v))

    observed_stats = extract_all_stats(initial_states, all_observations)
    logger.info("Observed stats: survival=%.2f, expansion=%.1f, population=%.2f, "
                "food=%.2f, wealth=%.2f, ports=%.1f, ruins=%.1f",
                observed_stats.get("survival_rate", -1) or -1,
                observed_stats.get("avg_expansion", -1) or -1,
                observed_stats.get("avg_population", -1) or -1,
                observed_stats.get("avg_food", -1) or -1,
                observed_stats.get("avg_wealth", -1) or -1,
                observed_stats.get("avg_ports", -1) or -1,
                observed_stats.get("avg_ruins", -1) or -1)

    # Pick seed(s) to simulate against (ones with most observations)
    eval_seeds = []
    for seed_idx, obs_list in sorted(all_observations.items()):
        if len(obs_list) >= 3:
            eval_seeds.append(seed_idx)
    if not eval_seeds:
        eval_seeds = [max(all_observations, key=lambda k: len(all_observations[k]))]

    logger.info("Evaluating against seeds: %s", eval_seeds)

    # Prepare grids and settlements for eval seeds
    eval_data = []
    for seed_idx in eval_seeds:
        state = initial_states[seed_idx]
        grid = np.array(state["grid"])
        settlements = state.get("settlements", [])
        # Determine viewport from observations (union of all viewports)
        obs_list = all_observations[seed_idx]
        if obs_list:
            vp = obs_list[0]["viewport"]  # Use first viewport as reference
        else:
            vp = None
        eval_data.append((grid, settlements, vp))

    # Observed per-seed stats
    obs_per_seed = {}
    for seed_idx in eval_seeds:
        state = initial_states[seed_idx]
        obs_per_seed[seed_idx] = extract_obs_stats(state, all_observations[seed_idx])

    bounds = [(low, high) for _, low, high in INFERENCE_PARAMS]
    default_values = np.array([DEFAULT_PARAMS[name] for name, _, _ in INFERENCE_PARAMS])

    best = {"dist": float("inf"), "values": default_values.copy()}
    eval_count = [0]

    def objective(x):
        params = dict(DEFAULT_PARAMS)
        for i, (name, _, _) in enumerate(INFERENCE_PARAMS):
            params[name] = float(x[i])

        total_dist = 0.0
        for idx, seed_idx in enumerate(eval_seeds):
            grid, settlements, vp = eval_data[idx]
            sim_stats = simulate_stats(grid, settlements, params,
                                       n_sims=n_sims_per_eval, viewport=vp)
            total_dist += _stats_distance(obs_per_seed[seed_idx], sim_stats)

        avg_dist = total_dist / len(eval_seeds)
        eval_count[0] += 1

        if avg_dist < best["dist"]:
            best["dist"] = avg_dist
            best["values"] = x.copy()
            if eval_count[0] <= 5 or eval_count[0] % 10 == 0:
                logger.info("Eval %d: new best dist=%.4f", eval_count[0], avg_dist)
        elif eval_count[0] % 20 == 0:
            logger.info("Eval %d: dist=%.4f (best=%.4f)", eval_count[0], avg_dist, best["dist"])

        return avg_dist

    # Evaluate baseline
    baseline_dist = objective(default_values)
    logger.info("Baseline distance (default params): %.4f", baseline_dist)

    logger.info("Running differential evolution (%d params, popsize=%d, maxiter=%d)...",
                len(INFERENCE_PARAMS), popsize, maxiter)

    result = differential_evolution(
        objective,
        bounds=bounds,
        x0=default_values,
        maxiter=maxiter,
        popsize=popsize,
        tol=0.005,
        seed=42,
        init="sobol",
    )

    logger.info("Inference complete after %d evaluations", eval_count[0])
    logger.info("Best distance: %.4f (baseline: %.4f, improvement: %.1f%%)",
                best["dist"], baseline_dist,
                100 * (baseline_dist - best["dist"]) / max(baseline_dist, 1e-10))

    optimized = dict(DEFAULT_PARAMS)
    for i, (name, _, _) in enumerate(INFERENCE_PARAMS):
        optimized[name] = float(best["values"][i])

    return optimized


def infer_and_save(initial_states, all_observations, save_path="data/inferred_params.json",
                   **kwargs) -> dict:
    """Infer params and save to disk."""
    params = infer_params(initial_states, all_observations, **kwargs)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(params, f, indent=2)
    logger.info("Inferred params saved to %s", save_path)
    return params
