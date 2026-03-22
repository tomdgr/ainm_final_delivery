"""Diagnostic viewport strategy for Astar Island.

Concentrates queries on carefully chosen viewports that isolate specific
game mechanics. Repeats each viewport 5-10 times to get real probability
distributions instead of single noisy samples.

Strategy:
1. Analyze initial state to find diagnostic settlement types:
   - Isolated settlement (measures growth/winter survival)
   - Clustered settlements (measures raiding/faction dynamics)
   - Coastal settlement (measures port development)
   - Forest-surrounded settlement (measures food production)
2. Place viewports to capture each type, repeat 5-10 times
3. Use remaining budget on highest-uncertainty areas

Usage:
    from nm_ai_ml.astar.diagnostic_strategy import plan_diagnostic_viewports
    viewports = plan_diagnostic_viewports(detail, seed_idx=0, budget=50)
"""
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

OCEAN = 10
PLAINS = 11
EMPTY = 0
SETTLEMENT = 1
PORT = 2
FOREST = 4
MOUNTAIN = 5


def _classify_settlements(grid: np.ndarray, settlements: list[dict]) -> dict[str, list[dict]]:
    """Classify each settlement by its spatial context.

    Returns dict mapping category → list of settlements with metadata.
    """
    H, W = grid.shape
    sett_positions = set((s["x"], s["y"]) for s in settlements)

    classified = {
        "isolated": [],      # no settlement neighbors — pure growth/winter
        "clustered": [],     # 2+ settlement neighbors — raiding/faction
        "coastal": [],       # adjacent to ocean — port development
        "coastal_port": [],  # already a port — port mechanics
        "forested": [],      # 3+ forest neighbors — food production
        "open": [],          # mostly plains — expansion space
    }

    for se in settlements:
        x, y = se["x"], se["y"]
        n_forest, n_plains, n_ocean, n_sett = 0, 0, 0, 0

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    if grid[ny, nx] == FOREST:
                        n_forest += 1
                    elif grid[ny, nx] in (EMPTY, PLAINS):
                        n_plains += 1
                    elif grid[ny, nx] == OCEAN:
                        n_ocean += 1
                    if (nx, ny) in sett_positions and (nx, ny) != (x, y):
                        n_sett += 1

        info = {**se, "n_forest": n_forest, "n_plains": n_plains,
                "n_ocean": n_ocean, "n_sett": n_sett}

        if se.get("has_port"):
            classified["coastal_port"].append(info)
        elif n_ocean > 0:
            classified["coastal"].append(info)

        if n_sett == 0:
            classified["isolated"].append(info)
        if n_sett >= 2:
            classified["clustered"].append(info)
        if n_forest >= 3:
            classified["forested"].append(info)
        if n_plains >= 5 and n_sett == 0:
            classified["open"].append(info)

    return classified


def _find_best_viewport(grid: np.ndarray, target_x: int, target_y: int,
                        map_width: int = 40, map_height: int = 40) -> tuple[int, int]:
    """Find 15x15 viewport that centers on target cell as much as possible."""
    vx = max(0, min(target_x - 7, map_width - 15))
    vy = max(0, min(target_y - 7, map_height - 15))
    return vx, vy


def _find_viewport_covering_multiple(settlements: list[dict],
                                      map_width: int = 40,
                                      map_height: int = 40) -> Optional[tuple[int, int]]:
    """Find 15x15 viewport that covers the most settlements from the list."""
    if not settlements:
        return None

    best_pos = None
    best_count = 0

    max_vx = map_width - 15
    max_vy = map_height - 15

    for vy in range(max_vy + 1):
        for vx in range(max_vx + 1):
            count = sum(
                1 for s in settlements
                if vx <= s["x"] < vx + 15 and vy <= s["y"] < vy + 15
            )
            if count > best_count:
                best_count = count
                best_pos = (vx, vy)

    return best_pos


def plan_diagnostic_viewports(
    detail: dict,
    seed_idx: int = 0,
    budget: int = 50,
    repeats_per_diagnostic: int = 8,
) -> list[tuple[int, int, str]]:
    """Plan viewport positions for diagnostic observation strategy.

    Args:
        detail: Round detail from API (contains initial_states).
        seed_idx: Which seed to use for diagnostics.
        budget: Total query budget.
        repeats_per_diagnostic: How many times to repeat each diagnostic viewport.

    Returns:
        List of (viewport_x, viewport_y, label) tuples.
        Label describes what this viewport is diagnosing.
    """
    state = detail["initial_states"][seed_idx]
    grid = np.array(state["grid"])
    settlements = state.get("settlements", [])
    H, W = grid.shape

    classified = _classify_settlements(grid, settlements)

    logger.info("Settlement classification for seed %d:", seed_idx)
    for cat, setts in classified.items():
        if setts:
            logger.info("  %s: %d settlements", cat, len(setts))

    viewports = []  # (vx, vy, label)
    used_viewports = set()  # track to avoid duplicates

    def _add_diagnostic(settlements_list, label_prefix, sort_key, n_repeats):
        """Add a diagnostic viewport, avoiding duplicates."""
        if not settlements_list:
            return
        # Sort by preference, try until we find one not already used
        sorted_setts = sorted(settlements_list, key=sort_key, reverse=True)
        for best in sorted_setts:
            vx, vy = _find_best_viewport(grid, best["x"], best["y"], W, H)
            if (vx, vy) not in used_viewports:
                used_viewports.add((vx, vy))
                for _ in range(n_repeats):
                    viewports.append((vx, vy, f"{label_prefix}({best['x']},{best['y']})"))
                logger.info("Diagnostic %s: (%d,%d) viewport (%d,%d) f=%d p=%d o=%d s=%d",
                            label_prefix, best["x"], best["y"], vx, vy,
                            best["n_forest"], best["n_plains"], best["n_ocean"], best["n_sett"])
                return
        # All viewports taken — use first anyway
        best = sorted_setts[0]
        vx, vy = _find_best_viewport(grid, best["x"], best["y"], W, H)
        for _ in range(n_repeats):
            viewports.append((vx, vy, f"{label_prefix}({best['x']},{best['y']})"))

    # === Diagnostic 1: Isolated settlement with food (growth/winter) ===
    _add_diagnostic(classified["isolated"], "isolated",
                    lambda s: s["n_forest"] + 0.3 * s["n_plains"], repeats_per_diagnostic)

    # === Diagnostic 2: Densest settlement area (raiding/faction) ===
    clustered = classified["clustered"]
    if clustered:
        vp = _find_viewport_covering_multiple(clustered, W, H)
    else:
        vp = _find_viewport_covering_multiple(settlements, W, H)
    if vp and vp not in used_viewports:
        vx, vy = vp
        used_viewports.add((vx, vy))
        for _ in range(repeats_per_diagnostic):
            viewports.append((vx, vy, f"clustered({vx},{vy})"))
        n_in = sum(1 for s in settlements if vx <= s["x"] < vx+15 and vy <= s["y"] < vy+15)
        logger.info("Diagnostic clustered: viewport (%d,%d) covers %d settlements", vx, vy, n_in)

    # === Diagnostic 3: Coastal settlement (port development) ===
    coastal = classified["coastal_port"] or classified["coastal"]
    _add_diagnostic(coastal, "coastal",
                    lambda s: s["n_ocean"] + (5 if s.get("has_port") else 0), repeats_per_diagnostic)

    # === Diagnostic 4: Forest-surrounded (food production) ===
    _add_diagnostic(classified["forested"], "forested",
                    lambda s: s["n_forest"], repeats_per_diagnostic)

    # === Diagnostic 5: Open plains settlement (expansion space) ===
    _add_diagnostic(classified["open"], "open",
                    lambda s: s["n_plains"], repeats_per_diagnostic)

    # === Remaining budget: highest-settlement-density areas not yet covered ===
    remaining = budget - len(viewports)
    if remaining > 0:
        covered_vps = set((vx, vy) for vx, vy, _ in viewports)

        # Find viewports with most settlements that aren't already covered
        max_vx = W - 15
        max_vy = H - 15
        vp_scores = []
        for vy in range(max_vy + 1):
            for vx in range(max_vx + 1):
                if (vx, vy) in covered_vps:
                    continue
                count = sum(
                    1 for s in settlements
                    if vx <= s["x"] < vx + 15 and vy <= s["y"] < vy + 15
                )
                if count > 0:
                    vp_scores.append((vx, vy, count))

        vp_scores.sort(key=lambda t: t[2], reverse=True)

        # Add remaining viewports, repeating top ones for better statistics
        repeats_remaining = max(3, remaining // max(len(vp_scores[:5]), 1))
        for vx, vy, count in vp_scores[:remaining]:
            viewports.append((vx, vy, f"coverage({vx},{vy},n={count})"))

        # If still budget left, repeat the densest viewport
        while len(viewports) < budget and vp_scores:
            vx, vy, count = vp_scores[0]
            viewports.append((vx, vy, f"repeat_dense({vx},{vy})"))

    viewports = viewports[:budget]

    logger.info("Viewport plan: %d total (%d diagnostic + %d coverage)",
                len(viewports),
                sum(1 for _, _, l in viewports if not l.startswith("coverage") and not l.startswith("repeat")),
                sum(1 for _, _, l in viewports if l.startswith("coverage") or l.startswith("repeat")))

    return viewports


def extract_diagnostic_params(observations: list[dict], grid: np.ndarray,
                               viewport_labels: list[str]) -> dict:
    """Extract hidden parameter estimates from diagnostic observations.

    Groups observations by diagnostic label and computes statistics:
    - Survival rates per settlement type
    - Expansion rates (new settlements per observation)
    - Port development rate
    - Food levels (proxy for winter severity)
    - Population levels (proxy for growth rate)

    Returns dict of estimated parameters.
    """
    H, W = grid.shape
    GRID_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}

    # Group observations by label
    groups = {}
    for obs, label in zip(observations, viewport_labels):
        base_label = label.split("(")[0]
        if base_label not in groups:
            groups[base_label] = []
        groups[base_label].append(obs)

    params = {}

    # From all observations: basic stats
    all_pops, all_foods, all_defs, all_wealths = [], [], [], []
    total_sett_cells, total_cells = 0, 0
    total_alive, total_ports = 0, 0

    for obs in observations:
        for se in obs.get("settlements", []):
            if se.get("alive"):
                all_pops.append(se["population"])
                all_foods.append(se["food"])
                all_defs.append(se["defense"])
                all_wealths.append(se["wealth"])
                total_alive += 1
                if se.get("has_port"):
                    total_ports += 1
        for row in obs["grid"]:
            for c in row:
                total_cells += 1
                if GRID_TO_CLASS.get(c, 0) == 1:
                    total_sett_cells += 1

    if all_pops:
        params["avg_population"] = float(np.mean(all_pops))
        params["avg_food"] = float(np.mean(all_foods))
        params["avg_defense"] = float(np.mean(all_defs))
        params["avg_wealth"] = float(np.mean(all_wealths))
        params["settlement_pct"] = total_sett_cells / max(total_cells, 1) * 100
        params["port_rate"] = total_ports / max(total_alive, 1)

    # Estimate simulator parameters from observations
    food = params.get("avg_food", 0.5)
    pop = params.get("avg_population", 1.0)
    sett_pct = params.get("settlement_pct", 5.0)
    defense = params.get("avg_defense", 0.5)

    # Growth rate: inversely related to food (high growth consumes food)
    params["est_growth_rate"] = max(0.02, 0.35 - 0.45 * food)

    # Winter severity: inversely related to food
    params["est_winter_severity"] = max(0.1, 1.0 - 0.85 * food)

    # Founding: proportional to settlement expansion
    if sett_pct > 15:
        params["est_founding_prob"] = 0.25
        params["est_founding_threshold"] = 1.2
    elif sett_pct > 8:
        params["est_founding_prob"] = 0.15
        params["est_founding_threshold"] = 1.5
    elif sett_pct > 3:
        params["est_founding_prob"] = 0.10
        params["est_founding_threshold"] = 2.0
    else:
        params["est_founding_prob"] = 0.05
        params["est_founding_threshold"] = 3.0

    # Raid strength: proportional to defense
    params["est_raid_strength"] = max(0.1, defense * 0.8)

    # Per-diagnostic-group analysis
    for group_name, obs_list in groups.items():
        survival_rates = []
        for obs in obs_list:
            alive = sum(1 for s in obs.get("settlements", []) if s.get("alive"))
            total = len(obs.get("settlements", []))
            if total > 0:
                survival_rates.append(alive / total)
        if survival_rates:
            params[f"{group_name}_survival_rate"] = float(np.mean(survival_rates))
            params[f"{group_name}_n_observations"] = len(obs_list)

    logger.info("Extracted parameters:")
    for k, v in sorted(params.items()):
        if isinstance(v, float):
            logger.info("  %s: %.4f", k, v)
        else:
            logger.info("  %s: %s", k, v)

    return params


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    round_dir = sys.argv[1] if len(sys.argv) > 1 else "data/rounds/round_14"

    with open(f"{round_dir}/round_detail.json") as f:
        detail = json.load(f)

    # Plan viewports for seed 0
    viewports = plan_diagnostic_viewports(detail, seed_idx=0, budget=50)

    print(f"\nViewport plan ({len(viewports)} queries):")
    seen = {}
    for vx, vy, label in viewports:
        key = f"({vx},{vy})"
        seen[key] = seen.get(key, 0) + 1

    for key, count in sorted(seen.items(), key=lambda x: -x[1]):
        labels = [l for vx, vy, l in viewports if f"({vx},{vy})" == key]
        print(f"  {key} × {count}: {labels[0]}")
