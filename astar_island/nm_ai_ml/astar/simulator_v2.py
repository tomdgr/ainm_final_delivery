"""V2 Local simulator — physics-based, calibrated against ground truth patterns.

Key differences from v1:
1. Population has carrying capacity (logistic growth, not exponential)
2. Food consumption scales with population (settlements starve under load)
3. Port formation is rare (matches GT: 7.5% of coastal settlements)
4. Settlements collapse frequently (matches GT: ~69% collapse rate)
5. Ruin reclamation is fast (ruins → forest/empty quickly)
6. Wealth barely accumulates (matches GT: ~0.01 mean wealth)
7. Forests get cleared by nearby settlements (matches GT: 29% near-sett clearing)

GT target statistics:
- Settlement survival: 31%
- Port formation (coastal non-port): 7.5%
- Population: mean 1.09, median 0.80
- Food: mean 0.67
- Wealth: mean 0.01
- Forest near sett stays forest: 71.4%
- Ruin rate: 2.4% (ruins are transient)
"""
from __future__ import annotations

import numpy as np
from numpy.random import Generator, default_rng

# Terrain constants
OCEAN = 10
PLAINS = 11
EMPTY = 0
SETTLEMENT = 1
PORT = 2
RUIN = 3
FOREST = 4
MOUNTAIN = 5

STATIC_TERRAIN = {OCEAN, MOUNTAIN}
BUILDABLE_TERRAIN = {PLAINS, EMPTY, FOREST}

# Neighbour offsets (8-connected)
_OFFSETS_8 = np.array(
    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
    dtype=np.int32,
)

DEFAULT_PARAMS: dict[str, float] = {
    # Food & Growth
    "food_per_forest": 0.22,       # food per adjacent forest tile per year
    "food_per_plains": 0.07,       # food per adjacent plains/empty tile per year
    "food_consumption": 0.28,      # food consumed per unit population per year
    "carrying_capacity": 2.0,      # max sustainable population (lower = more deaths)
    "growth_rate": 0.08,           # base population growth rate
    # Port & Ships
    "port_threshold": 1.0,         # population needed for port
    "port_prob": 0.05,             # probability of port formation per year when eligible
    "longship_threshold": 2.0,     # pop threshold for longship
    "longship_prob": 0.03,         # probability of longship per year
    # Expansion
    "founding_threshold": 1.2,     # population needed to found (lower = more expansion)
    "founding_prob": 0.08,         # probability of founding per year when eligible
    "founding_range": 3,           # max distance for new settlement (GT: steep falloff)
    "founding_cost": 0.35,         # fraction of pop/food sent to new settlement
    # Conflict
    "raid_range": 3,
    "longship_raid_bonus": 3,
    "raid_strength": 0.15,
    "raid_prob": 0.15,             # base chance of raiding per year
    "desperate_raid_mult": 2.0,
    # Trade
    "trade_range": 4,
    "trade_food": 0.05,            # very small trade benefit (wealth ≈ 0 in GT)
    "trade_wealth": 0.003,         # nearly zero wealth generation
    "tech_diffusion": 0.02,
    # Winter
    "winter_severity": 0.30,       # base food loss in winter
    "winter_variance": 0.20,       # high variance — harsh winters kill settlements
    "collapse_threshold": -0.2,    # food level for collapse check
    "min_pop_survive": 0.4,        # minimum population to survive
    # Environment
    "ruin_reclaim_prob": 0.25,     # fast: ruins disappear quickly (GT: only 2.4% ruins)
    "rebuild_prob": 0.10,          # probability of rebuilding on a ruin
    "forest_clear_prob": 0.005,    # very low: forests near sett stay 71% in GT
}

# Settlement structured array dtype
_SETTLEMENT_DTYPE = np.dtype([
    ("y", np.int32),
    ("x", np.int32),
    ("population", np.float32),
    ("food", np.float32),
    ("wealth", np.float32),
    ("defense", np.float32),
    ("tech_level", np.float32),
    ("has_port", np.bool_),
    ("has_longship", np.bool_),
    ("owner_id", np.int32),
    ("alive", np.bool_),
])


def _make_settlement_array(settlements: list[dict]) -> np.ndarray:
    n = len(settlements)
    arr = np.zeros(n, dtype=_SETTLEMENT_DTYPE)
    for i, s in enumerate(settlements):
        arr[i]["y"] = s.get("y", 0)
        arr[i]["x"] = s.get("x", 0)
        arr[i]["population"] = s.get("population", 1.0)
        arr[i]["food"] = s.get("food", 1.0)
        arr[i]["wealth"] = s.get("wealth", 0.0)
        arr[i]["defense"] = s.get("defense", 1.0)
        arr[i]["tech_level"] = s.get("tech_level", 1.0)
        arr[i]["has_port"] = s.get("has_port", False)
        arr[i]["has_longship"] = s.get("has_longship", False)
        arr[i]["owner_id"] = s.get("owner_id", i)
        arr[i]["alive"] = s.get("alive", True)
    return arr


class Simulator:
    def __init__(
        self,
        grid: np.ndarray,
        settlements: list[dict],
        params: dict | None = None,
        seed: int | None = None,
    ):
        self.grid = grid.copy()
        self.H, self.W = self.grid.shape
        self.sett = _make_settlement_array(settlements)
        self.p = {**DEFAULT_PARAMS, **(params or {})}
        self.rng: Generator = default_rng(seed)
        self.year = 0

        self._rebuild_pos_index()

        # Pre-compute static masks
        ocean_mask = self.grid == OCEAN
        self._coastal = np.zeros((self.H, self.W), dtype=np.bool_)
        for dy, dx in _OFFSETS_8:
            shifted = np.roll(np.roll(ocean_mask, dy, axis=0), dx, axis=1)
            self._coastal |= shifted
        self._coastal &= ~ocean_mask

        self._recount_adjacency()

    def _rebuild_pos_index(self) -> None:
        self._pos_to_idx: dict[tuple[int, int], int] = {}
        for i in range(len(self.sett)):
            if self.sett[i]["alive"]:
                pos = (int(self.sett[i]["y"]), int(self.sett[i]["x"]))
                self._pos_to_idx[pos] = i

    def _recount_adjacency(self) -> None:
        f = (self.grid == FOREST).astype(np.int32)
        p = ((self.grid == PLAINS) | (self.grid == EMPTY)).astype(np.int32)
        self._adj_forest = np.zeros((self.H, self.W), dtype=np.int32)
        self._adj_plains = np.zeros((self.H, self.W), dtype=np.int32)
        for dy, dx in _OFFSETS_8:
            self._adj_forest += np.roll(np.roll(f, dy, axis=0), dx, axis=1)
            self._adj_plains += np.roll(np.roll(p, dy, axis=0), dx, axis=1)

    def _alive_indices(self) -> np.ndarray:
        return np.where(self.sett["alive"])[0]

    def _add_settlement(
        self, y: int, x: int, owner_id: int,
        population: float = 0.5, food: float = 0.3,
        has_port: bool = False,
    ) -> None:
        new = np.zeros(1, dtype=_SETTLEMENT_DTYPE)
        new[0]["y"] = y
        new[0]["x"] = x
        new[0]["population"] = population
        new[0]["food"] = food
        new[0]["wealth"] = 0.0
        new[0]["defense"] = 0.3
        new[0]["tech_level"] = 1.0
        new[0]["has_port"] = has_port
        new[0]["has_longship"] = False
        new[0]["owner_id"] = owner_id
        new[0]["alive"] = True
        self.sett = np.concatenate([self.sett, new])
        terrain = PORT if has_port else SETTLEMENT
        self.grid[y, x] = terrain
        self._pos_to_idx[(y, x)] = len(self.sett) - 1

    # ------------------------------------------------------------------
    # Phase 1: Growth — logistic growth with carrying capacity
    # ------------------------------------------------------------------

    def _phase_growth(self) -> None:
        p = self.p
        alive = self._alive_indices()
        if len(alive) == 0:
            return

        s = self.sett
        rng = self.rng

        for i in alive:
            y, x = int(s[i]["y"]), int(s[i]["x"])

            # Food is a flow, not a stock — produced and consumed each year
            # Reset food to this year's net production
            food_production = (
                self._adj_forest[y, x] * p["food_per_forest"]
                + self._adj_plains[y, x] * p["food_per_plains"]
            )
            # Consumption scales with population
            food_needed = s[i]["population"] * p["food_consumption"]
            # Food is net surplus/deficit THIS YEAR, with small carry-over
            s[i]["food"] = s[i]["food"] * 0.3 + (food_production - food_needed)

            # Logistic population growth (carrying capacity limits growth)
            pop = s[i]["population"]
            cap = p["carrying_capacity"]
            if s[i]["food"] > 0 and pop < cap:
                growth = p["growth_rate"] * pop * (1.0 - pop / cap)
                s[i]["population"] = pop + growth
            elif s[i]["food"] < -0.1:
                # Starvation: population declines proportionally to deficit
                decline = min(0.15, 0.05 * abs(s[i]["food"]))
                s[i]["population"] *= (1.0 - decline)

            # Wealth: essentially zero (GT mean = 0.01)
            s[i]["wealth"] = max(0, s[i]["wealth"] * 0.8 + 0.001 * s[i]["population"])

            # Defense: proportional to population
            s[i]["defense"] = min(s[i]["population"] * 0.4, 1.5)

            # Port development (rare — GT shows only 7.5% of coastal)
            if not s[i]["has_port"] and self._coastal[y, x]:
                if s[i]["population"] >= p["port_threshold"]:
                    if rng.random() < p["port_prob"]:
                        s[i]["has_port"] = True
                        self.grid[y, x] = PORT

            # Longship construction (also rare)
            if not s[i]["has_longship"] and s[i]["has_port"]:
                if s[i]["population"] >= p["longship_threshold"]:
                    if rng.random() < p["longship_prob"]:
                        s[i]["has_longship"] = True

        # Founding new settlements
        founders = [
            i for i in alive
            if s[i]["population"] >= p["founding_threshold"]
        ]
        rng.shuffle(np.array(founders)) if founders else None

        for i in founders:
            if rng.random() > p["founding_prob"]:
                continue
            y0, x0 = int(s[i]["y"]), int(s[i]["x"])
            fr = int(p["founding_range"])

            candidates = []
            for dy in range(-fr, fr + 1):
                for dx in range(-fr, fr + 1):
                    ny, nx = y0 + dy, x0 + dx
                    if 0 <= ny < self.H and 0 <= nx < self.W:
                        if self.grid[ny, nx] in BUILDABLE_TERRAIN:
                            if (ny, nx) not in self._pos_to_idx:
                                candidates.append((ny, nx))

            if candidates:
                # Prefer closer cells AND forest cells (better food sources)
                dists = np.array([max(abs(c[0]-y0), abs(c[1]-x0)) for c in candidates])
                is_forest = np.array([1.0 if self.grid[c[0], c[1]] == FOREST else 0.5 for c in candidates])
                weights = is_forest / (dists + 0.5)
                weights /= weights.sum()
                idx = rng.choice(len(candidates), p=weights)
                cy, cx = candidates[idx]

                cost = p["founding_cost"]
                coastal = self._coastal[cy, cx]
                self._add_settlement(
                    cy, cx,
                    owner_id=int(s[i]["owner_id"]),
                    population=cost * s[i]["population"],
                    food=cost * s[i]["food"] if s[i]["food"] > 0 else 0.1,
                    has_port=False,  # new settlements don't start as ports
                )
                s[i]["population"] *= (1 - cost)
                s[i]["food"] *= (1 - cost)

    # ------------------------------------------------------------------
    # Phase 2: Conflict
    # ------------------------------------------------------------------

    def _phase_conflict(self) -> None:
        p = self.p
        s = self.sett
        alive = self._alive_indices()
        if len(alive) < 2:
            return

        rng = self.rng
        positions = np.column_stack([s[alive]["y"], s[alive]["x"]]).astype(np.float32)

        for idx_a, i in enumerate(alive):
            # Base raid probability
            if rng.random() > p["raid_prob"]:
                continue

            raid_range = p["raid_range"]
            if s[i]["has_longship"]:
                raid_range += p["longship_raid_bonus"]

            desperate = s[i]["food"] < 0
            mult = p["desperate_raid_mult"] if desperate else 1.0

            ya, xa = positions[idx_a]
            dists = np.maximum(
                np.abs(positions[:, 0] - ya), np.abs(positions[:, 1] - xa)
            )
            in_range = np.where((dists > 0) & (dists <= raid_range))[0]

            if len(in_range) == 0:
                continue

            targets_idx = alive[in_range]
            diff_faction = [
                j for j in targets_idx if s[j]["owner_id"] != s[i]["owner_id"]
            ]
            if diff_faction:
                j = diff_faction[rng.integers(len(diff_faction))]
            elif desperate:
                j = targets_idx[rng.integers(len(targets_idx))]
            else:
                continue

            attack = s[i]["population"] * p["raid_strength"] * mult
            defend = s[j]["defense"] + 0.3 * s[j]["population"]

            if attack > defend * (0.5 + rng.random()):
                # Successful raid — significant damage
                loot_food = min(s[j]["food"] * 0.4, 0.5)
                s[i]["food"] += loot_food
                s[j]["food"] -= loot_food
                s[j]["population"] *= 0.8  # heavy population loss
                s[j]["defense"] *= 0.6

                if s[j]["population"] < 0.3 and rng.random() < 0.5:
                    s[j]["owner_id"] = s[i]["owner_id"]
            else:
                s[i]["population"] *= 0.9
                s[i]["defense"] *= 0.8

    # ------------------------------------------------------------------
    # Phase 3: Trade (minimal — GT shows wealth ≈ 0)
    # ------------------------------------------------------------------

    def _phase_trade(self) -> None:
        p = self.p
        s = self.sett
        alive = self._alive_indices()
        ports = [i for i in alive if s[i]["has_port"]]
        if len(ports) < 2:
            return

        port_arr = np.array(ports)
        pos = np.column_stack([s[port_arr]["y"], s[port_arr]["x"]]).astype(np.float32)

        for idx_a in range(len(ports)):
            i = ports[idx_a]
            ya, xa = pos[idx_a]
            dists = np.maximum(
                np.abs(pos[:, 0] - ya), np.abs(pos[:, 1] - xa)
            )
            in_range = np.where((dists > 0) & (dists <= p["trade_range"]))[0]

            for idx_b in in_range:
                j = ports[idx_b]
                if s[i]["owner_id"] != s[j]["owner_id"]:
                    continue
                if i < j:
                    s[i]["food"] += p["trade_food"] * 0.5
                    s[j]["food"] += p["trade_food"] * 0.5

    # ------------------------------------------------------------------
    # Phase 4: Winter — harsh, many settlements should die
    # ------------------------------------------------------------------

    def _phase_winter(self) -> None:
        p = self.p
        s = self.sett
        alive = self._alive_indices()
        if len(alive) == 0:
            return

        rng = self.rng
        severity = p["winter_severity"] + rng.uniform(
            -p["winter_variance"], p["winter_variance"]
        )
        severity = max(severity, 0.05)

        # Winter food cost: base cost + per-capita cost
        # This ensures even small settlements face survival pressure
        for i in alive:
            base_cost = severity * 0.6  # fixed cost of surviving winter
            per_cap = severity * 0.4 * s[i]["population"]
            s[i]["food"] -= (base_cost + per_cap)

            # Population hit from harsh winter
            if severity > 0.4:
                s[i]["population"] *= (1.0 - 0.05 * (severity - 0.3))

        # Collapse: any settlement with negative food faces risk each year
        # Over 50 years, this gives ~69% collapse rate when food is marginal
        collapsed = []
        for i in alive:
            if s[i]["population"] < 0.1:
                collapsed.append(i)
            elif s[i]["food"] < p["collapse_threshold"]:
                # Base collapse probability per year when food is negative
                # We need ~69% to collapse over 50 years
                collapse_prob = 0.04 + 0.15 * max(0, -s[i]["food"])
                if s[i]["population"] < p["min_pop_survive"]:
                    collapse_prob += 0.05
                collapse_prob = min(collapse_prob, 0.5)
                if rng.random() < collapse_prob:
                    collapsed.append(i)

        for i in collapsed:
            y, x = int(s[i]["y"]), int(s[i]["x"])
            s[i]["alive"] = False
            self.grid[y, x] = RUIN
            self._pos_to_idx.pop((y, x), None)

    # ------------------------------------------------------------------
    # Phase 5: Environment — ruin reclamation + forest clearing
    # ------------------------------------------------------------------

    def _phase_environment(self) -> None:
        p = self.p
        rng = self.rng

        # --- Ruin reclamation (fast — GT shows only 2.4% ruin rate) ---
        ruin_ys, ruin_xs = np.where(self.grid == RUIN)
        alive = self._alive_indices()
        s = self.sett

        if len(alive) > 0:
            alive_pos = np.column_stack([s[alive]["y"], s[alive]["x"]]).astype(np.float32)
            alive_pop = s[alive]["population"]
            alive_owners = s[alive]["owner_id"]
        else:
            alive_pos = np.empty((0, 2), dtype=np.float32)
            alive_pop = np.empty(0)
            alive_owners = np.empty(0, dtype=np.int32)

        for ry, rx in zip(ruin_ys, ruin_xs):
            ry, rx = int(ry), int(rx)

            # Rebuild by thriving neighbor
            if len(alive_pos) > 0:
                dists = np.maximum(
                    np.abs(alive_pos[:, 0] - ry),
                    np.abs(alive_pos[:, 1] - rx),
                )
                near = np.where((dists <= 3) & (alive_pop > 1.5))[0]
                if len(near) > 0 and rng.random() < p["rebuild_prob"]:
                    patron = near[rng.integers(len(near))]
                    patron_idx = alive[patron]
                    self._add_settlement(
                        ry, rx,
                        owner_id=int(alive_owners[patron]),
                        population=0.3,
                        food=0.2,
                        has_port=False,
                    )
                    continue

            # Natural reclamation — ruins become forest or empty
            if rng.random() < p["ruin_reclaim_prob"]:
                # Adjacent forest influences outcome
                adj_f = self._adj_forest[ry, rx]
                forest_prob = min(0.7, 0.25 + 0.08 * adj_f)
                if rng.random() < forest_prob:
                    self.grid[ry, rx] = FOREST
                else:
                    self.grid[ry, rx] = EMPTY

        # --- Forest clearing near settlements ---
        # GT: forests near settlements (d≤3) → 71.4% forest, 9.3% empty, 16.8% settlement
        # Most forest change near settlements is due to FOUNDING, not explicit clearing
        # So keep forest_clear_prob very low; founding handles the forest→settlement transition
        if len(alive_pos) > 0 and p["forest_clear_prob"] > 0:
            forest_ys, forest_xs = np.where(self.grid == FOREST)
            for fy, fx in zip(forest_ys, forest_xs):
                fy, fx = int(fy), int(fx)
                dists = np.maximum(
                    np.abs(alive_pos[:, 0] - fy),
                    np.abs(alive_pos[:, 1] - fx),
                )
                min_dist = dists.min()
                if min_dist <= 1 and rng.random() < p["forest_clear_prob"]:
                    self.grid[fy, fx] = EMPTY

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> None:
        self._phase_growth()
        self._phase_conflict()
        self._phase_trade()
        self._phase_winter()
        self._phase_environment()
        self._recount_adjacency()
        self.year += 1

    def run(self, years: int = 50) -> np.ndarray:
        for _ in range(years):
            self.step()
        return self.get_grid()

    def get_grid(self) -> np.ndarray:
        return self.grid.copy()

    def get_settlements(self) -> list[dict]:
        alive = self._alive_indices()
        result = []
        for i in alive:
            s = self.sett[i]
            result.append({
                "y": int(s["y"]),
                "x": int(s["x"]),
                "population": float(s["population"]),
                "food": float(s["food"]),
                "wealth": float(s["wealth"]),
                "defense": float(s["defense"]),
                "tech_level": float(s["tech_level"]),
                "has_port": bool(s["has_port"]),
                "has_longship": bool(s["has_longship"]),
                "owner_id": int(s["owner_id"]),
                "alive": True,
            })
        return result


# ---------------------------------------------------------------------------
# Monte-Carlo helper
# ---------------------------------------------------------------------------

_LUT = np.zeros(12, dtype=np.int32)
_LUT[OCEAN] = 0
_LUT[PLAINS] = 0
_LUT[EMPTY] = 0
_LUT[SETTLEMENT] = 1
_LUT[PORT] = 2
_LUT[RUIN] = 3
_LUT[FOREST] = 4
_LUT[MOUNTAIN] = 5


def _run_sim_batch(args: tuple) -> np.ndarray:
    grid, settlements, params, years, seed_start, n = args
    H, W = grid.shape
    counts = np.zeros((H, W, 6), dtype=np.int32)
    for i in range(n):
        sim = Simulator(grid, settlements, params=params, seed=seed_start + i)
        final_grid = sim.run(years)
        class_grid = _LUT[final_grid]
        for c in range(6):
            counts[:, :, c] += (class_grid == c)
    return counts


def monte_carlo_predict(
    grid: np.ndarray,
    settlements: list[dict],
    n_sims: int = 1000,
    years: int = 50,
    params: dict | None = None,
    base_seed: int = 42,
    n_workers: int | None = None,
) -> np.ndarray:
    """Run sims and return (H, W, 6) probability tensor."""
    import multiprocessing as mp
    from scipy.special import digamma as _digamma

    H, W = grid.shape

    if n_workers is None:
        n_workers = min(mp.cpu_count(), n_sims)

    batch_size = n_sims // n_workers
    remainder = n_sims % n_workers
    batches = []
    seed_offset = base_seed
    for w in range(n_workers):
        n = batch_size + (1 if w < remainder else 0)
        if n > 0:
            batches.append((grid, settlements, params, years, seed_offset, n))
            seed_offset += n

    if n_workers > 1 and n_sims >= 4:
        with mp.Pool(n_workers) as pool:
            results = pool.map(_run_sim_batch, batches)
    else:
        results = [_run_sim_batch(b) for b in batches]

    counts = sum(results)

    alpha = 0.5  # Jeffreys prior
    beta = counts.astype(np.float64) + alpha
    beta_sum = beta.sum(axis=2, keepdims=True)
    log_q = _digamma(beta) - _digamma(beta_sum)
    probs = np.exp(log_q)
    probs = np.maximum(probs, 1e-6)
    probs /= probs.sum(axis=2, keepdims=True)
    return probs.astype(np.float32)
