"""Local approximation of the Astar Island Norse civilization simulator.

Designed for speed: runs thousands of 50-year simulations to build
empirical probability distributions over final grid states.

Terrain codes: Ocean(10), Plains(11), Empty(0), Settlement(1),
Port(2), Ruin(3), Forest(4), Mountain(5).
"""
from __future__ import annotations

import numpy as np
from numpy.random import Generator, default_rng

# ---------------------------------------------------------------------------
# Terrain constants
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Default simulation parameters (reasonable guesses for hidden params)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS: dict[str, float] = {
    "food_per_forest": 0.3,
    "food_per_plains": 0.1,
    "growth_rate": 0.1,
    "port_threshold": 2.0,
    "longship_threshold": 3.0,
    "founding_threshold": 4.0,
    "founding_range": 5,
    "raid_range": 3,
    "longship_raid_bonus": 4,
    "raid_strength": 0.3,
    "desperate_raid_mult": 2.0,
    "trade_range": 5,
    "trade_food": 0.2,
    "trade_wealth": 0.1,
    "tech_diffusion": 0.05,
    "winter_severity": 0.4,
    "winter_variance": 0.2,
    "collapse_threshold": -0.5,
    "ruin_reclaim_prob": 0.1,
    "rebuild_prob": 0.15,
}

# Neighbour offsets (8-connected)
_OFFSETS_8 = np.array(
    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
    dtype=np.int32,
)


# ---------------------------------------------------------------------------
# Settlement data stored as structured numpy arrays for speed
# ---------------------------------------------------------------------------
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
    """Convert list-of-dicts to structured numpy array."""
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


# ---------------------------------------------------------------------------
# Pre-computed adjacency helpers
# ---------------------------------------------------------------------------

def _adjacent_terrain_counts(grid: np.ndarray, y: int, x: int) -> dict[int, int]:
    """Count each terrain type in the 8-neighbourhood of (y, x)."""
    h, w = grid.shape
    counts: dict[int, int] = {}
    for dy, dx in _OFFSETS_8:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w:
            t = int(grid[ny, nx])
            counts[t] = counts.get(t, 0) + 1
    return counts


def _is_coastal(grid: np.ndarray, y: int, x: int) -> bool:
    h, w = grid.shape
    for dy, dx in _OFFSETS_8:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == OCEAN:
            return True
    return False


def _dist(y1: int, x1: int, y2: int, x2: int) -> float:
    """Chebyshev (L-inf) distance."""
    return max(abs(y1 - y2), abs(x1 - x2))


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class Simulator:
    """Fast local approximation of the Astar Island simulator.

    Parameters
    ----------
    grid : np.ndarray
        Initial 40x40 terrain grid (terrain codes).
    settlements : list[dict]
        Settlement dicts from the API.
    params : dict | None
        Simulation parameters. Missing keys filled from DEFAULT_PARAMS.
    seed : int | None
        Random seed for reproducibility.
    """

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

        # Build a lookup: (y, x) -> index in self.sett (only alive)
        self._rebuild_pos_index()

        # Pre-compute coastal mask (static — ocean never changes)
        self._coastal = np.zeros((self.H, self.W), dtype=np.bool_)
        ocean_mask = self.grid == OCEAN
        for dy, dx in _OFFSETS_8:
            shifted = np.roll(np.roll(ocean_mask, dy, axis=0), dx, axis=1)
            self._coastal |= shifted
        # Exclude ocean cells themselves
        self._coastal &= ~ocean_mask

        # Pre-compute adjacent forest / plains counts for all cells (static
        # terrain is cached; dynamic terrain recounted each year).
        self._adj_forest = np.zeros((self.H, self.W), dtype=np.int32)
        self._adj_plains = np.zeros((self.H, self.W), dtype=np.int32)
        self._recount_adjacency()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rebuild_pos_index(self) -> None:
        self._pos_to_idx: dict[tuple[int, int], int] = {}
        for i in range(len(self.sett)):
            if self.sett[i]["alive"]:
                pos = (int(self.sett[i]["y"]), int(self.sett[i]["x"]))
                self._pos_to_idx[pos] = i

    def _recount_adjacency(self) -> None:
        """Recount adjacent forest and plains for every cell."""
        f = (self.grid == FOREST).astype(np.int32)
        p = ((self.grid == PLAINS) | (self.grid == EMPTY)).astype(np.int32)
        self._adj_forest[:] = 0
        self._adj_plains[:] = 0
        for dy, dx in _OFFSETS_8:
            self._adj_forest += np.roll(np.roll(f, dy, axis=0), dx, axis=1)
            self._adj_plains += np.roll(np.roll(p, dy, axis=0), dx, axis=1)

    def _alive_indices(self) -> np.ndarray:
        return np.where(self.sett["alive"])[0]

    def _add_settlement(
        self, y: int, x: int, owner_id: int,
        population: float = 0.5, food: float = 0.5,
        has_port: bool = False,
    ) -> None:
        """Append a new settlement to the array."""
        new = np.zeros(1, dtype=_SETTLEMENT_DTYPE)
        new[0]["y"] = y
        new[0]["x"] = x
        new[0]["population"] = population
        new[0]["food"] = food
        new[0]["wealth"] = 0.0
        new[0]["defense"] = 0.5
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
    # Phase 1: Growth
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

            # Food production from adjacent terrain
            food_gain = (
                self._adj_forest[y, x] * p["food_per_forest"]
                + self._adj_plains[y, x] * p["food_per_plains"]
            )
            s[i]["food"] += food_gain

            # Population growth when food is sufficient
            if s[i]["food"] > 0:
                s[i]["population"] += p["growth_rate"] * s[i]["population"] * (
                    1.0 + 0.1 * s[i]["tech_level"]
                )

            # Wealth accumulates slowly with population
            s[i]["wealth"] += 0.02 * s[i]["population"]

            # Defense scales with population
            s[i]["defense"] = max(s[i]["defense"], 0.3 * s[i]["population"])

            # Port development (coastal + population threshold)
            if not s[i]["has_port"] and self._coastal[y, x]:
                if s[i]["population"] >= p["port_threshold"]:
                    if rng.random() < 0.3:
                        s[i]["has_port"] = True
                        self.grid[y, x] = PORT

            # Longship construction
            if not s[i]["has_longship"] and s[i]["has_port"]:
                if (s[i]["population"] + s[i]["wealth"]) >= p["longship_threshold"]:
                    if rng.random() < 0.2:
                        s[i]["has_longship"] = True

        # Founding new settlements (longship owners with high stats)
        founders = [
            i for i in alive
            if s[i]["has_longship"]
            and (s[i]["population"] + s[i]["wealth"]) >= p["founding_threshold"]
        ]
        rng.shuffle(np.array(founders))  # randomise order

        for i in founders:
            if rng.random() > 0.15:  # founding is rare per year
                continue
            y0, x0 = int(s[i]["y"]), int(s[i]["x"])
            fr = int(p["founding_range"])

            # Collect candidate cells
            candidates = []
            for dy in range(-fr, fr + 1):
                for dx in range(-fr, fr + 1):
                    ny, nx = y0 + dy, x0 + dx
                    if 0 <= ny < self.H and 0 <= nx < self.W:
                        if self.grid[ny, nx] in BUILDABLE_TERRAIN:
                            if (ny, nx) not in self._pos_to_idx:
                                candidates.append((ny, nx))

            if candidates:
                cy, cx = candidates[rng.integers(len(candidates))]
                coastal = self._coastal[cy, cx]
                self._add_settlement(
                    cy, cx,
                    owner_id=int(s[i]["owner_id"]),
                    population=0.3 * s[i]["population"],
                    food=0.3 * s[i]["food"],
                    has_port=coastal and rng.random() < 0.3,
                )
                # Founding costs the parent
                s[i]["population"] *= 0.7
                s[i]["food"] *= 0.7

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
        # Pre-compute positions for distance checks
        positions = np.column_stack([s[alive]["y"], s[alive]["x"]]).astype(np.float32)

        for idx_a, i in enumerate(alive):
            raid_range = p["raid_range"]
            if s[i]["has_longship"]:
                raid_range += p["longship_raid_bonus"]

            desperate = s[i]["food"] < 0
            mult = p["desperate_raid_mult"] if desperate else 1.0

            # Only raid if there is some reason (desperate or wealthy targets)
            if not desperate and rng.random() > 0.3:
                continue

            ya, xa = positions[idx_a]
            dists = np.maximum(
                np.abs(positions[:, 0] - ya), np.abs(positions[:, 1] - xa)
            )
            in_range = np.where((dists > 0) & (dists <= raid_range))[0]

            if len(in_range) == 0:
                continue

            # Pick a target — prefer different faction
            targets_idx = alive[in_range]
            diff_faction = [
                j for j in targets_idx if s[j]["owner_id"] != s[i]["owner_id"]
            ]
            if diff_faction:
                j = diff_faction[rng.integers(len(diff_faction))]
            elif desperate:
                j = targets_idx[rng.integers(len(targets_idx))]
            else:
                continue  # don't raid own faction unless desperate

            # Resolve raid
            attack = (
                s[i]["population"] * p["raid_strength"] * mult
                * (1.0 + 0.1 * s[i]["tech_level"])
            )
            defend = s[j]["defense"] + 0.2 * s[j]["population"]

            if attack > defend * (0.5 + rng.random()):
                # Successful raid
                loot_food = min(s[j]["food"] * 0.3, 1.0)
                loot_wealth = min(s[j]["wealth"] * 0.3, 0.5)
                s[i]["food"] += loot_food
                s[i]["wealth"] += loot_wealth
                s[j]["food"] -= loot_food
                s[j]["wealth"] -= loot_wealth
                s[j]["defense"] *= 0.7
                s[j]["population"] *= 0.9

                # Allegiance shift for weak settlements
                if s[j]["population"] < 0.5 and rng.random() < 0.4:
                    s[j]["owner_id"] = s[i]["owner_id"]
            else:
                # Failed raid — attacker takes some losses
                s[i]["population"] *= 0.95
                s[i]["defense"] *= 0.9

    # ------------------------------------------------------------------
    # Phase 3: Trade
    # ------------------------------------------------------------------

    def _phase_trade(self) -> None:
        p = self.p
        s = self.sett
        alive = self._alive_indices()

        # Only ports trade
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
            in_range = np.where(
                (dists > 0) & (dists <= p["trade_range"])
            )[0]

            for idx_b in in_range:
                j = ports[idx_b]
                # Only trade with same faction (not at war)
                if s[i]["owner_id"] != s[j]["owner_id"]:
                    continue

                # Trade benefits (applied symmetrically once — we iterate
                # both directions so halve the amounts)
                if i < j:  # process each pair once
                    s[i]["food"] += p["trade_food"] * 0.5
                    s[j]["food"] += p["trade_food"] * 0.5
                    s[i]["wealth"] += p["trade_wealth"] * 0.5
                    s[j]["wealth"] += p["trade_wealth"] * 0.5

                    # Tech diffusion
                    avg_tech = (s[i]["tech_level"] + s[j]["tech_level"]) * 0.5
                    s[i]["tech_level"] += (avg_tech - s[i]["tech_level"]) * p["tech_diffusion"]
                    s[j]["tech_level"] += (avg_tech - s[j]["tech_level"]) * p["tech_diffusion"]

    # ------------------------------------------------------------------
    # Phase 4: Winter
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

        # Food loss for all settlements
        s["food"][alive] -= severity * (1.0 + 0.05 * s["population"][alive])

        # Check for collapse
        collapsed = []
        for i in alive:
            if s[i]["food"] < p["collapse_threshold"] and s[i]["population"] < 1.0:
                collapsed.append(i)
            elif s[i]["population"] < 0.1:
                collapsed.append(i)

        for i in collapsed:
            y, x = int(s[i]["y"]), int(s[i]["x"])
            s[i]["alive"] = False
            self.grid[y, x] = RUIN
            self._pos_to_idx.pop((y, x), None)

            # Disperse population to nearby friendly settlements
            pop = s[i]["population"]
            owner = s[i]["owner_id"]
            if pop <= 0:
                continue
            alive_now = self._alive_indices()
            friendly = [
                j for j in alive_now
                if s[j]["owner_id"] == owner
                and _dist(int(s[j]["y"]), int(s[j]["x"]), y, x) <= 5
            ]
            if friendly:
                share = pop / len(friendly)
                for j in friendly:
                    s[j]["population"] += share

    # ------------------------------------------------------------------
    # Phase 5: Environment
    # ------------------------------------------------------------------

    def _phase_environment(self) -> None:
        p = self.p
        rng = self.rng

        # Find ruin cells
        ruin_ys, ruin_xs = np.where(self.grid == RUIN)
        if len(ruin_ys) == 0:
            return

        alive = self._alive_indices()
        s = self.sett
        if len(alive) > 0:
            alive_pos = np.column_stack([s[alive]["y"], s[alive]["x"]]).astype(np.float32)
            alive_pop = s[alive]["population"]
            alive_owners = s[alive]["owner_id"]
            alive_ports = s[alive]["has_port"]
        else:
            alive_pos = np.empty((0, 2), dtype=np.float32)
            alive_pop = np.empty(0, dtype=np.float32)
            alive_owners = np.empty(0, dtype=np.int32)
            alive_ports = np.empty(0, dtype=np.bool_)

        for ry, rx in zip(ruin_ys, ruin_xs):
            ry, rx = int(ry), int(rx)

            # Check for thriving neighbor that can rebuild
            if len(alive_pos) > 0:
                dists = np.maximum(
                    np.abs(alive_pos[:, 0] - ry),
                    np.abs(alive_pos[:, 1] - rx),
                )
                near = np.where((dists <= 3) & (alive_pop > 2.0))[0]

                if len(near) > 0 and rng.random() < p["rebuild_prob"]:
                    patron = near[rng.integers(len(near))]
                    patron_idx = alive[patron]
                    coastal = self._coastal[ry, rx]
                    is_port = coastal and bool(alive_ports[patron]) and rng.random() < 0.4

                    self._add_settlement(
                        ry, rx,
                        owner_id=int(alive_owners[patron]),
                        population=0.5,
                        food=0.3,
                        has_port=is_port,
                    )
                    continue

            # Unreclaimed ruins decay
            if rng.random() < p["ruin_reclaim_prob"]:
                if self._coastal[ry, rx] and rng.random() < 0.3:
                    self.grid[ry, rx] = PLAINS
                elif rng.random() < 0.6:
                    self.grid[ry, rx] = FOREST
                else:
                    self.grid[ry, rx] = PLAINS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Run one year of simulation (all 5 phases)."""
        self._phase_growth()
        self._phase_conflict()
        self._phase_trade()
        self._phase_winter()
        self._phase_environment()
        self._recount_adjacency()
        self.year += 1

    def run(self, years: int = 50) -> np.ndarray:
        """Run *years* of simulation and return the final grid."""
        for _ in range(years):
            self.step()
        return self.get_grid()

    def get_grid(self) -> np.ndarray:
        """Return a copy of the current terrain grid."""
        return self.grid.copy()

    def get_settlements(self) -> list[dict]:
        """Return alive settlements as list of dicts."""
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

# Lookup table: terrain code -> prediction class
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
    """Run a batch of simulations and return counts array. Used by multiprocessing."""
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
    """Run *n_sims* simulations and return (H, W, C) probability tensor.

    Terrain codes are mapped to prediction classes:
        0 -> Empty/Ocean/Plains   (class 0)
        1 -> Settlement           (class 1)
        2 -> Port                 (class 2)
        3 -> Ruin                 (class 3)
        4 -> Forest               (class 4)
        5 -> Mountain             (class 5)

    Uses multiprocessing to parallelize across CPU cores.
    Returns array of shape (H, W, 6) with normalised probabilities.
    """
    import multiprocessing as mp

    H, W = grid.shape

    if n_workers is None:
        n_workers = min(mp.cpu_count(), n_sims)

    # Split sims into batches for each worker
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

    # Convert counts to probabilities using digamma estimator
    # (KL-optimal under Dirichlet posterior with Jeffreys prior alpha=0.5)
    from scipy.special import digamma as _digamma
    alpha = 0.5  # Jeffreys prior
    beta = counts.astype(np.float64) + alpha  # posterior params
    beta_sum = beta.sum(axis=2, keepdims=True)
    log_q = _digamma(beta) - _digamma(beta_sum)
    probs = np.exp(log_q)
    probs = np.maximum(probs, 1e-6)
    probs /= probs.sum(axis=2, keepdims=True)
    return probs.astype(np.float32)
