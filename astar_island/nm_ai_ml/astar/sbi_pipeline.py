"""Simulation-Based Inference pipeline for Astar Island.

1. Sensitivity map: which cells change most across parameter settings
2. SBI with NPE: sample prior → run sim → extract stats → train neural posterior
3. Inference: observations → summary stats → posterior params → Monte Carlo predictions

Usage:
    from nm_ai_ml.astar.sbi_pipeline import (
        compute_sensitivity_map,
        build_sbi_training_data,
        train_npe,
        infer_posterior_params,
        predict_with_posterior,
    )
"""
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from nm_ai_ml.astar.simulator import Simulator, DEFAULT_PARAMS, _LUT, BUILDABLE_TERRAIN, PORT, RUIN

logger = logging.getLogger(__name__)

GRID_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}

# ============================================================
# Prior over simulator parameters
# ============================================================

PARAM_PRIOR = {
    # (name, low, high) — uniform prior ranges
    # Calibrated from observing historical rounds
    "growth_rate":       (0.02, 0.15),
    "winter_severity":   (0.10, 0.60),
    "food_per_forest":   (0.05, 0.35),
    "food_per_plains":   (0.01, 0.10),
    "founding_threshold":(1.0,  5.0),
    "founding_prob":     (0.02, 0.30),
    "founding_range":    (3,    8),
    "pop_cap":           (3.0,  8.0),
    "raid_strength":     (0.05, 0.60),
    "port_threshold":    (0.5,  4.0),
    "collapse_threshold":(-1.5, 0.0),
    "rebuild_prob":      (0.02, 0.30),
    "ruin_reclaim_prob": (0.02, 0.20),
    "winter_variance":   (0.05, 0.30),
}

PARAM_NAMES = list(PARAM_PRIOR.keys())
N_PARAMS = len(PARAM_NAMES)


def sample_prior(rng: np.random.Generator, n: int = 1) -> np.ndarray:
    """Sample n parameter vectors from the prior.

    Returns array of shape (n, N_PARAMS).
    """
    samples = np.zeros((n, N_PARAMS))
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = PARAM_PRIOR[name]
        samples[:, i] = rng.uniform(lo, hi, size=n)
    return samples


def params_vector_to_dict(vec: np.ndarray) -> dict:
    """Convert parameter vector to dict."""
    params = dict(DEFAULT_PARAMS)
    for i, name in enumerate(PARAM_NAMES):
        params[name] = float(vec[i])
    return params


# ============================================================
# Improved local simulator (with food cap + logistic growth)
# ============================================================

class LocalSim(Simulator):
    """Local simulator with food cap and logistic growth."""

    def _phase_growth(self):
        p = self.p; alive = self._alive_indices()
        if len(alive) == 0: return
        s = self.sett; rng = self.rng
        for i in alive:
            y, x = int(s[i]["y"]), int(s[i]["x"])
            s[i]["food"] += (self._adj_forest[y, x] * p["food_per_forest"]
                            + self._adj_plains[y, x] * p["food_per_plains"])
            s[i]["food"] = min(float(s[i]["food"]), 1.0)
            if s[i]["food"] > 0:
                cap = p.get("pop_cap", 5.0)
                growth = p["growth_rate"] * s[i]["population"] * (1.0 - s[i]["population"] / cap)
                s[i]["population"] += max(growth, 0)
            s[i]["wealth"] += 0.02 * s[i]["population"]
            s[i]["defense"] = max(s[i]["defense"], 0.3 * s[i]["population"])
            if not s[i]["has_port"] and self._coastal[y, x]:
                if s[i]["population"] >= p["port_threshold"] and rng.random() < 0.3:
                    s[i]["has_port"] = True; self.grid[y, x] = PORT
            if not s[i]["has_longship"] and s[i]["has_port"]:
                if (s[i]["population"] + s[i]["wealth"]) >= p["longship_threshold"] and rng.random() < 0.2:
                    s[i]["has_longship"] = True
        for i in alive:
            if s[i]["population"] < p["founding_threshold"]: continue
            if rng.random() > p.get("founding_prob", 0.15): continue
            y0, x0 = int(s[i]["y"]), int(s[i]["x"])
            fr = int(p.get("founding_range", 5))
            cands = [(y0+dy, x0+dx) for dy in range(-fr, fr+1) for dx in range(-fr, fr+1)
                     if 0 <= y0+dy < self.H and 0 <= x0+dx < self.W
                     and self.grid[y0+dy, x0+dx] in BUILDABLE_TERRAIN
                     and (y0+dy, x0+dx) not in self._pos_to_idx]
            if cands:
                cy, cx = cands[rng.integers(len(cands))]
                self._add_settlement(cy, cx, owner_id=int(s[i]["owner_id"]),
                    population=0.3*s[i]["population"], food=0.3*s[i]["food"],
                    has_port=self._coastal[cy, cx] and rng.random() < 0.3)
                s[i]["population"] *= 0.7; s[i]["food"] *= 0.7

    def _phase_winter(self):
        p = self.p; s = self.sett; alive = self._alive_indices()
        if len(alive) == 0: return
        rng = self.rng
        severity = p["winter_severity"] + rng.uniform(
            -p.get("winter_variance", 0.2), p.get("winter_variance", 0.2))
        severity = max(severity, 0.05)
        s["food"][alive] -= severity * (1.0 + 0.05 * s["population"][alive])
        s["food"][alive] = np.minimum(s["food"][alive], 1.0)
        collapsed = []
        for i in alive:
            if s[i]["food"] < p["collapse_threshold"] and s[i]["population"] < 1.0:
                collapsed.append(i)
            elif s[i]["population"] < 0.1:
                collapsed.append(i)
        for i in collapsed:
            y, x = int(s[i]["y"]), int(s[i]["x"])
            s[i]["alive"] = False; self.grid[y, x] = RUIN
            self._pos_to_idx.pop((y, x), None)


# ============================================================
# Summary statistics extraction
# ============================================================

def extract_summary_stats(grid_final: np.ndarray, settlements_final: list[dict],
                          grid_initial: np.ndarray) -> np.ndarray:
    """Extract summary statistics from a simulation result.

    Returns a fixed-length vector of statistics that characterize
    the simulation outcome.
    """
    H, W = grid_final.shape
    class_grid = np.vectorize(lambda c: GRID_TO_CLASS.get(int(c), 0))(grid_final)

    # Class fractions
    total = H * W
    frac_empty = np.sum(class_grid == 0) / total
    frac_sett = np.sum(class_grid == 1) / total
    frac_port = np.sum(class_grid == 2) / total
    frac_ruin = np.sum(class_grid == 3) / total
    frac_forest = np.sum(class_grid == 4) / total
    frac_mtn = np.sum(class_grid == 5) / total

    # Settlement stats
    alive = [s for s in settlements_final if s.get("alive", True)]
    if alive:
        avg_pop = np.mean([s["population"] for s in alive])
        avg_food = np.mean([s["food"] for s in alive])
        avg_def = np.mean([s["defense"] for s in alive])
        avg_wealth = np.mean([s["wealth"] for s in alive])
        n_alive = len(alive)
        n_ports = sum(1 for s in alive if s.get("has_port"))
        port_rate = n_ports / n_alive

        # Faction stats
        owners = {}
        for s in alive:
            oid = s.get("owner_id", 0)
            owners[oid] = owners.get(oid, 0) + 1
        n_factions = len(owners)
        max_faction = max(owners.values()) if owners else 0
        faction_concentration = max_faction / n_alive if n_alive > 0 else 0
    else:
        avg_pop = avg_food = avg_def = avg_wealth = 0
        n_alive = 0; port_rate = 0; n_factions = 0; faction_concentration = 0

    # Terrain change from initial
    init_class = np.vectorize(lambda c: GRID_TO_CLASS.get(int(c), 0))(grid_initial)
    changed = np.sum(class_grid != init_class) / total

    stats = np.array([
        frac_empty, frac_sett, frac_port, frac_ruin, frac_forest, frac_mtn,
        avg_pop, avg_food, avg_def, avg_wealth,
        n_alive / total, port_rate, n_factions / max(n_alive, 1),
        faction_concentration, changed,
    ], dtype=np.float32)

    return stats

N_STATS = 15  # length of summary stats vector


def extract_stats_from_observations(observations: list[dict],
                                     grid_initial: np.ndarray) -> np.ndarray:
    """Extract summary stats from API observations (averaging across obs)."""
    all_stats = []
    for o in observations:
        # Build a pseudo-final grid from viewport
        # (only partial, but stats should be representative)
        settlements = [s for s in o.get("settlements", []) if s.get("alive")]

        alive = settlements
        if alive:
            avg_pop = np.mean([s["population"] for s in alive])
            avg_food = np.mean([s["food"] for s in alive])
            avg_def = np.mean([s["defense"] for s in alive])
            avg_wealth = np.mean([s["wealth"] for s in alive])
            n_ports = sum(1 for s in alive if s.get("has_port"))
            port_rate = n_ports / len(alive)
            owners = {}
            for s in alive:
                owners[s.get("owner_id", 0)] = owners.get(s.get("owner_id", 0), 0) + 1
            n_factions = len(owners)
            max_faction = max(owners.values()) if owners else 0
            faction_concentration = max_faction / len(alive)
        else:
            avg_pop = avg_food = avg_def = avg_wealth = 0
            port_rate = 0; n_factions = 0; faction_concentration = 0

        # Grid stats from viewport
        vp = o["viewport"]
        H, W = grid_initial.shape
        total_vp = 0; sett_vp = 0; ruin_vp = 0; forest_vp = 0; changed_vp = 0
        for dy in range(len(o["grid"])):
            for dx in range(len(o["grid"][0])):
                gy, gx = vp["y"]+dy, vp["x"]+dx
                if gy >= H or gx >= W: continue
                total_vp += 1
                obs_class = GRID_TO_CLASS.get(o["grid"][dy][dx], 0)
                init_class = GRID_TO_CLASS.get(int(grid_initial[gy, gx]), 0)
                if obs_class == 1: sett_vp += 1
                elif obs_class == 3: ruin_vp += 1
                elif obs_class == 4: forest_vp += 1
                if obs_class != init_class: changed_vp += 1

        if total_vp > 0:
            all_stats.append([
                sett_vp/total_vp, ruin_vp/total_vp, forest_vp/total_vp,
                avg_pop, avg_food, avg_def, avg_wealth,
                port_rate, faction_concentration, changed_vp/total_vp,
            ])

    if not all_stats:
        return np.zeros(N_STATS, dtype=np.float32)

    avg = np.mean(all_stats, axis=0)

    # Pad to N_STATS length
    stats = np.zeros(N_STATS, dtype=np.float32)
    # Map to same positions as extract_summary_stats
    stats[1] = avg[0]   # frac_sett
    stats[3] = avg[1]   # frac_ruin
    stats[4] = avg[2]   # frac_forest
    stats[6] = avg[3]   # avg_pop
    stats[7] = avg[4]   # avg_food
    stats[8] = avg[5]   # avg_def
    stats[9] = avg[6]   # avg_wealth
    stats[11] = avg[7]  # port_rate
    stats[13] = avg[8]  # faction_concentration
    stats[14] = avg[9]  # changed

    return stats


# ============================================================
# Sensitivity map
# ============================================================

def compute_sensitivity_map(
    grid: np.ndarray,
    settlements: list[dict],
    n_param_samples: int = 20,
    n_sims_per_param: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Compute per-cell parameter sensitivity.

    For each cell, measures how much the predicted class distribution
    changes across different parameter settings. High sensitivity =
    the cell's outcome depends on the hidden parameters.

    Returns (H, W) array of sensitivity values.
    """
    H, W = grid.shape
    rng = np.random.default_rng(seed)

    # Sample diverse parameter sets
    param_samples = sample_prior(rng, n_param_samples)

    # For each param set, run sims and compute class distributions
    all_predictions = []  # list of (H, W, 6) arrays

    for pi in range(n_param_samples):
        params = params_vector_to_dict(param_samples[pi])
        counts = np.zeros((H, W, 6), dtype=np.int32)

        for si in range(n_sims_per_param):
            setts = [dict(s) for s in settlements]
            for j, se in enumerate(setts):
                se["owner_id"] = j
            sim = LocalSim(grid, setts, params=params, seed=seed + pi * 100 + si)
            sim.run(50)
            cg = _LUT[sim.grid]
            for c in range(6):
                counts[:, :, c] += (cg == c)

        probs = counts.astype(np.float64) / n_sims_per_param
        all_predictions.append(probs)

    # Sensitivity = variance of predictions across parameter settings
    stacked = np.stack(all_predictions)  # (n_params, H, W, 6)
    # Per-cell variance across parameter settings, summed over classes
    sensitivity = np.var(stacked, axis=0).sum(axis=2)  # (H, W)

    return sensitivity


# ============================================================
# Neural Posterior Estimation (NPE)
# ============================================================

class NPENetwork(nn.Module):
    """Simple MLP that maps summary statistics → parameter posterior.

    Outputs mean and log-variance for each parameter (Gaussian posterior).
    """
    def __init__(self, n_stats: int = N_STATS, n_params: int = N_PARAMS,
                 hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_stats, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, n_params)
        self.logvar_head = nn.Linear(hidden, n_params)

    def forward(self, stats):
        h = self.net(stats)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        return mean, logvar

    def sample(self, stats, n_samples=100):
        """Sample from the posterior given summary statistics."""
        mean, logvar = self.forward(stats)
        std = torch.exp(0.5 * logvar)
        # Reparameterization trick
        eps = torch.randn(n_samples, mean.shape[-1])
        samples = mean.unsqueeze(0) + eps * std.unsqueeze(0)
        return samples  # (n_samples, n_params)


def _extract_viewport_stats(sim, grid_initial: np.ndarray,
                            rng: np.random.Generator,
                            n_viewports: int = 10) -> np.ndarray:
    """Extract stats from random viewports of a sim result.

    Mirrors exactly what inference does with real API observations:
    sample viewports, count cells, gather settlement stats.
    """
    H, W = sim.grid.shape
    max_vy, max_vx = H - 15, W - 15

    all_sett_cells, all_total_cells = 0, 0
    all_ruin_cells, all_forest_cells = 0, 0
    all_pops, all_foods, all_defs, all_wealths = [], [], [], []
    all_ports, all_alive = 0, 0
    all_changed = 0
    owners = {}

    alive_setts = sim.get_settlements()

    for _ in range(n_viewports):
        vy = rng.integers(0, max_vy + 1)
        vx = rng.integers(0, max_vx + 1)

        for dy in range(15):
            for dx in range(15):
                gy, gx = vy + dy, vx + dx
                if gy >= H or gx >= W:
                    continue
                all_total_cells += 1
                obs_class = GRID_TO_CLASS.get(int(sim.grid[gy, gx]), 0)
                init_class = GRID_TO_CLASS.get(int(grid_initial[gy, gx]), 0)
                if obs_class == 1: all_sett_cells += 1
                elif obs_class == 3: all_ruin_cells += 1
                elif obs_class == 4: all_forest_cells += 1
                if obs_class != init_class: all_changed += 1

        # Settlements in viewport
        for s in alive_setts:
            if vx <= s["x"] < vx + 15 and vy <= s["y"] < vy + 15:
                all_pops.append(s["population"])
                all_foods.append(s["food"])
                all_defs.append(s["defense"])
                all_wealths.append(s.get("wealth", 0))
                all_alive += 1
                if s.get("has_port"): all_ports += 1
                oid = s.get("owner_id", 0)
                owners[oid] = owners.get(oid, 0) + 1

    total = max(all_total_cells, 1)
    n_alive = max(all_alive, 1)
    n_factions = len(owners)
    max_faction = max(owners.values()) if owners else 0

    stats = np.zeros(N_STATS, dtype=np.float32)
    stats[0] = 1.0 - all_sett_cells/total - all_ruin_cells/total - all_forest_cells/total
    stats[1] = all_sett_cells / total
    stats[3] = all_ruin_cells / total
    stats[4] = all_forest_cells / total
    stats[6] = np.mean(all_pops) if all_pops else 0
    stats[7] = np.mean(all_foods) if all_foods else 0
    stats[8] = np.mean(all_defs) if all_defs else 0
    stats[9] = np.mean(all_wealths) if all_wealths else 0
    stats[10] = all_alive / total
    stats[11] = all_ports / n_alive
    stats[12] = n_factions / n_alive
    stats[13] = max_faction / n_alive
    stats[14] = all_changed / total

    return stats


def build_sbi_training_data(
    grid: np.ndarray,
    settlements: list[dict],
    n_samples: int = 500,
    n_sims_per_sample: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate training data for NPE by sampling prior and running sims.

    Returns (params_array, stats_array) of shape (n_samples, N_PARAMS)
    and (n_samples, N_STATS).
    """
    rng = np.random.default_rng(seed)
    param_samples = sample_prior(rng, n_samples)

    all_params = []
    all_stats = []

    for i in range(n_samples):
        params = params_vector_to_dict(param_samples[i])

        # Run sims and extract stats from SAMPLED VIEWPORTS
        # (mirrors what inference does with real API observations)
        stats_list = []
        for si in range(n_sims_per_sample):
            setts_copy = [dict(s) for s in settlements]
            for j, se in enumerate(setts_copy):
                se["owner_id"] = j
            sim = LocalSim(grid, setts_copy, params=params, seed=seed + i * 100 + si)
            sim.run(50)

            # Sample random viewports from sim output (like API observations)
            stats = _extract_viewport_stats(sim, grid, rng, n_viewports=10)
            stats_list.append(stats)

        avg_stats = np.mean(stats_list, axis=0)
        all_params.append(param_samples[i])
        all_stats.append(avg_stats)

        if (i + 1) % 50 == 0:
            logger.info("SBI training: %d/%d samples generated", i + 1, n_samples)

    return np.array(all_params), np.array(all_stats)


def train_npe(params_data: np.ndarray, stats_data: np.ndarray,
              n_epochs: int = 500, lr: float = 0.001) -> NPENetwork:
    """Train Neural Posterior Estimation network.

    Args:
        params_data: (n_samples, N_PARAMS) — normalized to [0,1]
        stats_data: (n_samples, N_STATS)

    Returns trained NPENetwork.
    """
    # Normalize params to [0,1] for training
    param_min = np.array([PARAM_PRIOR[n][0] for n in PARAM_NAMES])
    param_max = np.array([PARAM_PRIOR[n][1] for n in PARAM_NAMES])
    params_norm = (params_data - param_min) / (param_max - param_min + 1e-8)

    # Normalize stats
    stats_mean = stats_data.mean(axis=0)
    stats_std = stats_data.std(axis=0) + 1e-8

    stats_norm = (stats_data - stats_mean) / stats_std

    X = torch.FloatTensor(stats_norm)
    Y = torch.FloatTensor(params_norm)

    model = NPENetwork(n_stats=N_STATS, n_params=N_PARAMS)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        mean, logvar = model(X)
        # Gaussian negative log-likelihood
        var = torch.exp(logvar)
        loss = 0.5 * torch.mean(logvar + (Y - mean) ** 2 / var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            logger.info("NPE epoch %d/%d: loss=%.4f", epoch + 1, n_epochs, loss.item())

    # Store normalization constants
    model._stats_mean = stats_mean
    model._stats_std = stats_std
    model._param_min = param_min
    model._param_max = param_max

    return model


def infer_posterior_params(
    model: NPENetwork,
    observations: list[dict],
    grid_initial: np.ndarray,
    n_samples: int = 200,
) -> list[dict]:
    """Infer posterior parameter samples from observations.

    Returns list of parameter dicts sampled from the posterior.
    """
    # Extract summary stats from observations
    obs_stats = extract_stats_from_observations(observations, grid_initial)

    # Normalize
    stats_norm = (obs_stats - model._stats_mean) / model._stats_std
    stats_tensor = torch.FloatTensor(stats_norm).unsqueeze(0)

    # Sample from posterior
    model.eval()
    with torch.no_grad():
        samples = model.sample(stats_tensor, n_samples)  # (n_samples, N_PARAMS)

    # Denormalize — squeeze out batch dim if present
    samples_np = samples.squeeze().numpy()
    if samples_np.ndim == 1:
        samples_np = samples_np.reshape(1, -1)
    samples_np = samples_np * (model._param_max - model._param_min) + model._param_min

    # Clip to prior bounds
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = PARAM_PRIOR[name]
        samples_np[:, i] = np.clip(samples_np[:, i], lo, hi)

    # Convert to dicts
    param_dicts = []
    for j in range(n_samples):
        param_dicts.append(params_vector_to_dict(samples_np[j]))

    return param_dicts


# ============================================================
# Prediction with posterior samples
# ============================================================

def predict_with_posterior(
    grid: np.ndarray,
    settlements: list[dict],
    posterior_params: list[dict],
    n_sims_per_param: int = 5,
    base_seed: int = 42,
) -> np.ndarray:
    """Run Monte Carlo predictions using posterior parameter samples.

    Args:
        grid: Initial terrain grid.
        settlements: Initial settlements.
        posterior_params: List of parameter dicts from posterior.
        n_sims_per_param: Sims per parameter sample.
        base_seed: Random seed base.

    Returns (H, W, 6) probability array.
    """
    H, W = grid.shape
    counts = np.zeros((H, W, 6), dtype=np.int32)
    total_sims = 0

    for pi, params in enumerate(posterior_params):
        for si in range(n_sims_per_param):
            setts = [dict(s) for s in settlements]
            for j, se in enumerate(setts):
                se["owner_id"] = j
            sim = LocalSim(grid, setts, params=params, seed=base_seed + pi * 100 + si)
            sim.run(50)
            cg = _LUT[sim.grid]
            for c in range(6):
                counts[:, :, c] += (cg == c)
            total_sims += 1

    probs = counts.astype(np.float64) / total_sims
    probs = np.maximum(probs, 0.005)
    probs /= probs.sum(axis=2, keepdims=True)

    return probs


# ============================================================
# Full pipeline
# ============================================================

def run_sbi_pipeline(
    round_dir: str | Path,
    n_prior_samples: int = 500,
    n_sims_per_sample: int = 3,
    n_posterior_samples: int = 100,
    n_sims_per_posterior: int = 5,
    npe_epochs: int = 500,
) -> list[np.ndarray]:
    """Full SBI pipeline: train NPE → infer params → predict.

    Args:
        round_dir: Path to round data directory.
        n_prior_samples: Number of prior samples for NPE training.
        n_sims_per_sample: Sims per prior sample during training.
        n_posterior_samples: Samples from posterior for prediction.
        n_sims_per_posterior: Sims per posterior sample for prediction.
        npe_epochs: Training epochs for NPE.

    Returns list of (H, W, 6) prediction arrays, one per seed.
    """
    round_dir = Path(round_dir)

    with open(round_dir / "round_detail.json") as f:
        detail = json.load(f)

    # Use seed 0 for NPE training (representative initial state)
    state0 = detail["initial_states"][0]
    grid0 = np.array(state0["grid"])
    setts0 = state0.get("settlements", [])

    # Step 1: Generate SBI training data
    logger.info("Step 1: Generating %d prior samples for NPE training...", n_prior_samples)
    params_data, stats_data = build_sbi_training_data(
        grid0, setts0, n_samples=n_prior_samples,
        n_sims_per_sample=n_sims_per_sample)

    # Step 2: Train NPE
    logger.info("Step 2: Training NPE (%d epochs)...", npe_epochs)
    model = train_npe(params_data, stats_data, n_epochs=npe_epochs)

    # Step 3: Load observations and infer posterior
    logger.info("Step 3: Inferring posterior from observations...")
    all_obs = []
    for si in range(detail["seeds_count"]):
        obs_file = round_dir / "observations" / f"seed_{si}.json"
        if obs_file.exists():
            with open(obs_file) as f:
                all_obs.extend(json.load(f))

    if not all_obs:
        logger.warning("No observations found!")
        return []

    posterior_params = infer_posterior_params(model, all_obs, grid0, n_posterior_samples)

    # Log posterior summary
    param_arrays = {name: [] for name in PARAM_NAMES}
    for p in posterior_params:
        for name in PARAM_NAMES:
            param_arrays[name].append(p[name])
    logger.info("Posterior summary:")
    for name in PARAM_NAMES:
        vals = param_arrays[name]
        logger.info("  %s: %.4f ± %.4f", name, np.mean(vals), np.std(vals))

    # Step 4: Predict for all seeds
    logger.info("Step 4: Predicting with %d posterior samples × %d sims...",
                n_posterior_samples, n_sims_per_posterior)
    predictions = []
    for si in range(detail["seeds_count"]):
        state = detail["initial_states"][si]
        grid = np.array(state["grid"])
        setts = state.get("settlements", [])

        pred = predict_with_posterior(
            grid, setts, posterior_params,
            n_sims_per_param=n_sims_per_posterior)
        predictions.append(pred)
        logger.info("  Seed %d: done", si)

    return predictions


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    round_dir = sys.argv[1] if len(sys.argv) > 1 else "data/rounds/round_13"

    # Quick test with small numbers
    predictions = run_sbi_pipeline(
        round_dir,
        n_prior_samples=100,
        n_sims_per_sample=2,
        n_posterior_samples=20,
        n_sims_per_posterior=3,
        npe_epochs=200,
    )

    print(f"\nPredictions generated for {len(predictions)} seeds")

    # Score against GT if available
    def score_prediction(pred, gt):
        eps = 1e-10
        pred = np.maximum(pred, 0.005)
        pred = pred / pred.sum(axis=2, keepdims=True)
        kl = np.sum(gt * np.log((gt+eps)/(pred+eps)), axis=2)
        entropy = -np.sum(gt * np.log(gt+eps), axis=2)
        te = entropy.sum()
        return max(0, min(100, 100*np.exp(-3*(entropy*kl).sum()/te))) if te > 1e-10 else 100

    for si, pred in enumerate(predictions):
        try:
            with open(f"{round_dir}/analysis/seed_{si}.json") as f:
                a = json.load(f)
            if a.get("ground_truth"):
                gt = np.array(a["ground_truth"])
                score = score_prediction(pred, gt)
                print(f"  Seed {si}: score={score:.1f}")
        except FileNotFoundError:
            pass
