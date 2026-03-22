"""Parallel synthetic data generation from local simulator.

Generates training episodes for ConvCNP by running local sim with
randomized params. Uses all CPU cores for parallelism.
"""
import json
import logging
import random
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np

from nm_ai_ml.astar.simulator_v2 import Simulator, DEFAULT_PARAMS, _LUT
from nm_ai_ml.astar.sbi_pipeline import sample_prior, params_vector_to_dict
from nm_ai_ml.astar.convcnp import build_observation_map, build_input_from_counts

logger = logging.getLogger(__name__)

GRID_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}


def _generate_one_episode(args):
    """Generate one synthetic episode. Designed for multiprocessing."""
    grid, settlements, initial_map, episode_seed = args
    rng = np.random.default_rng(episode_seed)
    py_rng = random.Random(episode_seed)

    H, W = grid.shape

    # Sample random params
    params_vec = sample_prior(rng, 1)[0]
    params = params_vector_to_dict(params_vec)

    # Run sim N times to get distribution
    n_sims = 30  # fewer sims per episode, more episodes for diversity
    counts = np.zeros((H, W, 6), dtype=np.int32)
    for i in range(n_sims):
        sim = Simulator(grid, settlements, params=params, seed=episode_seed * 100 + i)
        final = sim.run(50)
        class_grid = _LUT[final]
        for c in range(6):
            counts[:, :, c] += (class_grid == c)

    gt_dist = counts.astype(np.float32) / n_sims
    gt_dist = np.maximum(gt_dist, 0.005)
    gt_dist /= gt_dist.sum(axis=2, keepdims=True)

    # Create fake observations (random viewports from one sim run)
    fake_obs = []
    n_obs = py_rng.randint(3, 12)
    for _ in range(n_obs):
        vy = py_rng.randint(0, max(0, H - 15))
        vx = py_rng.randint(0, max(0, W - 15))
        sim = Simulator(grid, settlements, params=params, seed=episode_seed * 1000 + py_rng.randint(0, 999))
        final = sim.run(50)
        obs_grid = final[vy:vy + 15, vx:vx + 15].tolist()
        fake_obs.append({"viewport": {"y": vy, "x": vx, "h": 15, "w": 15}, "grid": obs_grid})

    obs_class, obs_mask, obs_counts, obs_total = build_observation_map(fake_obs)
    x = build_input_from_counts(obs_counts, obs_total, obs_mask, initial_map)
    y = gt_dist.transpose(2, 0, 1)  # (6, H, W)

    # Random augmentation
    aug = py_rng.randint(0, 7)
    k = aug % 4
    flip = aug >= 4
    x = np.rot90(x, k, axes=(1, 2)).copy()
    y = np.rot90(y, k, axes=(1, 2)).copy()
    if flip:
        x = np.flip(x, axis=2).copy()
        y = np.flip(y, axis=2).copy()

    return x.astype(np.float32), y.astype(np.float32)


def generate_synthetic_episodes(n_episodes=1000, n_workers=None):
    """Generate synthetic training episodes using all CPU cores.

    Returns list of (x, y) tuples where x is (8, H, W) and y is (6, H, W).
    """
    if n_workers is None:
        n_workers = min(cpu_count(), 32)

    rounds_dir = Path("data/rounds")
    gt_rounds = []
    for d in sorted(rounds_dir.iterdir()):
        if d.is_dir() and d.name != "rounds":
            r = int(d.name.split("_")[1])
            if (d / "round_detail.json").exists():
                gt_rounds.append(r)

    # Collect initial states from diverse rounds
    initial_states = []
    for r in gt_rounds[:10]:
        rd = rounds_dir / f"round_{r}"
        with open(rd / "round_detail.json") as f:
            detail = json.load(f)
        for si in range(min(detail["seeds_count"], 3)):
            state = detail["initial_states"][si]
            grid = np.array(state["grid"])
            settlements = state.get("settlements", [])
            initial_states.append((grid, settlements, grid.copy()))

    if not initial_states:
        logger.warning("No initial states found!")
        return []

    # Build args for parallel generation
    args_list = []
    for ep in range(n_episodes):
        grid, settlements, initial_map = initial_states[ep % len(initial_states)]
        args_list.append((grid, settlements, initial_map, ep + 42))

    logger.info("Generating %d synthetic episodes on %d workers...", n_episodes, n_workers)

    with Pool(n_workers) as pool:
        episodes = pool.map(_generate_one_episode, args_list)

    logger.info("Generated %d episodes", len(episodes))
    return episodes
