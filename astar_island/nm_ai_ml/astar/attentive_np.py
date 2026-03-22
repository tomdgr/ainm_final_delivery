"""Attentive Neural Process (ANP) for Astar Island predictions.

Uses cross-attention from target cells to observed cells, allowing the model
to learn which observations are most relevant for each prediction. Unlike
ConvCNP which uses spatial convolution (assumes stationarity), attention can
capture non-stationary patterns (e.g., coastal dynamics differ from inland).

Architecture:
  1. Encode each observed cell: (x, y, terrain_features, observed_class) → embedding
  2. For each target cell: cross-attend to all observed cell embeddings
  3. Decode attention output → 6-class probability distribution

Training:
  - Train on rounds 1-12 ground truth (11 rounds × 5 seeds = 55 episodes)
  - Each episode: initial_state + observations → predict full grid distribution
  - Loss: KL divergence (matching competition metric)

Key insight: observations give us (viewport_y, viewport_x, observed_grid) —
these are scattered "context points" and we predict at all "target points".
This is exactly the Neural Process framework.
"""
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

logger = logging.getLogger(__name__)

NUM_CLASSES = 6
GRID_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}


def _build_feature_grid(grid, settlements):
    """Build per-cell feature tensor from initial state.

    Returns: (H, W, F) feature tensor
    Features: terrain_onehot(6), settlement(1), port(1), dist_sett(1),
              dist_ocean(1), adj_forest(1), adj_ocean(1), adj_sett(1),
              coastal(1), y_coord(1), x_coord(1) = 16 features
    """
    H, W = grid.shape

    # Terrain one-hot (6)
    terrain = np.zeros((H, W, 6))
    for y in range(H):
        for x in range(W):
            cls = GRID_TO_CLASS.get(int(grid[y, x]), 0)
            terrain[y, x, cls] = 1.0

    # Settlement and port maps
    sett_map = np.zeros((H, W))
    port_map = np.zeros((H, W))
    for se in settlements:
        sett_map[se["y"], se["x"]] = 1.0
        if se.get("has_port"):
            port_map[se["y"], se["x"]] = 1.0

    # Distance transforms
    ocean_map = (grid == 10).astype(float)
    dist_sett = distance_transform_edt(1 - sett_map) if sett_map.sum() > 0 else np.full((H, W), 20.0)
    dist_ocean = distance_transform_edt(1 - ocean_map) if ocean_map.sum() > 0 else np.full((H, W), 20.0)
    # Normalize distances
    dist_sett = dist_sett / 20.0
    dist_ocean = dist_ocean / 20.0

    # Adjacent counts
    forest_map = (grid == 4).astype(float)
    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    adj_forest = np.zeros((H, W))
    adj_ocean = np.zeros((H, W))
    adj_sett = np.zeros((H, W))
    for dy, dx in offsets:
        adj_forest += np.roll(np.roll(forest_map, dy, axis=0), dx, axis=1)
        adj_ocean += np.roll(np.roll(ocean_map, dy, axis=0), dx, axis=1)
        adj_sett += np.roll(np.roll(sett_map, dy, axis=0), dx, axis=1)
    adj_forest /= 8.0
    adj_ocean /= 8.0
    adj_sett /= 8.0

    # Coastal mask
    coastal = np.zeros((H, W))
    for dy, dx in offsets:
        coastal = np.maximum(coastal, np.roll(np.roll(ocean_map, dy, axis=0), dx, axis=1))
    coastal *= (1 - ocean_map)

    # Coordinates (normalized)
    yy, xx = np.meshgrid(np.arange(H) / (H - 1), np.arange(W) / (W - 1), indexing='ij')

    features = np.stack([
        *[terrain[:, :, i] for i in range(6)],
        sett_map, port_map,
        dist_sett, dist_ocean,
        adj_forest, adj_ocean, adj_sett,
        coastal,
        yy, xx,
    ], axis=-1)

    return features.astype(np.float32)


class AttentiveNP(nn.Module):
    """Attentive Neural Process for grid prediction.

    Context: observed cells with their outcomes
    Target: all cells where we want predictions
    """

    def __init__(self, feature_dim=16, obs_dim=6, d_model=128, n_heads=4, n_layers=3):
        super().__init__()
        self.d_model = d_model

        # Context encoder: features + observed class → embedding
        self.context_encoder = nn.Sequential(
            nn.Linear(feature_dim + obs_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Target encoder: features → query embedding
        self.target_encoder = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Cross-attention layers (target attends to context)
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
            for _ in range(n_layers)
        ])
        self.cross_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        self.cross_ff = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model),
            )
            for _ in range(n_layers)
        ])
        self.ff_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

        # Decoder: attention output → class probabilities
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, NUM_CLASSES),
        )

    def forward(self, target_features, context_features, context_classes):
        """
        Args:
            target_features: (B, T, F) features for all target cells
            context_features: (B, C, F) features for observed cells
            context_classes: (B, C, 6) observed class distributions
        Returns:
            (B, T, 6) predicted class probabilities
        """
        # Encode context
        context_input = torch.cat([context_features, context_classes], dim=-1)
        context_emb = self.context_encoder(context_input)  # (B, C, d_model)

        # Encode targets
        target_emb = self.target_encoder(target_features)  # (B, T, d_model)

        # Cross-attention: targets attend to context
        x = target_emb
        for attn, norm1, ff, norm2 in zip(
            self.cross_attention_layers, self.cross_norms,
            self.cross_ff, self.ff_norms
        ):
            attended, _ = attn(x, context_emb, context_emb)
            x = norm1(x + attended)
            x = norm2(x + ff(x))

        # Decode to class probabilities
        logits = self.decoder(x)  # (B, T, 6)
        return F.softmax(logits, dim=-1)


def _prepare_episode(round_dir, detail, seed_idx, gt=None):
    """Prepare one training episode from a round/seed.

    Returns:
        features: (H*W, F) feature grid
        context_positions: list of (y, x) observed positions
        context_classes: (C, 6) observed class distributions
        target_gt: (H*W, 6) ground truth distributions (if gt provided)
    """
    state = detail["initial_states"][seed_idx]
    grid = np.array(state["grid"])
    H, W = grid.shape
    settlements = state.get("settlements", [])

    features = _build_feature_grid(grid, settlements)  # (H, W, F)
    features_flat = features.reshape(H * W, -1)

    # Load observations
    obs_file = round_dir / "observations" / f"seed_{seed_idx}.json"
    if not obs_file.exists():
        return None

    with open(obs_file) as f:
        observations = json.load(f)

    if not observations:
        return None

    # Build per-cell observation counts
    cell_counts = {}  # (y, x) -> [count_per_class]
    for obs in observations:
        vp = obs["viewport"]
        obs_grid = obs["grid"]
        for dy in range(len(obs_grid)):
            for dx in range(len(obs_grid[0])):
                gy = vp["y"] + dy
                gx = vp["x"] + dx
                if gy >= H or gx >= W:
                    continue
                key = (gy, gx)
                if key not in cell_counts:
                    cell_counts[key] = np.zeros(6)
                cls = GRID_TO_CLASS.get(int(obs_grid[dy][dx]), 0)
                cell_counts[key][cls] += 1

    # Normalize to probabilities
    context_positions = []
    context_classes = []
    for (y, x), counts in cell_counts.items():
        total = counts.sum()
        if total > 0:
            context_positions.append((y, x))
            context_classes.append(counts / total)

    context_classes = np.array(context_classes, dtype=np.float32)
    context_features = np.array([features[y, x] for y, x in context_positions], dtype=np.float32)

    result = {
        "features": features_flat,
        "context_features": context_features,
        "context_classes": context_classes,
        "context_positions": context_positions,
        "H": H, "W": W,
    }

    if gt is not None:
        result["target_gt"] = gt.reshape(H * W, 6).astype(np.float32)

    return result


def load_training_data():
    """Load all training episodes from rounds with both observations and GT."""
    episodes = []
    rounds_dir = Path("data/rounds")

    for round_dir in sorted(rounds_dir.iterdir()):
        if not (round_dir / "analysis" / "seed_0.json").exists():
            continue
        if not (round_dir / "observations" / "seed_0.json").exists():
            continue

        r = int(round_dir.name.split("_")[1])
        with open(round_dir / "round_detail.json") as f:
            detail = json.load(f)

        for seed_idx in range(detail["seeds_count"]):
            gt_file = round_dir / "analysis" / f"seed_{seed_idx}.json"
            if not gt_file.exists():
                continue

            with open(gt_file) as f:
                gt = np.array(json.load(f)["ground_truth"])

            episode = _prepare_episode(round_dir, detail, seed_idx, gt)
            if episode is not None:
                episode["round"] = r
                episode["seed"] = seed_idx
                episodes.append(episode)

    return episodes


def train_anp(episodes, n_epochs=100, lr=1e-3, device="cpu"):
    """Train the Attentive NP on all episodes."""
    feature_dim = episodes[0]["features"].shape[1]
    model = AttentiveNP(feature_dim=feature_dim, d_model=128, n_heads=4, n_layers=3)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    for epoch in range(n_epochs):
        total_loss = 0
        np.random.shuffle(episodes)

        for ep in episodes:
            target_features = torch.tensor(ep["features"], device=device).unsqueeze(0)
            context_features = torch.tensor(ep["context_features"], device=device).unsqueeze(0)
            context_classes = torch.tensor(ep["context_classes"], device=device).unsqueeze(0)
            target_gt = torch.tensor(ep["target_gt"], device=device).unsqueeze(0)

            pred = model(target_features, context_features, context_classes)

            # KL divergence loss (matching competition metric)
            eps = 1e-8
            pred_clamped = pred.clamp(min=eps)
            kl = (target_gt * (torch.log(target_gt + eps) - torch.log(pred_clamped))).sum(dim=-1)

            # Entropy weighting (match competition scoring)
            entropy = -(target_gt * torch.log(target_gt + eps)).sum(dim=-1)
            weighted_kl = (entropy * kl).sum() / (entropy.sum() + eps)

            loss = weighted_kl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(episodes)
            logger.info("Epoch %d/%d: avg_loss=%.4f", epoch + 1, n_epochs, avg_loss)

    return model


def predict_with_anp(model, round_dir, detail, seed_idx, device="cpu"):
    """Predict using trained ANP model."""
    episode = _prepare_episode(round_dir, detail, seed_idx)
    if episode is None:
        return None

    model.eval()
    with torch.no_grad():
        target_features = torch.tensor(episode["features"], device=device).unsqueeze(0)
        context_features = torch.tensor(episode["context_features"], device=device).unsqueeze(0)
        context_classes = torch.tensor(episode["context_classes"], device=device).unsqueeze(0)

        pred = model(target_features, context_features, context_classes)

    pred_np = pred.squeeze(0).cpu().numpy()
    H, W = episode["H"], episode["W"]
    pred_grid = pred_np.reshape(H, W, 6)

    # Apply probability floor
    pred_grid = np.maximum(pred_grid, 0.01)
    pred_grid /= pred_grid.sum(axis=2, keepdims=True)

    return pred_grid


def score_prediction(pred, gt):
    eps = 1e-10
    pred = np.maximum(pred, 0.01)
    pred = pred / pred.sum(axis=2, keepdims=True)
    kl = np.sum(gt * np.log((gt + eps) / (pred + eps)), axis=2)
    entropy = -np.sum(gt * np.log(gt + eps), axis=2)
    total_ent = entropy.sum()
    weighted_kl = (entropy * kl).sum() / total_ent if total_ent > eps else kl.mean()
    return max(0, min(100, 100 * np.exp(-3 * weighted_kl)))


if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # Load training data
    logger.info("Loading training episodes...")
    episodes = load_training_data()
    logger.info("Loaded %d episodes from %d rounds",
                len(episodes), len(set(e["round"] for e in episodes)))

    # Leave-one-round-out cross-validation
    rounds = sorted(set(e["round"] for e in episodes))
    logger.info("Rounds: %s", rounds)

    all_scores = []
    for test_round in rounds:
        train_eps = [e for e in episodes if e["round"] != test_round]
        test_eps = [e for e in episodes if e["round"] == test_round]

        if not test_eps:
            continue

        logger.info("=== Testing on round %d (train on %d episodes) ===",
                    test_round, len(train_eps))

        t0 = time.time()
        model = train_anp(train_eps, n_epochs=80, lr=5e-4, device=device)

        round_scores = []
        for ep in test_eps:
            round_dir = Path(f"data/rounds/round_{ep['round']}")
            with open(round_dir / "round_detail.json") as f:
                detail = json.load(f)

            pred = predict_with_anp(model, round_dir, detail, ep["seed"], device=device)
            if pred is not None:
                gt = ep["target_gt"].reshape(ep["H"], ep["W"], 6)
                score = score_prediction(pred, gt)
                round_scores.append(score)

        elapsed = time.time() - t0
        avg = np.mean(round_scores) if round_scores else 0
        all_scores.extend(round_scores)
        logger.info("Round %d: avg=%.1f [%s] (%.0fs)",
                    test_round, avg,
                    ", ".join(f"{s:.1f}" for s in round_scores),
                    elapsed)

    if all_scores:
        logger.info("OVERALL: %.1f", np.mean(all_scores))
