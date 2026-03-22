"""ConvCNP-inspired U-Net for Astar Island prediction.

Learns: "given these observed cells → predict entire map" using
episodic meta-learning from historical rounds with ground truth.

Input: 40×40×8 tensor (Dirichlet-smoothed obs + obs count + initial terrain)
Output: 40×40×6 probability distribution

Usage:
    from nm_ai_ml.astar.convcnp import AstarUNet, train, predict, build_input_tensor
    model = AstarUNet()
    train(model, historical_rounds)
    pred = predict(model, observations, initial_map)
"""
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

GRID_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}


# ============================================================
# Model
# ============================================================

class AstarUNet(nn.Module):
    """U-Net that maps 40×40×8 observation tensor → 40×40×6 probabilities."""

    def __init__(self, in_channels=8, base_filters=32):
        super().__init__()
        self.enc1 = self._block(in_channels, base_filters)
        self.enc2 = self._block(base_filters, base_filters * 2)
        self.enc3 = self._block(base_filters * 2, base_filters * 4)
        self.bottleneck = self._block(base_filters * 4, base_filters * 8)
        self.dec3 = self._block(base_filters * 8 + base_filters * 4, base_filters * 4)
        self.dec2 = self._block(base_filters * 4 + base_filters * 2, base_filters * 2)
        self.dec1 = self._block(base_filters * 2 + base_filters, base_filters)
        self.out = nn.Conv2d(base_filters, 6, kernel_size=1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def _forward_features(self, x):
        """Forward pass up to the last conv block (before final 1x1 conv)."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return d1

    def forward_logits(self, x):
        """Forward pass returning raw logits (before softmax)."""
        return self.out(self._forward_features(x))

    def forward(self, x):
        return F.softmax(self.forward_logits(x), dim=1)  # (B, 6, 40, 40)


# ============================================================
# Input/output construction
# ============================================================

def build_observation_map(observations: list[dict], grid_size: int = 40):
    """Build per-cell observation counts and class from API observations."""
    obs_counts = np.zeros((grid_size, grid_size, 6), dtype=np.float64)
    obs_total = np.zeros((grid_size, grid_size), dtype=np.float64)

    for o in observations:
        vp = o["viewport"]
        for dy in range(len(o["grid"])):
            for dx in range(len(o["grid"][0])):
                gy, gx = vp["y"] + dy, vp["x"] + dx
                if gy >= grid_size or gx >= grid_size:
                    continue
                cls = GRID_TO_CLASS.get(o["grid"][dy][dx], 0)
                obs_counts[gy, gx, cls] += 1
                obs_total[gy, gx] += 1

    obs_mask = obs_total > 0
    obs_class = np.argmax(obs_counts, axis=2)
    obs_class[~obs_mask] = -1

    return obs_class, obs_mask, obs_counts, obs_total


def build_input_tensor(obs_class: np.ndarray, obs_mask: np.ndarray,
                       obs_total: np.ndarray, initial_map: np.ndarray,
                       alpha: float = 0.5) -> np.ndarray:
    """Build 8-channel input tensor.

    Channels 0-5: Dirichlet-smoothed class probabilities (0 where unobserved)
    Channel 6: Observation count (normalized)
    Channel 7: Initial terrain type (normalized to [0,1])

    Returns (8, 40, 40) float32 array.
    """
    H, W = initial_map.shape
    tensor = np.zeros((8, H, W), dtype=np.float32)

    for cls in range(6):
        cls_obs = (obs_class == cls).astype(float)
        count = obs_total.copy()
        # Bayesian estimate: (n_cls + alpha) / (n_total + 6*alpha)
        tensor[cls] = np.where(
            obs_mask,
            (cls_obs * count + alpha) / (count + 6 * alpha),
            0.0  # unobserved: network learns to handle this
        )

    # Channel 6: observation count (normalized, max ~10)
    tensor[6] = np.minimum(obs_total / 10.0, 1.0)

    # Channel 7: initial terrain (normalized)
    tensor[7] = initial_map.astype(np.float32) / 11.0

    return tensor


def build_input_from_counts(obs_counts: np.ndarray, obs_total: np.ndarray,
                            obs_mask: np.ndarray, initial_map: np.ndarray,
                            alpha: float = 0.5) -> np.ndarray:
    """Build input tensor from observation counts (more precise than obs_class)."""
    H, W = initial_map.shape
    tensor = np.zeros((8, H, W), dtype=np.float32)

    for cls in range(6):
        tensor[cls] = np.where(
            obs_mask,
            (obs_counts[:, :, cls] + alpha) / (obs_total + 6 * alpha),
            0.0
        )

    tensor[6] = np.minimum(obs_total / 10.0, 1.0)
    tensor[7] = initial_map.astype(np.float32) / 11.0

    return tensor


# ============================================================
# Training data generation
# ============================================================

def _simulate_viewports(observations: list[dict], n_context: int,
                        rng: random.Random) -> list[dict]:
    """Sample a random subset of viewports as 'context' observations."""
    if n_context >= len(observations):
        return observations
    return rng.sample(observations, n_context)


def _augment_tensor(x: np.ndarray, y: np.ndarray, aug_type: int):
    """Apply rotation/flip augmentation. aug_type in [0,7]."""
    # x: (C, H, W), y: (6, H, W)
    k = aug_type % 4  # 0-3 rotations
    flip = aug_type >= 4
    x = np.rot90(x, k, axes=(1, 2)).copy()
    y = np.rot90(y, k, axes=(1, 2)).copy()
    if flip:
        x = np.flip(x, axis=2).copy()
        y = np.flip(y, axis=2).copy()
    return x, y


def build_training_episodes(
    round_dir: str | Path,
    n_episodes_per_round: int = 20,
    n_context_range: tuple[int, int] = (2, 50),
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build training episodes from one historical round.

    Each episode: sample random subset of observations as context,
    use ground truth as target. Augment with rotations/flips.

    Returns list of (input_tensor, target_tensor) pairs.
    input: (8, 40, 40), target: (6, 40, 40)
    """
    round_dir = Path(round_dir)
    rng = random.Random(seed)

    with open(round_dir / "round_detail.json") as f:
        detail = json.load(f)

    episodes = []

    for si in range(detail["seeds_count"]):
        # Need both observations and ground truth
        obs_file = round_dir / "observations" / f"seed_{si}.json"
        gt_file = round_dir / "analysis" / f"seed_{si}.json"
        if not obs_file.exists() or not gt_file.exists():
            continue

        with open(obs_file) as f:
            all_obs = json.load(f)
        with open(gt_file) as f:
            analysis = json.load(f)
        if not analysis.get("ground_truth"):
            continue

        gt = np.array(analysis["ground_truth"])  # (40, 40, 6)
        initial_map = np.array(detail["initial_states"][si]["grid"])

        # Generate episodes by sampling different context subsets
        for ep in range(n_episodes_per_round):
            n_context = rng.randint(*n_context_range)
            context_obs = _simulate_viewports(all_obs, n_context, rng)

            obs_class, obs_mask, obs_counts, obs_total = build_observation_map(context_obs)
            x = build_input_from_counts(obs_counts, obs_total, obs_mask, initial_map)
            y = gt.transpose(2, 0, 1).astype(np.float32)  # (6, 40, 40)

            # Random augmentation
            aug = rng.randint(0, 7)
            x, y = _augment_tensor(x, y, aug)

            episodes.append((x, y))

    return episodes


# ============================================================
# Training
# ============================================================

def train_model(
    model: AstarUNet,
    historical_round_dirs: list[str | Path],
    epochs: int = 200,
    lr: float = 1e-3,
    episodes_per_round: int = 20,
    n_context_range: tuple[int, int] = (2, 50),
    device: str = "cpu",
    batch_size: int = 8,
) -> AstarUNet:
    """Train the U-Net on historical rounds with episodic meta-learning.

    Supports GPU training and batched episodes for speed.
    """
    import random
    from tqdm import tqdm

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    for epoch in tqdm(range(epochs), desc="ConvCNP training", unit="epoch"):
        model.train()
        total_loss = 0
        n_episodes = 0

        # Collect all episodes for this epoch
        all_x, all_y = [], []
        for rd in historical_round_dirs:
            episodes = build_training_episodes(
                rd,
                n_episodes_per_round=episodes_per_round,
                n_context_range=n_context_range,
                seed=epoch * 1000 + hash(str(rd)) % 1000,
            )
            for x_np, y_np in episodes:
                all_x.append(x_np)
                all_y.append(y_np)

        # Shuffle and batch
        indices = list(range(len(all_x)))
        random.shuffle(indices)

        for batch_start in range(0, len(indices), batch_size):
            batch_idx = indices[batch_start:batch_start + batch_size]
            x = torch.stack([torch.FloatTensor(all_x[i]) for i in batch_idx]).to(device)
            y = torch.stack([torch.FloatTensor(all_y[i]) for i in batch_idx]).to(device)

            pred = model(x)

            loss = F.kl_div(
                torch.log(pred.clamp(min=1e-6)),
                y.clamp(min=1e-6),
                reduction='batchmean',
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_idx)
            n_episodes += len(batch_idx)

        scheduler.step()
        avg_loss = total_loss / max(n_episodes, 1)
        if (epoch + 1) % 10 == 0:
            tqdm.write(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} ({n_episodes} episodes)")

    return model


# ============================================================
# Inference
# ============================================================

def predict(
    model: AstarUNet,
    observations: list[dict],
    initial_map: np.ndarray,
    min_prob: float = 0.005,
    temperature: float = 1.0,
    tta: bool = True,
) -> np.ndarray:
    """Predict full map from observations.

    Args:
        model: Trained AstarUNet.
        observations: List of API observation dicts.
        initial_map: Initial terrain grid (raw codes).
        min_prob: Probability floor.
        temperature: Temperature scaling (>1 softens, <1 sharpens).
        tta: If True, use test-time augmentation (8-fold dihedral symmetry).

    Returns (40, 40, 6) probability array.
    """
    obs_class, obs_mask, obs_counts, obs_total = build_observation_map(observations)
    x = build_input_from_counts(obs_counts, obs_total, obs_mask, initial_map)

    device = next(model.parameters()).device
    model.eval()

    if tta:
        # 8-fold dihedral augmentation: 4 rotations x 2 flips
        preds = []
        for k in range(4):
            for flip in [False, True]:
                x_aug = np.rot90(x, k, axes=(1, 2)).copy()
                if flip:
                    x_aug = np.flip(x_aug, axis=2).copy()

                x_tensor = torch.FloatTensor(x_aug).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model.forward_logits(x_tensor)
                    scaled = F.softmax(logits / temperature, dim=1)
                    pred_np = scaled.squeeze(0).permute(1, 2, 0).cpu().numpy()

                # Inverse transform
                if flip:
                    pred_np = np.flip(pred_np, axis=1).copy()
                pred_np = np.rot90(pred_np, -k, axes=(0, 1)).copy()
                preds.append(pred_np)

        pred_np = np.mean(preds, axis=0)
    else:
        x_tensor = torch.FloatTensor(x).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x_tensor)
        pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()

    pred_np = np.maximum(pred_np, min_prob)
    pred_np /= pred_np.sum(axis=2, keepdims=True)
    return pred_np


# ============================================================
# Full pipeline
# ============================================================

def run_convcnp_pipeline(
    round_dir: str | Path,
    historical_dirs: list[str | Path] | None = None,
    epochs: int = 200,
    episodes_per_round: int = 20,
) -> list[np.ndarray]:
    """Full pipeline: train on history → predict current round.

    Args:
        round_dir: Current round directory.
        historical_dirs: Directories with ground truth for training.
            If None, auto-discovers from data/rounds/.
        epochs: Training epochs.
        episodes_per_round: Episodes per round per epoch.

    Returns list of (40, 40, 6) predictions, one per seed.
    """
    round_dir = Path(round_dir)

    # Auto-discover historical rounds with GT
    if historical_dirs is None:
        data_root = Path("data/rounds")
        historical_dirs = []
        for rd in sorted(data_root.iterdir()):
            if not rd.is_dir() or rd == round_dir:
                continue
            if (rd / "analysis" / "seed_0.json").exists() and (rd / "observations" / "seed_0.json").exists():
                historical_dirs.append(rd)
        logger.info("Found %d historical rounds with GT + observations", len(historical_dirs))

    if not historical_dirs:
        logger.warning("No historical training data found!")
        return []

    # Train model
    logger.info("Training ConvCNP on %d rounds (%d epochs)...",
                len(historical_dirs), epochs)
    model = AstarUNet()
    model = train_model(model, historical_dirs, epochs=epochs,
                        episodes_per_round=episodes_per_round)

    # Load current round
    with open(round_dir / "round_detail.json") as f:
        detail = json.load(f)

    # Predict each seed
    predictions = []
    for si in range(detail["seeds_count"]):
        initial_map = np.array(detail["initial_states"][si]["grid"])

        # Load observations for this seed (if available)
        obs = []
        obs_file = round_dir / "observations" / f"seed_{si}.json"
        if obs_file.exists():
            with open(obs_file) as f:
                obs = json.load(f)

        # If no obs for this seed, use all available obs
        if not obs:
            for sj in range(detail["seeds_count"]):
                f2 = round_dir / "observations" / f"seed_{sj}.json"
                if f2.exists():
                    with open(f2) as fh:
                        obs.extend(json.load(fh))

        pred = predict(model, obs, initial_map)
        predictions.append(pred)
        logger.info("Seed %d: predicted", si)

    return predictions


def score_prediction(pred, gt):
    eps = 1e-10
    pred = np.maximum(pred, 0.005)
    pred = pred / pred.sum(axis=2, keepdims=True)
    kl = np.sum(gt * np.log((gt + eps) / (pred + eps)), axis=2)
    entropy = -np.sum(gt * np.log(gt + eps), axis=2)
    te = entropy.sum()
    return max(0, min(100, 100 * np.exp(-3 * (entropy * kl).sum() / te))) if te > 1e-10 else 100


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    test_round = sys.argv[1] if len(sys.argv) > 1 else "data/rounds/round_13"

    # Leave-one-out: train on all rounds except test round
    data_root = Path("data/rounds")
    test_dir = Path(test_round)
    historical = [rd for rd in sorted(data_root.iterdir())
                  if rd.is_dir() and rd != test_dir
                  and (rd / "analysis" / "seed_0.json").exists()
                  and (rd / "observations" / "seed_0.json").exists()]

    print(f"Training on {len(historical)} rounds, testing on {test_dir.name}")

    predictions = run_convcnp_pipeline(
        test_dir,
        historical_dirs=historical,
        epochs=100,
        episodes_per_round=10,
    )

    # Score
    for si, pred in enumerate(predictions):
        try:
            with open(test_dir / "analysis" / f"seed_{si}.json") as f:
                a = json.load(f)
            if a.get("ground_truth"):
                gt = np.array(a["ground_truth"])
                s = score_prediction(pred, gt)
                print(f"  Seed {si}: {s:.1f}")
        except FileNotFoundError:
            pass
