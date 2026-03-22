"""Residual ConvCNP — learns to CORRECT RF predictions, not replace them.

RF scores 87-88 consistently. This model takes RF predictions as input
and learns spatial corrections from historical rounds. If uncertain,
it passes RF through unchanged (safe baseline). If it sees a clear
spatial pattern, it adjusts RF upward.

Input: 40×40×14 tensor
  - Channels 0-5: Dirichlet-smoothed observation probabilities
  - Channel 6: Observation count per cell
  - Channel 7: Initial terrain type
  - Channels 8-13: RF prediction (6 class probabilities)

Output: 40×40×6 corrected probability distribution

Usage:
    from nm_ai_ml.astar.convcnp_residual import ResidualUNet, train_residual, predict_residual
    model = ResidualUNet()
    train_residual(model, historical_rounds)
    pred = predict_residual(model, observations, initial_map, rf_pred)
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
# Model — smaller than ConvCNP, with residual connection
# ============================================================

class ResidualUNet(nn.Module):
    """Small U-Net that learns corrections on top of RF predictions.

    Key design choices:
    - Fewer filters than full ConvCNP (16 base vs 32) to reduce overfitting
    - Skip connection from RF input to output (residual learning)
    - Dropout for regularization
    - Weight decay applied during training
    """

    def __init__(self, in_channels=14, base_filters=16, dropout=0.1):
        super().__init__()
        bf = base_filters
        self.enc1 = self._block(in_channels, bf, dropout)
        self.enc2 = self._block(bf, bf * 2, dropout)
        self.enc3 = self._block(bf * 2, bf * 4, dropout)
        self.bottleneck = self._block(bf * 4, bf * 8, dropout)
        self.dec3 = self._block(bf * 8 + bf * 4, bf * 4, dropout)
        self.dec2 = self._block(bf * 4 + bf * 2, bf * 2, dropout)
        self.dec1 = self._block(bf * 2 + bf, bf, dropout)
        # Output: 6 channels (correction to add to RF)
        self.out = nn.Conv2d(bf, 6, kernel_size=1)
        # Learnable blend weight: how much to trust the correction
        self.blend_weight = nn.Parameter(torch.tensor(0.0))  # starts at sigmoid(0) = 0.5
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def _block(self, in_c, out_c, dropout=0.1):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        x: (B, 14, 40, 40)
        Channels 8-13 are RF predictions.
        Returns: (B, 6, 40, 40) corrected predictions.
        """
        rf_pred = x[:, 8:14, :, :]  # (B, 6, 40, 40)

        # U-Net on full input
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        correction = self.out(d1)  # (B, 6, 40, 40)

        # Residual blend: output = (1-w)*RF + w*softmax(correction)
        w = torch.sigmoid(self.blend_weight)
        corrected = (1 - w) * rf_pred + w * F.softmax(correction, dim=1)

        # Renormalize
        return corrected / corrected.sum(dim=1, keepdim=True).clamp(min=1e-6)


# ============================================================
# Input construction
# ============================================================

def build_observation_map(observations: list[dict], grid_size: int = 40):
    """Build per-cell observation counts from API observations."""
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
    return obs_counts, obs_total, obs_mask


def build_input_tensor(obs_counts: np.ndarray, obs_total: np.ndarray,
                       obs_mask: np.ndarray, initial_map: np.ndarray,
                       rf_pred: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Build 14-channel input tensor.

    Channels 0-5: Dirichlet-smoothed observation probabilities
    Channel 6: Observation count (normalized)
    Channel 7: Initial terrain (normalized)
    Channels 8-13: RF prediction probabilities

    Returns (14, H, W) float32 array.
    """
    H, W = initial_map.shape
    tensor = np.zeros((14, H, W), dtype=np.float32)

    # Channels 0-5: Bayesian observation estimate
    for cls in range(6):
        tensor[cls] = np.where(
            obs_mask,
            (obs_counts[:, :, cls] + alpha) / (obs_total + 6 * alpha),
            0.0
        )

    # Channel 6: observation count
    tensor[6] = np.minimum(obs_total / 10.0, 1.0)

    # Channel 7: initial terrain
    tensor[7] = initial_map.astype(np.float32) / 11.0

    # Channels 8-13: RF predictions
    for cls in range(6):
        tensor[8 + cls] = rf_pred[:, :, cls]

    return tensor


# ============================================================
# Training episode generation
# ============================================================

def _augment(x: np.ndarray, y: np.ndarray, aug: int):
    """Rotation/flip augmentation."""
    k = aug % 4
    flip = aug >= 4
    x = np.rot90(x, k, axes=(1, 2)).copy()
    y = np.rot90(y, k, axes=(1, 2)).copy()
    if flip:
        x = np.flip(x, axis=2).copy()
        y = np.flip(y, axis=2).copy()
    return x, y


def build_training_episodes(
    round_dir: str | Path,
    rf_predictions: list[np.ndarray],
    n_episodes: int = 20,
    n_context_range: tuple[int, int] = (3, 8),
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build training episodes from one round.

    Each episode: sample random observation subset as context,
    include RF prediction as input, use ground truth as target.

    Args:
        round_dir: Round directory.
        rf_predictions: RF predictions for each seed (list of (40,40,6)).
        n_episodes: Episodes to generate per seed.
        n_context_range: Range of context viewports to sample.
        seed: Random seed.

    Returns list of (input_tensor, target_tensor) pairs.
    """
    round_dir = Path(round_dir)
    rng = random.Random(seed)

    with open(round_dir / "round_detail.json") as f:
        detail = json.load(f)

    episodes = []

    for si in range(detail["seeds_count"]):
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

        gt = np.array(analysis["ground_truth"]).transpose(2, 0, 1).astype(np.float32)
        initial_map = np.array(detail["initial_states"][si]["grid"])
        rf_pred = rf_predictions[si]

        for ep in range(n_episodes):
            # Sample random context subset
            n_ctx = rng.randint(*n_context_range)
            ctx_obs = rng.sample(all_obs, min(n_ctx, len(all_obs)))

            obs_counts, obs_total, obs_mask = build_observation_map(ctx_obs)
            x = build_input_tensor(obs_counts, obs_total, obs_mask,
                                   initial_map, rf_pred)

            # Random augmentation
            aug = rng.randint(0, 7)
            x, y = _augment(x, gt.copy(), aug)

            episodes.append((x, y))

    return episodes


# ============================================================
# Training
# ============================================================

def train_residual(
    model: ResidualUNet,
    historical_round_dirs: list[str | Path],
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    episodes_per_round: int = 20,
    n_context_range: tuple[int, int] = (3, 8),
) -> ResidualUNet:
    """Train residual U-Net on historical rounds.

    Requires RF predictions for each historical round.
    """
    from nm_ai_ml.astar.spatial_predictor_rf import predict_round as predict_rf

    # Cache RF predictions — compute once, reuse every epoch
    logger.info("Caching RF predictions for %d rounds...", len(historical_round_dirs))
    rf_cache = {}
    for rd in historical_round_dirs:
        try:
            rf_cache[str(rd)] = predict_rf(str(rd))
        except Exception:
            pass
    logger.info("Cached RF for %d rounds", len(rf_cache))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_eps = 0

        for rd in historical_round_dirs:
            rd_key = str(rd)
            if rd_key not in rf_cache:
                continue
            rf_preds = rf_cache[rd_key]

            episodes = build_training_episodes(
                rd, rf_preds,
                n_episodes=episodes_per_round,
                n_context_range=n_context_range,
                seed=epoch * 1000 + hash(rd_key) % 1000,
            )

            for x_np, y_np in episodes:
                x = torch.FloatTensor(x_np).unsqueeze(0)
                y = torch.FloatTensor(y_np).unsqueeze(0)

                pred = model(x)

                loss = F.kl_div(
                    torch.log(pred.clamp(min=1e-6)),
                    y.clamp(min=1e-6),
                    reduction='batchmean',
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_eps += 1

        scheduler.step()

        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / max(n_eps, 1)
            w = torch.sigmoid(model.blend_weight).item()
            logger.info("Epoch %d/%d: loss=%.4f blend_weight=%.3f (%d episodes)",
                        epoch + 1, epochs, avg_loss, w, n_eps)

    return model


# ============================================================
# Inference
# ============================================================

def predict_residual(
    model: ResidualUNet,
    observations: list[dict],
    initial_map: np.ndarray,
    rf_pred: np.ndarray,
    min_prob: float = 0.005,
) -> np.ndarray:
    """Predict using residual correction on RF."""
    obs_counts, obs_total, obs_mask = build_observation_map(observations)
    x = build_input_tensor(obs_counts, obs_total, obs_mask, initial_map, rf_pred)
    x_tensor = torch.FloatTensor(x).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        pred = model(x_tensor)

    pred_np = pred.squeeze(0).permute(1, 2, 0).numpy()
    pred_np = np.maximum(pred_np, min_prob)
    pred_np /= pred_np.sum(axis=2, keepdims=True)

    return pred_np


# ============================================================
# Full pipeline
# ============================================================

def run_residual_pipeline(
    round_dir: str | Path,
    historical_dirs: list[str | Path] | None = None,
    epochs: int = 200,
    episodes_per_round: int = 20,
) -> list[np.ndarray]:
    """Full pipeline: train residual model → correct RF predictions."""
    from nm_ai_ml.astar.spatial_predictor_rf import predict_round as predict_rf

    round_dir = Path(round_dir)

    # Auto-discover historical rounds
    if historical_dirs is None:
        data_root = Path("data/rounds")
        historical_dirs = []
        for rd in sorted(data_root.iterdir()):
            if not rd.is_dir() or rd == round_dir or rd.name == 'rounds':
                continue
            if ((rd / "analysis" / "seed_0.json").exists()
                    and (rd / "observations" / "seed_0.json").exists()):
                historical_dirs.append(rd)
        logger.info("Found %d historical rounds for training", len(historical_dirs))

    # Train
    logger.info("Training residual U-Net (%d epochs)...", epochs)
    model = ResidualUNet()
    model = train_residual(model, historical_dirs, epochs=epochs,
                           episodes_per_round=episodes_per_round)

    w = torch.sigmoid(model.blend_weight).item()
    logger.info("Learned blend weight: %.3f (0=pure RF, 1=pure correction)", w)

    # Get RF predictions for current round
    rf_preds = predict_rf(str(round_dir))

    # Load current round detail
    with open(round_dir / "round_detail.json") as f:
        detail = json.load(f)

    # Predict each seed
    predictions = []
    for si in range(detail["seeds_count"]):
        initial_map = np.array(detail["initial_states"][si]["grid"])

        obs = []
        obs_file = round_dir / "observations" / f"seed_{si}.json"
        if obs_file.exists():
            with open(obs_file) as f:
                obs = json.load(f)
        if not obs:
            for sj in range(detail["seeds_count"]):
                f2 = round_dir / "observations" / f"seed_{sj}.json"
                if f2.exists():
                    with open(f2) as fh:
                        obs.extend(json.load(fh))

        pred = predict_residual(model, obs, initial_map, rf_preds[si])
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
    test_dir = Path(test_round)

    # Leave-one-out
    data_root = Path("data/rounds")
    historical = [rd for rd in sorted(data_root.iterdir())
                  if rd.is_dir() and rd != test_dir and rd.name != 'rounds'
                  and (rd / "analysis" / "seed_0.json").exists()
                  and (rd / "observations" / "seed_0.json").exists()]

    print(f"Training on {len(historical)} rounds, testing on {test_dir.name}")

    predictions = run_residual_pipeline(
        test_dir,
        historical_dirs=historical,
        epochs=150,
        episodes_per_round=15,
    )

    # Also get RF baseline for comparison
    from nm_ai_ml.astar.spatial_predictor_rf import predict_round as predict_rf
    rf_preds = predict_rf(str(test_dir))

    for si, pred in enumerate(predictions):
        try:
            with open(test_dir / "analysis" / f"seed_{si}.json") as f:
                a = json.load(f)
            if a.get("ground_truth"):
                gt = np.array(a["ground_truth"])
                res_score = score_prediction(pred, gt)
                rf_score = score_prediction(rf_preds[si], gt)
                print(f"  Seed {si}: RF={rf_score:.1f}  Residual={res_score:.1f}  diff={res_score-rf_score:+.1f}")
        except FileNotFoundError:
            pass
