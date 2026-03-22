"""Experiment v7: Per-cell gated ensemble + deep ConvCNP ensemble.

Improvements over v6:
1. Deep ensemble: 3 ConvCNPs with different seeds (stabilizes variance)
2. Per-cell gated ensemble: trust neural model where it agrees with RF
3. Observation-based model weighting per seed
4. XGBoost with sim-derived features (run sim, use distributions as extra features)

Usage:
  uv run python experiment_v7.py --round 18
"""
import argparse
import json
import logging
import time
import random
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def score_prediction(pred, gt):
    eps = 1e-10
    pred = np.maximum(pred, 0.005)
    pred = pred / pred.sum(axis=2, keepdims=True)
    kl = np.sum(gt * np.log((gt + eps) / (pred + eps)), axis=2)
    entropy = -np.sum(gt * np.log(gt + eps), axis=2)
    total_ent = entropy.sum()
    weighted_kl = (entropy * kl).sum() / total_ent if total_ent > eps else kl.mean()
    return max(0, min(100, 100 * np.exp(-3 * weighted_kl)))


def geo_blend(preds, weights):
    eps = 1e-8
    log_sum = sum(w * np.log(np.maximum(p, eps)) for w, p in zip(weights, preds))
    b = np.exp(log_sum)
    b = np.maximum(b, 0.005)
    b /= b.sum(axis=2, keepdims=True)
    return b


def find_gt_rounds(exclude=None):
    rounds = []
    for d in sorted(Path("data/rounds").iterdir()):
        if d.is_dir() and d.name != "rounds":
            r = int(d.name.split("_")[1])
            if r == exclude:
                continue
            if (d / "analysis" / "seed_0.json").exists() and (d / "observations" / "seed_0.json").exists():
                rounds.append(r)
    return rounds


# ============================================================
# Deep ConvCNP ensemble (3 models)
# ============================================================

def train_deep_cnp_ensemble(train_rounds, device="cpu", n_models=3,
                             synth_episodes=1000, synth_epochs=30, finetune_epochs=60):
    """Train N independent ConvCNPs, return list of models."""
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    from nm_ai_ml.astar.convcnp import AstarUNet, build_training_episodes
    from nm_ai_ml.astar.synth_data import generate_synthetic_episodes

    rounds_dir = Path("data/rounds")

    # Generate synthetic data once (shared across all models)
    logger.info("Generating %d synthetic episodes...", synth_episodes)
    synth = generate_synthetic_episodes(n_episodes=synth_episodes, n_workers=32)

    models = []
    for model_idx in range(n_models):
        logger.info("Training CNP ensemble member %d/%d...", model_idx + 1, n_models)
        model = AstarUNet().to(device)

        # Pre-train on synthetic (different shuffle per model)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(synth_epochs):
            model.train()
            random.seed(42 + model_idx * 1000 + epoch)
            random.shuffle(synth)
            for bs in range(0, len(synth), 16):
                batch = synth[bs:bs + 16]
                x = torch.stack([torch.FloatTensor(e[0]) for e in batch]).to(device)
                y = torch.stack([torch.FloatTensor(e[1]) for e in batch]).to(device)
                pred = model(x)
                loss = F.kl_div(torch.log(pred.clamp(min=1e-6)), y.clamp(min=1e-6), reduction='batchmean')
                opt.zero_grad(); loss.backward(); opt.step()

        # Fine-tune on real (different seed per model for data ordering)
        train_dirs = [rounds_dir / f"round_{r}" for r in train_rounds]
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        for epoch in range(finetune_epochs):
            model.train()
            for rd in train_dirs:
                try:
                    episodes = build_training_episodes(
                        rd, n_episodes_per_round=8,
                        seed=epoch * 1000 + model_idx * 10000 + hash(str(rd)) % 1000)
                    for x_np, y_np in episodes:
                        x = torch.FloatTensor(x_np).unsqueeze(0).to(device)
                        y = torch.FloatTensor(y_np).unsqueeze(0).to(device)
                        pred = model(x)
                        loss = F.kl_div(torch.log(pred.clamp(min=1e-6)), y.clamp(min=1e-6), reduction='batchmean')
                        opt.zero_grad(); loss.backward(); opt.step()
                except:
                    pass

        models.append(model)

    return models


def predict_deep_ensemble(models, round_dir, detail, device="cpu"):
    """Average predictions across ensemble members, each with TTA."""
    from nm_ai_ml.astar.convcnp import predict

    predictions = []
    for si in range(detail["seeds_count"]):
        initial_map = np.array(detail["initial_states"][si]["grid"])
        obs_file = Path(round_dir) / "observations" / f"seed_{si}.json"
        obs = json.load(open(obs_file)) if obs_file.exists() else []

        # Average across models (each with TTA)
        model_preds = []
        for model in models:
            pred = predict(model, obs, initial_map, temperature=1.1, tta=True)
            model_preds.append(pred)

        avg = np.mean(model_preds, axis=0)
        avg = np.maximum(avg, 0.005)
        avg /= avg.sum(axis=2, keepdims=True)
        predictions.append(avg)

    return predictions


# ============================================================
# Per-cell gated ensemble
# ============================================================

def gated_ensemble(rf_pred, neural_pred, obs_count=None):
    """Per-cell adaptive ensemble: trust neural where it agrees with RF.

    For each cell:
    - Compute Jensen-Shannon divergence between RF and neural
    - Low JSD (agreement) → blend 50/50
    - High JSD (disagreement) → use 90% RF
    - Cells with observations → trust neural more (it has evidence)
    """
    eps = 1e-8
    rf = np.maximum(rf_pred, 0.005)
    rf /= rf.sum(axis=2, keepdims=True)
    neural = np.maximum(neural_pred, 0.005)
    neural /= neural.sum(axis=2, keepdims=True)

    # Per-cell JSD
    m = 0.5 * (rf + neural)
    jsd = 0.5 * np.sum(rf * np.log((rf + eps) / (m + eps)), axis=2) + \
          0.5 * np.sum(neural * np.log((neural + eps) / (m + eps)), axis=2)

    # Gate: low JSD → high neural weight, high JSD → low neural weight
    # Sigmoid-like mapping: w_neural = 0.5 * exp(-10 * jsd)
    w_neural = 0.5 * np.exp(-10 * jsd)

    # Boost neural weight where we have observations
    if obs_count is not None:
        obs_boost = np.minimum(obs_count / 5.0, 1.0)  # saturates at 5 observations
        w_neural = w_neural + 0.2 * obs_boost
        w_neural = np.minimum(w_neural, 0.6)  # cap at 60% neural

    w_rf = 1.0 - w_neural

    # Per-cell blend
    blend = w_rf[:, :, np.newaxis] * rf + w_neural[:, :, np.newaxis] * neural
    blend = np.maximum(blend, 0.005)
    blend /= blend.sum(axis=2, keepdims=True)

    return blend, w_neural.mean()


# ============================================================
# XGBoost with sim-derived features
# ============================================================

def train_predict_xgb_sim(round_dir_str):
    """XGBoost with sim distributions as extra features."""
    from nm_ai_ml.astar.xgb_predictor import predict_round
    return predict_round(round_dir_str, n_seeds=3, use_sim=True, temperature=1.0)


# ============================================================
# Main pipeline
# ============================================================

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--synth-episodes", type=int, default=1000)
    parser.add_argument("--n-cnp-models", type=int, default=3)
    parser.add_argument("--output-dir", default="data/model_predictions_v7")
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("V7 Pipeline — Device: %s, Round: %d", device, args.round)

    gt_rounds = find_gt_rounds(exclude=args.round)
    logger.info("GT rounds: %s", gt_rounds)

    out = Path(args.output_dir)
    t0 = time.time()

    # 1. Train deep ConvCNP ensemble
    logger.info("\n=== DEEP CNP ENSEMBLE (%d models) ===", args.n_cnp_models)
    cnp_models = train_deep_cnp_ensemble(
        gt_rounds, device=device, n_models=args.n_cnp_models,
        synth_episodes=args.synth_episodes, synth_epochs=30, finetune_epochs=60)

    # 2. Predict all rounds with deep ensemble
    rounds_dir = Path("data/rounds")
    all_rounds = gt_rounds + [args.round]

    cnp_predictions = {}
    for r in all_rounds:
        rd = rounds_dir / f"round_{r}"
        with open(rd / "round_detail.json") as f:
            detail = json.load(f)
        cnp_predictions[r] = predict_deep_ensemble(cnp_models, rd, detail, device=device)
        logger.info("R%d: deep CNP ensemble predicted", r)

    # 3. RF predictions
    logger.info("\n=== RF PREDICTIONS ===")
    from nm_ai_ml.astar.spatial_predictor_rf import predict_round as rf_predict
    rf_predictions = {}
    for r in all_rounds:
        try:
            rf_predictions[r] = rf_predict(f"data/rounds/round_{r}")
        except:
            pass

    # 4. Per-cell gated ensemble on GT rounds (validation)
    logger.info("\n=== GATED ENSEMBLE VALIDATION ===")
    rf_scores, cnp_scores, gated_scores, simple_scores = [], [], [], []

    for r in gt_rounds:
        rd = rounds_dir / f"round_{r}"
        with open(rd / "round_detail.json") as f:
            detail = json.load(f)

        for si in range(detail["seeds_count"]):
            gt_file = rd / "analysis" / f"seed_{si}.json"
            if not gt_file.exists() or r not in rf_predictions or r not in cnp_predictions:
                continue
            gt = np.array(json.load(open(gt_file))["ground_truth"])

            rf_p = rf_predictions[r][si]
            cnp_p = cnp_predictions[r][si]

            # Build obs count map
            obs_count = np.zeros((40, 40))
            obs_file = rd / "observations" / f"seed_{si}.json"
            if obs_file.exists():
                for o in json.load(open(obs_file)):
                    vp = o["viewport"]
                    obs_count[vp["y"]:vp["y"] + vp["h"], vp["x"]:vp["x"] + vp["w"]] += 1

            rf_scores.append(score_prediction(rf_p, gt))
            cnp_scores.append(score_prediction(cnp_p, gt))

            # Gated ensemble
            gated, avg_w = gated_ensemble(rf_p, cnp_p, obs_count)
            gated_scores.append(score_prediction(gated, gt))

            # Simple blend for comparison
            simple = geo_blend([rf_p, cnp_p], [0.55, 0.45])
            simple_scores.append(score_prediction(simple, gt))

    logger.info("\n" + "=" * 50)
    logger.info("VALIDATION RESULTS (trained on all, scored on all)")
    logger.info("=" * 50)
    logger.info("RF:           avg=%.1f  std=%.1f", np.mean(rf_scores), np.std(rf_scores))
    logger.info("Deep CNP:     avg=%.1f  std=%.1f", np.mean(cnp_scores), np.std(cnp_scores))
    logger.info("Simple blend: avg=%.1f  std=%.1f", np.mean(simple_scores), np.std(simple_scores))
    logger.info("Gated blend:  avg=%.1f  std=%.1f", np.mean(gated_scores), np.std(gated_scores))

    # 5. Save predictions for target round
    logger.info("\n=== SAVING R%d PREDICTIONS ===", args.round)
    for name, preds in [("rf", rf_predictions), ("cnp_deep", cnp_predictions)]:
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        if args.round in preds:
            for si, p in enumerate(preds[args.round]):
                np.save(d / f"round_{args.round}_seed_{si}.npy", p)

    # Save gated ensemble
    if args.round in rf_predictions and args.round in cnp_predictions:
        d = out / "gated_ensemble"
        d.mkdir(parents=True, exist_ok=True)
        rd = rounds_dir / f"round_{args.round}"
        with open(rd / "round_detail.json") as f:
            detail = json.load(f)
        for si in range(detail["seeds_count"]):
            obs_count = np.zeros((40, 40))
            obs_file = rd / "observations" / f"seed_{si}.json"
            if obs_file.exists():
                for o in json.load(open(obs_file)):
                    vp = o["viewport"]
                    obs_count[vp["y"]:vp["y"] + vp["h"], vp["x"]:vp["x"] + vp["w"]] += 1
            gated, _ = gated_ensemble(rf_predictions[args.round][si],
                                       cnp_predictions[args.round][si], obs_count)
            np.save(d / f"round_{args.round}_seed_{si}.npy", gated)

    total = time.time() - t0
    logger.info("\nV7 COMPLETE — %.0fs (%.1f min)", total, total / 60)
