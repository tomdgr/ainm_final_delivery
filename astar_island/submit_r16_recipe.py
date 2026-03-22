"""R16 recipe: RF(50%) + ConvCNP(35%) + ANP(15%).
Proven: scored 86.2 on R16 (#24/272).

Usage: uv run python submit_r16_recipe.py --round 23
"""
import argparse
import json
import logging
import time
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
    return sorted(rounds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("R16 Recipe — Device: %s, Round: %d", device, args.round)

    rounds_dir = Path("data/rounds")
    target_dir = rounds_dir / f"round_{args.round}"
    gt_rounds = find_gt_rounds(exclude=args.round)
    logger.info("Training on %d GT rounds: %s", len(gt_rounds), gt_rounds)

    with open(target_dir / "round_detail.json") as f:
        detail = json.load(f)

    t0 = time.time()

    # 1. RF (per-round, fast)
    logger.info("\n=== RF ===")
    from nm_ai_ml.astar.spatial_predictor_rf import predict_round as rf_predict
    rf_preds = rf_predict(str(target_dir))
    logger.info("RF done (%.0fs)", time.time() - t0)

    # 2. ConvCNP (trained on all GT rounds)
    logger.info("\n=== ConvCNP ===")
    t1 = time.time()
    from nm_ai_ml.astar.convcnp import AstarUNet, train_model, predict, build_training_episodes
    import torch.nn.functional as F
    from tqdm import tqdm

    train_dirs = [rounds_dir / f"round_{r}" for r in gt_rounds]
    model = AstarUNet()
    model = train_model(model, train_dirs, epochs=100, episodes_per_round=12,
                        device=device, batch_size=16)

    cnp_preds = []
    for si in range(detail["seeds_count"]):
        initial_map = np.array(detail["initial_states"][si]["grid"])
        obs_file = target_dir / "observations" / f"seed_{si}.json"
        obs = json.load(open(obs_file)) if obs_file.exists() else []
        pred = predict(model, obs, initial_map, temperature=1.0, tta=False)
        cnp_preds.append(pred)
    logger.info("ConvCNP done (%.0fs)", time.time() - t1)

    # 3. ANP (trained on all GT rounds)
    logger.info("\n=== ANP ===")
    t2 = time.time()
    from nm_ai_ml.astar.attentive_np import load_training_data, train_anp, predict_with_anp

    all_episodes = load_training_data()
    anp_model = train_anp(all_episodes, n_epochs=80, lr=5e-4, device=device)

    anp_preds = []
    for si in range(detail["seeds_count"]):
        pred = predict_with_anp(anp_model, target_dir, detail, si, device=device)
        if pred is None:
            pred = rf_preds[si].copy()
        anp_preds.append(pred)
    logger.info("ANP done (%.0fs)", time.time() - t2)

    # 4. Ensemble: RF(50%) + CNP(35%) + ANP(15%)
    logger.info("\n=== ENSEMBLE ===")
    final_preds = []
    for si in range(detail["seeds_count"]):
        rf = np.maximum(rf_preds[si], 0.005)
        rf /= rf.sum(axis=2, keepdims=True)
        cnp = np.maximum(cnp_preds[si], 0.005)
        cnp /= cnp.sum(axis=2, keepdims=True)
        anp = np.maximum(anp_preds[si], 0.005)
        anp /= anp.sum(axis=2, keepdims=True)

        blend = geo_blend([rf, cnp, anp], [0.50, 0.35, 0.15])
        final_preds.append(blend)
        logger.info("Seed %d: blended", si)

    # Save predictions
    out_dir = target_dir / "predictions"
    out_dir.mkdir(exist_ok=True)
    for si, pred in enumerate(final_preds):
        np.save(out_dir / f"seed_{si}_r16_recipe.npy", pred)

    total = time.time() - t0
    logger.info("\nTotal: %.0fs (%.1f min)", total, total / 60)
    logger.info("Predictions saved to %s", out_dir)

    # Score if GT available
    if (target_dir / "analysis" / "seed_0.json").exists():
        logger.info("\n=== SCORES ===")
        for name, preds in [("RF", rf_preds), ("ConvCNP", cnp_preds), ("ANP", anp_preds), ("Ensemble", final_preds)]:
            scores = []
            for si in range(detail["seeds_count"]):
                gt = np.array(json.load(open(target_dir / f"analysis/seed_{si}.json"))["ground_truth"])
                scores.append(score_prediction(preds[si], gt))
            logger.info("%-10s avg=%.1f [%s]", name, np.mean(scores), ", ".join(f"{s:.1f}" for s in scores))
