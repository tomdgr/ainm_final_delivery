"""LOO evaluation of R16 recipe + weight variants. Fast, runs on GPU.

Usage:
  uv run python loo_eval.py                    # R16 recipe LOO
  uv run python loo_eval.py --variant weights  # try different weights
  uv run python loo_eval.py --variant r17      # R17 recipe (4 models)
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


def find_gt_rounds():
    rounds = []
    for d in sorted(Path("data/rounds").iterdir()):
        if d.is_dir() and d.name != "rounds":
            r = int(d.name.split("_")[1])
            if (d / "analysis" / "seed_0.json").exists() and (d / "observations" / "seed_0.json").exists():
                rounds.append(r)
    return sorted(rounds)


def train_cnp_loo(train_rounds, device, epochs=80):
    import torch.nn.functional as F
    from nm_ai_ml.astar.convcnp import AstarUNet, train_model
    rounds_dir = Path("data/rounds")
    train_dirs = [rounds_dir / f"round_{r}" for r in train_rounds]
    model = AstarUNet()
    model = train_model(model, train_dirs, epochs=epochs, episodes_per_round=10, device=device, batch_size=16)
    return model


def train_cnp_synth_loo(train_rounds, device, epochs=60, synth_episodes=300):
    """CNP with synthetic pretraining (R17 variant)."""
    import torch, torch.nn.functional as F, random
    from nm_ai_ml.astar.convcnp import AstarUNet, train_model, build_training_episodes
    from nm_ai_ml.astar.synth_data import generate_synthetic_episodes
    rounds_dir = Path("data/rounds")

    synth = generate_synthetic_episodes(n_episodes=synth_episodes, n_workers=8)
    model = AstarUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(20):
        model.train()
        random.shuffle(synth)
        for bs in range(0, len(synth), 16):
            batch = synth[bs:bs+16]
            x = torch.stack([torch.FloatTensor(e[0]) for e in batch]).to(device)
            y = torch.stack([torch.FloatTensor(e[1]) for e in batch]).to(device)
            pred = model(x)
            loss = torch.nn.functional.kl_div(torch.log(pred.clamp(min=1e-6)), y.clamp(min=1e-6), reduction='batchmean')
            opt.zero_grad(); loss.backward(); opt.step()

    train_dirs = [rounds_dir / f"round_{r}" for r in train_rounds]
    model = train_model(model, train_dirs, epochs=epochs, episodes_per_round=10, device=device, batch_size=16)
    return model


def predict_cnp(model, round_dir, detail):
    from nm_ai_ml.astar.convcnp import predict
    preds = []
    for si in range(detail["seeds_count"]):
        initial_map = np.array(detail["initial_states"][si]["grid"])
        obs_file = Path(round_dir) / "observations" / f"seed_{si}.json"
        obs = json.load(open(obs_file)) if obs_file.exists() else []
        pred = predict(model, obs, initial_map, temperature=1.0, tta=False)
        preds.append(pred)
    return preds


def train_anp_loo(train_rounds, device, epochs=60):
    from nm_ai_ml.astar.attentive_np import load_training_data, train_anp
    all_eps = load_training_data()
    train_eps = [e for e in all_eps if e["round"] in train_rounds]
    return train_anp(train_eps, n_epochs=epochs, lr=5e-4, device=device)


def predict_anp(model, round_dir, detail, device):
    from nm_ai_ml.astar.attentive_np import predict_with_anp
    preds = []
    for si in range(detail["seeds_count"]):
        pred = predict_with_anp(model, Path(round_dir), detail, si, device=device)
        if pred is None:
            pred = np.ones((40, 40, 6)) / 6
        preds.append(pred)
    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="r16", choices=["r16", "weights", "r17"])
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gt_rounds = find_gt_rounds()
    rounds_dir = Path("data/rounds")
    logger.info("LOO eval — variant=%s, device=%s, %d GT rounds", args.variant, device, len(gt_rounds))

    from tqdm import tqdm
    from nm_ai_ml.astar.spatial_predictor_rf import predict_round as rf_predict

    results = {}

    for test_r in tqdm(gt_rounds, desc="LOO rounds"):
        train_r = [r for r in gt_rounds if r != test_r]
        rd = rounds_dir / f"round_{test_r}"
        detail = json.load(open(rd / "round_detail.json"))

        # RF (always honest)
        rf_preds = rf_predict(str(rd))

        # CNP LOO
        cnp_model = train_cnp_loo(train_r, device, epochs=80)
        cnp_preds = predict_cnp(cnp_model, rd, detail)

        # ANP LOO
        anp_model = train_anp_loo(train_r, device, epochs=60)
        anp_preds = predict_anp(anp_model, rd, detail, device)

        # CNP_v5 synth (only for r17 variant)
        cnp5_preds = None
        if args.variant == "r17":
            cnp5_model = train_cnp_synth_loo(train_r, device, epochs=40, synth_episodes=300)
            cnp5_preds = predict_cnp(cnp5_model, rd, detail)

        # Score
        rf_s, cnp_s, anp_s = [], [], []
        blend_configs = {}

        for si in range(detail["seeds_count"]):
            gt = np.array(json.load(open(rd / f"analysis/seed_{si}.json"))["ground_truth"])
            rf = np.maximum(rf_preds[si], 0.005); rf /= rf.sum(axis=2, keepdims=True)
            cnp = np.maximum(cnp_preds[si], 0.005); cnp /= cnp.sum(axis=2, keepdims=True)
            anp = np.maximum(anp_preds[si], 0.005); anp /= anp.sum(axis=2, keepdims=True)

            rf_s.append(score_prediction(rf, gt))
            cnp_s.append(score_prediction(cnp, gt))
            anp_s.append(score_prediction(anp, gt))

            if args.variant == "r16":
                blend = geo_blend([rf, cnp, anp], [0.50, 0.35, 0.15])
                blend_configs.setdefault("R16(50/35/15)", []).append(score_prediction(blend, gt))

            elif args.variant == "weights":
                for w_rf, w_cnp, w_anp, name in [
                    (0.50, 0.35, 0.15, "50/35/15"),
                    (0.45, 0.35, 0.20, "45/35/20"),
                    (0.40, 0.40, 0.20, "40/40/20"),
                    (0.55, 0.30, 0.15, "55/30/15"),
                    (0.60, 0.25, 0.15, "60/25/15"),
                    (0.45, 0.40, 0.15, "45/40/15"),
                ]:
                    blend = geo_blend([rf, cnp, anp], [w_rf, w_cnp, w_anp])
                    blend_configs.setdefault(name, []).append(score_prediction(blend, gt))

            elif args.variant == "r17" and cnp5_preds:
                cnp5 = np.maximum(cnp5_preds[si], 0.005); cnp5 /= cnp5.sum(axis=2, keepdims=True)
                blend3 = geo_blend([rf, cnp, anp], [0.50, 0.35, 0.15])
                blend4 = geo_blend([rf, cnp, anp, cnp5], [0.45, 0.25, 0.15, 0.15])
                blend_configs.setdefault("R16(3mod)", []).append(score_prediction(blend3, gt))
                blend_configs.setdefault("R17(4mod)", []).append(score_prediction(blend4, gt))

        results[test_r] = {
            "rf": np.mean(rf_s), "cnp": np.mean(cnp_s), "anp": np.mean(anp_s),
            **{k: np.mean(v) for k, v in blend_configs.items()}
        }
        tqdm.write(f"R{test_r}: RF={np.mean(rf_s):.1f} CNP={np.mean(cnp_s):.1f} ANP={np.mean(anp_s):.1f} " +
                   " ".join(f"{k}={np.mean(v):.1f}" for k, v in blend_configs.items()))

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("LOO RESULTS — %s", args.variant)
    logger.info("=" * 70)

    all_keys = ["rf", "cnp", "anp"] + sorted(set(k for r in results.values() for k in r if k not in ["rf", "cnp", "anp"]))
    header = f"{'Round':<8}" + "".join(f"{k:>12}" for k in all_keys)
    logger.info(header)
    logger.info("-" * len(header))

    for r in sorted(results):
        row = f"R{r:<6}"
        for k in all_keys:
            row += f"{results[r].get(k, 0):>11.1f} "
        logger.info(row)

    logger.info("-" * len(header))
    row = f"{'AVG':<8}"
    for k in all_keys:
        vals = [results[r].get(k, 0) for r in results if results[r].get(k, 0) > 0]
        row += f"{np.mean(vals):>11.1f} " if vals else f"{'':>12}"
    logger.info(row)
