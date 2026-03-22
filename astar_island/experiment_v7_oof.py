"""V7 with proper OOF evaluation. Deep CNP ensemble scored honestly via LOO."""
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
    return sorted(rounds)


def gated_ensemble(rf_pred, neural_pred, obs_count=None):
    eps = 1e-8
    rf = np.maximum(rf_pred, 0.005); rf /= rf.sum(axis=2, keepdims=True)
    neural = np.maximum(neural_pred, 0.005); neural /= neural.sum(axis=2, keepdims=True)
    m = 0.5 * (rf + neural)
    jsd = 0.5 * np.sum(rf * np.log((rf + eps) / (m + eps)), axis=2) + \
          0.5 * np.sum(neural * np.log((neural + eps) / (m + eps)), axis=2)
    w_neural = 0.5 * np.exp(-10 * jsd)
    if obs_count is not None:
        obs_boost = np.minimum(obs_count / 5.0, 1.0)
        w_neural = w_neural + 0.2 * obs_boost
        w_neural = np.minimum(w_neural, 0.6)
    w_rf = 1.0 - w_neural
    blend = w_rf[:, :, np.newaxis] * rf + w_neural[:, :, np.newaxis] * neural
    blend = np.maximum(blend, 0.005); blend /= blend.sum(axis=2, keepdims=True)
    return blend, w_neural.mean()


def train_deep_cnp(train_rounds, device, n_models=3, synth_episodes=500, synth_epochs=30, finetune_epochs=60):
    """Train deep CNP ensemble on train_rounds only."""
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    from nm_ai_ml.astar.convcnp import AstarUNet, build_training_episodes
    from nm_ai_ml.astar.synth_data import generate_synthetic_episodes

    rounds_dir = Path("data/rounds")
    synth = generate_synthetic_episodes(n_episodes=synth_episodes, n_workers=32)

    models = []
    for mi in range(n_models):
        model = AstarUNet().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for ep in range(synth_epochs):
            model.train()
            random.seed(42 + mi * 1000 + ep)
            random.shuffle(synth)
            for bs in range(0, len(synth), 16):
                batch = synth[bs:bs+16]
                x = torch.stack([torch.FloatTensor(e[0]) for e in batch]).to(device)
                y = torch.stack([torch.FloatTensor(e[1]) for e in batch]).to(device)
                pred = model(x)
                loss = F.kl_div(torch.log(pred.clamp(min=1e-6)), y.clamp(min=1e-6), reduction='batchmean')
                opt.zero_grad(); loss.backward(); opt.step()

        train_dirs = [rounds_dir / f"round_{r}" for r in train_rounds]
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        for ep in range(finetune_epochs):
            model.train()
            for rd in train_dirs:
                try:
                    episodes = build_training_episodes(rd, n_episodes_per_round=8,
                                                       seed=ep*1000+mi*10000+hash(str(rd))%1000)
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


def predict_with_deep_cnp(models, round_dir, detail, device):
    from nm_ai_ml.astar.convcnp import predict
    preds = []
    for si in range(detail["seeds_count"]):
        initial_map = np.array(detail["initial_states"][si]["grid"])
        obs_file = Path(round_dir) / "observations" / f"seed_{si}.json"
        obs = json.load(open(obs_file)) if obs_file.exists() else []
        model_preds = [predict(m, obs, initial_map, temperature=1.1, tta=True) for m in models]
        avg = np.mean(model_preds, axis=0)
        avg = np.maximum(avg, 0.005); avg /= avg.sum(axis=2, keepdims=True)
        preds.append(avg)
    return preds


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--synth-episodes", type=int, default=500)
    parser.add_argument("--n-cnp-models", type=int, default=3)
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("V7 OOF — Device: %s, Round: %d", device, args.round)

    gt_rounds = find_gt_rounds(exclude=args.round)
    rounds_dir = Path("data/rounds")
    logger.info("GT rounds: %s (%d)", gt_rounds, len(gt_rounds))

    # ============================================================
    # Phase 1: OOF evaluation — LOO for deep CNP, honest for RF
    # ============================================================
    logger.info("\n=== PHASE 1: OOF EVALUATION ===")

    from tqdm import tqdm
    from nm_ai_ml.astar.spatial_predictor_rf import predict_round as rf_predict

    rf_oof = {}
    cnp_oof = {}

    # RF: per-round training, always honest
    logger.info("RF OOF predictions...")
    for r in tqdm(gt_rounds, desc="RF OOF"):
        try:
            rf_oof[r] = rf_predict(f"data/rounds/round_{r}")
        except:
            pass

    # Deep CNP: LOO — train on all EXCEPT test round
    logger.info("Deep CNP OOF (LOO, %d models per fold)...", args.n_cnp_models)
    for r in tqdm(gt_rounds, desc="CNP LOO"):
        train_r = [x for x in gt_rounds if x != r]
        t0 = time.time()
        models = train_deep_cnp(train_r, device, n_models=args.n_cnp_models,
                                synth_episodes=args.synth_episodes, synth_epochs=20, finetune_epochs=30)
        rd = rounds_dir / f"round_{r}"
        with open(rd / "round_detail.json") as f:
            detail = json.load(f)
        cnp_oof[r] = predict_with_deep_cnp(models, rd, detail, device)
        tqdm.write(f"R{r}: deep CNP LOO done ({time.time()-t0:.0f}s)")

    # Score everything
    logger.info("\n=== OOF SCORES ===")
    rf_scores, cnp_scores, gated_scores, simple_scores = {}, {}, {}, {}

    for r in gt_rounds:
        if r not in rf_oof or r not in cnp_oof:
            continue
        rd = rounds_dir / f"round_{r}"
        with open(rd / "round_detail.json") as f:
            detail = json.load(f)

        r_rf, r_cnp, r_gated, r_simple = [], [], [], []
        for si in range(detail["seeds_count"]):
            gt_file = rd / "analysis" / f"seed_{si}.json"
            if not gt_file.exists():
                continue
            gt = np.array(json.load(open(gt_file))["ground_truth"])

            rf_p = rf_oof[r][si]
            cnp_p = cnp_oof[r][si]

            obs_count = np.zeros((40, 40))
            obs_file = rd / "observations" / f"seed_{si}.json"
            if obs_file.exists():
                for o in json.load(open(obs_file)):
                    vp = o["viewport"]
                    obs_count[vp["y"]:vp["y"]+vp["h"], vp["x"]:vp["x"]+vp["w"]] += 1

            r_rf.append(score_prediction(rf_p, gt))
            r_cnp.append(score_prediction(cnp_p, gt))
            gated, _ = gated_ensemble(rf_p, cnp_p, obs_count)
            r_gated.append(score_prediction(gated, gt))
            simple = geo_blend([rf_p, cnp_p], [0.55, 0.45])
            r_simple.append(score_prediction(simple, gt))

        rf_scores[r] = np.mean(r_rf)
        cnp_scores[r] = np.mean(r_cnp)
        gated_scores[r] = np.mean(r_gated)
        simple_scores[r] = np.mean(r_simple)

    # Report
    logger.info("\n" + "=" * 70)
    logger.info("V7 OOF RESULTS (HONEST — LOO for CNP)")
    logger.info("=" * 70)
    logger.info("%-8s %8s %8s %10s %10s", "Round", "RF", "DeepCNP", "Gated", "Simple")
    logger.info("-" * 50)
    for r in sorted(gt_rounds):
        logger.info("R%-7d %7.1f  %7.1f  %9.1f  %9.1f",
                    r, rf_scores.get(r, 0), cnp_scores.get(r, 0),
                    gated_scores.get(r, 0), simple_scores.get(r, 0))
    logger.info("-" * 50)
    rf_avg = np.mean(list(rf_scores.values()))
    cnp_avg = np.mean(list(cnp_scores.values()))
    gated_avg = np.mean(list(gated_scores.values()))
    simple_avg = np.mean(list(simple_scores.values()))
    logger.info("%-8s %7.1f  %7.1f  %9.1f  %9.1f", "AVG", rf_avg, cnp_avg, gated_avg, simple_avg)
    logger.info("\nGated vs RF: %+.1f", gated_avg - rf_avg)
    logger.info("Simple vs RF: %+.1f", simple_avg - rf_avg)

    # ============================================================
    # Phase 2: Full prediction for target round
    # ============================================================
    logger.info("\n=== PHASE 2: R%d PREDICTION ===", args.round)
    models = train_deep_cnp(gt_rounds, device, n_models=args.n_cnp_models,
                            synth_episodes=args.synth_episodes, synth_epochs=30, finetune_epochs=60)

    target_dir = rounds_dir / f"round_{args.round}"
    with open(target_dir / "round_detail.json") as f:
        detail = json.load(f)

    cnp_preds = predict_with_deep_cnp(models, target_dir, detail, device)
    rf_preds = rf_predict(str(target_dir))

    out = Path("data/model_predictions_v7_oof")
    for name, preds in [("rf", rf_preds), ("cnp_deep", cnp_preds)]:
        d = out / name; d.mkdir(parents=True, exist_ok=True)
        for si, p in enumerate(preds):
            np.save(d / f"round_{args.round}_seed_{si}.npy", p)

    # Gated ensemble
    d = out / "gated"; d.mkdir(parents=True, exist_ok=True)
    for si in range(detail["seeds_count"]):
        obs_count = np.zeros((40, 40))
        obs_file = target_dir / "observations" / f"seed_{si}.json"
        if obs_file.exists():
            for o in json.load(open(obs_file)):
                vp = o["viewport"]
                obs_count[vp["y"]:vp["y"]+vp["h"], vp["x"]:vp["x"]+vp["w"]] += 1
        gated, w = gated_ensemble(rf_preds[si], cnp_preds[si], obs_count)
        np.save(d / f"round_{args.round}_seed_{si}.npy", gated)
        logger.info("Seed %d: gated (avg_w_neural=%.3f)", si, w)

    total = time.time() - time.time()
    logger.info("\nV7 OOF COMPLETE")
