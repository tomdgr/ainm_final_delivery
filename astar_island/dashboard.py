"""Experiment tracker dashboard. Run: uv run streamlit run dashboard.py"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NM i AI — Experiment Tracker", layout="wide")

ROUNDS_DIR = Path("data/rounds")

# Official server scores
SERVER_SCORES = {
    1: 19.3, 2: 48.9, 5: 43.9, 6: 6.6, 7: 29.5, 8: 85.9,
    9: 86.6, 10: 82.6, 11: 77.6, 12: 59.2, 13: 87.8,
    14: 82.4, 15: 74.2, 16: 86.2,
}
SERVER_RANKS = {
    1: "#72/117", 2: "#79/153", 5: "#104/144", 6: "#174/186", 7: "#162/199",
    8: "#43/214", 9: "#61/221", 10: "#57/238", 11: "#67/171", 12: "#39/146",
    13: "#48/186", 14: "#31/244", 15: "#167/262", 16: "#24/272",
}
ROUND_WEIGHTS = {r: 1.05 * 1.05 ** (r - 1) for r in range(1, 20)}


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
    for d in sorted(ROUNDS_DIR.iterdir()):
        if d.is_dir() and d.name != "rounds":
            r = int(d.name.split("_")[1])
            if (d / "analysis" / "seed_0.json").exists():
                rounds.append(r)
    return rounds


def load_pred(path):
    if path.exists():
        p = np.load(path)
        p = np.maximum(p, 0.005)
        p /= p.sum(axis=2, keepdims=True)
        return p
    return None


def load_gt(r, si):
    f = ROUNDS_DIR / f"round_{r}" / "analysis" / f"seed_{si}.json"
    if f.exists():
        return np.array(json.load(open(f))["ground_truth"])
    return None


def get_seeds(r):
    with open(ROUNDS_DIR / f"round_{r}" / "round_detail.json") as f:
        return json.load(f)["seeds_count"]


# ============================================================
# Define experiments
# ============================================================
EXPERIMENTS = {
    "Server (submitted)": {
        "description": "What we actually submitted. RF-based auto-submit pipeline.",
        "color": "🟢",
        "scores": SERVER_SCORES,
    },
    "RF (v2 pipeline)": {
        "description": "Random Forest spatial predictor. 16 features, trained per-round on observations.",
        "color": "🔵",
        "pred_dir": Path("data/model_predictions/rf"),
        "pattern": "round_{r}_seed_{si}.npy",
    },
    "ConvCNP (v2, no synth)": {
        "description": "ConvCNP U-Net trained on real GT only. 80 epochs. Known to overfit.",
        "color": "🟡",
        "pred_dir": Path("data/model_predictions/cnp"),
        "pattern": "round_{r}_seed_{si}.npy",
    },
    "ANP (Attentive NP)": {
        "description": "Transformer attention over observations. 60 epochs on GPU.",
        "color": "🟠",
        "pred_dir": Path("data/model_predictions/anp"),
        "pattern": "round_{r}_seed_{si}.npy",
    },
    "ConvCNP v5 (synth pretrained)": {
        "description": "ConvCNP pre-trained on 500 synthetic episodes (randomized params), then fine-tuned on real. 30+40 epochs.",
        "color": "🔴",
        "pred_dir": Path("data/model_predictions_v5/cnp_v5"),
        "pattern": "round_{r}_seed_{si}.npy",
    },
    "Ensemble R16 (RF+CNP+ANP)": {
        "description": "Geometric mean blend: 50% RF + 35% ConvCNP + 15% ANP. Submitted for R16.",
        "color": "🟣",
        "blend": [("data/model_predictions/rf", 0.50),
                  ("data/model_predictions/cnp", 0.35),
                  ("data/model_predictions/anp", 0.15)],
    },
    "Ensemble R17 (RF+CNP_v5)": {
        "description": "Geometric mean: 70% RF + 30% ConvCNP v5 (synth pretrained). Current best candidate.",
        "color": "⭐",
        "blend": [("data/model_predictions/rf", 0.70),
                  ("data/model_predictions_v5/cnp_v5", 0.30)],
    },
}

# ============================================================
# Score all experiments on GT rounds
# ============================================================
gt_rounds = find_gt_rounds()


@st.cache_data
def compute_all_scores():
    results = {}
    for exp_name, exp in EXPERIMENTS.items():
        exp_scores = {}

        if "scores" in exp:
            exp_scores = exp["scores"]
        elif "blend" in exp:
            for r in gt_rounds:
                ns = get_seeds(r)
                scores = []
                for si in range(ns):
                    gt = load_gt(r, si)
                    if gt is None:
                        continue
                    preds = []
                    for pred_dir, w in exp["blend"]:
                        p = load_pred(Path(pred_dir) / f"round_{r}_seed_{si}.npy")
                        if p is not None:
                            preds.append((p, w))
                    if len(preds) == len(exp["blend"]):
                        blend = geo_blend([p for p, _ in preds], [w for _, w in preds])
                        scores.append(score_prediction(blend, gt))
                if scores:
                    exp_scores[r] = np.mean(scores)
        elif "pred_dir" in exp:
            for r in gt_rounds:
                ns = get_seeds(r)
                scores = []
                for si in range(ns):
                    gt = load_gt(r, si)
                    if gt is None:
                        continue
                    p = load_pred(exp["pred_dir"] / f"round_{r}_seed_{si}.npy")
                    if p is not None:
                        scores.append(score_prediction(p, gt))
                if scores:
                    exp_scores[r] = np.mean(scores)

        results[exp_name] = exp_scores
    return results


all_scores = compute_all_scores()

# ============================================================
# Dashboard
# ============================================================
st.title("NM i AI — Experiment Tracker")

# Top metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Best Server Score", f"{max(SERVER_SCORES.values()):.1f}",
            f"R{max(SERVER_SCORES, key=SERVER_SCORES.get)}")
col2.metric("Best Rank", min(SERVER_RANKS.values(), key=lambda x: int(x.split('/')[0].replace('#', ''))))
recent = [SERVER_SCORES.get(r, 0) for r in [14, 15, 16]]
col3.metric("Recent Avg (R14-16)", f"{np.mean(recent):.1f}")
col4.metric("R17 Weight", "2.29x")

st.markdown("---")

# Experiment comparison table
st.header("Experiment Comparison")

# Build comparison table
table_data = []
for r in gt_rounds:
    row = {"Round": f"R{r}"}
    for exp_name in EXPERIMENTS:
        s = all_scores[exp_name].get(r, None)
        row[exp_name] = f"{s:.1f}" if s else "-"
    table_data.append(row)

# Averages
avg_row = {"Round": "**AVG**"}
for exp_name in EXPERIMENTS:
    vals = [v for v in all_scores[exp_name].values() if v]
    avg_row[exp_name] = f"**{np.mean(vals):.1f}**" if vals else "-"
table_data.append(avg_row)

st.dataframe(table_data, use_container_width=True)

# Chart
st.subheader("Scores per Round")
chart_data = {}
for exp_name in EXPERIMENTS:
    chart_data[exp_name] = [all_scores[exp_name].get(r, None) for r in gt_rounds]
df_chart = pd.DataFrame(chart_data, index=[f"R{r}" for r in gt_rounds])
st.line_chart(df_chart)

st.markdown("---")

# Experiment details
st.header("Experiment Details")
for exp_name, exp in EXPERIMENTS.items():
    vals = [v for v in all_scores[exp_name].values() if v]
    avg = np.mean(vals) if vals else 0
    best = max(vals) if vals else 0
    worst = min(vals) if vals else 0

    with st.expander(f"{exp.get('color', '')} **{exp_name}** — avg: {avg:.1f}, best: {best:.1f}, worst: {worst:.1f}"):
        st.write(exp["description"])
        if vals:
            st.write(f"- **Average**: {avg:.1f}")
            st.write(f"- **Best**: {best:.1f}")
            st.write(f"- **Worst**: {worst:.1f}")
            st.write(f"- **Std**: {np.std(vals):.1f}")
            st.write(f"- **Rounds scored**: {len(vals)}")

            # Does it beat server?
            server_avg = np.mean([SERVER_SCORES.get(r, 0) for r in all_scores[exp_name] if r in SERVER_SCORES])
            if server_avg > 0:
                delta = avg - server_avg
                st.write(f"- **vs Server avg**: {delta:+.1f}")

st.markdown("---")

# Submission decision
st.header("Round 17 Submission")
st.write("**Available predictions for R17:**")
for exp_name, exp in EXPERIMENTS.items():
    if "pred_dir" in exp:
        f = exp["pred_dir"] / "round_17_seed_0.npy"
        st.write(f"- {exp.get('color', '')} {exp_name}: {'✅ Ready' if f.exists() else '❌ Missing'}")
    elif "blend" in exp:
        all_ready = all(
            (Path(d) / "round_17_seed_0.npy").exists()
            for d, _ in exp["blend"]
        )
        st.write(f"- {exp.get('color', '')} {exp_name}: {'✅ Ready' if all_ready else '❌ Missing components'}")
