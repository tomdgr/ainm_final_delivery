from dataclasses import dataclass, field
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

_MIN_TRIALS_FOR_PLOTS = 2
_MIN_TRIALS_FOR_IMPORTANCE = 4
_PLOT_DPI = 150
_TOP_N_TRIALS = 5


def format_run_name(prefix: str, index: int, params: dict) -> str:
    parts = [f"{prefix}_{index:03d}"]
    for k, v in sorted(params.items()):
        if isinstance(v, list):
            val_str = "-".join(str(x) for x in v)
        elif isinstance(v, float):
            val_str = f"{v:.3g}"
        else:
            val_str = str(v)
        parts.append(f"{k}_{val_str}")
    return "_".join(parts)


def flatten_params(params: dict) -> dict:
    flat = {}
    for k, v in params.items():
        if isinstance(v, list):
            flat[k] = str(v)
        elif isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat[f"{k}_{sub_k}"] = sub_v
        else:
            flat[k] = v
    return flat


def _build_trials_dataframe(study: optuna.Study) -> pd.DataFrame:
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    records = []
    for t in complete:
        row = dict(t.params)
        row["value"] = t.value
        row["trial"] = t.number
        records.append(row)
    return pd.DataFrame(records)


def _plot_trial_progression(df: pd.DataFrame, direction: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))

    values = df["value"].values
    trials = df["trial"].values

    if direction == "minimize":
        running_best = np.minimum.accumulate(values)
        best_val = values.min()
    else:
        running_best = np.maximum.accumulate(values)
        best_val = values.max()

    ax.scatter(trials, values, alpha=0.5, s=30, c="steelblue", label="Trials")
    ax.plot(trials, running_best, color="red", linewidth=2, label="Running best")
    ax.axhline(y=best_val, color="green", linestyle="--", alpha=0.5,
               label=f"Best: {best_val:.4f}")

    ax.set_xlabel("Trial")
    ax.set_ylabel("Objective Value")
    ax.set_title("Optimization Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def _plot_param_heatmap(df: pd.DataFrame) -> plt.Figure:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Not enough numeric parameters", ha="center", va="center")
        return fig

    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(max(6, len(corr.columns) * 1.2), max(5, len(corr.columns))))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, ax=ax, vmin=-1, vmax=1,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Parameter-Metric Correlation")
    plt.tight_layout()
    return fig


def _plot_parallel_coordinates(df: pd.DataFrame, direction: str) -> plt.Figure:
    param_cols = [c for c in df.columns if c not in ("value", "trial")]
    if not param_cols:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No parameters to plot", ha="center", va="center")
        return fig

    norm_df = df[param_cols].copy()
    tick_labels = {}
    for col in param_cols:
        if norm_df[col].dtype == object:
            categories = sorted(norm_df[col].unique())
            mapping = {v: i / max(len(categories) - 1, 1) for i, v in enumerate(categories)}
            norm_df[col] = norm_df[col].map(mapping)
            tick_labels[col] = categories
        else:
            col_min, col_max = norm_df[col].min(), norm_df[col].max()
            if col_max > col_min:
                norm_df[col] = (norm_df[col] - col_min) / (col_max - col_min)
            else:
                norm_df[col] = 0.5

    fig, ax = plt.subplots(figsize=(max(10, len(param_cols) * 2.5), 6))

    cmap = plt.cm.RdYlGn_r if direction == "minimize" else plt.cm.RdYlGn
    norm = Normalize(vmin=df["value"].min(), vmax=df["value"].max())

    x = range(len(param_cols))
    for _, row in norm_df.iterrows():
        idx = row.name
        color = cmap(norm(df.loc[idx, "value"]))
        ax.plot(x, [row[c] for c in param_cols], alpha=0.25, color=color, linewidth=1)

    best_idx = df["value"].idxmin() if direction == "minimize" else df["value"].idxmax()
    best_row = norm_df.loc[best_idx]
    ax.plot(x, [best_row[c] for c in param_cols], color="blue", linewidth=3,
            label=f"Best (val={df.loc[best_idx, 'value']:.4f})", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(param_cols, rotation=45, ha="right")
    ax.set_ylabel("Normalized Value")
    ax.set_title("Parallel Coordinates (colored by metric)")
    ax.legend(loc="upper right")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, label="Objective Value")

    plt.tight_layout()
    return fig


def _plot_param_importance(study: optuna.Study) -> plt.Figure:
    importances = optuna.importance.get_param_importances(study)

    names = list(importances.keys())
    values = list(importances.values())

    fig, ax = plt.subplots(figsize=(10, max(3, len(names) * 0.5)))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(names)))
    bars = ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Hyperparameter Importance (fANOVA)")
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)

    plt.tight_layout()
    return fig


def _plot_top_trials_table(study: optuna.Study) -> plt.Figure:
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    sorted_trials = sorted(complete, key=lambda t: t.value)[:_TOP_N_TRIALS]
    best = sorted_trials[0]

    param_names = list(best.params.keys())
    col_labels = ["#", "Value"] + param_names

    cell_data = []
    for t in sorted_trials:
        row = [str(t.number), f"{t.value:.4f}"]
        for p in param_names:
            v = t.params.get(p, "")
            row.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        cell_data.append(row)

    n_cols = len(col_labels)
    n_rows = len(cell_data)
    fig_width = max(8, n_cols * 1.8)
    fig_height = max(2.5, n_rows * 0.5 + 1.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for j in range(n_cols):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for j in range(n_cols):
        table[1, j].set_facecolor("#E2EFDA")

    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    subtitle = f"{len(study.trials)} trials ({len(complete)} complete, {n_pruned} pruned)"
    ax.set_title(f"Top {min(_TOP_N_TRIALS, n_rows)} Trials\n{subtitle}",
                 fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    return fig


def generate_tuning_plots(study: optuna.Study, output_dir: Path, direction: str = "minimize") -> list[Path]:
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(complete) < _MIN_TRIALS_FOR_PLOTS:
        return []

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = _build_trials_dataframe(study)
    saved = []

    plots = [
        ("trial_progression.png", lambda: _plot_trial_progression(df, direction)),
        ("param_correlation.png", lambda: _plot_param_heatmap(df)),
        ("parallel_coordinates.png", lambda: _plot_parallel_coordinates(df, direction)),
        ("top_trials.png", lambda: _plot_top_trials_table(study)),
    ]

    if len(complete) >= _MIN_TRIALS_FOR_IMPORTANCE:
        plots.append(("param_importance.png", lambda: _plot_param_importance(study)))

    for filename, plot_fn in plots:
        fig = plot_fn()
        path = plots_dir / filename
        fig.savefig(path, dpi=_PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    logger.info("Saved %d tuning plots to %s", len(saved), plots_dir)
    return saved


@dataclass
class StudyResult:
    best_params: dict
    best_value: float
    n_trials: int
    n_pruned: int
    method: str
    model_name: str
    metric: str
    study: optuna.Study | None = field(default=None, repr=False)

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "n_trials": self.n_trials,
            "n_pruned": self.n_pruned,
            "method": self.method,
            "model_name": self.model_name,
            "metric": self.metric,
        }

        with open(output_dir / "best_trial.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved best trial to %s", output_dir / "best_trial.json")

        if self.study is not None and self.study.trials:
            trials_data = []
            for trial in self.study.trials:
                trials_data.append({
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "state": trial.state.name,
                })
            with open(output_dir / "all_trials.json", "w") as f:
                json.dump(trials_data, f, indent=2)

        if self.study is not None:
            generate_tuning_plots(self.study, output_dir)
