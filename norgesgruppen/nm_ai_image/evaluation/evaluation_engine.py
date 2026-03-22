"""Evaluation engine for image classification analysis and visualization."""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns

from nm_ai_image.config.task import TaskConfig, TaskType
from nm_ai_image.evaluation.metrics import calculate_all_metrics

logger = logging.getLogger(__name__)

_PLOT_DPI = 150


class EvaluationEngine:
    """Generates evaluation plots, logs metrics and artifacts to MLflow."""

    def __init__(self, config: TaskConfig, y_true: np.ndarray, y_pred: np.ndarray,
                 y_prob: np.ndarray | None = None, class_names: list[str] | None = None):
        self.config = config
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.class_names = class_names or [str(i) for i in range(config.num_classes)]

        self.plots_dir = Path(config.output_dir) / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def generate_report(self) -> dict[str, float]:
        task_str = self.config.task_type.value
        metrics = calculate_all_metrics(task_str, self.y_pred, self.y_true, self.config.num_classes, self.y_prob)

        # Log metrics to MLflow
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                try:
                    mlflow.log_metric(f"eval_{key}", val)
                except Exception:
                    pass

        # Generate plots
        if self.config.task_type == TaskType.CLASSIFICATION:
            self._generate_classification_plots(metrics)
        elif self.config.task_type == TaskType.SEMANTIC_SEGMENTATION:
            self._generate_segmentation_plots(metrics)

        # Log plots as MLflow artifacts
        try:
            for plot_file in self.plots_dir.glob("*.png"):
                mlflow.log_artifact(str(plot_file), "evaluation_plots")
        except Exception:
            pass

        # Save metrics JSON
        metrics_path = Path(self.config.output_dir) / "evaluation_metrics.json"
        serializable = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str, list))}
        with open(metrics_path, "w") as f:
            json.dump(serializable, f, indent=2)

        logger.info("Evaluation complete — %s", {k: f"{v:.4f}" for k, v in metrics.items() if isinstance(v, float)})
        return metrics

    def _generate_classification_plots(self, metrics: dict):
        plots = [
            ("confusion_matrix.png", self._plot_confusion_matrix),
            ("class_distribution.png", self._plot_class_distribution),
            ("per_class_f1.png", self._plot_per_class_f1),
        ]

        if self.y_prob is not None:
            plots.append(("confidence_distribution.png", self._plot_confidence_distribution))
            plots.append(("predicted_vs_actual_dist.png", self._plot_prediction_distribution))

        for filename, plot_fn in plots:
            try:
                fig = plot_fn()
                fig.savefig(self.plots_dir / filename, dpi=_PLOT_DPI, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                logger.warning("Failed to generate %s: %s", filename, e)

    def _generate_segmentation_plots(self, metrics: dict):
        try:
            fig = self._plot_segmentation_summary(metrics)
            fig.savefig(self.plots_dir / "segmentation_summary.png", dpi=_PLOT_DPI, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            logger.warning("Failed to generate segmentation plot: %s", e)

    def _plot_confusion_matrix(self) -> plt.Figure:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(self.y_true, self.y_pred)
        n_classes = len(cm)

        fig, ax = plt.subplots(figsize=(max(8, n_classes * 0.5), max(6, n_classes * 0.4)))

        # Normalize for display
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        sns.heatmap(cm_norm, annot=True if n_classes <= 20 else False, fmt=".2f",
                    cmap="Blues", xticklabels=self.class_names[:n_classes],
                    yticklabels=self.class_names[:n_classes], ax=ax,
                    vmin=0, vmax=1, square=True)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix (n={len(self.y_true)}, acc={np.mean(self.y_true == self.y_pred):.3f})")

        if n_classes > 10:
            plt.xticks(rotation=90, fontsize=7)
            plt.yticks(fontsize=7)

        plt.tight_layout()
        return fig

    def _plot_class_distribution(self) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, data, title in [
            (axes[0], self.y_true, "True Label Distribution"),
            (axes[1], self.y_pred, "Predicted Label Distribution"),
        ]:
            unique, counts = np.unique(data, return_counts=True)
            ax.bar(unique, counts, alpha=0.7)
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return fig

    def _plot_per_class_f1(self) -> plt.Figure:
        from sklearn.metrics import f1_score

        n_classes = self.config.num_classes
        f1_per_class = f1_score(self.y_true, self.y_pred, average=None, zero_division=0, labels=range(n_classes))

        fig, ax = plt.subplots(figsize=(max(10, n_classes * 0.3), 5))

        colors = plt.cm.RdYlGn(f1_per_class)
        bars = ax.bar(range(n_classes), f1_per_class, color=colors, edgecolor="black", linewidth=0.3)

        ax.axhline(y=np.mean(f1_per_class), color="red", linestyle="--", linewidth=2,
                   label=f"Mean F1: {np.mean(f1_per_class):.3f}")
        ax.set_xlabel("Class")
        ax.set_ylabel("F1 Score")
        ax.set_title("Per-Class F1 Score")
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")

        if n_classes <= 30:
            ax.set_xticks(range(n_classes))
            ax.set_xticklabels(self.class_names[:n_classes], rotation=90, fontsize=7)

        plt.tight_layout()
        return fig

    def _plot_confidence_distribution(self) -> plt.Figure:
        max_probs = np.max(self.y_prob, axis=1)
        correct = self.y_true == self.y_pred

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(max_probs[correct], bins=50, alpha=0.6, label="Correct", color="green", density=True)
        ax.hist(max_probs[~correct], bins=50, alpha=0.6, label="Incorrect", color="red", density=True)
        ax.set_xlabel("Max Predicted Probability")
        ax.set_ylabel("Density")
        ax.set_title("Confidence Distribution: Correct vs Incorrect Predictions")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def _plot_prediction_distribution(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 5))
        mean_probs = np.mean(self.y_prob, axis=0)
        ax.bar(range(len(mean_probs)), mean_probs, alpha=0.7)
        ax.set_xlabel("Class")
        ax.set_ylabel("Mean Predicted Probability")
        ax.set_title("Average Prediction Distribution Across Classes")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        return fig

    def _plot_segmentation_summary(self, metrics: dict) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 4))
        metric_names = ["pixel_accuracy", "mIoU", "dice"]
        values = [metrics.get(m, 0) for m in metric_names]
        colors = plt.cm.RdYlGn(np.array(values))

        bars = ax.barh(metric_names, values, color=colors)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center")

        ax.set_xlim(0, 1.1)
        ax.set_title("Segmentation Metrics")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        return fig


def log_sample_predictions(images: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                           class_names: list[str] | None = None, n_samples: int = 16,
                           output_dir: str = "outputs"):
    """Log a grid of sample predictions as an image artifact."""
    n = min(n_samples, len(images))
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
    axes = axes.flatten() if n > 1 else [axes]

    for i in range(n):
        ax = axes[i]
        img = images[i]
        if img.shape[0] == 3:  # CHW -> HWC
            img = np.transpose(img, (1, 2, 0))
        # Denormalize
        img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)

        ax.imshow(img)
        true_label = class_names[y_true[i]] if class_names else str(y_true[i])
        pred_label = class_names[y_pred[i]] if class_names else str(y_pred[i])
        color = "green" if y_true[i] == y_pred[i] else "red"
        ax.set_title(f"T:{true_label}\nP:{pred_label}", fontsize=8, color=color)
        ax.axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Sample Predictions (Green=Correct, Red=Wrong)", fontsize=12)
    plt.tight_layout()

    out_path = Path(output_dir) / "plots" / "sample_predictions.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    try:
        mlflow.log_artifact(str(out_path), "evaluation_plots")
    except Exception:
        pass
