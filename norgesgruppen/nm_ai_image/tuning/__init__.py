import logging
from pathlib import Path

import mlflow

from nm_ai_image.config.task import TaskConfig
from nm_ai_image.config.tuning import TuningConfig
from nm_ai_image.tuning.beam_search import run_beam_search
from nm_ai_image.tuning.evolutionary_tuner import run_evolutionary
from nm_ai_image.tuning.monte_carlo import run_monte_carlo
from nm_ai_image.tuning.objective import TuningObjective
from nm_ai_image.tuning.optuna_tuner import run_optuna_study
from nm_ai_image.tuning.results import StudyResult, flatten_params
from nm_ai_image.tuning.sensitivity import run_sensitivity

logger = logging.getLogger(__name__)

OPTUNA_METHODS = {"tpe", "random", "grid", "cmaes"}
EVOLUTIONARY_METHODS = {"ga", "pso", "gjo", "macla"}


class HyperparameterOptimizer:
    def __init__(self, task_config: TaskConfig, tuning_config: TuningConfig):
        self.task_config = task_config
        self.tuning_config = tuning_config

    def run(self, output_dir: str | Path | None = None) -> StudyResult:
        if self.tuning_config.method == "beam":
            return self._run_beam(output_dir)

        run_name = (
            f"{self.tuning_config.method}_{self.task_config.backbone_name}"
            f"_{self.tuning_config.tune_scope}_{self.tuning_config.n_trials}t"
        )

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "method": self.tuning_config.method,
                "backbone_name": self.task_config.backbone_name,
                "tune_scope": self.tuning_config.tune_scope,
                "n_trials": self.tuning_config.n_trials,
                "metric": self.tuning_config.metric,
            })

            logger.info(
                "Starting hyperparameter optimization: method=%s, backbone=%s, n_trials=%d",
                self.tuning_config.method,
                self.task_config.backbone_name,
                self.tuning_config.n_trials,
            )

            objective = TuningObjective(self.task_config, self.tuning_config)
            result = self._dispatch(objective)

            mlflow.log_metric("best_value", result.best_value)
            if result.method != "sensitivity":
                best_prefixed = {f"best_{k}": v for k, v in result.best_params.items()}
                mlflow.log_params(flatten_params(best_prefixed))

            if output_dir is not None:
                result.save(output_dir)
                self._log_artifacts(Path(output_dir))

            return result

    def _run_beam(self, output_dir: str | Path | None = None) -> StudyResult:
        run_name = (
            f"beam_{self.task_config.backbone_name}"
            f"_w{self.tuning_config.beam_width}"
        )

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "method": "beam",
                "backbone_name": self.task_config.backbone_name,
                "beam_width": self.tuning_config.beam_width,
                "beam_trial_epochs": self.tuning_config.beam_trial_epochs,
                "beam_trial_patience": self.tuning_config.beam_trial_patience,
                "beam_hpo_method": self.tuning_config.beam_hpo_method or "none",
                "metric": self.tuning_config.metric,
            })

            logger.info(
                "Starting beam search: width=%d, epochs=%d",
                self.tuning_config.beam_width,
                self.tuning_config.beam_trial_epochs,
            )

            result = run_beam_search(
                self.task_config,
                self.tuning_config,
                output_dir=output_dir,
            )

            mlflow.log_metric("best_value", result.best_value)
            best_prefixed = {f"best_{k}": v for k, v in result.best_params.items()}
            mlflow.log_params(flatten_params(best_prefixed))

            if output_dir is not None:
                result.save(output_dir)
                self._log_artifacts(Path(output_dir))

            return result

    def _dispatch(self, objective: TuningObjective) -> StudyResult:
        method = self.tuning_config.method
        if method in OPTUNA_METHODS:
            return run_optuna_study(objective, self.tuning_config)
        if method in EVOLUTIONARY_METHODS:
            return run_evolutionary(objective, self.tuning_config)
        if method == "sensitivity":
            return run_sensitivity(objective, self.tuning_config)
        if method == "monte_carlo":
            return run_monte_carlo(objective, self.tuning_config)
        raise ValueError(f"Unknown tuning method: {method}")

    @staticmethod
    def _log_artifacts(output_dir: Path) -> None:
        try:
            for json_file in output_dir.glob("*.json"):
                mlflow.log_artifact(str(json_file))

            plots_dir = output_dir / "plots"
            if plots_dir.exists():
                for plot_file in plots_dir.glob("*.png"):
                    mlflow.log_artifact(str(plot_file), "plots")
        except Exception:
            logger.warning("Could not log MLflow artifacts — files saved to outputs/ instead")


__all__ = [
    "HyperparameterOptimizer",
    "TuningObjective",
    "StudyResult",
]
