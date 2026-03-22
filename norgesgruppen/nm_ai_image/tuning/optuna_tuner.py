import logging

import mlflow
import numpy as np
import optuna
from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner
from optuna.samplers import CmaEsSampler, GridSampler, RandomSampler, TPESampler

from nm_ai_image.config.tuning import TuningConfig
from nm_ai_image.tuning.objective import TuningObjective
from nm_ai_image.tuning.results import StudyResult, format_run_name
from nm_ai_image.tuning.search_spaces import MODEL_SPACES, TRAINING_SPACE

logger = logging.getLogger(__name__)

_HYPERBAND_REDUCTION_FACTOR = 3


def _build_sampler(tuning_config: TuningConfig, model_name: str) -> optuna.samplers.BaseSampler:
    seed = tuning_config.seed
    if tuning_config.method == "tpe":
        return TPESampler(seed=seed)
    if tuning_config.method == "random":
        return RandomSampler(seed=seed)
    if tuning_config.method == "cmaes":
        return CmaEsSampler(seed=seed)
    if tuning_config.method == "grid":
        return GridSampler(_build_grid_space(model_name, tuning_config.tune_scope, tuning_config.grid_points), seed=seed)
    raise ValueError(f"Unknown Optuna method: {tuning_config.method}")


def _build_pruner(tuning_config: TuningConfig) -> optuna.pruners.BasePruner:
    if tuning_config.pruner == "median":
        return MedianPruner(n_warmup_steps=tuning_config.pruner_warmup_steps)
    if tuning_config.pruner == "hyperband":
        return HyperbandPruner(
            min_resource=tuning_config.pruner_warmup_steps,
            max_resource=tuning_config.trial_epochs,
            reduction_factor=_HYPERBAND_REDUCTION_FACTOR,
        )
    return NopPruner()


def _build_grid_space(model_name: str, scope: str, grid_points: int = 5) -> dict[str, list]:
    space = {}
    if scope in ("all", "training"):
        for name, spec in TRAINING_SPACE.items():
            space[name] = _discretize(spec, grid_points)
    if scope in ("all", "architecture"):
        for name, spec in MODEL_SPACES.get(model_name, {}).items():
            space[name] = _discretize(spec, grid_points)
    return space


def _discretize(spec: tuple, grid_points: int = 5) -> list:
    kind = spec[0]
    if kind == "categorical":
        return spec[1]
    if kind == "int":
        return list(range(spec[1], spec[2] + 1))
    if kind in ("float", "float_log"):
        if kind == "float_log":
            return list(np.logspace(np.log10(spec[1]), np.log10(spec[2]), grid_points))
        return list(np.linspace(spec[1], spec[2], grid_points))
    raise ValueError(f"Cannot discretize spec type: {kind}")


def run_optuna_study(
    objective: TuningObjective,
    tuning_config: TuningConfig,
) -> StudyResult:
    study_name = f"tune_{objective.model_name}_{tuning_config.method}"

    sampler = _build_sampler(tuning_config, objective.model_name)
    pruner = _build_pruner(tuning_config)

    study = optuna.create_study(
        study_name=study_name,
        direction=tuning_config.direction,
        sampler=sampler,
        pruner=pruner,
        storage=tuning_config.storage,
        load_if_exists=True,
    )

    logger.info(
        "Starting %s study '%s' with %d trials",
        tuning_config.method, study_name, tuning_config.n_trials,
    )

    def logged_objective(trial):
        try:
            value = objective(trial)
        except optuna.TrialPruned:
            with mlflow.start_run(run_name=format_run_name("trial", trial.number, trial.params), nested=True):
                mlflow.log_params(trial.params)
                mlflow.set_tag("status", "pruned")
            raise
        with mlflow.start_run(run_name=format_run_name("trial", trial.number, trial.params), nested=True):
            mlflow.log_params(trial.params)
            mlflow.log_metric(tuning_config.metric, value)
        return value

    study.optimize(logged_objective, n_trials=tuning_config.n_trials)

    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    logger.info("Study complete — %d finished, %d pruned", n_complete, n_pruned)

    return StudyResult(
        best_params=study.best_params,
        best_value=study.best_value,
        n_trials=len(study.trials),
        n_pruned=n_pruned,
        method=tuning_config.method,
        model_name=objective.model_name,
        metric=tuning_config.metric,
        study=study,
    )
