import logging

import mlflow
import numpy as np

from nm_ai_image.config.tuning import TuningConfig
from nm_ai_image.tuning.evolutionary_tuner import ParameterEncoder
from nm_ai_image.tuning.objective import TuningObjective
from nm_ai_image.tuning.results import StudyResult, flatten_params, format_run_name

logger = logging.getLogger(__name__)

_MLFLOW_CHILD_RUN_THRESHOLD = 100
_LOG_INTERVAL = 10


def run_monte_carlo(objective: TuningObjective, tuning_config: TuningConfig) -> StudyResult:
    encoder = ParameterEncoder(objective.model_name, tuning_config.tune_scope)
    rng = np.random.default_rng(tuning_config.seed)

    n_simulations = tuning_config.n_trials

    logger.info(
        "Starting Monte Carlo simulation: %d simulations, %d dimensions",
        n_simulations, encoder.n_dims,
    )

    all_results = []
    best_value = np.inf
    best_params = {}

    for i in range(n_simulations):
        vector = rng.uniform(0, 1, size=encoder.n_dims)
        training_params, model_kwargs = encoder.decode(vector)
        value = objective.evaluate(training_params, model_kwargs)

        merged_params = {**training_params, **model_kwargs}
        all_results.append({"params": merged_params, "value": value})

        if n_simulations <= _MLFLOW_CHILD_RUN_THRESHOLD:
            with mlflow.start_run(run_name=format_run_name("sim", i + 1, merged_params), nested=True):
                mlflow.log_params(flatten_params(merged_params))
                mlflow.log_metric(tuning_config.metric, value)

        if value < best_value:
            best_value = value
            best_params = merged_params

        if (i + 1) % _LOG_INTERVAL == 0:
            logger.info("MCS simulation %d/%d — best so far: %.4f", i + 1, n_simulations, best_value)

    values = [r["value"] for r in all_results]
    logger.info(
        "Monte Carlo complete — mean: %.4f, std: %.4f, min: %.4f, max: %.4f",
        np.mean(values), np.std(values), np.min(values), np.max(values),
    )

    return StudyResult(
        best_params=best_params,
        best_value=float(best_value),
        n_trials=n_simulations,
        n_pruned=0,
        method="monte_carlo",
        model_name=objective.model_name,
        metric=tuning_config.metric,
    )
