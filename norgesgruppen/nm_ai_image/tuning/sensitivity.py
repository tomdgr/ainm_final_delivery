import logging

import mlflow
import numpy as np
from SALib.analyze import sobol
from SALib.sample import saltelli

from nm_ai_image.config.tuning import TuningConfig
from nm_ai_image.tuning.evolutionary_tuner import ParameterEncoder
from nm_ai_image.tuning.objective import TuningObjective
from nm_ai_image.tuning.results import StudyResult

logger = logging.getLogger(__name__)

_LOG_INTERVAL = 10


def run_sensitivity(objective: TuningObjective, tuning_config: TuningConfig) -> StudyResult:
    encoder = ParameterEncoder(objective.model_name, tuning_config.tune_scope)

    param_names = [name for name, _ in encoder.params]

    problem = {
        "num_vars": encoder.n_dims,
        "names": param_names,
        "bounds": [[0.0, 1.0]] * encoder.n_dims,
    }

    param_values = saltelli.sample(problem, tuning_config.n_sa_samples)
    n_evaluations = len(param_values)

    logger.info(
        "Starting Sobol sensitivity analysis: %d parameters, %d samples -> %d evaluations",
        encoder.n_dims, tuning_config.n_sa_samples, n_evaluations,
    )

    Y = np.zeros(n_evaluations)
    for i, row in enumerate(param_values):
        training_params, model_kwargs = encoder.decode(row)
        Y[i] = objective.evaluate(training_params, model_kwargs)
        if (i + 1) % _LOG_INTERVAL == 0:
            logger.info("Sensitivity eval %d/%d", i + 1, n_evaluations)

    si = sobol.analyze(problem, Y)

    s1_dict = dict(zip(param_names, si["S1"]))
    st_dict = dict(zip(param_names, si["ST"]))

    sorted_st = sorted(st_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    logger.info("Total-order sensitivity indices (sorted):")
    for name, val in sorted_st:
        logger.info("  %s: %.4f", name, val)

    mlflow.log_metrics({f"S1_{name}": float(val) for name, val in s1_dict.items()})
    mlflow.log_metrics({f"ST_{name}": float(val) for name, val in st_dict.items()})

    return StudyResult(
        best_params={
            "S1": s1_dict,
            "ST": st_dict,
        },
        best_value=0.0,
        n_trials=n_evaluations,
        n_pruned=0,
        method="sensitivity",
        model_name=objective.model_name,
        metric=tuning_config.metric,
    )
