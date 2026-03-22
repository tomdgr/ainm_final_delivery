import logging
import math
from typing import Callable

import mlflow
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from nm_ai_image.config.tuning import TuningConfig
from nm_ai_image.tuning.objective import TuningObjective
from nm_ai_image.tuning.results import StudyResult, flatten_params, format_run_name
from nm_ai_image.tuning.search_spaces import MODEL_SPACES, TRAINING_SPACE

logger = logging.getLogger(__name__)

_CATEGORICAL_CLIP_UPPER = 1.0 - 1e-4
_LEVY_BETA = 1.5
_LOG_INTERVAL = 10


class ParameterEncoder:
    def __init__(self, model_name: str, scope: str = "all"):
        self.params: list[tuple[str, tuple]] = []
        if scope in ("all", "training"):
            for name, spec in TRAINING_SPACE.items():
                self.params.append((name, spec))
        if scope in ("all", "architecture"):
            for name, spec in MODEL_SPACES.get(model_name, {}).items():
                self.params.append((name, spec))
        self.n_dims = len(self.params)

    @property
    def lower_bounds(self) -> np.ndarray:
        return np.zeros(self.n_dims)

    @property
    def upper_bounds(self) -> np.ndarray:
        return np.ones(self.n_dims)

    def decode(self, vector: np.ndarray) -> tuple[dict, dict]:
        training_params = {}
        model_kwargs = {}
        training_names = set(TRAINING_SPACE.keys())

        for i, (name, spec) in enumerate(self.params):
            val = self._decode_param(vector[i], spec)
            if name in training_names:
                training_params[name] = val
            else:
                model_kwargs[name] = val

        if "hidden_size_0" in model_kwargs:
            model_kwargs["hidden_sizes"] = [
                model_kwargs.pop("hidden_size_0"),
                model_kwargs.pop("hidden_size_1"),
            ]

        if "d_model" in model_kwargs and "nhead" in model_kwargs:
            d_model = model_kwargs["d_model"]
            nhead = model_kwargs["nhead"]
            if d_model % nhead != 0:
                valid = [n for n in [1, 2, 4, 8] if d_model % n == 0]
                model_kwargs["nhead"] = min(valid, key=lambda n: abs(n - nhead))

        return training_params, model_kwargs

    def _decode_param(self, value: float, spec: tuple):
        value = np.clip(value, 0.0, _CATEGORICAL_CLIP_UPPER)
        kind = spec[0]
        if kind == "categorical":
            choices = spec[1]
            idx = int(value * len(choices))
            return choices[idx]
        if kind in ("float", "float_log"):
            return spec[1] + value * (spec[2] - spec[1])
        if kind == "int":
            return int(round(spec[1] + value * (spec[2] - spec[1])))
        raise ValueError(f"Unknown spec type: {kind}")


class _HPOProblem(ElementwiseProblem):
    def __init__(self, encoder: ParameterEncoder, evaluate_fn: Callable):
        super().__init__(
            n_var=encoder.n_dims,
            n_obj=1,
            xl=encoder.lower_bounds,
            xu=encoder.upper_bounds,
        )
        self.encoder = encoder
        self.evaluate_fn = evaluate_fn

    def _evaluate(self, x, out, *args, **kwargs):
        training_params, model_kwargs = self.encoder.decode(x)
        out["F"] = self.evaluate_fn(training_params, model_kwargs)


class _MLflowGenerationCallback(Callback):
    def __init__(self, encoder: ParameterEncoder, metric: str):
        super().__init__()
        self.encoder = encoder
        self.metric = metric

    def notify(self, algorithm):
        gen = algorithm.n_gen
        F = algorithm.pop.get("F").flatten()
        X = algorithm.pop.get("X")
        best_idx = np.argmin(F)
        training_params, model_kwargs = self.encoder.decode(X[best_idx])
        merged = {**training_params, **model_kwargs}
        with mlflow.start_run(run_name=format_run_name("gen", gen, merged), nested=True):
            mlflow.log_params(flatten_params(merged))
            mlflow.log_metric(self.metric, float(F[best_idx]))


def run_evolutionary(objective: TuningObjective, tuning_config: TuningConfig) -> StudyResult:
    method = tuning_config.method
    if method == "ga":
        return run_ga(objective, tuning_config)
    if method == "pso":
        return run_pso(objective, tuning_config)
    if method == "gjo":
        return run_gjo(objective, tuning_config)
    if method == "macla":
        return run_macla(objective, tuning_config)
    raise ValueError(f"Unknown evolutionary method: {method}")


def run_ga(objective: TuningObjective, tuning_config: TuningConfig) -> StudyResult:
    encoder = ParameterEncoder(objective.model_name, tuning_config.tune_scope)
    problem = _HPOProblem(encoder, objective.evaluate)

    algorithm = GA(pop_size=tuning_config.pop_size)
    termination = get_termination("n_gen", tuning_config.n_generations)

    callback = _MLflowGenerationCallback(encoder, tuning_config.metric)
    logger.info("Starting GA: pop_size=%d, n_generations=%d", tuning_config.pop_size, tuning_config.n_generations)
    result = minimize(problem, algorithm, termination, seed=tuning_config.seed, verbose=False, callback=callback)

    best_training, best_kwargs = encoder.decode(result.X)
    return StudyResult(
        best_params={**best_training, **best_kwargs},
        best_value=float(result.F[0]),
        n_trials=result.algorithm.evaluator.n_eval,
        n_pruned=0,
        method="ga",
        model_name=objective.model_name,
        metric=tuning_config.metric,
    )


def run_pso(objective: TuningObjective, tuning_config: TuningConfig) -> StudyResult:
    encoder = ParameterEncoder(objective.model_name, tuning_config.tune_scope)
    problem = _HPOProblem(encoder, objective.evaluate)

    algorithm = PSO(pop_size=tuning_config.pop_size)
    termination = get_termination("n_gen", tuning_config.n_generations)

    callback = _MLflowGenerationCallback(encoder, tuning_config.metric)
    logger.info("Starting PSO: pop_size=%d, n_generations=%d", tuning_config.pop_size, tuning_config.n_generations)
    result = minimize(problem, algorithm, termination, seed=tuning_config.seed, verbose=False, callback=callback)

    best_training, best_kwargs = encoder.decode(result.X)
    return StudyResult(
        best_params={**best_training, **best_kwargs},
        best_value=float(result.F[0]),
        n_trials=result.algorithm.evaluator.n_eval,
        n_pruned=0,
        method="pso",
        model_name=objective.model_name,
        metric=tuning_config.metric,
    )


def run_gjo(objective: TuningObjective, tuning_config: TuningConfig) -> StudyResult:
    encoder = ParameterEncoder(objective.model_name, tuning_config.tune_scope)
    rng = np.random.default_rng(tuning_config.seed)

    pop_size = tuning_config.pop_size
    n_gen = tuning_config.n_generations
    n_dims = encoder.n_dims

    positions = rng.uniform(0, 1, size=(pop_size, n_dims))
    fitness = np.full(pop_size, np.inf)

    for i in range(pop_size):
        training_params, model_kwargs = encoder.decode(positions[i])
        fitness[i] = objective.evaluate(training_params, model_kwargs)

    all_evals = pop_size
    logger.info("Starting GJO: pop_size=%d, n_generations=%d", pop_size, n_gen)

    for gen in range(n_gen):
        sorted_idx = np.argsort(fitness)
        male = positions[sorted_idx[0]]
        female = positions[sorted_idx[1]] if pop_size > 1 else male.copy()

        escaping_energy = 2.0 * (1.0 - (gen + 1) / n_gen)

        for i in range(pop_size):
            r1 = rng.random()
            r2 = rng.random()

            male_influence = male - r1 * abs(male - positions[i]) * escaping_energy
            female_influence = female - r2 * abs(female - positions[i]) * escaping_energy
            new_pos = (male_influence + female_influence) / 2.0

            new_pos = np.clip(new_pos, 0.0, 1.0)

            training_params, model_kwargs = encoder.decode(new_pos)
            new_fitness = objective.evaluate(training_params, model_kwargs)
            all_evals += 1

            if new_fitness < fitness[i]:
                positions[i] = new_pos
                fitness[i] = new_fitness

        best_idx = np.argmin(fitness)
        best_training, best_kwargs = encoder.decode(positions[best_idx])
        merged = {**best_training, **best_kwargs}
        with mlflow.start_run(run_name=format_run_name("gen", gen + 1, merged), nested=True):
            mlflow.log_params(flatten_params(merged))
            mlflow.log_metric(tuning_config.metric, float(fitness[best_idx]))
        logger.info("GJO gen %d/%d — best: %.4f", gen + 1, n_gen, fitness[best_idx])

    best_idx = np.argmin(fitness)
    best_training, best_kwargs = encoder.decode(positions[best_idx])

    return StudyResult(
        best_params={**best_training, **best_kwargs},
        best_value=float(fitness[best_idx]),
        n_trials=all_evals,
        n_pruned=0,
        method="gjo",
        model_name=objective.model_name,
        metric=tuning_config.metric,
    )


def run_macla(objective: TuningObjective, tuning_config: TuningConfig) -> StudyResult:
    encoder = ParameterEncoder(objective.model_name, tuning_config.tune_scope)
    rng = np.random.default_rng(tuning_config.seed)

    pop_size = tuning_config.pop_size
    n_gen = tuning_config.n_generations
    n_dims = encoder.n_dims

    positions = rng.uniform(0, 1, size=(pop_size, n_dims))
    fitness = np.full(pop_size, np.inf)

    for i in range(pop_size):
        training_params, model_kwargs = encoder.decode(positions[i])
        fitness[i] = objective.evaluate(training_params, model_kwargs)

    all_evals = pop_size
    logger.info("Starting MACLA: pop_size=%d, n_generations=%d", pop_size, n_gen)

    for gen in range(n_gen):
        best_idx = np.argmin(fitness)
        prey = positions[best_idx].copy()

        t_ratio = (gen + 1) / n_gen

        for i in range(pop_size):
            if i == best_idx:
                continue

            r = rng.random(n_dims)

            if rng.random() < 0.5:
                step_size = 2.0 * r * (prey - positions[i])
                new_pos = positions[i] + step_size * (1.0 - t_ratio)
            else:
                random_idx = rng.integers(0, pop_size)
                levy = self_adaptive_levy(n_dims, rng)
                new_pos = positions[i] + levy * (positions[random_idx] - positions[i]) * t_ratio

            new_pos = np.clip(new_pos, 0.0, 1.0)

            training_params, model_kwargs = encoder.decode(new_pos)
            new_fitness = objective.evaluate(training_params, model_kwargs)
            all_evals += 1

            if new_fitness < fitness[i]:
                positions[i] = new_pos
                fitness[i] = new_fitness

        gen_best_idx = np.argmin(fitness)
        gen_best_training, gen_best_kwargs = encoder.decode(positions[gen_best_idx])
        gen_merged = {**gen_best_training, **gen_best_kwargs}
        with mlflow.start_run(run_name=format_run_name("gen", gen + 1, gen_merged), nested=True):
            mlflow.log_params(flatten_params(gen_merged))
            mlflow.log_metric(tuning_config.metric, float(fitness[gen_best_idx]))
        logger.info("MACLA gen %d/%d — best: %.4f", gen + 1, n_gen, fitness[gen_best_idx])

    best_idx = np.argmin(fitness)
    best_training, best_kwargs = encoder.decode(positions[best_idx])

    return StudyResult(
        best_params={**best_training, **best_kwargs},
        best_value=float(fitness[best_idx]),
        n_trials=all_evals,
        n_pruned=0,
        method="macla",
        model_name=objective.model_name,
        metric=tuning_config.metric,
    )


def self_adaptive_levy(n_dims: int, rng: np.random.Generator) -> np.ndarray:
    sigma_u = (
        math.gamma(1 + _LEVY_BETA) * np.sin(np.pi * _LEVY_BETA / 2)
        / (math.gamma((1 + _LEVY_BETA) / 2) * _LEVY_BETA * 2 ** ((_LEVY_BETA - 1) / 2))
    ) ** (1 / _LEVY_BETA)
    u = rng.normal(0, sigma_u, n_dims)
    v = rng.normal(0, 1, n_dims)
    return u / (np.abs(v) ** (1 / _LEVY_BETA))
