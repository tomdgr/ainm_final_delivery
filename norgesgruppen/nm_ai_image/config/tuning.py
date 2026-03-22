from dataclasses import dataclass
from typing import Literal


@dataclass
class TuningConfig:
    method: Literal[
        "tpe", "random", "grid", "cmaes",
        "ga", "pso", "gjo", "macla",
        "sensitivity", "monte_carlo",
        "beam",
    ] = "tpe"

    n_trials: int = 50
    metric: str = "val_f1"
    direction: Literal["minimize", "maximize"] = "maximize"
    seed: int = 42

    tune_scope: Literal["all", "training", "architecture"] = "all"

    pruner: Literal["median", "hyperband", "none"] = "median"
    pruner_warmup_steps: int = 10

    grid_points: int = 5

    pop_size: int = 20
    n_generations: int = 50

    n_sa_samples: int = 64

    trial_epochs: int = 100
    trial_patience: int = 30

    beam_width: int = 5
    beam_trial_epochs: int = 50
    beam_trial_patience: int = 15
    beam_hpo_method: str | None = None
    beam_hpo_trials: int = 30

    storage: str | None = None

    generate_plots: bool = True
