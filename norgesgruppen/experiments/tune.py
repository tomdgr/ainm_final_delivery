#!/usr/bin/env python3
"""Hyperparameter tuning entry point for experiment suite."""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.library import ALL_MODELS, MODEL_LR
from nm_ai_image.config import TaskConfig, TaskType, TuningConfig
from nm_ai_image.config.image import ImageConfig
from nm_ai_image.tuning import HyperparameterOptimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augmentation", default="medium")
    parser.add_argument("--model", default="resnet50", choices=list(ALL_MODELS.keys()))
    parser.add_argument("--task-type", default="classification")
    parser.add_argument("--method", default="tpe")
    parser.add_argument("--scope", default="architecture")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--trial-epochs", type=int, default=50)
    parser.add_argument("--trial-patience", type=int, default=20)
    parser.add_argument("--model-kwargs", type=str, default=None)
    parser.add_argument("--pop-size", type=int, default=20)
    parser.add_argument("--n-generations", type=int, default=50)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--beam-trial-epochs", type=int, default=50)
    parser.add_argument("--beam-trial-patience", type=int, default=15)
    parser.add_argument("--num-classes", type=int, default=10)
    args = parser.parse_args()

    model_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else None

    config_data = {
        "augmentation": args.augmentation,
        "model": args.model,
        "method": args.method,
        "scope": args.scope,
        "n_trials": args.n_trials,
    }
    print(f"CONFIG_JSON: {json.dumps(config_data)}")

    image_config = ImageConfig(augmentation_policy=args.augmentation)

    config = TaskConfig(
        task_type=TaskType(args.task_type),
        backbone_name=args.model,
        num_classes=args.num_classes,
        image_config=image_config,
        model_kwargs=model_kwargs or {},
    )

    tuning = TuningConfig(
        method=args.method,
        tune_scope=args.scope,
        n_trials=args.n_trials,
        trial_epochs=args.trial_epochs,
        trial_patience=args.trial_patience,
        pop_size=args.pop_size,
        n_generations=args.n_generations,
        beam_width=args.beam_width,
        beam_trial_epochs=args.beam_trial_epochs,
        beam_trial_patience=args.beam_trial_patience,
    )

    optimizer = HyperparameterOptimizer(config, tuning)
    result = optimizer.run(output_dir=config.output_dir + "/tuning")

    print(f"BEST_PARAMS_JSON: {json.dumps(result.best_params, default=str)}")
    print(f"BEST_VALUE: {result.best_value:.4f}")


if __name__ == "__main__":
    main()
