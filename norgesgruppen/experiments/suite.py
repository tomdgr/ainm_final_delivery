#!/usr/bin/env python3
"""Experiment suite runner — submit ablation studies to Azure ML.

Unified CLI for submitting experiments. Supports single runs,
dimension sweeps, cross-product searches, and model sweeps.

Usage:
    python -m experiments.suite run --augmentation heavy --model resnet50
    python -m experiments.suite sweep --dim augmentation
    python -m experiments.suite model-sweep --augmentation medium
    python -m experiments.suite tune --model resnet50 --method tpe --n-trials 50
"""

import argparse
import json
import subprocess
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.library import (
    ALL_MODELS,
    MODEL_LR,
)
from nm_ai_image.tuning.search_spaces import AUGMENTATION_OPTIONS

EXPERIMENT_NAME = "experiment_suite"
IMAGE = "mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:31"

COMPUTE = {
    "cpu": "ainmxperis",
    "gpu": "ainmxperis",
}

DATA_INPUTS: dict = {}

DEFAULTS = {
    "augmentation": "medium",
    "model": "resnet50",
}

DIMENSIONS = {
    "augmentation": AUGMENTATION_OPTIONS,
    "model": ALL_MODELS,
}

ABBREVIATIONS = {
    "resnet50": "r50",
    "efficientnet_b0": "effb0",
    "efficientnet_b3": "effb3",
    "convnext_tiny": "cnxt",
    "vit_base_patch16_224": "vit",
}


def _abbrev(name: str) -> str:
    return ABBREVIATIONS.get(name, name)


def build_name(config: dict) -> str:
    preproc_dims = [d for d in DIMENSIONS if d != "model"]
    deltas = []
    for dim in preproc_dims:
        val = config.get(dim, DEFAULTS[dim])
        if val != DEFAULTS[dim]:
            deltas.append(_abbrev(val))

    model = config.get("model", DEFAULTS["model"])
    model_abbrev = _abbrev(model)

    if not deltas:
        return f"baseline_{model_abbrev}"
    return f"{'+'.join(deltas)}_{model_abbrev}"


def _build_env_preamble() -> str:
    return (
        "curl -LsSf https://astral.sh/uv/install.sh | sh && "
        "source $HOME/.local/bin/env && "
        "uv sync && "
    )


def build_command(config: dict, epochs: int = 100, patience: int = 20) -> str:
    cmd = _build_env_preamble()
    cmd += "uv run python experiments/train.py"
    for dim in DIMENSIONS:
        if dim == "model":
            continue
        cmd += f" --{dim} {config.get(dim, DEFAULTS[dim])}"

    model = config.get("model", DEFAULTS["model"])
    cmd += f" --model {model}"
    cmd += f" --epochs {epochs}"
    cmd += f" --patience {patience}"
    if model in MODEL_LR:
        cmd += f" --lr {MODEL_LR[model]}"

    return cmd


def build_tune_command(config: dict) -> str:
    cmd = _build_env_preamble()
    cmd += "uv run python experiments/tune.py"
    for dim in DIMENSIONS:
        if dim == "model":
            continue
        cmd += f" --{dim} {config.get(dim, DEFAULTS[dim])}"

    cmd += f" --model {config.get('model', DEFAULTS['model'])}"
    cmd += f" --method {config['method']}"
    cmd += f" --scope {config['scope']}"
    cmd += f" --n-trials {config['n_trials']}"
    cmd += f" --trial-epochs {config['trial_epochs']}"
    cmd += f" --trial-patience {config['trial_patience']}"
    cmd += f" --pop-size {config.get('pop_size', 20)}"
    cmd += f" --n-generations {config.get('n_generations', 50)}"
    cmd += f" --beam-width {config.get('beam_width', 5)}"
    cmd += f" --beam-trial-epochs {config.get('beam_trial_epochs', 50)}"
    cmd += f" --beam-trial-patience {config.get('beam_trial_patience', 15)}"

    model_kwargs = config.get("model_kwargs")
    if model_kwargs:
        cmd += f" --model-kwargs '{json.dumps(model_kwargs)}'"

    return cmd


def build_tune_name(config: dict) -> str:
    model = _abbrev(config.get("model", DEFAULTS["model"]))
    method = config["method"]
    if method == "beam":
        beam_width = config.get("beam_width", 5)
        return f"beam_w{beam_width}_{model}"
    scope = config["scope"]
    n_trials = config["n_trials"]
    return f"{method}_{scope}_{n_trials}t_{model}"


def get_completed_names(experiment: str = EXPERIMENT_NAME) -> set[str]:
    cmd = [
        "az", "ml", "job", "list",
        "--resource-group", "rg-nmai-workspace",
        "--workspace-name", "nmai-experis",
        "--query",
        f"[?experiment_name=='{experiment}' && status=='Completed'].display_name",
        "-o", "json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: could not query completed jobs: {result.stderr[:200]}")
        return set()
    return set(json.loads(result.stdout))


def _get_ml_client():
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    if not hasattr(_get_ml_client, "_client"):
        _get_ml_client._client = MLClient.from_config(
            credential=DefaultAzureCredential()
        )
    return _get_ml_client._client


def submit(config: dict, compute: str = "gpu", epochs: int = 100, patience: int = 20, dry_run: bool = False) -> str | None:
    name = build_name(config)
    cmd = build_command(config, epochs, patience)

    if dry_run:
        print(f"  [dry-run] {name}")
        return None

    from azure.ai.ml import command as aml_command
    from azure.ai.ml.entities import Environment

    ml_client = _get_ml_client()
    job = aml_command(
        command=cmd, code=".", environment=Environment(image=IMAGE),
        compute=COMPUTE[compute], experiment_name=EXPERIMENT_NAME,
        display_name=name, inputs=DATA_INPUTS if DATA_INPUTS else None,
    )

    returned_job = ml_client.jobs.create_or_update(job)
    print(f"  Submitted: {name} -> {returned_job.name}")
    return returned_job.name


def submit_tune(config: dict, compute: str = "gpu", dry_run: bool = False) -> str | None:
    name = build_tune_name(config)
    cmd = build_tune_command(config)
    model = config.get("model", DEFAULTS["model"])
    experiment = f"hpo_{model}"

    if dry_run:
        print(f"  [dry-run] {experiment}/{name}")
        return None

    from azure.ai.ml import command as aml_command
    from azure.ai.ml.entities import Environment

    ml_client = _get_ml_client()
    job = aml_command(
        command=cmd, code=".", environment=Environment(image=IMAGE),
        compute=COMPUTE[compute], experiment_name=experiment,
        display_name=name, inputs=DATA_INPUTS if DATA_INPUTS else None,
    )

    returned_job = ml_client.jobs.create_or_update(job)
    print(f"  Submitted: {experiment}/{name} -> {returned_job.name}")
    return returned_job.name


def configs_for_run(args) -> list[dict]:
    config = dict(DEFAULTS)
    for dim in DIMENSIONS:
        val = getattr(args, dim, None)
        if val is not None:
            config[dim] = val
    return [config]


def configs_for_sweep(args) -> list[dict]:
    dim = args.dim
    if dim not in DIMENSIONS:
        print(f"Error: unknown dimension '{dim}'. Choose from: {list(DIMENSIONS.keys())}")
        sys.exit(1)
    configs = []
    for value in DIMENSIONS[dim]:
        config = dict(DEFAULTS)
        config[dim] = value
        configs.append(config)
    return configs


def configs_for_cross(args) -> list[dict]:
    dim_values = {}
    for dim in DIMENSIONS:
        val = getattr(args, dim, None)
        if val is not None:
            dim_values[dim] = val.split(",")
        else:
            dim_values[dim] = [DEFAULTS[dim]]
    configs = []
    dim_names = list(dim_values.keys())
    for combo in product(*dim_values.values()):
        config = dict(zip(dim_names, combo))
        configs.append(config)
    return configs


def configs_for_model_sweep(args) -> list[dict]:
    base = dict(DEFAULTS)
    for dim in DIMENSIONS:
        if dim == "model":
            continue
        val = getattr(args, dim, None)
        if val is not None:
            base[dim] = val
    configs = []
    for model_name in ALL_MODELS:
        config = dict(base)
        config["model"] = model_name
        configs.append(config)
    return configs


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--compute", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)


def add_dimension_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--augmentation", default=None)
    parser.add_argument("--model", default=None)


def run_configs(configs: list[dict], args) -> None:
    if args.skip_completed:
        completed = get_completed_names()
        before = len(configs)
        configs = [c for c in configs if build_name(c) not in completed]
        skipped = before - len(configs)
        if skipped:
            print(f"Skipping {skipped} already-completed experiments")

    if not configs:
        print("Nothing to submit.")
        return

    action = "Preview" if args.dry_run else "Submitting"
    print(f"\n{action}: {len(configs)} experiment(s)\n")

    job_names = []
    for config in configs:
        name = submit(config, compute=args.compute, epochs=args.epochs,
                      patience=args.patience, dry_run=args.dry_run)
        if name:
            job_names.append(name)

    if not args.dry_run:
        print(f"\nSubmitted {len(job_names)} job(s).")


def main():
    parser = argparse.ArgumentParser(prog="experiments.suite")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_run = subparsers.add_parser("run")
    add_dimension_args(p_run)
    add_common_args(p_run)

    p_sweep = subparsers.add_parser("sweep")
    p_sweep.add_argument("--dim", required=True, choices=list(DIMENSIONS.keys()))
    add_common_args(p_sweep)

    p_cross = subparsers.add_parser("cross")
    add_dimension_args(p_cross)
    add_common_args(p_cross)

    p_model = subparsers.add_parser("model-sweep")
    add_dimension_args(p_model)
    add_common_args(p_model)

    p_tune = subparsers.add_parser("tune")
    add_dimension_args(p_tune)
    p_tune.add_argument("--method", default="tpe")
    p_tune.add_argument("--scope", default="architecture")
    p_tune.add_argument("--n-trials", type=int, default=50)
    p_tune.add_argument("--trial-epochs", type=int, default=50)
    p_tune.add_argument("--trial-patience", type=int, default=20)
    p_tune.add_argument("--model-kwargs", type=str, default=None)
    p_tune.add_argument("--pop-size", type=int, default=20)
    p_tune.add_argument("--n-generations", type=int, default=50)
    p_tune.add_argument("--beam-width", type=int, default=5)
    p_tune.add_argument("--beam-trial-epochs", type=int, default=50)
    p_tune.add_argument("--beam-trial-patience", type=int, default=15)
    p_tune.add_argument("--dry-run", action="store_true")
    p_tune.add_argument("--compute", choices=["cpu", "gpu"], default="gpu")

    args = parser.parse_args()

    if args.command == "tune":
        config = dict(DEFAULTS)
        for dim in DIMENSIONS:
            val = getattr(args, dim, None)
            if val is not None:
                config[dim] = val
        config["method"] = args.method
        config["scope"] = args.scope
        config["n_trials"] = args.n_trials
        config["trial_epochs"] = args.trial_epochs
        config["trial_patience"] = args.trial_patience
        config["pop_size"] = args.pop_size
        config["n_generations"] = args.n_generations
        config["beam_width"] = args.beam_width
        config["beam_trial_epochs"] = args.beam_trial_epochs
        config["beam_trial_patience"] = args.beam_trial_patience
        if args.model_kwargs:
            config["model_kwargs"] = json.loads(args.model_kwargs)
        submit_tune(config, compute=args.compute, dry_run=args.dry_run)
        return

    generators = {
        "run": configs_for_run,
        "sweep": configs_for_sweep,
        "cross": configs_for_cross,
        "model-sweep": configs_for_model_sweep,
    }

    configs = generators[args.command](args)
    run_configs(configs, args)


if __name__ == "__main__":
    main()
