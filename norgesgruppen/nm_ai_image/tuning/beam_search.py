from dataclasses import dataclass, field
import json
import logging
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
import mlflow
import torch

from nm_ai_image.config.task import TaskConfig
from nm_ai_image.config.tuning import TuningConfig
from nm_ai_image.model.lightning.lightning_module import ImageTask
from nm_ai_image.training.data import ImageDataModule
from nm_ai_image.tuning.results import StudyResult
from nm_ai_image.tuning.search_spaces import (
    AUGMENTATION_OPTIONS,
    DIMENSION_REGISTRY,
    MODEL_DEFAULTS,
    MODEL_LR,
)

logger = logging.getLogger(__name__)

DEFAULT_STAGES = [
    "augmentation",
    "model",
]


@dataclass
class BeamCandidate:
    dimensions: dict = field(default_factory=dict)
    model_kwargs: dict = field(default_factory=dict)
    score: float = float("inf")


def _get_model_options() -> list[str]:
    return list(MODEL_DEFAULTS.keys())


def _evaluate_candidate(
    candidate: BeamCandidate,
    tuning_config: TuningConfig,
    base_config: TaskConfig,
) -> float:
    dimensions = candidate.dimensions
    model_name = dimensions.get("model", base_config.backbone_name)
    model_kwargs = candidate.model_kwargs
    aug_policy = dimensions.get("augmentation", "medium")

    lr = MODEL_LR.get(model_name, 1e-3)
    config = TaskConfig(
        task_type=base_config.task_type,
        backbone_name=model_name,
        num_classes=base_config.num_classes,
        pretrained=base_config.pretrained,
        model_kwargs=model_kwargs,
        epochs=tuning_config.beam_trial_epochs,
        early_stopping_patience=tuning_config.beam_trial_patience,
        lr=lr,
    )
    config.image_config.augmentation_policy = aug_policy

    data_module = ImageDataModule(config, num_workers=0)
    data_module.setup()

    num_classes = data_module.num_classes or config.num_classes
    model = config.build_model(num_classes)
    task = ImageTask(model=model, config=config)

    callbacks = [
        EarlyStopping(
            patience=tuning_config.beam_trial_patience,
            monitor=tuning_config.metric,
            mode="min" if tuning_config.direction == "minimize" else "max",
        ),
    ]

    precision = config.precision if torch.cuda.is_available() else "32-true"

    trainer = L.Trainer(
        max_epochs=tuning_config.beam_trial_epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        precision=precision,
        gradient_clip_val=1.0,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
    )

    trainer.fit(task, datamodule=data_module)

    metric_value = trainer.callback_metrics.get(tuning_config.metric)
    if metric_value is None:
        logger.warning("Metric '%s' not found for candidate %s", tuning_config.metric, dimensions)
        return float("inf") if tuning_config.direction == "minimize" else float("-inf")

    return metric_value.item()


def _select_top_k(
    candidates: list[BeamCandidate],
    beam_width: int,
    direction: str,
) -> list[BeamCandidate]:
    reverse = direction == "maximize"
    ranked = sorted(candidates, key=lambda c: c.score, reverse=reverse)
    return ranked[:beam_width]


class BeamSearch:
    def __init__(
        self,
        base_config: TaskConfig,
        tuning_config: TuningConfig,
        stages: list[str] | None = None,
    ):
        self.base_config = base_config
        self.tuning_config = tuning_config
        self.stages = stages or DEFAULT_STAGES
        self.history: list[dict] = []

    def run(self) -> list[BeamCandidate]:
        beam = [BeamCandidate(dimensions={}, model_kwargs={})]
        total_evals = 0

        for stage_idx, stage_name in enumerate(self.stages, 1):
            logger.info(
                "=== Stage %d/%d: %s (beam size: %d) ===",
                stage_idx, len(self.stages), stage_name, len(beam),
            )

            expanded = self._expand_stage(beam, stage_name)
            logger.info("Evaluating %d candidates for stage '%s'", len(expanded), stage_name)

            for i, candidate in enumerate(expanded):
                choice = candidate.dimensions.get(stage_name, "?")
                run_name = f"stage_{stage_idx}_{stage_name}/{choice}"

                with mlflow.start_run(run_name=run_name, nested=True):
                    mlflow.log_params({
                        "stage": stage_name,
                        "stage_idx": stage_idx,
                        "candidate_idx": i,
                        **{f"dim_{k}": v for k, v in candidate.dimensions.items()},
                    })

                    try:
                        candidate.score = _evaluate_candidate(candidate, self.tuning_config, self.base_config)
                    except Exception as e:
                        logger.error("Candidate %s failed: %s", candidate.dimensions, e)
                        candidate.score = float("inf") if self.tuning_config.direction == "minimize" else float("-inf")

                    mlflow.log_metric("score", candidate.score)

                total_evals += 1
                logger.info(
                    "  [%d/%d] %s=%s -> %.4f",
                    i + 1, len(expanded), stage_name, choice, candidate.score,
                )

            beam = _select_top_k(expanded, self.tuning_config.beam_width, self.tuning_config.direction)

            stage_summary = {
                "stage": stage_name,
                "stage_idx": stage_idx,
                "n_candidates": len(expanded),
                "survivors": [
                    {"dimensions": c.dimensions.copy(), "score": c.score}
                    for c in beam
                ],
            }
            self.history.append(stage_summary)

            logger.info(
                "Stage '%s' complete — top %d survivors:",
                stage_name, len(beam),
            )
            for rank, c in enumerate(beam, 1):
                logger.info("  #%d: score=%.4f dims=%s", rank, c.score, c.dimensions)

            mlflow.log_metric(f"best_after_stage_{stage_idx}", beam[0].score)

        logger.info("Beam search complete: %d total evaluations", total_evals)
        return beam

    def _expand_stage(
        self,
        beam: list[BeamCandidate],
        stage_name: str,
    ) -> list[BeamCandidate]:
        if stage_name == "model":
            return self._expand_model_stage(beam)

        options = DIMENSION_REGISTRY.get(stage_name, {})
        if not options:
            logger.warning("No options for stage '%s', skipping", stage_name)
            return beam

        expanded = []
        for parent in beam:
            for choice_name in options:
                new_dims = {**parent.dimensions, stage_name: choice_name}
                expanded.append(BeamCandidate(
                    dimensions=new_dims,
                    model_kwargs=parent.model_kwargs.copy(),
                    score=float("inf"),
                ))
        return expanded

    def _expand_model_stage(self, beam: list[BeamCandidate]) -> list[BeamCandidate]:
        expanded = []
        for parent in beam:
            for model_name in _get_model_options():
                new_dims = {**parent.dimensions, "model": model_name}
                kwargs = MODEL_DEFAULTS.get(model_name, {}).copy()
                expanded.append(BeamCandidate(
                    dimensions=new_dims,
                    model_kwargs=kwargs,
                    score=float("inf"),
                ))
        return expanded

    def save_history(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "beam_history.json", "w") as f:
            json.dump(self.history, f, indent=2, default=str)
        logger.info("Saved beam history to %s", output_dir / "beam_history.json")


def run_beam_search(
    task_config: TaskConfig,
    tuning_config: TuningConfig,
    output_dir: str | Path | None = None,
) -> StudyResult:
    beam = BeamSearch(task_config, tuning_config)
    survivors = beam.run()

    best = survivors[0]

    if tuning_config.beam_hpo_method and output_dir:
        logger.info("Running HPO refinement on top %d survivors", len(survivors))
        best = _refine_with_hpo(survivors, tuning_config, task_config, output_dir)

    best_params = {
        **{f"beam_{k}": v for k, v in best.dimensions.items()},
        **{f"model_kwarg_{k}": v for k, v in best.model_kwargs.items()},
    }

    if output_dir:
        beam.save_history(Path(output_dir))

    return StudyResult(
        best_params=best_params,
        best_value=best.score,
        n_trials=sum(s["n_candidates"] for s in beam.history),
        n_pruned=0,
        method="beam",
        model_name=best.dimensions.get("model", task_config.backbone_name),
        metric=tuning_config.metric,
        study=None,
    )


def _refine_with_hpo(
    survivors: list[BeamCandidate],
    tuning_config: TuningConfig,
    base_config: TaskConfig,
    output_dir: str | Path,
) -> BeamCandidate:
    from nm_ai_image.tuning import HyperparameterOptimizer

    best_candidate = survivors[0]

    for i, candidate in enumerate(survivors):
        logger.info(
            "HPO refinement %d/%d: model=%s, score=%.4f",
            i + 1, len(survivors),
            candidate.dimensions.get("model", "?"),
            candidate.score,
        )

        model_name = candidate.dimensions.get("model", base_config.backbone_name)
        lr = MODEL_LR.get(model_name, 1e-3)

        config = TaskConfig(
            task_type=base_config.task_type,
            backbone_name=model_name,
            num_classes=base_config.num_classes,
            pretrained=base_config.pretrained,
            model_kwargs=candidate.model_kwargs,
            lr=lr,
        )

        hpo_tuning = TuningConfig(
            method=tuning_config.beam_hpo_method,
            n_trials=tuning_config.beam_hpo_trials,
            metric=tuning_config.metric,
            direction=tuning_config.direction,
            tune_scope="all",
            trial_epochs=tuning_config.trial_epochs,
            trial_patience=tuning_config.trial_patience,
        )

        optimizer = HyperparameterOptimizer(config, hpo_tuning)
        hpo_output = Path(output_dir) / f"hpo_survivor_{i}"
        result = optimizer.run(output_dir=str(hpo_output))

        is_better = (
            (tuning_config.direction == "maximize" and result.best_value > candidate.score) or
            (tuning_config.direction == "minimize" and result.best_value < candidate.score)
        )
        if is_better:
            candidate.score = result.best_value
            candidate.model_kwargs.update(
                {k: v for k, v in result.best_params.items() if not k.startswith("best_")}
            )

        compare_better = (
            (tuning_config.direction == "maximize" and candidate.score > best_candidate.score) or
            (tuning_config.direction == "minimize" and candidate.score < best_candidate.score)
        )
        if compare_better:
            best_candidate = candidate

    return best_candidate
