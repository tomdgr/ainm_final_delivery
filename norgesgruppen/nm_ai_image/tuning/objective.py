import logging

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
import optuna
from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback
import torch

from nm_ai_image.config.task import TaskConfig
from nm_ai_image.config.tuning import TuningConfig
from nm_ai_image.model.lightning.lightning_module import ImageTask
from nm_ai_image.training.data import ImageDataModule
from nm_ai_image.tuning.search_spaces import suggest_params

logger = logging.getLogger(__name__)


class TuningObjective:
    def __init__(
        self,
        task_config: TaskConfig,
        tuning_config: TuningConfig,
        data_dir: str | None = None,
    ):
        self.model_name = task_config.backbone_name
        self.tuning_config = tuning_config
        self.base_config = task_config

        self._data_module = ImageDataModule(task_config, data_dir=data_dir, num_workers=0)
        self._data_module.setup()
        self._trial_count = 0

    def __call__(self, trial: optuna.Trial) -> float:
        training_params, model_kwargs = suggest_params(
            trial, self.model_name, self.tuning_config.tune_scope,
        )
        merged_kwargs = {**self.base_config.model_kwargs, **model_kwargs}
        return self._train_and_evaluate(training_params, merged_kwargs, trial=trial)

    def evaluate(self, training_params: dict, model_kwargs: dict) -> float:
        return self._train_and_evaluate(training_params, model_kwargs, trial=None)

    def _train_and_evaluate(
        self,
        training_params: dict,
        model_kwargs: dict,
        trial: optuna.Trial | None = None,
    ) -> float:
        self._trial_count += 1
        logger.info("Trial %d — params: %s | kwargs: %s", self._trial_count, training_params, model_kwargs)

        batch_size = training_params.pop("batch_size", self.base_config.batch_size)
        image_size = training_params.pop("image_size", self.base_config.image_size)
        augmentation_policy = training_params.pop("augmentation_policy", self.base_config.image_config.augmentation_policy)
        freeze_backbone = training_params.pop("freeze_backbone", self.base_config.freeze_backbone)

        config = TaskConfig(
            task_type=self.base_config.task_type,
            backbone_name=self.model_name,
            num_classes=self.base_config.num_classes,
            pretrained=self.base_config.pretrained,
            freeze_backbone=freeze_backbone,
            model_kwargs=model_kwargs,
            epochs=self.tuning_config.trial_epochs,
            early_stopping_patience=self.tuning_config.trial_patience,
            batch_size=batch_size,
            image_size=image_size,
            **training_params,
        )
        config.image_config.augmentation_policy = augmentation_policy
        config.image_config.image_size = image_size

        # Re-setup data module with potentially new config
        self._data_module.config = config
        self._data_module.train_ds = None
        self._data_module.setup()

        num_classes = self._data_module.num_classes or config.num_classes
        model = config.build_model(num_classes)
        task = ImageTask(model=model, config=config)

        callbacks = [
            EarlyStopping(
                patience=self.tuning_config.trial_patience,
                monitor=self.tuning_config.metric,
                mode="min" if self.tuning_config.direction == "minimize" else "max",
            ),
        ]
        if trial is not None and self.tuning_config.pruner != "none":
            callbacks.append(
                PyTorchLightningPruningCallback(trial, monitor=self.tuning_config.metric)
            )

        precision = config.precision if torch.cuda.is_available() else "32-true"

        trainer = L.Trainer(
            max_epochs=self.tuning_config.trial_epochs,
            accelerator="auto",
            devices=1,
            callbacks=callbacks,
            precision=precision,
            gradient_clip_val=1.0,
            enable_model_summary=False,
            enable_progress_bar=False,
            logger=False,
        )

        trainer.fit(task, datamodule=self._data_module)

        metric_value = trainer.callback_metrics.get(self.tuning_config.metric)
        if metric_value is None:
            logger.warning("Trial %d — metric '%s' not found", self._trial_count, self.tuning_config.metric)
            if self.tuning_config.direction == "minimize":
                return float("inf")
            return float("-inf")

        value = metric_value.item()
        logger.info("Trial %d — %s: %.4f", self._trial_count, self.tuning_config.metric, value)
        return value
