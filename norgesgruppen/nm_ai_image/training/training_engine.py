import logging
import os

import lightning as L
import mlflow
import torch

from nm_ai_image.config.task import TaskConfig, TaskType
from nm_ai_image.logging.aml_logging import initialize_aml_logging, is_running_in_azure_ml
from nm_ai_image.model.lightning.lightning_module import ImageTask
from nm_ai_image.training.data import ImageDataModule

logger = logging.getLogger(__name__)


def _get_lightning_logger(config: TaskConfig):
    """Create a logger that works in both local and Azure ML environments."""
    if is_running_in_azure_ml():
        try:
            from lightning.pytorch.loggers import MLFlowLogger
            return MLFlowLogger(
                experiment_name=config.output_dir,
                tracking_uri=mlflow.get_tracking_uri(),
            )
        except Exception:
            pass

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not tracking_uri or tracking_uri.startswith("file://") or tracking_uri.startswith("http"):
        try:
            from lightning.pytorch.loggers import MLFlowLogger
            uri = tracking_uri or "file:./mlruns"
            return MLFlowLogger(experiment_name=config.output_dir, tracking_uri=uri)
        except Exception:
            pass

    from lightning.pytorch.loggers import CSVLogger
    return CSVLogger(save_dir=config.output_dir, name="training")


class LightningTrainer:
    def __init__(self, config: TaskConfig, data_module: ImageDataModule):
        self.config = config
        self.data_module = data_module

        # Initialize Azure ML / MLflow logging
        initialize_aml_logging(experiment_name=config.output_dir)

        num_classes = data_module.num_classes or config.num_classes
        training_module = ImageTask(
            model=config.build_model(num_classes),
            config=config,
        )
        self.training_module = training_module

        # Log hyperparams
        try:
            mlflow.log_params({
                "backbone": config.backbone_name,
                "task_type": config.task_type.value,
                "num_classes": num_classes,
                "image_size": config.image_config.image_size,
                "augmentation": config.image_config.augmentation_policy,
                "batch_size": config.batch_size,
                "lr": config.lr,
                "epochs": config.epochs,
                "loss_fn": config.loss_fn,
                "freeze_backbone": config.freeze_backbone,
            })
        except Exception:
            pass

        pl_logger = _get_lightning_logger(config)

        precision_strategy = config.precision if torch.cuda.is_available() else "32-true"
        self.trainer = L.Trainer(
            max_epochs=config.epochs,
            accelerator="auto",
            logger=pl_logger,
            devices=1,
            callbacks=config.get_callbacks(),
            precision=precision_strategy,
            gradient_clip_val=1.0,
            enable_model_summary=True,
        )

    def run_training(self):
        self.trainer.fit(self.training_module, datamodule=self.data_module)

    def run_test(self):
        self.trainer.test(self.training_module, datamodule=self.data_module, ckpt_path="best")

    def run_evaluation(self, class_names: list[str] | None = None):
        """Run full evaluation with plots and artifact logging."""
        if self.config.task_type != TaskType.CLASSIFICATION:
            logger.info("Full evaluation only supported for classification tasks")
            return

        import numpy as np
        from nm_ai_image.evaluation.evaluation_engine import EvaluationEngine, log_sample_predictions

        # Collect predictions from test set
        self.training_module.eval()
        self.training_module.test_outputs.clear()
        self.trainer.test(self.training_module, datamodule=self.data_module, ckpt_path="best")

        if not self.training_module.test_outputs:
            logger.warning("No test outputs collected for evaluation")
            return

        y_true = np.concatenate([x["y_true"] for x in self.training_module.test_outputs])
        y_pred = np.concatenate([x["y_pred"] for x in self.training_module.test_outputs])

        engine = EvaluationEngine(
            config=self.config,
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
        )
        metrics = engine.generate_report()
        logger.info("Full evaluation report generated with %d plots", len(list(engine.plots_dir.glob("*.png"))))
        return metrics
