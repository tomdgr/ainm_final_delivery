#!/usr/bin/env python3
"""Training entry point for experiment suite."""

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
from nm_ai_image.config import TaskConfig, TaskType
from nm_ai_image.config.image import ImageConfig
from nm_ai_image.training import ImageDataModule, LightningTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augmentation", default="medium")
    parser.add_argument("--model", default="resnet50", choices=list(ALL_MODELS.keys()))
    parser.add_argument("--task-type", default="classification")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    config_data = {
        "augmentation": args.augmentation,
        "model": args.model,
        "task_type": args.task_type,
    }
    print(f"CONFIG_JSON: {json.dumps(config_data)}")

    lr = args.lr or MODEL_LR.get(args.model, 1e-3)

    image_config = ImageConfig(
        augmentation_policy=args.augmentation,
        image_size=args.image_size,
    )

    config = TaskConfig(
        task_type=TaskType(args.task_type),
        backbone_name=args.model,
        num_classes=args.num_classes,
        image_config=image_config,
        model_kwargs=ALL_MODELS[args.model],
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        lr=lr,
        image_size=args.image_size,
    )

    data = ImageDataModule(config, data_dir=args.data_dir)
    data.setup()

    trainer = LightningTrainer(config, data)
    trainer.run_training()
    trainer.run_test()

    result_summary = {"model": args.model, "augmentation": args.augmentation}
    print(f"RESULTS_JSON: {json.dumps(result_summary)}")


if __name__ == "__main__":
    main()
