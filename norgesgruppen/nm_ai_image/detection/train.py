"""Ultralytics YOLOv8/RT-DETR training wrapper."""
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DetectionTrainConfig:
    """Training configuration for ultralytics models."""
    model: str = "yolov8x.pt"
    data: str = "data/yolo/data.yaml"
    imgsz: int = 640
    epochs: int = 150
    batch: int = 8
    device: str = "0"
    project: str = "runs/detect"
    name: str = "train"
    patience: int = 30
    cos_lr: bool = True
    close_mosaic: int = 20
    multi_scale: bool = False
    # Augmentation
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 5.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 2.0
    flipud: float = 0.5
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.15
    copy_paste: float = 0.1
    # Loss weights
    cls: float = 0.5
    box: float = 7.5
    # Optimizer
    optimizer: str = "AdamW"
    lr0: float = 0.001
    lrf: float = 0.01
    weight_decay: float = 0.0005
    warmup_epochs: int = 5
    seed: int = 42
    # Advanced
    freeze: int | None = None
    rect: bool = False
    save_period: int = 25
    nbs: int = 64

    @property
    def best_weights(self) -> Path:
        return Path(self.project) / self.name / "weights" / "best.pt"


def train(config: DetectionTrainConfig) -> Path:
    """Train an ultralytics model. Returns path to best weights."""
    from ultralytics import YOLO

    model = YOLO(config.model)
    train_kwargs = dict(
        data=config.data,
        imgsz=config.imgsz,
        epochs=config.epochs,
        batch=config.batch,
        device=config.device,
        project=config.project,
        name=config.name,
        patience=config.patience,
        hsv_h=config.hsv_h, hsv_s=config.hsv_s, hsv_v=config.hsv_v,
        degrees=config.degrees, translate=config.translate,
        scale=config.scale, shear=config.shear,
        flipud=config.flipud, fliplr=config.fliplr,
        mosaic=config.mosaic, mixup=config.mixup, copy_paste=config.copy_paste,
        close_mosaic=config.close_mosaic,
        cos_lr=config.cos_lr,
        optimizer=config.optimizer,
        lr0=config.lr0, lrf=config.lrf,
        weight_decay=config.weight_decay,
        warmup_epochs=config.warmup_epochs,
        warmup_momentum=0.5,
        save=True, save_period=config.save_period,
        plots=True, val=True,
        workers=4, seed=config.seed,
        multi_scale=config.multi_scale,
        cls=config.cls, box=config.box,
        rect=config.rect, nbs=config.nbs,
    )
    if config.freeze is not None:
        train_kwargs["freeze"] = config.freeze
    model.train(**train_kwargs)

    best_path = config.best_weights
    if best_path.exists():
        logger.info("Best weights saved to %s", best_path)
        best_model = YOLO(str(best_path))
        metrics = best_model.val(data=config.data, imgsz=config.imgsz, device=config.device)
        logger.info("Best mAP50: %.4f, mAP50-95: %.4f", metrics.box.map50, metrics.box.map)
    else:
        logger.warning("Best weights not found at %s", best_path)

    return best_path
