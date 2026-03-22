from dataclasses import dataclass, field
from enum import Enum

import torch.nn as nn

from nm_ai_image.config.image import ImageConfig
from nm_ai_image.config.training import TrainingConfig


class TaskType(Enum):
    CLASSIFICATION = "classification"
    OBJECT_DETECTION = "object_detection"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"


@dataclass
class TaskConfig(TrainingConfig):
    task_type: TaskType = TaskType.CLASSIFICATION
    multi_label: bool = False
    backbone_name: str = "resnet50"
    pretrained: bool = True
    freeze_backbone: bool = False
    num_classes: int = 10
    image_config: ImageConfig = field(default_factory=ImageConfig)
    model_kwargs: dict = field(default_factory=dict)

    def build_model(self, num_classes: int | None = None) -> nn.Module:
        from nm_ai_image.model.networks.heads import build_model_for_task

        nc = num_classes if num_classes is not None else self.num_classes
        return build_model_for_task(
            task_type=self.task_type,
            backbone_name=self.backbone_name,
            num_classes=nc,
            pretrained=self.pretrained,
            **self.model_kwargs,
        )
