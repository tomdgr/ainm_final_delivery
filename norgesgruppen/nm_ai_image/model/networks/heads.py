import torch.nn as nn

from nm_ai_image.config.task import TaskType
from nm_ai_image.model.networks.backbones import get_backbone


class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, drop_rate: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(self.drop(x))


class ClassificationModel(nn.Module):
    def __init__(self, backbone: nn.Module, num_features: int, num_classes: int, drop_rate: float = 0.0):
        super().__init__()
        self.backbone = backbone
        self.head = ClassificationHead(num_features, num_classes, drop_rate)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def build_model_for_task(
    task_type: TaskType,
    backbone_name: str,
    num_classes: int,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    if task_type == TaskType.CLASSIFICATION:
        return _build_classification_model(backbone_name, num_classes, pretrained, **kwargs)
    elif task_type == TaskType.OBJECT_DETECTION:
        return _build_detection_model(backbone_name, num_classes, pretrained, **kwargs)
    elif task_type == TaskType.SEMANTIC_SEGMENTATION:
        return _build_segmentation_model(backbone_name, num_classes, pretrained, **kwargs)
    elif task_type == TaskType.INSTANCE_SEGMENTATION:
        return _build_instance_segmentation_model(num_classes, pretrained, **kwargs)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def _build_classification_model(
    backbone_name: str, num_classes: int, pretrained: bool = True, **kwargs
) -> nn.Module:
    drop_rate = kwargs.pop("drop_rate", 0.0)
    backbone = get_backbone(backbone_name, pretrained=pretrained, **kwargs)
    return ClassificationModel(backbone, backbone.num_features, num_classes, drop_rate)


def _build_detection_model(
    backbone_name: str,
    num_classes: int,
    pretrained: bool = True,
    detector: str = "fasterrcnn",
    **kwargs,
) -> nn.Module:
    if detector == "fcos":
        return _build_fcos_detection_model(num_classes, pretrained, **kwargs)
    return _build_fasterrcnn_detection_model(num_classes, pretrained, **kwargs)


def _build_fasterrcnn_detection_model(
    num_classes: int, pretrained: bool = True, **kwargs
) -> nn.Module:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def _build_fcos_detection_model(
    num_classes: int, pretrained: bool = True, **kwargs
) -> nn.Module:
    """Build an FCOS (Fully Convolutional One-Stage) detector via torchvision."""
    from torchvision.models.detection import fcos_resnet50_fpn

    model = fcos_resnet50_fpn(weights="DEFAULT" if pretrained else None, num_classes=num_classes)
    return model


def _build_segmentation_model(
    backbone_name: str,
    num_classes: int,
    pretrained: bool = True,
    seg_arch: str = "unet",
    **kwargs,
) -> nn.Module:
    import segmentation_models_pytorch as smp

    encoder_weights = "imagenet" if pretrained else None
    # Map our backbone names to smp encoder names
    encoder_map = {
        "resnet50": "resnet50",
        "efficientnet_b0": "efficientnet-b0",
        "efficientnet_b3": "efficientnet-b3",
        "convnext_tiny": "tu-convnext_tiny",
        "vit_base_patch16_224": "tu-vit_base_patch16_224",
        "efficientnet_v2_s": "tu-tf_efficientnetv2_s",
        "efficientnet_v2_m": "tu-tf_efficientnetv2_m",
        "convnext_base": "tu-convnext_base",
        "swin_base_patch4_window7_224": "tu-swin_base_patch4_window7_224",
        "convnextv2_base": "tu-convnextv2_base",
        "mobilenetv4_conv_large": "tu-mobilenetv4_conv_large",
    }
    smp_encoder = encoder_map.get(backbone_name, backbone_name)

    seg_architectures = {
        "unet": smp.Unet,
        "deeplabv3plus": smp.DeepLabV3Plus,
        "fpn": smp.FPN,
    }
    arch_cls = seg_architectures.get(seg_arch, smp.Unet)

    return arch_cls(
        encoder_name=smp_encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )


def _build_instance_segmentation_model(
    num_classes: int, pretrained: bool = True, **kwargs
) -> nn.Module:
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    model = maskrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model
