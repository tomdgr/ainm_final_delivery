from nm_ai_image.tuning.search_spaces import (
    AUGMENTATION_OPTIONS,
    BACKBONE_REGISTRY,
    MODEL_DEFAULTS,
    MODEL_LR,
)

ALL_MODELS = MODEL_DEFAULTS

DISPLAY_NAMES: dict[str, str] = {
    "resnet50": "ResNet-50",
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnet_b3": "EfficientNet-B3",
    "convnext_tiny": "ConvNeXt-Tiny",
    "vit_base_patch16_224": "ViT-Base/16",
    "efficientnet_v2_s": "EfficientNetV2-S",
    "efficientnet_v2_m": "EfficientNetV2-M",
    "convnext_base": "ConvNeXt-Base",
    "swin_base_patch4_window7_224": "Swin-Base",
    "eva02_base_patch14_224": "EVA-02-Base",
    "maxvit_tiny_224": "MaxViT-Tiny",
    "coatnet_0_224": "CoAtNet-0",
    "caformer_b36": "CAFormer-B36",
    "mobilenetv4_conv_large": "MobileNetV4-Large",
    "convnextv2_base": "ConvNeXtV2-Base",
    "mambaout_base": "MambaOut-Base",
    "mambaout_small": "MambaOut-Small",
    "siglip_base_224": "SigLIP-Base/16-224",
    "siglip_so400m_224": "SigLIP-So400m/14-224",
    "dinov2_base": "DINOv2-Base/14",
    "dinov2_large": "DINOv2-Large/14",
    "swinv2_base_256": "SwinV2-Base-256",
    "eva02_large_patch14_224": "EVA-02-Large/14",
    "convnextv2_large": "ConvNeXtV2-Large",
    "eva02_small_patch14_224": "EVA-02-Small/14",
}


def build_image_config(augmentation: str = "medium", image_size: int = 224) -> dict:
    return {
        "augmentation_policy": augmentation,
        "image_size": image_size,
    }
