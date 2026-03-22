from typing import Literal

import optuna


TRAINING_SPACE = {
    "lr": ("float_log", 1e-5, 1e-2),
    "batch_size": ("categorical", [8, 16, 32, 64]),
    "loss_fn": ("categorical", ["cross_entropy", "focal"]),
    "warmup_epochs": ("int", 1, 5),
    "image_size": ("categorical", [224, 256, 384, 512]),
    "freeze_backbone": ("categorical", [True, False]),
    "augmentation_policy": ("categorical", ["none", "light", "medium", "heavy"]),
}

MODEL_SPACES = {
    "resnet50": {"drop_rate": ("float", 0.0, 0.5)},
    "efficientnet_b0": {"drop_rate": ("float", 0.0, 0.5)},
    "efficientnet_b3": {"drop_rate": ("float", 0.0, 0.5)},
    "convnext_tiny": {"drop_rate": ("float", 0.0, 0.5)},
    "vit_base_patch16_224": {"drop_rate": ("float", 0.0, 0.3)},
    "efficientnet_v2_s": {"drop_rate": ("float", 0.0, 0.5)},
    "efficientnet_v2_m": {"drop_rate": ("float", 0.0, 0.5)},
    "convnext_base": {"drop_rate": ("float", 0.0, 0.5)},
    "swin_base_patch4_window7_224": {"drop_rate": ("float", 0.0, 0.3)},
    "eva02_base_patch14_224": {"drop_rate": ("float", 0.0, 0.3)},
    "maxvit_tiny_224": {"drop_rate": ("float", 0.0, 0.3)},
    "coatnet_0_224": {"drop_rate": ("float", 0.0, 0.3)},
    "caformer_b36": {"drop_rate": ("float", 0.0, 0.3)},
    "mobilenetv4_conv_large": {"drop_rate": ("float", 0.0, 0.5)},
    "convnextv2_base": {"drop_rate": ("float", 0.0, 0.5)},
    "mambaout_base": {"drop_rate": ("float", 0.0, 0.5)},
    "mambaout_small": {"drop_rate": ("float", 0.0, 0.5)},
    "siglip_base_224": {"drop_rate": ("float", 0.0, 0.3)},
    "siglip_so400m_224": {"drop_rate": ("float", 0.0, 0.3)},
    "dinov2_base": {"drop_rate": ("float", 0.0, 0.3)},
    "dinov2_large": {"drop_rate": ("float", 0.0, 0.3)},
    "swinv2_base_256": {"drop_rate": ("float", 0.0, 0.3)},
    "eva02_large_patch14_224": {"drop_rate": ("float", 0.0, 0.3)},
    "convnextv2_large": {"drop_rate": ("float", 0.0, 0.5)},
    "eva02_small_patch14_224": {"drop_rate": ("float", 0.0, 0.3)},
}

MODEL_DEFAULTS = {
    "resnet50": {"drop_rate": 0.1},
    "efficientnet_b0": {"drop_rate": 0.2},
    "efficientnet_b3": {"drop_rate": 0.3},
    "convnext_tiny": {"drop_rate": 0.1},
    "vit_base_patch16_224": {"drop_rate": 0.1},
    "efficientnet_v2_s": {"drop_rate": 0.2},
    "efficientnet_v2_m": {"drop_rate": 0.3},
    "convnext_base": {"drop_rate": 0.1},
    "swin_base_patch4_window7_224": {"drop_rate": 0.1},
    "eva02_base_patch14_224": {"drop_rate": 0.1},
    "maxvit_tiny_224": {"drop_rate": 0.1},
    "coatnet_0_224": {"drop_rate": 0.1},
    "caformer_b36": {"drop_rate": 0.1},
    "mobilenetv4_conv_large": {"drop_rate": 0.2},
    "convnextv2_base": {"drop_rate": 0.1},
    "mambaout_base": {"drop_rate": 0.1},
    "mambaout_small": {"drop_rate": 0.1},
    "siglip_base_256": {"drop_rate": 0.1},
    "siglip_large_256": {"drop_rate": 0.1},
    "siglip_so400m_224": {"drop_rate": 0.1},
    "dinov2_base": {"drop_rate": 0.1},
    "dinov2_large": {"drop_rate": 0.1},
    "swinv2_base_256": {"drop_rate": 0.1},
    "eva02_large_patch14_224": {"drop_rate": 0.1},
    "convnextv2_large": {"drop_rate": 0.1},
    "eva02_small_patch14_224": {"drop_rate": 0.1},
}

MODEL_LR: dict[str, float] = {
    "vit_base_patch16_224": 1e-4,
    "swin_base_patch4_window7_224": 1e-4,
    "eva02_base_patch14_224": 5e-5,
    "maxvit_tiny_224": 1e-4,
    "coatnet_0_224": 1e-4,
    "caformer_b36": 5e-5,
    "siglip_base_224": 5e-5,
    "siglip_so400m_224": 3e-5,
    "dinov2_base": 5e-5,
    "dinov2_large": 3e-5,
    "swinv2_base_256": 1e-4,
    "eva02_large_patch14_224": 3e-5,
    "eva02_small_patch14_224": 5e-5,
}

BACKBONE_REGISTRY = {
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

AUGMENTATION_OPTIONS = {
    "none": {"augmentation_policy": "none"},
    "light": {"augmentation_policy": "light"},
    "medium": {"augmentation_policy": "medium"},
    "heavy": {"augmentation_policy": "heavy"},
}

DIMENSION_REGISTRY = {
    "augmentation": AUGMENTATION_OPTIONS,
}


def _suggest(trial: optuna.Trial, name: str, spec: tuple):
    kind = spec[0]
    if kind == "float":
        return trial.suggest_float(name, spec[1], spec[2])
    if kind == "float_log":
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    if kind == "int":
        return trial.suggest_int(name, spec[1], spec[2])
    if kind == "categorical":
        return trial.suggest_categorical(name, spec[1])
    raise ValueError(f"Unknown search space type: {kind}")


def suggest_params(
    trial: optuna.Trial,
    model_name: str,
    scope: Literal["all", "training", "architecture"] = "all",
) -> tuple[dict, dict]:
    training_params = {}
    if scope in ("all", "training"):
        for name, spec in TRAINING_SPACE.items():
            training_params[name] = _suggest(trial, name, spec)

    model_kwargs = {}
    if scope in ("all", "architecture"):
        model_space = MODEL_SPACES.get(model_name, {})
        for name, spec in model_space.items():
            model_kwargs[name] = _suggest(trial, name, spec)

    return training_params, model_kwargs


def get_search_space(
    model_name: str,
    scope: Literal["all", "training", "architecture"] = "all",
) -> dict[str, tuple]:
    space = {}
    if scope in ("all", "training"):
        space.update(TRAINING_SPACE)
    if scope in ("all", "architecture"):
        space.update(MODEL_SPACES.get(model_name, {}))
    return space
