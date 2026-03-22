import timm
import torch.nn as nn

from nm_ai_image.model.networks.registry import register_model

_BACKBONE_REGISTRY: dict[str, str] = {}


def register_timm_backbone(name: str, timm_name: str):
    _BACKBONE_REGISTRY[name] = timm_name

    @register_model(name)
    class TimmBackbone(nn.Module):
        def __init__(self, pretrained: bool = True, **kwargs):
            super().__init__()
            self.backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0, **kwargs)
            self.num_features = self.backbone.num_features

        def forward(self, x):
            return self.backbone(x)

    TimmBackbone.__name__ = f"TimmBackbone_{name}"
    TimmBackbone.__qualname__ = f"TimmBackbone_{name}"
    return TimmBackbone


register_timm_backbone("resnet50", "resnet50")
register_timm_backbone("efficientnet_b0", "efficientnet_b0")
register_timm_backbone("efficientnet_b3", "efficientnet_b3")
register_timm_backbone("convnext_tiny", "convnext_tiny")
register_timm_backbone("vit_base_patch16_224", "vit_base_patch16_224")

# --- Cutting-edge models ---
register_timm_backbone("efficientnet_v2_s", "tf_efficientnetv2_s")
register_timm_backbone("efficientnet_v2_m", "tf_efficientnetv2_m")
register_timm_backbone("convnext_base", "convnext_base")
register_timm_backbone("swin_base_patch4_window7_224", "swin_base_patch4_window7_224")
register_timm_backbone("eva02_base_patch14_224", "eva02_base_patch14_224.mim_in22k_ft_in1k")
register_timm_backbone("maxvit_tiny_224", "maxvit_tiny_tf_224")
register_timm_backbone("coatnet_0_224", "coatnet_0_224")
register_timm_backbone("caformer_b36", "caformer_b36.sail_in22k_ft_in1k")
register_timm_backbone("mobilenetv4_conv_large", "mobilenetv4_conv_large.e600_r384_in1k")
register_timm_backbone("convnextv2_base", "convnextv2_base.fcmae_ft_in22k_in1k")

# --- 2025 SOTA: SSM/Mamba, SigLIP, DINOv2, SwinV2, EVA-02-Large ---
register_timm_backbone("mambaout_base", "mambaout_base")
register_timm_backbone("mambaout_small", "mambaout_small")
register_timm_backbone("siglip_base_224", "vit_base_patch16_siglip_224")
register_timm_backbone("siglip_so400m_224", "vit_so400m_patch14_siglip_224")
register_timm_backbone("dinov2_base", "vit_base_patch14_dinov2")
register_timm_backbone("dinov2_large", "vit_large_patch14_dinov2")
register_timm_backbone("swinv2_base_256", "swinv2_base_window8_256")
register_timm_backbone("eva02_large_patch14_224", "eva02_large_patch14_224")
register_timm_backbone("convnextv2_large", "convnextv2_large.fcmae_ft_in22k_in1k")
register_timm_backbone("eva02_small_patch14_224", "eva02_small_patch14_224")


def get_backbone(name: str, pretrained: bool = True, **kwargs) -> nn.Module:
    from nm_ai_image.model.networks.registry import get_model

    model_cls = get_model(name)
    return model_cls(pretrained=pretrained, **kwargs)


def get_backbone_features(name: str, pretrained: bool = False) -> int:
    backbone = get_backbone(name, pretrained=pretrained)
    return backbone.num_features
