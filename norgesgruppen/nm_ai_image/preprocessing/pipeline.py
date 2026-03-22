import albumentations as A
from albumentations.pytorch import ToTensorV2

from nm_ai_image.config.image import ImageConfig
from nm_ai_image.model.augmentation import build_train_transforms, build_val_transforms


def build_augmentation_pipeline(config: ImageConfig, train: bool = True) -> A.Compose:
    if train:
        return build_train_transforms(config)
    return build_val_transforms(config)
