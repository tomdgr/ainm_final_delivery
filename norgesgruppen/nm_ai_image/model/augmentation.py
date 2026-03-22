import albumentations as A
from albumentations.pytorch import ToTensorV2

from nm_ai_image.config.image import ImageConfig
from nm_ai_image.config.task import TaskType


def build_train_transforms(config: ImageConfig, task_type: TaskType = TaskType.CLASSIFICATION) -> A.Compose:
    policy = config.augmentation_policy
    size = config.image_size

    sz = (size, size)

    if policy == "none":
        transforms = [A.Resize(*sz)]
    elif policy == "light":
        transforms = [
            A.Resize(*sz),
            A.HorizontalFlip(p=0.5),
        ]
    elif policy == "medium":
        transforms = [
            A.RandomResizedCrop(sz, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.1),
        ]
    elif policy == "heavy":
        transforms = [
            A.RandomResizedCrop(sz, scale=(0.5, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=0.7),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.GaussNoise(p=0.2),
            A.CoarseDropout(max_holes=8, max_height=size // 8, max_width=size // 8, p=0.3),
        ]
    else:
        raise ValueError(f"Unknown augmentation policy: {policy}")

    transforms.extend([
        A.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ToTensorV2(),
    ])

    additional_targets = {}
    bbox_params = None

    if task_type in (TaskType.OBJECT_DETECTION, TaskType.INSTANCE_SEGMENTATION):
        bbox_params = A.BboxParams(format="pascal_voc", label_fields=["labels"])

    return A.Compose(transforms, bbox_params=bbox_params, additional_targets=additional_targets)


def build_val_transforms(config: ImageConfig, task_type: TaskType = TaskType.CLASSIFICATION) -> A.Compose:
    transforms = [
        A.Resize(config.image_size, config.image_size),
        A.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ToTensorV2(),
    ]

    bbox_params = None
    if task_type in (TaskType.OBJECT_DETECTION, TaskType.INSTANCE_SEGMENTATION):
        bbox_params = A.BboxParams(format="pascal_voc", label_fields=["labels"])

    return A.Compose(transforms, bbox_params=bbox_params)
