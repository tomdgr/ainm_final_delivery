from dataclasses import dataclass


@dataclass
class ImageConfig:
    image_size: int = 224
    normalize_mean: tuple = (0.485, 0.456, 0.406)
    normalize_std: tuple = (0.229, 0.224, 0.225)
    augmentation_policy: str = "medium"  # none|light|medium|heavy
    num_channels: int = 3
    random_seed: int = 42
