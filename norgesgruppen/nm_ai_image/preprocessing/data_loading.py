import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def load_image_folder(root: str | Path, task_type: str = "classification") -> dict:
    root = Path(root)

    if task_type == "classification":
        return _load_classification_folder(root)
    elif task_type in ("object_detection", "instance_segmentation"):
        return _load_detection_folder(root)
    elif task_type == "semantic_segmentation":
        return _load_segmentation_folder(root)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def _load_classification_folder(root: Path) -> dict:
    """Load ImageFolder-style directory: root/class_name/image.jpg"""
    images = []
    labels = []
    class_names = []

    if not root.exists():
        logger.warning("Data directory %s does not exist", root)
        return {"images": [], "labels": [], "class_names": []}

    subdirs = sorted([d for d in root.iterdir() if d.is_dir()])

    if subdirs:
        class_names = [d.name for d in subdirs]
        for class_idx, class_dir in enumerate(subdirs):
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
                    images.append(str(img_path))
                    labels.append(class_idx)
    else:
        # Flat directory — all images, no labels
        for img_path in sorted(root.iterdir()):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
                images.append(str(img_path))
                labels.append(0)

    logger.info("Loaded %d images, %d classes from %s", len(images), len(class_names), root)
    return {"images": images, "labels": labels, "class_names": class_names}


def _load_detection_folder(root: Path) -> dict:
    """Load COCO or YOLO-style detection data."""
    images = []
    annotations = []

    if not root.exists():
        return {"images": [], "annotations": []}

    images_dir = root / "images"
    labels_dir = root / "labels"

    if images_dir.exists() and labels_dir.exists():
        # YOLO format
        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                label_path = labels_dir / (img_path.stem + ".txt")
                images.append(str(img_path))
                annotations.append(str(label_path) if label_path.exists() else None)
    else:
        for img_path in sorted(root.iterdir()):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                images.append(str(img_path))
                annotations.append(None)

    return {"images": images, "annotations": annotations}


def _load_segmentation_folder(root: Path) -> dict:
    """Load segmentation data: root/images/ + root/masks/"""
    images = []
    masks = []

    images_dir = root / "images"
    masks_dir = root / "masks"

    if not images_dir.exists():
        return {"images": [], "masks": []}

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            mask_path = masks_dir / img_path.name
            if not mask_path.exists():
                mask_path = masks_dir / (img_path.stem + ".png")
            images.append(str(img_path))
            masks.append(str(mask_path) if mask_path.exists() else None)

    return {"images": images, "masks": masks}


def generate_synthetic_data(
    n_samples: int = 100,
    num_classes: int = 5,
    image_size: int = 64,
    output_dir: str | Path = "data/synthetic",
) -> Path:
    """Generate synthetic classification data for smoke tests."""
    output_dir = Path(output_dir)
    rng = np.random.default_rng(42)

    for class_idx in range(num_classes):
        class_dir = output_dir / f"class_{class_idx}"
        class_dir.mkdir(parents=True, exist_ok=True)

        n_per_class = n_samples // num_classes
        for i in range(n_per_class):
            # Create simple colored images per class
            img = rng.integers(0, 255, (image_size, image_size, 3), dtype=np.uint8)
            # Add class-specific color bias
            img[:, :, class_idx % 3] = np.clip(
                img[:, :, class_idx % 3].astype(int) + 50 * (class_idx + 1), 0, 255
            ).astype(np.uint8)

            Image.fromarray(img).save(class_dir / f"img_{i:04d}.png")

    logger.info("Generated %d synthetic images in %s", n_samples, output_dir)
    return output_dir
