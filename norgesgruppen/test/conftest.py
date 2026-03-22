import numpy as np
import pytest
import torch


@pytest.fixture
def sample_images():
    """Batch of random images (B, C, H, W)."""
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def sample_images_small():
    """Batch of small random images for fast tests."""
    return torch.randn(4, 3, 64, 64)


@pytest.fixture
def sample_classification_labels():
    return torch.tensor([0, 1, 2, 3])


@pytest.fixture
def sample_detection_targets():
    return [
        {"boxes": torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
         "labels": torch.tensor([1])},
        {"boxes": torch.tensor([[20, 20, 60, 60]], dtype=torch.float32),
         "labels": torch.tensor([2])},
    ]


@pytest.fixture
def sample_masks():
    """Batch of segmentation masks (B, H, W)."""
    return torch.randint(0, 5, (4, 224, 224))


@pytest.fixture
def tmp_image_folder(tmp_path):
    """Create a temporary ImageFolder-style directory."""
    rng = np.random.default_rng(42)
    for class_idx in range(3):
        class_dir = tmp_path / f"class_{class_idx}"
        class_dir.mkdir()
        for i in range(10):
            img = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
            from PIL import Image
            Image.fromarray(img).save(class_dir / f"img_{i:03d}.png")
    return tmp_path
