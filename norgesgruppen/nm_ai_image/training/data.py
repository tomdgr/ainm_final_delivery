import logging
from pathlib import Path

import lightning as L
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from nm_ai_image.config.task import TaskConfig, TaskType
from nm_ai_image.model.augmentation import build_train_transforms, build_val_transforms
from nm_ai_image.preprocessing.data_loading import generate_synthetic_data, load_image_folder

logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    def __init__(self, image_paths: list[str], labels: list, transform=None, task_type: TaskType = TaskType.CLASSIFICATION):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.task_type = task_type

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = np.array(img)

        if self.task_type == TaskType.CLASSIFICATION:
            if self.transform:
                transformed = self.transform(image=img)
                img = transformed["image"]
            label = self.labels[idx]
            return img, label

        elif self.task_type == TaskType.SEMANTIC_SEGMENTATION:
            mask_path = self.labels[idx]
            if mask_path and Path(mask_path).exists():
                mask = np.array(Image.open(mask_path).convert("L"))
            else:
                mask = np.zeros(img.shape[:2], dtype=np.uint8)

            if self.transform:
                transformed = self.transform(image=img, mask=mask)
                img = transformed["image"]
                mask = transformed["mask"]

            return img, mask.long() if isinstance(mask, torch.Tensor) else torch.tensor(mask, dtype=torch.long)

        elif self.task_type in (TaskType.OBJECT_DETECTION, TaskType.INSTANCE_SEGMENTATION):
            target = self.labels[idx] if self.labels[idx] else {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.long)}
            if self.transform:
                transformed = self.transform(image=img)
                img = transformed["image"]
            return img, target

        return img, self.labels[idx]


class ImageDataModule(L.LightningDataModule):
    def __init__(self, config: TaskConfig, data_dir: str | None = None, num_workers: int | None = None):
        super().__init__()
        self.config = config
        self.data_dir = data_dir or "data/raw"
        self.num_workers = num_workers if num_workers is not None else config.num_workers
        self.pin_memory = torch.cuda.is_available()

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self._num_classes: int | None = None

    @property
    def num_classes(self) -> int:
        if self._num_classes is not None:
            return self._num_classes
        return self.config.num_classes

    def setup(self, stage=None):
        if self.train_ds is not None:
            return

        data_dir = Path(self.data_dir)

        # Check for train/val/test split dirs
        train_dir = data_dir / "train"
        val_dir = data_dir / "val"
        test_dir = data_dir / "test"

        task_type_str = self.config.task_type.value

        if train_dir.exists():
            train_data = load_image_folder(train_dir, task_type_str)
            val_data = load_image_folder(val_dir, task_type_str) if val_dir.exists() else None
            test_data = load_image_folder(test_dir, task_type_str) if test_dir.exists() else None
        elif data_dir.exists() and any(data_dir.iterdir()):
            all_data = load_image_folder(data_dir, task_type_str)
            train_data, val_data, test_data = self._split_data(all_data)
        else:
            logger.info("No data found at %s, generating synthetic data", data_dir)
            synthetic_dir = generate_synthetic_data(
                n_samples=200,
                num_classes=self.config.num_classes,
                image_size=self.config.image_config.image_size,
                output_dir=data_dir,
            )
            all_data = load_image_folder(synthetic_dir, task_type_str)
            train_data, val_data, test_data = self._split_data(all_data)

        if train_data and "class_names" in train_data and train_data["class_names"]:
            self._num_classes = len(train_data["class_names"])

        train_transform = build_train_transforms(self.config.image_config, self.config.task_type)
        val_transform = build_val_transforms(self.config.image_config, self.config.task_type)

        self.train_ds = ImageDataset(
            train_data["images"], train_data["labels"],
            transform=train_transform, task_type=self.config.task_type,
        )

        if val_data and val_data["images"]:
            self.val_ds = ImageDataset(
                val_data["images"], val_data["labels"],
                transform=val_transform, task_type=self.config.task_type,
            )
        else:
            self.val_ds = self.train_ds

        if test_data and test_data["images"]:
            self.test_ds = ImageDataset(
                test_data["images"], test_data["labels"],
                transform=val_transform, task_type=self.config.task_type,
            )
        else:
            self.test_ds = self.val_ds

    def _split_data(self, data: dict) -> tuple[dict, dict, dict]:
        n = len(data["images"])
        indices = list(range(n))
        rng = np.random.default_rng(self.config.image_config.random_seed)
        rng.shuffle(indices)

        train_end = int(n * self.config.train_split)
        val_end = int(n * (self.config.train_split + self.config.val_split))

        def subset(idxs):
            return {
                "images": [data["images"][i] for i in idxs],
                "labels": [data["labels"][i] for i in idxs],
                "class_names": data.get("class_names", []),
            }

        return subset(indices[:train_end]), subset(indices[train_end:val_end]), subset(indices[val_end:])

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=self._collate_fn(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=self._collate_fn(),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=self._collate_fn(),
        )

    def _collate_fn(self):
        if self.config.task_type in (TaskType.OBJECT_DETECTION, TaskType.INSTANCE_SEGMENTATION):
            return _detection_collate_fn
        return None


def _detection_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets
