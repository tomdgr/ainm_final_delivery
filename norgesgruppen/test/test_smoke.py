import numpy as np
import torch

from nm_ai_image.config.image import ImageConfig
from nm_ai_image.config.task import TaskConfig, TaskType
from nm_ai_image.ensemble.blending import BlendingEnsemble, blend_predictions
from nm_ai_image.evaluation.metrics import calculate_all_metrics
from nm_ai_image.model.augmentation import build_train_transforms, build_val_transforms
from nm_ai_image.model.networks import get_model, list_models


def test_model_registry():
    models = list_models()
    assert len(models) > 0
    assert "resnet50" in models
    assert "efficientnet_b0" in models
    assert "vit_base_patch16_224" in models


def test_backbone_forward():
    model_cls = get_model("resnet50")
    model = model_cls(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    features = model(x)
    assert features.ndim == 2
    assert features.shape[0] == 2
    assert features.shape[1] > 0  # num_features


def test_classification_head():
    from nm_ai_image.model.networks.heads import build_model_for_task

    model = build_model_for_task(
        task_type=TaskType.CLASSIFICATION,
        backbone_name="resnet50",
        num_classes=10,
        pretrained=False,
    )
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 10)


def test_image_dataset(tmp_image_folder):
    from nm_ai_image.training.data import ImageDataModule

    config = TaskConfig(
        task_type=TaskType.CLASSIFICATION,
        backbone_name="resnet50",
        num_classes=3,
        batch_size=4,
        num_workers=0,
    )
    config.image_config.image_size = 32

    dm = ImageDataModule(config, data_dir=str(tmp_image_folder), num_workers=0)
    dm.setup()

    assert dm.train_ds is not None
    assert len(dm.train_ds) > 0

    img, label = dm.train_ds[0]
    assert img.shape[0] == 3  # channels
    assert isinstance(label, int)


def test_classification_metrics():
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 1, 2, 0])
    m = calculate_all_metrics("classification", y_pred, y_true, num_classes=3)
    assert "accuracy" in m
    assert "f1" in m
    assert "precision" in m
    assert "recall" in m
    assert m["accuracy"] > 0.8


def test_augmentation_pipeline():
    config = ImageConfig(image_size=224, augmentation_policy="medium")
    train_transform = build_train_transforms(config)
    val_transform = build_val_transforms(config)

    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    train_out = train_transform(image=img)["image"]
    val_out = val_transform(image=img)["image"]

    assert train_out.shape == (3, 224, 224)
    assert val_out.shape == (3, 224, 224)


def test_blending_ensemble():
    rng = np.random.default_rng(42)
    # Two model outputs: (N, C) probability arrays
    out1 = rng.random((20, 5))
    out2 = rng.random((20, 5))

    # Simple weighted blend
    weights = np.array([0.6, 0.4])
    blended = blend_predictions([out1, out2], weights)
    assert blended.shape == (20, 5)

    # BlendingEnsemble
    y_true = np.argmax(out1 + out2, axis=1)  # synthetic labels
    ensemble = BlendingEnsemble(method="linear")
    ensemble.fit([out1, out2], y_true)
    preds = ensemble.predict([out1, out2])
    assert preds.shape == (20, 5)
