# NM i AI 2026 - NorgesGruppen Detection Pipeline
import argparse
import logging

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(): pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


def run_training(config, data_dir: str | None = None):
    from nm_ai_image.training import ImageDataModule, LightningTrainer

    data = ImageDataModule(config, data_dir=data_dir)
    data.setup()

    trainer = LightningTrainer(config, data)
    trainer.run_training()
    trainer.run_test()


def run_hyperparameter_tuning(config, tuning, data_dir: str | None = None):
    from nm_ai_image.tuning import HyperparameterOptimizer

    optimizer = HyperparameterOptimizer(config, tuning)
    result = optimizer.run(output_dir=config.output_dir + "/tuning")

    logger.info("Best value: %.4f", result.best_value)
    logger.info("Best params: %s", result.best_params)


def run_detection_training(args):
    """Train YOLOv8/RT-DETR for object detection."""
    from nm_ai_image.detection.data import COCOToYOLO
    from nm_ai_image.detection.evaluate import evaluate_model
    from nm_ai_image.detection.train import DetectionTrainConfig, train

    # Convert COCO to YOLO format
    converter = COCOToYOLO(args.coco_dir, args.yolo_dir, val_ratio=args.val_ratio, seed=args.seed)
    stats = converter.convert()
    logger.info("Dataset: %d images, %d classes, %d annotations", stats.num_images, stats.num_categories, stats.num_annotations)
    logger.info("Rare classes (<10 annotations): %d", len(stats.rare_classes))

    config = DetectionTrainConfig(
        model=args.model,
        data=str((converter.output_dir / "data.yaml").resolve()),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        name=args.name,
        patience=args.patience,
        multi_scale=args.multi_scale,
        seed=args.seed,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        scale=args.scale,
        degrees=args.degrees,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        warmup_epochs=args.warmup_epochs,
        flipud=args.flipud,
        cls=args.cls,
        box=args.box,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        freeze=args.freeze,
        rect=args.rect,
        save_period=args.save_period,
        nbs=args.nbs,
        cos_lr=args.cos_lr,
    )
    best_path = train(config)
    logger.info("Training complete. Best weights: %s", best_path)

    # Run competition-style evaluation on full training set
    if best_path.exists():
        coco_json = str((converter.coco_dir / "annotations.json").resolve())
        image_dir = str((converter.coco_dir / "images").resolve())
        eval_dir = str(best_path.parent.parent / "eval")
        result = evaluate_model(best_path, coco_json, image_dir, output_dir=eval_dir, imgsz=args.imgsz)
        logger.info("Competition score: %.4f", result.competition_score)


def run_evaluate(args):
    """Evaluate a trained model using competition scoring."""
    from nm_ai_image.detection.evaluate import evaluate_model

    result = evaluate_model(
        weights=args.weights,
        coco_json=args.coco_json,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        imgsz=args.imgsz,
    )
    logger.info("Competition score: %.4f", result.competition_score)


def run_build_gallery(args):
    """Build embedding gallery from reference + training images."""
    from nm_ai_image.detection.classifier import GalleryBuilder

    builder = GalleryBuilder(backbone=args.backbone, max_crops_per_class=args.max_crops)
    builder.build(
        coco_json=args.coco_json,
        image_dir=args.image_dir,
        product_dir=args.product_dir,
        output_path=args.output,
    )


def run_build_submission(args):
    """Build a competition submission ZIP."""
    from nm_ai_image.detection.submission import SubmissionBuilder

    builder = SubmissionBuilder(output_dir=args.output_dir)

    if args.onnx_ensemble:
        zip_path = builder.build_onnx_ensemble(
            onnx_paths=args.weights,
            name=args.name,
            imgsz=args.imgsz,
            conf=args.conf,
            wbf_iou_thr=args.wbf_iou,
            nms_iou=args.nms_iou,
        )
    elif args.ensemble:
        zip_path = builder.build_ensemble(
            weight_paths=args.weights,
            name=args.name,
            imgsz=args.imgsz,
        )
    elif args.gallery:
        zip_path = builder.build_twostage(
            detector_weights=args.weights[0],
            gallery_path=args.gallery,
            name=args.name,
            imgsz=args.imgsz,
            backbone_weights=args.backbone_weights,
        )
    elif args.sahi:
        zip_path = builder.build_sahi(
            weights_path=args.weights[0],
            name=args.name,
            imgsz=args.imgsz,
        )
    elif args.onnx:
        zip_path = builder.build_onnx(
            onnx_path=args.weights[0],
            name=args.name,
            imgsz=args.imgsz,
            conf=args.conf,
            nms_iou=args.nms_iou,
            use_soft_nms=args.soft_nms,
            soft_sigma=args.soft_sigma,
        )
    else:
        zip_path = builder.build_single_model(
            weights_path=args.weights[0],
            name=args.name,
            imgsz=args.imgsz,
        )
    logger.info("Submission ZIP: %s", zip_path)


def main():
    parser = argparse.ArgumentParser(prog="nm_ai_image")
    subparsers = parser.add_subparsers(dest="command")

    # Classification training (legacy)
    cls_parser = subparsers.add_parser("classify", help="Train classification model")
    cls_parser.add_argument("--backbone", default="resnet50")
    cls_parser.add_argument("--data-dir", default=None)
    cls_parser.add_argument("--num-classes", type=int, default=10)
    cls_parser.add_argument("--image-size", type=int, default=224)
    cls_parser.add_argument("--epochs", type=int, default=100)
    cls_parser.add_argument("--batch-size", type=int, default=32)
    cls_parser.add_argument("--lr", type=float, default=1e-3)
    cls_parser.add_argument("--augmentation", default="medium", choices=["none", "light", "medium", "heavy"])
    cls_parser.add_argument("--hyperparamsweep", action="store_true")

    # Detection training
    det_parser = subparsers.add_parser("detect", help="Train YOLOv8/RT-DETR detector")
    det_parser.add_argument("--model", default="yolov8x.pt", help="Ultralytics model name or path")
    det_parser.add_argument("--coco-dir", default="data/raw/coco_dataset/train", help="COCO dataset directory")
    det_parser.add_argument("--yolo-dir", default="data/yolo", help="YOLO output directory")
    det_parser.add_argument("--imgsz", type=int, default=640)
    det_parser.add_argument("--epochs", type=int, default=150)
    det_parser.add_argument("--batch", type=int, default=8)
    det_parser.add_argument("--name", default="train")
    det_parser.add_argument("--patience", type=int, default=30)
    det_parser.add_argument("--multi-scale", action="store_true")
    det_parser.add_argument("--seed", type=int, default=42)
    det_parser.add_argument("--val-ratio", type=float, default=0.15)
    det_parser.add_argument("--mixup", type=float, default=0.15)
    det_parser.add_argument("--copy-paste", type=float, default=0.1)
    det_parser.add_argument("--scale", type=float, default=0.5)
    det_parser.add_argument("--degrees", type=float, default=5.0)
    det_parser.add_argument("--optimizer", default="AdamW")
    det_parser.add_argument("--lr0", type=float, default=0.001)
    det_parser.add_argument("--lrf", type=float, default=0.01)
    det_parser.add_argument("--warmup-epochs", type=int, default=5)
    det_parser.add_argument("--flipud", type=float, default=0.5)
    det_parser.add_argument("--fliplr", type=float, default=0.5)
    det_parser.add_argument("--cls", type=float, default=0.5)
    det_parser.add_argument("--box", type=float, default=7.5)
    det_parser.add_argument("--mosaic", type=float, default=1.0)
    det_parser.add_argument("--close-mosaic", type=int, default=20)
    det_parser.add_argument("--freeze", type=int, default=None)
    det_parser.add_argument("--rect", action="store_true")
    det_parser.add_argument("--save-period", type=int, default=25)
    det_parser.add_argument("--nbs", type=int, default=64)
    det_parser.add_argument("--cos-lr", action="store_true", default=True)
    det_parser.add_argument("--no-cos-lr", dest="cos_lr", action="store_false")

    # Build gallery
    gal_parser = subparsers.add_parser("gallery", help="Build embedding gallery")
    gal_parser.add_argument("--coco-json", default="data/raw/coco_dataset/train/annotations.json")
    gal_parser.add_argument("--image-dir", default="data/raw/coco_dataset/train/images")
    gal_parser.add_argument("--product-dir", default="data/raw/product_images")
    gal_parser.add_argument("--output", default="data/reference_embeddings.pt")
    gal_parser.add_argument("--backbone", default="resnet50")
    gal_parser.add_argument("--max-crops", type=int, default=20)

    # Evaluate model
    eval_parser = subparsers.add_parser("eval", help="Evaluate model with competition scoring")
    eval_parser.add_argument("weights", help="Model weight file")
    eval_parser.add_argument("--coco-json", default="data/raw/coco_dataset/train/annotations.json")
    eval_parser.add_argument("--image-dir", default="data/raw/coco_dataset/train/images")
    eval_parser.add_argument("--imgsz", type=int, default=640)
    eval_parser.add_argument("--output-dir", default="outputs/eval")

    # Build submission
    sub_parser = subparsers.add_parser("submission", help="Build submission ZIP")
    sub_parser.add_argument("weights", nargs="+", help="Model weight file(s)")
    sub_parser.add_argument("--name", default="submission")
    sub_parser.add_argument("--imgsz", type=int, default=640)
    sub_parser.add_argument("--ensemble", action="store_true", help="WBF ensemble mode")
    sub_parser.add_argument("--gallery", default=None, help="Gallery .pt for two-stage mode")
    sub_parser.add_argument("--backbone-weights", default=None, help="ResNet50 backbone .pt for offline two-stage")
    sub_parser.add_argument("--sahi", action="store_true", help="SAHI tiled inference mode")
    sub_parser.add_argument("--onnx", action="store_true", help="ONNX inference mode")
    sub_parser.add_argument("--onnx-ensemble", action="store_true", help="ONNX WBF ensemble mode")
    sub_parser.add_argument("--conf", type=float, default=0.01, help="Confidence threshold")
    sub_parser.add_argument("--nms-iou", type=float, default=0.5, help="NMS IoU threshold")
    sub_parser.add_argument("--wbf-iou", type=float, default=0.55, help="WBF IoU threshold (ensemble)")
    sub_parser.add_argument("--soft-nms", action="store_true", help="Use soft-NMS instead of hard NMS")
    sub_parser.add_argument("--soft-sigma", type=float, default=0.5, help="Soft-NMS sigma")
    sub_parser.add_argument("--output-dir", default="submissions")

    args = parser.parse_args()
    load_dotenv()

    if args.command == "classify":
        from nm_ai_image.config import TaskConfig, TaskType, TuningConfig
        from nm_ai_image.config.image import ImageConfig

        task_type = TaskType.CLASSIFICATION
        image_config = ImageConfig(image_size=args.image_size, augmentation_policy=args.augmentation)
        config = TaskConfig(
            task_type=task_type, backbone_name=args.backbone, num_classes=args.num_classes,
            image_config=image_config, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, image_size=args.image_size,
        )
        if args.hyperparamsweep:
            run_hyperparameter_tuning(config, TuningConfig(), data_dir=args.data_dir)
        else:
            run_training(config, data_dir=args.data_dir)
    elif args.command == "detect":
        run_detection_training(args)
    elif args.command == "eval":
        run_evaluate(args)
    elif args.command == "gallery":
        run_build_gallery(args)
    elif args.command == "submission":
        run_build_submission(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
