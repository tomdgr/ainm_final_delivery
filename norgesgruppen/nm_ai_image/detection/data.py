"""COCO-to-YOLO data conversion and dataset utilities."""
import json
import logging
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DatasetStats:
    num_images: int = 0
    num_categories: int = 0
    num_annotations: int = 0
    annotations_per_class: dict[int, int] = field(default_factory=dict)
    image_dimensions: list[tuple[int, int]] = field(default_factory=list)

    @property
    def rare_classes(self) -> list[int]:
        return [c for c, n in self.annotations_per_class.items() if n < 10]


class COCOToYOLO:
    """Convert COCO annotations to YOLO format with train/val split."""

    def __init__(self, coco_dir: str | Path, output_dir: str | Path, val_ratio: float = 0.15, seed: int = 42):
        self.coco_dir = Path(coco_dir)
        self.output_dir = Path(output_dir)
        self.val_ratio = val_ratio
        self.seed = seed
        self._coco: dict | None = None

    @property
    def coco(self) -> dict:
        if self._coco is None:
            ann_file = self.coco_dir / "annotations.json"
            with open(ann_file) as f:
                self._coco = json.load(f)
        return self._coco

    def convert(self) -> DatasetStats:
        """Run the full conversion pipeline."""
        images = {img["id"]: img for img in self.coco["images"]}
        anns_by_image = defaultdict(list)
        for ann in self.coco["annotations"]:
            anns_by_image[ann["image_id"]].append(ann)

        # Split
        rng = random.Random(self.seed)
        image_ids = list(images.keys())
        rng.shuffle(image_ids)
        val_count = max(1, int(len(image_ids) * self.val_ratio))
        val_ids = set(image_ids[:val_count])
        train_ids = set(image_ids[val_count:])

        logger.info("Train: %d, Val: %d", len(train_ids), len(val_ids))

        for split, ids in [("train", train_ids), ("val", val_ids)]:
            img_out = self.output_dir / "images" / split
            lbl_out = self.output_dir / "labels" / split
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)

            for img_id in ids:
                img_info = images[img_id]
                w, h = img_info["width"], img_info["height"]
                fname = img_info["file_name"]

                src = self.coco_dir / "images" / fname
                if src.exists():
                    shutil.copy2(src, img_out / fname)

                label_file = lbl_out / (Path(fname).stem + ".txt")
                lines = []
                for ann in anns_by_image.get(img_id, []):
                    bx, by, bw, bh = ann["bbox"]
                    cx = max(0, min(1, (bx + bw / 2) / w))
                    cy = max(0, min(1, (by + bh / 2) / h))
                    nw = max(0, min(1, bw / w))
                    nh = max(0, min(1, bh / h))
                    lines.append(f"{ann['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

                with open(label_file, "w") as f:
                    f.write("\n".join(lines))

        self._write_data_yaml()
        stats = self._compute_stats(anns_by_image)
        logger.info("Converted %d images, %d annotations, %d classes", stats.num_images, stats.num_annotations, stats.num_categories)
        return stats

    def _write_data_yaml(self):
        cat_names = {c["id"]: c["name"] for c in self.coco["categories"]}
        nc = len(self.coco["categories"])
        names = [cat_names.get(i, f"class_{i}") for i in range(nc)]
        yaml_content = f"path: {self.output_dir.resolve()}\ntrain: images/train\nval: images/val\n\nnc: {nc}\nnames: {names}\n"
        with open(self.output_dir / "data.yaml", "w") as f:
            f.write(yaml_content)

    def _compute_stats(self, anns_by_image: dict) -> DatasetStats:
        ann_per_cat = defaultdict(int)
        for anns in anns_by_image.values():
            for ann in anns:
                ann_per_cat[ann["category_id"]] += 1
        return DatasetStats(
            num_images=len(self.coco["images"]),
            num_categories=len(self.coco["categories"]),
            num_annotations=len(self.coco["annotations"]),
            annotations_per_class=dict(ann_per_cat),
            image_dimensions=[(img["width"], img["height"]) for img in self.coco["images"]],
        )

    def get_category_names(self) -> dict[int, str]:
        return {c["id"]: c["name"] for c in self.coco["categories"]}
