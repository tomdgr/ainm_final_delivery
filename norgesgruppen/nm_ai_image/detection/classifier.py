"""Embedding-based product classifier for two-stage detection.

Stage 1 (Detector) finds boxes. Stage 2 (this) classifies crops
by matching against a gallery of reference + training embeddings.
"""
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)

BACKBONES = {
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048, "fc"),
    "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT, 2048, "fc"),
    "efficientnet_v2_s": (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.DEFAULT, 1280, "classifier"),
}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load_backbone(name: str, pretrained: bool = True) -> tuple[torch.nn.Module, int]:
    factory, weights, feat_dim, head_attr = BACKBONES[name]
    model = factory(weights=weights if pretrained else None)
    setattr(model, head_attr, torch.nn.Identity())
    model.eval()
    return model, feat_dim


class GalleryBuilder:
    """Build a gallery of per-class embeddings from reference images and training crops."""

    def __init__(self, backbone: str = "resnet50", device: str | None = None, max_crops_per_class: int = 20):
        self.backbone_name = backbone
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_crops = max_crops_per_class
        self.model, self.feat_dim = _load_backbone(backbone)
        self.model = self.model.to(self.device)

    @torch.no_grad()
    def _embed(self, img: Image.Image) -> torch.Tensor:
        tensor = TRANSFORM(img).unsqueeze(0).to(self.device)
        feat = self.model(tensor)
        return F.normalize(feat, dim=1).cpu()

    def build(
        self,
        coco_json: str | Path,
        image_dir: str | Path,
        product_dir: str | Path,
        output_path: str | Path,
    ) -> Path:
        """Build gallery and save to disk."""
        output_path = Path(output_path)

        with open(coco_json) as f:
            coco = json.load(f)

        images = {img["id"]: img for img in coco["images"]}
        catid_to_name = {c["id"]: c["name"] for c in coco["categories"]}
        nc = len(coco["categories"])
        name_to_catid = {c["name"].strip().upper(): c["id"] for c in coco["categories"]}

        cat_embeddings: dict[int, list[torch.Tensor]] = defaultdict(list)

        # Reference images
        product_dir = Path(product_dir)
        meta_path = product_dir / "metadata.json"
        ref_count = 0
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            for p in meta.get("products", []):
                if not p.get("has_images"):
                    continue
                cat_id = name_to_catid.get(p["product_name"].strip().upper())
                if cat_id is None:
                    continue
                pdir = product_dir / p["product_code"]
                if not pdir.exists():
                    continue
                for img_path in sorted(pdir.glob("*.jpg")):
                    try:
                        cat_embeddings[cat_id].append(self._embed(Image.open(img_path).convert("RGB")))
                        ref_count += 1
                    except Exception as e:
                        logger.warning("Error processing %s: %s", img_path, e)

        logger.info("Extracted %d reference embeddings for %d categories", ref_count, len(cat_embeddings))

        # Training crops
        anns_by_cat: dict[int, list] = defaultdict(list)
        for ann in coco["annotations"]:
            anns_by_cat[ann["category_id"]].append(ann)

        image_dir = Path(image_dir)
        crop_count = 0
        rng = np.random.RandomState(42)
        for cat_id in range(nc):
            anns = anns_by_cat.get(cat_id, [])
            if len(anns) > self.max_crops:
                indices = rng.choice(len(anns), self.max_crops, replace=False)
                anns = [anns[i] for i in indices]
            for ann in anns:
                img_info = images[ann["image_id"]]
                img_path = image_dir / img_info["file_name"]
                if not img_path.exists():
                    continue
                try:
                    img = Image.open(img_path).convert("RGB")
                    x, y, w, h = ann["bbox"]
                    pad_x, pad_y = w * 0.1, h * 0.1
                    crop = img.crop((
                        max(0, x - pad_x), max(0, y - pad_y),
                        min(img.width, x + w + pad_x), min(img.height, y + h + pad_y),
                    ))
                    if crop.width >= 10 and crop.height >= 10:
                        cat_embeddings[cat_id].append(self._embed(crop))
                        crop_count += 1
                except Exception as e:
                    logger.warning("Error cropping ann %d: %s", ann["id"], e)

        logger.info("Extracted %d crop embeddings", crop_count)

        # Average per category
        gallery = torch.zeros(nc, self.feat_dim)
        gallery_counts = torch.zeros(nc)
        for cat_id in range(nc):
            embs = cat_embeddings.get(cat_id, [])
            if embs:
                stacked = torch.cat(embs, dim=0)
                gallery[cat_id] = F.normalize(stacked.mean(dim=0, keepdim=True), dim=1).squeeze(0)
                gallery_counts[cat_id] = len(embs)
            else:
                logger.warning("No embeddings for cat %d (%s)", cat_id, catid_to_name.get(cat_id, "?"))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "gallery": gallery,
            "gallery_counts": gallery_counts,
            "feat_dim": self.feat_dim,
            "nc": nc,
            "backbone": self.backbone_name,
            "catid_to_name": catid_to_name,
        }, output_path)

        cats_with = int((gallery_counts > 0).sum().item())
        logger.info("Saved gallery to %s (%d/%d categories covered)", output_path, cats_with, nc)
        return output_path


class EmbeddingClassifier:
    """Classify image crops against a pre-built gallery via cosine similarity."""

    def __init__(self, gallery_path: str | Path, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        data = torch.load(gallery_path, map_location=self.device)
        self.gallery = data["gallery"].to(self.device)
        self.nc = data["nc"]
        self.backbone_name = data["backbone"]

        self.model, _ = _load_backbone(self.backbone_name, pretrained=True)
        self.model = self.model.to(self.device)

    @torch.no_grad()
    def classify(self, crops: list[Image.Image], batch_size: int = 32) -> list[tuple[int, float]]:
        """Classify crops. Returns list of (category_id, confidence)."""
        if not crops:
            return []

        results = []
        for i in range(0, len(crops), batch_size):
            batch = crops[i:i + batch_size]
            tensors = torch.stack([TRANSFORM(c) for c in batch]).to(self.device)
            feats = F.normalize(self.model(tensors), dim=1)
            sims = feats @ self.gallery.T
            best_cats = sims.argmax(dim=1)
            best_scores = sims.gather(1, best_cats.unsqueeze(1)).squeeze(1)
            for cat_id, score in zip(best_cats.tolist(), best_scores.tolist()):
                results.append((cat_id, max(0, (score + 1) / 2)))
        return results
