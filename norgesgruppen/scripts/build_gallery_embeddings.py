"""Build gallery embeddings using timm models for crop re-ranking.

Usage:
  uv run python scripts/build_gallery_embeddings.py
  uv run python scripts/build_gallery_embeddings.py --model efficientnet_b3.ra2_in1k --output gallery_effb3.npz
"""
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent


def build_gallery(
    model_name: str = "convnext_base.fb_in22k_ft_in1k",
    output_path: str = "gallery_convnext.npz",
    max_ref_per_class: int = 7,
    max_crops_per_class: int = 10,
    image_size: int = 224,
    batch_size: int = 32,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load model
    logger.info("Loading model: %s", model_name)
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model = model.to(device)
    model.eval()

    # Get the model's native transform
    config = resolve_data_config(model.pretrained_cfg)
    config["input_size"] = (3, image_size, image_size)
    transform = create_transform(**config, is_training=False)
    logger.info("Transform: input_size=%s", config["input_size"])

    # Test feature dim
    with torch.no_grad():
        dummy = torch.randn(1, 3, image_size, image_size, device=device)
        feat_dim = model(dummy).shape[1]
    logger.info("Feature dim: %d", feat_dim)

    # Load annotations
    coco_json = ROOT / "data" / "raw" / "coco_dataset" / "train" / "annotations.json"
    with open(coco_json) as f:
        coco = json.load(f)

    categories = coco["categories"]
    nc = len(categories)
    name_to_catid = {c["name"].strip().upper(): c["id"] for c in categories}
    catid_to_name = {c["id"]: c["name"] for c in categories}
    images = {img["id"]: img for img in coco["images"]}
    logger.info("Categories: %d, Images: %d", nc, len(images))

    # Load product metadata
    product_dir = ROOT / "data" / "raw" / "product_images"
    with open(product_dir / "metadata.json") as f:
        meta = json.load(f)

    @torch.no_grad()
    def embed_batch(pil_images: list[Image.Image]) -> np.ndarray:
        """Embed a batch of images. Returns (N, feat_dim) array."""
        if not pil_images:
            return np.zeros((0, feat_dim), dtype=np.float32)
        tensors = torch.stack([transform(img) for img in pil_images]).to(device)
        feats = model(tensors)
        feats = F.normalize(feats, dim=1)
        return feats.cpu().numpy()

    # Collect all images to embed with their category labels
    ref_images = []  # (PIL.Image, cat_id)
    crop_images = []  # (PIL.Image, cat_id)

    # 1. Reference images from product_images/
    logger.info("Loading reference images...")
    t0 = time.time()
    for pi, p in enumerate(meta["products"]):
        if not p.get("has_images"):
            continue
        cat_id = name_to_catid.get(p["product_name"].strip().upper())
        if cat_id is None:
            continue
        pdir = product_dir / p["product_code"]
        if not pdir.exists():
            continue
        img_files = sorted(pdir.glob("*.jpg"))
        priority = {"main.jpg": 0, "front.jpg": 1, "back.jpg": 2}
        img_files.sort(key=lambda x: priority.get(x.name, 5))
        for img_path in img_files[:max_ref_per_class]:
            try:
                img = Image.open(img_path).convert("RGB")
                ref_images.append((img, cat_id))
            except Exception as e:
                logger.warning("Error loading %s: %s", img_path, e)
        if (pi + 1) % 100 == 0:
            logger.info("  Loaded %d/%d products...", pi + 1, len(meta["products"]))

    logger.info("Loaded %d reference images in %.1fs", len(ref_images), time.time() - t0)

    # 2. Training crops
    logger.info("Loading training crops...")
    t0 = time.time()
    anns_by_cat: dict[int, list] = {i: [] for i in range(nc)}
    for ann in coco["annotations"]:
        anns_by_cat[ann["category_id"]].append(ann)

    image_dir = ROOT / "data" / "raw" / "coco_dataset" / "train" / "images"
    # Cache opened images to avoid re-opening
    img_cache: dict[int, Image.Image] = {}
    rng = np.random.RandomState(42)

    for cat_id in range(nc):
        anns = anns_by_cat[cat_id]
        if len(anns) > max_crops_per_class:
            indices = rng.choice(len(anns), max_crops_per_class, replace=False)
            anns = [anns[i] for i in indices]
        for ann in anns:
            img_id = ann["image_id"]
            if img_id not in img_cache:
                img_info = images[img_id]
                img_path = image_dir / img_info["file_name"]
                if img_path.exists():
                    img_cache[img_id] = Image.open(img_path).convert("RGB")
                else:
                    img_cache[img_id] = None
            img = img_cache[img_id]
            if img is None:
                continue
            try:
                x, y, w, h = ann["bbox"]
                pad_x, pad_y = w * 0.1, h * 0.1
                crop = img.crop((
                    max(0, x - pad_x), max(0, y - pad_y),
                    min(img.width, x + w + pad_x), min(img.height, y + h + pad_y),
                ))
                if crop.width >= 10 and crop.height >= 10:
                    crop_images.append((crop, cat_id))
            except Exception as e:
                logger.warning("Error cropping ann %d: %s", ann["id"], e)

    logger.info("Loaded %d training crops in %.1fs", len(crop_images), time.time() - t0)
    # Free image cache
    del img_cache

    # Embed everything in batches
    all_items = ref_images + crop_images
    logger.info("Embedding %d total images in batches of %d...", len(all_items), batch_size)
    t0 = time.time()

    cat_embeddings: dict[int, list[np.ndarray]] = {i: [] for i in range(nc)}
    for batch_start in range(0, len(all_items), batch_size):
        batch = all_items[batch_start:batch_start + batch_size]
        batch_imgs = [item[0] for item in batch]
        batch_cats = [item[1] for item in batch]
        embs = embed_batch(batch_imgs)
        for emb, cat_id in zip(embs, batch_cats):
            cat_embeddings[cat_id].append(emb)
        if (batch_start // batch_size + 1) % 20 == 0:
            elapsed = time.time() - t0
            done = batch_start + len(batch)
            rate = done / elapsed
            remaining = (len(all_items) - done) / rate
            logger.info("  Embedded %d/%d (%.0f img/s, ~%.0fs remaining)",
                        done, len(all_items), rate, remaining)

    elapsed = time.time() - t0
    logger.info("Embedded %d images in %.1fs (%.0f img/s)", len(all_items), elapsed, len(all_items) / elapsed)

    # Build gallery
    gallery_mean = np.zeros((nc, feat_dim), dtype=np.float32)
    gallery_counts = np.zeros(nc, dtype=np.int32)
    all_embs = []
    all_labels = []

    for cat_id in range(nc):
        embs = cat_embeddings[cat_id]
        if embs:
            stacked = np.stack(embs)
            mean = stacked.mean(axis=0)
            mean = mean / (np.linalg.norm(mean) + 1e-8)
            gallery_mean[cat_id] = mean
            gallery_counts[cat_id] = len(embs)
            all_embs.append(stacked)
            all_labels.extend([cat_id] * len(embs))
        else:
            logger.warning("No embeddings for cat %d (%s)", cat_id, catid_to_name.get(cat_id, "?"))

    if all_embs:
        all_embs_arr = np.vstack(all_embs).astype(np.float32)
        all_labels_arr = np.array(all_labels, dtype=np.int32)
    else:
        all_embs_arr = np.zeros((0, feat_dim), dtype=np.float32)
        all_labels_arr = np.zeros(0, dtype=np.int32)

    cats_with = int((gallery_counts > 0).sum())
    logger.info("Gallery: %d/%d categories covered, %d total embeddings", cats_with, nc, len(all_labels_arr))

    # Save
    output = Path(output_path)
    np.savez_compressed(
        output,
        gallery_mean=gallery_mean,
        gallery_counts=gallery_counts,
        all_embs=all_embs_arr,
        all_labels=all_labels_arr,
        feat_dim=feat_dim,
        nc=nc,
        model_name=model_name,
    )
    size_mb = output.stat().st_size / 1024 / 1024
    logger.info("Saved gallery to %s (%.1f MB)", output, size_mb)
    logger.info("  Mean gallery: %s", gallery_mean.shape)
    logger.info("  All embeddings: %s", all_embs_arr.shape)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="convnext_base.fb_in22k_ft_in1k")
    parser.add_argument("--output", default="gallery_convnext.npz")
    parser.add_argument("--max-ref", type=int, default=7)
    parser.add_argument("--max-crops", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    build_gallery(
        model_name=args.model,
        output_path=args.output,
        max_ref_per_class=args.max_ref,
        max_crops_per_class=args.max_crops,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )
