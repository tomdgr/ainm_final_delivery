# Task 2 Research: Shelf Product Recognition (Low Data)

## The Task
- Predict items on NorgesGruppen shelves
- Low training data
- 3 submissions/day against server validation
- Hidden test set at competition end
- This is fine-grained classification on real retail shelf images

---

## Relevant Competitions & Datasets

### Kaggle
| Name | Type | Link |
|------|------|------|
| **iMaterialist Product Recognition (FGVC6, CVPR 2019)** | Fine-grained product classification | https://www.kaggle.com/c/imaterialist-product-2019 |
| **Products-10K** | SKU-level product recognition (10K classes) | https://products-10k.github.io/ |
| **Retail Products Classification** | Product classification by image | https://www.kaggle.com/competitions/retail-products-classification |
| **Retail Product Checkout Dataset** | Self-checkout product detection | https://www.kaggle.com/datasets/diyer22/retail-product-checkout-dataset |
| **Grocery Store Dataset** | 81 classes, fruits/veg/packages | https://www.kaggle.com/datasets/validmodel/grocery-store-dataset |
| **Supermarket Shelves Dataset** | Shelf product detection | https://www.kaggle.com/datasets/humansintheloop/supermarket-shelves-dataset |
| **Visual Product Recognition (Products-10K)** | 10K classes product images | https://www.kaggle.com/datasets/warcoder/visual-product-recognition |

### CVPR / Academic
| Name | Year | Key Insight |
|------|------|-------------|
| **FGVC12 Workshop** | CVPR 2025 | Fine-grained visual categorization, few-shot challenges |
| **Retail Vision Workshop** | CVPR 2020-2024 | Amazon retail shelf + action recognition |
| **NTIRE 2025 Cross-Domain Few-Shot Object Detection** | CVPR 2025 | Few-shot detection across domains |
| **Visual RAG Pipeline** (Lamm & Keuper) | CVPR 2025 | RAG + VLM for few-shot product classification, 86.8% acc without retraining |

### GitHub Datasets
| Name | Link |
|------|------|
| **GroceryStoreDataset** (81 classes) | https://github.com/marcusklasson/GroceryStoreDataset |
| **ObjectDetectionGroceryProducts** (vending machines) | https://github.com/tobiagru/ObjectDetectionGroceryProducts |
| **grocerydataset** (shelf recognition, 2014) | https://github.com/gulvarol/grocerydataset |

---

## Winning Strategies for Low-Data Shelf Recognition

### 1. Strong Pretrained Backbones (CRITICAL)
- Use ImageNet-22K pretrained models (EVA-02, ConvNeXtV2, SigLIP)
- Freeze backbone, only train classification head initially
- Gradually unfreeze layers (discriminative fine-tuning)
- Lower LR for backbone (1e-5), higher for head (1e-3)

### 2. Heavy Data Augmentation
- **Must-have:** RandomResizedCrop, HorizontalFlip, ColorJitter, GaussianBlur
- **Advanced:** MixUp, CutMix, CutOut (CoarseDropout), RandomErasing
- **Shelf-specific:** Perspective transform, slight rotation (shelves aren't perfectly aligned)
- Use "heavy" augmentation policy in our scaffold

### 3. Few-Shot / Metric Learning
- **Siamese Networks:** One-shot learning with iconic product images
- **Prototypical Networks:** Compute class prototypes, nearest-centroid classification
- **Nearest Prototype Classifier (NPC):** Image embeddings + KNN
- **Visual RAG (CVPR 2025):** Use VLM + retrieval for zero/few-shot without retraining

### 4. Test-Time Augmentation (TTA)
- Horizontal flip + multi-scale + slight rotation
- Average predictions across augmentations
- Already built in our scaffold: `ensemble/tta.py`

### 5. Ensemble of Diverse Models
- Blend CNN (ConvNeXt) + ViT (EVA-02) + VL model (SigLIP)
- Different architectures make different errors
- Weighted voting or stacking

### 6. Knowledge Distillation
- Train large model (EVA-02-Large) on all data
- Distill to smaller model for faster inference
- Pseudo-labeling: use confident predictions to expand training set

### 7. External Data Augmentation
- Scrape product images from Norwegian grocery websites
- Oda.com (formerly Kolonial.no) has product catalogs
- NorgesGruppen's own Meny/Kiwi/SPAR websites may have product images
- Generate synthetic shelf images with known products

---

## Norwegian Product Image Sources

| Source | Notes |
|--------|-------|
| **Oda.com** | Online grocery, has product images in catalog |
| **Meny.no** | NorgesGruppen premium chain, product images in webshop |
| **Kolonial.no** → Oda | Previously had detailed product images |
| **Kasserolle.no** | Recipe site with ingredient images |
| **Matvaretabellen.no** | Norwegian food database (Mattilsynet) |

---

## Priority Actions

1. **Download and explore the competition training data** — understand class distribution, image quality, shelf layout
2. **Start with EVA-02-Small or ConvNeXt-Base** — best transfer learning for low data
3. **Use heavy augmentation + MixUp/CutMix** — maximize effective training set
4. **Build a simple KNN baseline** using frozen backbone features — often surprisingly strong with few shots
5. **Try Visual RAG approach** with a VLM if traditional classification underperforms
6. **Ensemble 3 models** for each of the 3 daily submissions

---

## Key Papers

- Lamm & Keuper, "A Visual RAG Pipeline for Few-Shot Fine-Grained Product Classification", CVPR 2025 FGVC12
- "Toward Retail Product Recognition on Grocery Shelves" (Varol & Salah, 2015)
- "Few-Shot Image Classification: Current Status and Research Trends" (MDPI Electronics, 2022)
- "Fine-grained Few-Shot Classification with Part Matching" (Black et al., CVPR 2025 FGVC12)
