# Norwegian Grocery Product Image Sources

## Top Sources (Priority Order)

### 1. KASSALAPP (kassal.app) — BEST SOURCE
- **~100,000 Norwegian grocery products** with images
- Free API for hobby/hackathon use (60 req/min)
- Python client: https://github.com/bendikrb/kassalappy
- Search by product name or EAN barcode
- Images sourced from Norwegian retailers (Meny, Kiwi, Rema 1000)
- API docs: https://kassal.app/docs/api
- **Action: Register API key, bulk download product images with labels**

### 2. Open Food Facts — Norway Subset
- **~21,054 Norwegian products** with photos
- CC BY-SA license — completely free
- Bulk download via AWS S3: `openfoodfacts-images`
- Download guide: https://openfoodfacts.github.io/openfoodfacts-server/api/how-to-download-images/
- Hugging Face: https://huggingface.co/datasets/openfoodfacts/product-database
- **Action: Bulk download Norwegian subset**

### 3. SKU-110K — Shelf Detection
- **11,762 shelf images, 1.73M bounding box annotations**
- 110K+ SKU categories from US/Europe/Asia
- GitHub: https://github.com/eg4000/SKU110K_CVPR19
- Academic license
- **Action: Download for shelf detection pretraining**

### 4. GroceryStoreDataset — Swedish (similar market)
- **81 classes, 5,125 images** from Swedish grocery stores
- High product overlap with Norwegian market
- GitHub: https://github.com/marcusklasson/GroceryStoreDataset
- **Action: Already using this for Azure training jobs**

### 5. Tradesolution MediaStore — Highest Quality (Restricted)
- **~100K products**, professional 360-degree photography
- Industry standard for Norwegian grocery
- API: https://mediastore.tradesolution.no/api/swagger/index.html
- Requires commercial agreement
- Contact: salgmediastore@tradesolution.no
- **Action: Email asking for competition/academic access**

### 6. Roboflow Universe — Supplementary
- Multiple grocery datasets with annotations
- https://universe.roboflow.com/browse/retail
- Pre-annotated in YOLO/COCO format
- **Action: Browse for relevant datasets**

## Potential Combined Dataset: ~130,000+ labeled product images

| Source | Products | Images | License |
|--------|----------|--------|---------|
| Kassalapp | ~100K | ~100K | Free tier |
| Open Food Facts NO | ~21K | ~21K+ | CC BY-SA |
| SKU-110K | 110K SKUs | 11.7K shelf images | Academic |
| GroceryStoreDataset | 81 classes | 5.1K | Academic |
| Roboflow/Kaggle | Various | ~5K+ | Varies |

## Quick Start Scripts

### Kassalapp API
```python
# pip install kassalappy
from kassalappy import Kassalapp
client = Kassalapp(api_key="YOUR_KEY")
products = client.search("melk")
for p in products:
    print(p.name, p.image_url, p.ean)
```

### Open Food Facts Norway
```python
import requests
# Search Norwegian products
url = "https://world.openfoodfacts.org/cgi/search.pl"
params = {"countries_tags": "norway", "json": True, "page_size": 100}
r = requests.get(url, params=params)
products = r.json()["products"]
```
