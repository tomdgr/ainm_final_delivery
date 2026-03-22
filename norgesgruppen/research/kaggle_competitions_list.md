# Kaggle Object Detection Competitions — Full List

Download winning notebooks from these into `research/winning_code/`

## Most Relevant to Our Task (dense detection, many classes, low data)

| Competition | Year | Slug | Why Relevant |
|---|---|---|---|
| **Global Wheat Detection** | 2020 | `global-wheat-detection` | Dense, mAP@0.5-0.75, domain shift, ~3K images |
| **Great Barrier Reef** | 2021 | `tensorflow-great-barrier-reef` | Dense single-class, tiled training, WBF |
| **VinBigData Chest X-ray** | 2021 | `vinbigdata-chest-xray-abnormalities-detection` | Low data, 14 classes, mAP@0.4 |
| **Sartorius Cell Segmentation** | 2021 | `sartorius-cell-instance-segmentation` | 606 images only, dense, external pretraining |
| **SIIM-FISABIO COVID-19** | 2021 | `siim-covid19-detection` | Detection + classification, medical |
| **NFL Helmet Detection** | 2021 | `nfl-impact-detection` | Dense detection in video |
| **Airbus Ship Detection** | 2018 | `airbus-ship-detection` | Dense detection in satellite, 192K images |
| **RSNA Pneumonia** | 2018 | `rsna-pneumonia-detection-challenge` | Medical detection, mAP@IoU |
| **Severstal Steel Defect** | 2019 | `severstal-steel-defect-detection` | Defect detection + segmentation |
| **Open Images Detection** | 2019 | `open-images-2019-object-detection` | 500 classes, mAP, WBF invented here |

## All Detection Competitions (newest first)

| # | Competition | Slug | Year | Metric |
|---|---|---|---|---|
| 1 | BYU Flagellar Motors 2025 | `byu-locating-bacterial-flagellar-motors-2025` | 2025 | 3D localization |
| 2 | FathomNet 2025 | `fathomnet-2025` | 2025 | Hierarchical F1 |
| 3 | CZII CryoET Object ID | `czii-cryo-et-object-identification` | 2024 | 3D detection |
| 4 | RSNA 2024 Lumbar Spine | `rsna-2024-lumbar-spine-degenerative-classification` | 2024 | Weighted log loss |
| 5 | RSNA Mammography | `rsna-breast-cancer-detection` | 2023 | pF1 |
| 6 | HuBMAP Vasculature | `hubmap-hacking-the-human-vasculature` | 2023 | Dice |
| 7 | NFL Player Contact | `nfl-player-contact-detection` | 2023 | MCC |
| 8 | Great Barrier Reef | `tensorflow-great-barrier-reef` | 2021 | F2@IoU |
| 9 | Sartorius Cell | `sartorius-cell-instance-segmentation` | 2021 | mAP@0.5:0.95 |
| 10 | VinBigData X-ray | `vinbigdata-chest-xray-abnormalities-detection` | 2021 | mAP@0.4 |
| 11 | SIIM COVID-19 | `siim-covid19-detection` | 2021 | mAP |
| 12 | NFL Impact Detection | `nfl-impact-detection` | 2021 | F1@IoU |
| 13 | Global Wheat | `global-wheat-detection` | 2020 | mAP@0.5:0.75 |
| 14 | Open Images 2019 Det | `open-images-2019-object-detection` | 2019 | mAP |
| 15 | Open Images 2019 Seg | `open-images-2019-instance-segmentation` | 2019 | mAP mask |
| 16 | iMaterialist Fashion | `imaterialist-fashion-2019-FGVC6` | 2019 | mAP mask |
| 17 | Severstal Steel | `severstal-steel-defect-detection` | 2019 | Dice+class |
| 18 | Airbus Ship | `airbus-ship-detection` | 2018 | F2 |
| 19 | RSNA Pneumonia | `rsna-pneumonia-detection-challenge` | 2018 | mAP@IoU |
| 20 | Google Open Images 2018 | `google-ai-open-images-object-detection-track` | 2018 | mAP |

## Where to Find Winning Solutions

For any competition: `https://www.kaggle.com/competitions/{SLUG}/discussion`

Search for posts titled "1st place", "2nd place", "winning solution", "gold medal"

Most winners also have GitHub repos linked in their discussion posts.
