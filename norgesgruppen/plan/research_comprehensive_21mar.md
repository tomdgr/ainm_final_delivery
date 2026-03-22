# Comprehensive Research Sweep: Grocery Shelf Detection Competition Techniques

**Date**: 2026-03-21
**Competition**: NM i AI 2026 -- Task 2 (NorgesGruppen shelf product detection)
**Current standing**: 4th place, 0.9210 (score = 0.7*det_mAP@0.5 + 0.3*cls_mAP@0.5)
**Current setup**: 2x YOLOv8x + 1x YOLOv8l, multi-scale 640+800, TTA hflip, WBF
**Constraints**: 3 ONNX weight files max, 420MB, 300s on NVIDIA L4, no network

---

## TABLE OF CONTENTS
1. [High-Priority Actionable Techniques](#1-high-priority-actionable)
2. [Kaggle Competition Insights](#2-kaggle-competitions)
3. [Academic Paper Findings](#3-academic-papers)
4. [Blog/Medium Insights](#4-blog-insights)
5. [Training Optimization](#5-training-optimization)
6. [Inference/Post-Processing Optimization](#6-inference-optimization)
7. [Augmentation Strategies](#7-augmentation)
8. [Model Architecture Alternatives](#8-model-alternatives)
9. [Anti-Shakeup / Generalization Strategies](#9-generalization)
10. [Implementation Priority Matrix](#10-priority-matrix)

---

## 1. HIGH-PRIORITY ACTIONABLE TECHNIQUES <a name="1-high-priority-actionable"></a>

### 1.1 WBF Parameter Tuning (HIGH impact, EASY to implement)
- **Default**: iou_thr=0.5, skip_box_thr=0.0001
- **Optimal found in literature**: iou_thr=0.43, skip_box_thr=0.21
- **Action**: Grid search iou_thr in [0.3, 0.4, 0.43, 0.5, 0.55, 0.6] and skip_box_thr in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25] on validation set
- **Helps generalization**: Moderate -- tuned on val, but robust settings transfer
- **Implementation**: Just change parameters in WBF call, run eval loop
- **Expected impact**: +0.5-2% mAP improvement

### 1.2 Model Soup / Weight Averaging (HIGH impact, MODERATE effort)
- **Technique**: Average weights of multiple fine-tuned checkpoints from same architecture
- **Greedy soup**: Sequentially add checkpoints, keep only if val improves
- **Key insight**: NO increase in inference time or model count -- still 1 model
- **Literature result**: Improves both in-distribution and out-of-distribution performance
- **Action**: Train 3-5 YOLOv8x runs with different seeds/hyperparameters, average best checkpoints
- **Helps generalization**: YES -- finds wider optima, more robust to distribution shift
- **Implementation**: Average .pt state_dicts, export to ONNX
- **Expected impact**: +1-3% mAP, especially on private test

### 1.3 Multi-Seed Training Diversity (HIGH impact, MODERATE effort)
- **Technique**: Train same architecture with different random seeds
- **Key finding**: Ensembles of models from different seeds significantly improve certified robustness
- **Your setup already partially does this: But maximize seed diversity across your 3 models
- **Action**: Ensure all 3 ONNX models use maximally different seeds, hyperparameters, and potentially different training splits
- **Helps generalization**: YES -- reduces variance, improves robustness
- **Expected impact**: +0.5-1.5% with better seed selection

### 1.4 Confidence Score Calibration (MEDIUM impact, EASY to implement)
- **Technique**: Post-process raw confidence scores using Platt scaling or isotonic regression
- **Key insight**: Calibrated scores improve WBF fusion quality
- **NorCal method**: Reweight predicted scores by training sample size per class -- improves rare and common classes
- **Action**: Fit calibration on validation set, apply during inference before WBF
- **Helps generalization**: Moderate -- better-calibrated ensembles are more robust
- **Expected impact**: +0.3-1% mAP

### 1.5 Progressive Resolution / Multi-Resolution Training (MEDIUM impact, MODERATE effort)
- **Technique**: Train at 640, then fine-tune at 800 or 1280
- **Key finding**: Starting at lower resolution learns coarse features faster, higher resolution fine-tunes details
- **Literature**: +1.8 mAP improvement vs standard ImageNet pretraining on COCO
- **Action**: Train YOLOv8x at 640 for 100 epochs, then fine-tune at 800 for 50 epochs
- **Helps generalization**: YES -- exposes model to multiple scales during training
- **Expected impact**: +0.5-2% mAP

---

## 2. KAGGLE COMPETITION INSIGHTS <a name="2-kaggle-competitions"></a>

### 2.1 Global Wheat Detection (previously researched -- key additions)
- **Winners used**: EfficientDet, Faster-RCNN, YOLOv3 -- NO domain adaptation modules
- **Critical lesson**: Generalization came from model diversity, NOT domain-specific tricks
- **1st place (dungnb1333)**: Available on GitHub -- used ensemble of standard architectures
- **Shakeup**: Significant -- models evaluated on unseen domains (Japan, China, Australia vs training on France, UK, Switzerland, Canada)
- **Relevance**: Very similar to our setup -- small labeled dataset, unseen test distribution

### 2.2 Great Barrier Reef Starfish Detection
- **Setup**: Dense object detection in underwater video frames
- **Top solutions**: YOLOv5 variants with tracking, YOLOX, Swin Transformer
- **Key technique**: Tracking across frames improved temporal consistency
- **Relevance**: Dense detection similar to shelf images; WBF was widely used

### 2.3 VinBigData Chest X-ray Detection (previously researched -- key additions)
- **2nd place (ZFTurbo)**: Published solution on GitHub
- **Two-stage approach**: Classifier to determine if image has abnormalities, then detector
- **Threshold**: 60-70% for normal/abnormal classification
- **Blending**: Results of two-stage and one-stage blended via NMS
- **Relevance**: Two-stage concept applicable -- classify shelf region first, then detect products

### 2.4 SKU-110K Dense Product Detection
- **Dataset**: 110K images, 1M objects -- densely packed retail shelves
- **Key challenge**: Identical products side-by-side create overlapping detections
- **Original solution**: Soft-IoU + Mixture of Gaussians + EM clustering
- **YOLOv5x result**: mAP50 ~0.6 at 30 epochs (showing difficulty of dense scenes)
- **DeIoU method**: Novel loss function for dense scenes -- suppress unnecessary overlap
- **Relevance**: DIRECTLY applicable -- same domain (retail shelves, dense products)

### 2.5 FathomNet 2025 (Fine-Grained Classification)
- **Winner (Yonsei+SSL)**:
  - Multi-scale image inputs (ROI + environmental context via ViT)
  - Multi-Context Environmental Attention Module (MCEAM)
  - Hierarchical auxiliary classifier using taxonomic relationships
- **Key insight**: Context around objects improved fine-grained classification significantly
- **Relevance**: Medium -- our 356 classes include similar-looking products, context from shelf position could help

### 2.6 Retail Products Classification (Kaggle)
- **Competition**: Image classification of retail products
- **Relevance**: Classification component of our score (30% weight)

### 2.7 Products-10K Dataset Challenge
- **Scale**: 10,000 product categories, largest product recognition dataset
- **Key challenge**: Fine-grained differences between similar products
- **Relevance**: Techniques for distinguishing similar products applicable to our 356 classes

---

## 3. ACADEMIC PAPER FINDINGS <a name="3-academic-papers"></a>

### 3.1 NTIRE 2025: Cross-Domain Few-Shot Object Detection (CVPR 2025W)
- **Setup**: Detect objects in new domains with few labeled examples
- **Top methods**: Feature alignment, meta-learning, domain-invariant representations
- **Relevance**: HIGH -- our test set may have different lighting/angles/products than training

### 3.2 Copy-Paste Augmentation (CVPR 2021, Ghiasi et al.)
- **Result**: +10 Box AP in low-data regime, 2x data efficiency
- **Background-aware variant**: +1.1% mAP vs random copy-paste
- **How it works**: Cut object instances from one image, paste onto another randomly
- **Relevance**: HIGH for 248-image dataset -- effectively multiplies training data
- **Implementation**: Requires instance masks (may need to create from bounding boxes)
- **Expected impact**: HIGH if instance masks available

### 3.3 Soft-NMS (ICCV 2017, Bodla et al.)
- **Result**: +1.7% mAP on PASCAL VOC, +1.1-1.3% on COCO
- **How it works**: Instead of hard suppression, decay confidence of overlapping boxes
- **Stable performance**: IoU threshold 0.4-0.7 all give ~1% improvement
- **Softer-NMS extension**: Adds variance-weighted voting for box coordinates
- **Relevance**: HIGH for dense shelf images with overlapping products
- **Implementation**: One line of code change
- **Expected impact**: +1-2% on dense scenes

### 3.4 Semi-Supervised Object Detection (SSOD)
- **FixMatch approach**: Consistency between weak and strong augmentations on unlabeled data
- **Result**: Up to +20.4 percentage points with pseudo-labeled images
- **Key challenge**: Initial model quality determines pseudo-label quality
- **Action**: Use current best model to pseudo-label test images, retrain
- **Relevance**: MEDIUM -- risky with small dataset, but potential high reward
- **Expected impact**: +2-5% if done carefully, but risk of degradation

### 3.5 Source-Free Domain Adaptation (ECCV 2024)
- **Key method**: Mean-Teacher self-training with high-confidence pseudo labels
- **Innovation**: Low-confidence pseudo label distillation for small/rare objects
- **TEPLS**: Temporal Ensemble Pseudo-Label Selection for cross-domain robustness
- **Relevance**: MEDIUM -- applicable if test domain differs from training domain

### 3.6 Shelf Management System (Expert Systems with Applications, 2024)
- **Deployed**: 7,000+ 7-Eleven stores in Taiwan
- **Method**: Multi-image stitching + virtual shelves + deep learning detection
- **Accuracy**: 99% planogram adherence detection
- **Relevance**: LOW for competition, but validates our detection approach

### 3.7 Model Soups (ICML 2022, Wortsman et al.)
- **Uniform soup**: Average all fine-tuned models -- simple, effective
- **Greedy soup**: Sequentially add models only if val improves -- better performance
- **Key finding**: Improves both in-distribution AND out-of-distribution
- **No inference cost**: Same model size, same speed
- **Relevance**: VERY HIGH -- free performance gain, especially for generalization

### 3.8 EMA (Exponential Moving Average) Optimization
- **Finding**: EMA models generalize better, more robust to noisy labels, better calibration
- **Already in YOLO**: YOLOv8 uses EMA during training
- **Action**: Ensure EMA is enabled (default), consider tuning decay from default 0.9999
- **Expected impact**: Already included in baseline, but verify settings

---

## 4. BLOG/MEDIUM INSIGHTS <a name="4-blog-insights"></a>

### 4.1 RF-DETR (Roboflow, 2025)
- **Performance**: First real-time detector to exceed 60 mAP on COCO
- **RF-DETR-S**: 53.0 AP (vs YOLOv11-X 51.2 AP) at FASTER inference
- **Key advantage**: DINOv2 backbone generalizes better with fewer training images
- **NMS-free**: Eliminates NMS post-processing entirely
- **Relevance**: HIGH as a potential replacement model, BUT:
  - Must verify ONNX export compatibility
  - Must verify it fits in 420MB with other models
  - Unknown compatibility with ultralytics submission format
- **Risk**: Untested in our pipeline; may not integrate easily in 300s budget

### 4.2 YOLOv12 and YOLO11 Improvements
- **YOLO11**: 22% fewer parameters than YOLOv8m at higher mAP
- **YOLO11m**: 50.3% mAP on COCO -- strong accuracy/efficiency balance
- **YOLOv12**: Attention-based design with Area Attention, FlashAttention
- **Action consideration**: Test YOLO11x or YOLOv12x as replacement for one model
- **Risk**: May not generalize better on our specific small dataset despite better COCO numbers

### 4.3 Competition Winning Patterns (Multiple Sources)
- **Diversity over specialization**: Winning ensembles use different architectures, not just different seeds
- **Cross-validation discipline**: Exact same folds for all models
- **Hill-climbing ensemble selection**: Greedy selection of models that maximize val metric
- **300+ model ensembles**: Not applicable to our 3-model constraint, but principle of diversity matters

---

## 5. TRAINING OPTIMIZATION <a name="5-training-optimization"></a>

### 5.1 close_mosaic Parameter
- **Default**: close_mosaic=10 (disable mosaic last 10 epochs)
- **Rationale**: Mosaic creates unrealistic training images; disabling late lets model converge on natural distribution
- **YOLOX finding**: Turn off strong augmentation in last 15 epochs
- **Action**: Try close_mosaic=15 or close_mosaic=20 for longer convergence on clean data
- **Risk**: If turned off too early, loses augmentation benefit; too late, already overfit
- **Expected impact**: +0.3-1% mAP

### 5.2 Label Smoothing
- **Technique**: Replace hard labels [0,1] with soft labels [0.05, 0.95]
- **Already in YOLOv4+**: Part of "Bag of Freebies"
- **Action**: Set label_smoothing=0.05 or 0.1 in training config
- **Helps generalization**: YES -- prevents overconfident predictions
- **Expected impact**: +0.2-0.5% mAP

### 5.3 Frozen Backbone Strategy
- **Finding**: Freezing first 4 blocks reduces GPU usage 28% while maintaining accuracy
- **For small datasets**: Prevents overfitting the pretrained backbone features
- **Literature**: Freeze backbone for first N epochs, then unfreeze for fine-tuning
- **Action**: Try freeze=10 (first 10 layers) for initial training, then unfreeze
- **Expected impact**: +0.3-1% on small datasets

### 5.4 Cosine Annealing with Warm Restarts
- **Benefit**: Escapes local minima by periodically resetting learning rate
- **Configuration**: T_0 (initial period), T_mult (period multiplier)
- **For small datasets**: Short T_0 works better
- **Already in YOLO**: Default uses cosine schedule, but warm restarts are separate
- **Action**: Experiment with warm restarts if retraining

### 5.5 K-Fold Cross-Validation for Model Selection
- **Standard**: 5-fold CV with stratified splits
- **Key for small datasets**: Maximizes data utilization
- **Competition approach**: Train on all folds, ensemble fold-specific models
- **Action**: If retraining, use 5-fold to get 5 models, then select best 3 or soup them
- **Expected impact**: +1-3% through better data utilization

---

## 6. INFERENCE/POST-PROCESSING OPTIMIZATION <a name="6-inference-optimization"></a>

### 6.1 SAHI (Slicing Aided Hyper Inference)
- **Result**: +6.8% AP on FCOS, +5.1% on VFNet, +5.3% on TOOD
- **With slicing-aided fine-tuning**: Cumulative +12.7-14.5% AP
- **Optimal overlap**: 6-9% overlap between tiles
- **Implementation**: `from sahi.predict import get_sliced_prediction`
- **Concern for our setup**: Increases inference time significantly (multiple passes per image)
- **Action**: Test with 2x2 tiles at 640 (effective 1280 resolution) -- check if fits in 300s
- **Expected impact**: HIGH for small products, but time budget is the limiting factor

### 6.2 Soft-NMS in Post-Processing
- **Action**: Replace standard NMS with Soft-NMS in inference pipeline
- **Gaussian decay**: sigma=0.5 is a good default
- **Linear decay**: Also effective, simpler
- **Expected impact**: +1-2% on dense shelf scenes

### 6.3 Multi-Scale TTA Optimization
- **Current**: 640 + 800 + hflip
- **Research finding**: Optimal TTA scales should be centered around training resolution
- **Consideration**: Adding 1024 scale (if time permits) for catching small products
- **Key insight**: More TTA scales = more time; optimize for best 2-3 within 300s budget
- **Action**: Profile exact inference time, maximize TTA within budget

### 6.4 Confidence Threshold Optimization
- **Current**: Likely using default thresholds
- **Action**: Per-class confidence threshold optimization on validation set
- **For rare classes**: Lower threshold to increase recall
- **For common classes**: Standard threshold
- **Expected impact**: +0.5-1% mAP

### 6.5 Class-Agnostic vs Class-Specific NMS
- **Class-agnostic**: Suppresses across all classes -- better for dense overlapping different products
- **Class-specific**: Suppresses within each class -- standard approach
- **For retail shelf**: Products of different classes are often adjacent/overlapping
- **Action**: Test class-agnostic NMS -- may reduce duplicate detections of similar products
- **Expected impact**: +0.3-1% on dense scenes

---

## 7. AUGMENTATION STRATEGIES <a name="7-augmentation"></a>

### 7.1 Already Applied (verify settings)
- Mosaic augmentation (default in YOLOv8)
- Horizontal flip (keep; vertical flip disabled -- correct for shelf images)
- Scale augmentation
- Color jitter (hsv_h, hsv_s, hsv_v)

### 7.2 Consider Adding
- **CutMix**: cutmix=0.5 -- forces learning from partial features, ~0.5% improvement
- **MixUp**: mixup=0.15 -- gentle blending, regularization effect
- **Copy-Paste**: If instance masks available -- significant gains on small datasets
- **Perspective transforms**: perspective=0.001 -- subtle viewing angle changes
- **Erasing/cutout**: erasing=0.3 -- occlusion robustness

### 7.3 Augmentations to AVOID
- **Vertical flip (flipud)**: Already disabled -- correct for grocery shelves
- **Aggressive rotation**: Products have fixed orientation on shelves
- **Heavy color distortion**: Could confuse product labels/branding

### 7.4 Mosaic Settings
- **mosaic=1.0**: Default, all images get mosaic
- **close_mosaic=10-20**: Disable late for clean convergence
- **Consideration**: mosaic=0.8 to have 20% of images train without mosaic throughout

---

## 8. MODEL ARCHITECTURE ALTERNATIVES <a name="8-model-alternatives"></a>

### 8.1 Current: YOLOv8x + YOLOv8l
- Proven to work, established pipeline
- YOLOv8x generalizes better than YOLOv8l (validated by leaderboard)

### 8.2 RF-DETR (Potential game-changer)
- **Pro**: 60.5% mAP on COCO, DINOv2 backbone, NMS-free, better with few images
- **Con**: Unproven in our pipeline, ONNX export uncertainty, time budget unknown
- **Risk assessment**: HIGH reward, MEDIUM risk
- **Action**: Only try if time permits and ONNX export is verified

### 8.3 YOLO11x
- **Pro**: Better COCO accuracy than YOLOv8x, 22% fewer params
- **Con**: Marginal improvement on COCO may not transfer to our domain
- **Action**: Quick test -- train YOLO11x with same settings as YOLOv8x baseline
- **Expected impact**: +0-1% (uncertain)

### 8.4 Co-DETR / DINO
- **Pro**: Strong on dense detection, transformer attention
- **Con**: Slow inference, complex pipeline, likely won't fit 300s budget
- **Action**: Skip for this competition

### 8.5 Mixed Architecture Ensemble
- **Principle**: Maximum diversity in ensemble improves robustness
- **Option A**: 2x YOLOv8x (different seeds) + 1x YOLO11x
- **Option B**: 2x YOLOv8x (different seeds/scales) + 1x RF-DETR
- **Option C**: Keep current 2x YOLOv8x + 1x YOLOv8l (proven)
- **Recommendation**: Stick with proven setup unless quick test shows clear improvement

---

## 9. ANTI-SHAKEUP / GENERALIZATION STRATEGIES <a name="9-generalization"></a>

### 9.1 Key Lesson from Global Wheat Detection
- Winners used STANDARD architectures, no domain adaptation
- Generalization came from: model diversity, robust augmentation, proper validation
- Our val set is too small (37 images) to be reliable

### 9.2 Strategies for Private Test Robustness
1. **Model soup**: Average multiple checkpoints -- finds wider, more robust optima
2. **Multi-seed ensemble**: Different random initializations capture different features
3. **Conservative augmentation**: Match real-world conditions, don't over-augment
4. **Label smoothing**: Prevent overconfident predictions
5. **Weight decay**: Proper regularization (0.0005 default is usually good)
6. **EMA**: Already default in YOLOv8, ensure enabled
7. **Don't over-optimize WBF on val**: Use reasonable defaults rather than val-overfit params

### 9.3 Val Score vs Test Score Relationship
- **Our experience**: YOLOv8x with LOWER val mAP scored HIGHER on test (0.7802 vs 0.7700)
- **Implication**: Val is not reliable -- favor techniques known to improve generalization
- **Strategy**: Prefer model soup / multi-seed over val-score chasing
- **Risk mitigation**: Submit diverse solutions across remaining submissions

---

## 10. IMPLEMENTATION PRIORITY MATRIX <a name="10-priority-matrix"></a>

### TIER 1: Do Now (High impact, Low effort, fits 300s budget)
| Technique | Expected Gain | Time to Implement | Risk |
|-----------|--------------|-------------------|------|
| WBF iou_thr grid search | +0.5-2% | 30 min | Low |
| Soft-NMS in post-processing | +1-2% | 15 min | Low |
| Confidence threshold per class | +0.5-1% | 1 hour | Low |
| skip_box_thr tuning | +0.3-1% | 15 min | Low |

### TIER 2: Do If Time (High impact, Moderate effort)
| Technique | Expected Gain | Time to Implement | Risk |
|-----------|--------------|-------------------|------|
| Model soup (weight averaging) | +1-3% | 2-3 hours training | Low |
| Multi-seed retraining | +0.5-1.5% | 3-4 hours training | Low |
| Label smoothing retrain | +0.2-0.5% | 3-4 hours training | Low |
| close_mosaic=15-20 retrain | +0.3-1% | 3-4 hours training | Low |

### TIER 3: High Potential but Risky
| Technique | Expected Gain | Time to Implement | Risk |
|-----------|--------------|-------------------|------|
| SAHI tiled inference | +3-5% | 2-3 hours | HIGH (time budget) |
| RF-DETR replacement model | +1-3% | 4-6 hours | HIGH (untested) |
| Pseudo-labeling test data | +2-5% | 4-6 hours | MEDIUM (noise) |
| Copy-paste augmentation | +2-4% | 3-4 hours | MEDIUM (needs masks) |

### TIER 4: Skip for This Competition
| Technique | Reason to Skip |
|-----------|---------------|
| Knowledge distillation | Not applicable -- we want maximum accuracy, not compression |
| Co-DETR/DINO | Too slow for 300s budget |
| Domain adaptation modules | Winners don't use them (GWD lesson) |
| 70+ model ensembles | Limited to 3 ONNX models |

---

## IMMEDIATE ACTION PLAN (Next 2-3 hours)

1. **WBF parameter grid search** on validation set -- try all combinations of iou_thr and skip_box_thr
2. **Soft-NMS** -- implement as alternative to standard NMS in post-processing
3. **Per-class confidence threshold optimization** on validation set
4. **Model soup** -- if we have multiple checkpoint files from different training runs, average their weights
5. **Profile inference time** precisely to know exactly how much TTA headroom remains within 300s

---

## SOURCES

### Kaggle Competitions
- [Global Wheat Detection](https://www.kaggle.com/competitions/global-wheat-detection) - 1st place: github.com/dungnb1333/global-wheat-dection-2020
- [SKU-110K Dense Detection](https://github.com/eg4000/SKU110K_CVPR19)
- [Great Barrier Reef Detection](https://www.kaggle.com/competitions/tensorflow-great-barrier-reef)
- [VinBigData Chest X-ray](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection) - 2nd place: github.com/ZFTurbo/2nd-place-solution-for-VinBigData
- [FathomNet 2025](https://www.fathomnet.org/news/looking-at-the-bigger-picture:-results-from-fathomnet%E2%80%99s-2025-kaggle-competition)
- [Products-10K Dataset](https://products-10k.github.io/)
- [Retail Products Classification](https://www.kaggle.com/competitions/retail-products-classification)
- [Kaggle Solutions Database](https://farid.one/kaggle-solutions/)
- [Kaggle Image Competition Solutions](https://github.com/jayinai/kaggle-image)

### Academic Papers
- [Copy-Paste Augmentation (CVPR 2021)](https://arxiv.org/abs/2012.07177)
- [Soft-NMS (ICCV 2017)](https://arxiv.org/abs/1704.04503)
- [Model Soups (ICML 2022)](https://arxiv.org/pdf/2203.05482)
- [WBF Paper](https://arxiv.org/abs/1910.13302)
- [SAHI Paper](https://arxiv.org/abs/2202.06934)
- [EMA in Deep Learning](https://arxiv.org/abs/2411.18704)
- [NTIRE 2025 Cross-Domain Few-Shot Detection](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/)
- [SKU-110K Dense Detection](https://www.researchgate.net/publication/338512014_Precise_Detection_in_Densely_Packed_Scenes)
- [Semi-Supervised Object Detection Survey](https://pmc.ncbi.nlm.nih.gov/articles/PMC12788260/)
- [Source-Free Domain Adaptation (ECCV 2024)](https://dl.acm.org/doi/10.1007/978-3-031-72907-2_20)
- [Layer-Freezing YOLO Transfer Learning](https://www.mdpi.com/2227-7390/13/15/2539)
- [Shelf Management Deep Learning (2024)](https://www.sciencedirect.com/science/article/pii/S0957417424015021)
- [Confidence Calibration for Detection](https://arxiv.org/pdf/2202.12785)
- [Small Object Detection Survey 2025](https://arxiv.org/pdf/2503.20516)

### Blogs & Tutorials
- [RF-DETR vs YOLO (Roboflow 2025)](https://blog.roboflow.com/best-object-detection-models/)
- [WBF Detailed View (Analytics Vidhya)](https://medium.com/analytics-vidhya/weighted-boxes-fusion-86fad2c6be16)
- [WBF Optimization (Towards Data Science)](https://towardsdatascience.com/wbf-optimizing-object-detection-fusing-filtering-predicted-boxes-7dc5c02ca6d3/)
- [Image Classification Tips from 13 Kaggle Competitions (Neptune.ai)](https://neptune.ai/blog/image-classification-tips-and-tricks-from-13-kaggle-competitions)
- [YOLO Data Augmentation (Ultralytics)](https://docs.ultralytics.com/guides/yolo-data-augmentation/)
- [SAHI Tutorial (Ultralytics)](https://docs.ultralytics.com/guides/sahi-tiled-inference/)
- [Hyperparameter Tuning (Ultralytics)](https://docs.ultralytics.com/guides/hyperparameter-tuning/)
- [Transfer Learning Frozen Layers (Ultralytics)](https://docs.ultralytics.com/yolov5/tutorials/transfer_learning_with_frozen_layers/)
- [K-Fold Cross-Validation YOLO (Ultralytics)](https://docs.ultralytics.com/guides/kfold-cross-validation/)
- [Progressive Resizing (MosaicML)](https://docs.mosaicml.com/projects/composer/en/latest/method_cards/progressive_resizing.html)
- [YOLOv8 Best Practices](https://medium.com/internet-of-technology/yolov8-best-practices-for-training-cdb6eacf7e4f)
- [Mosaic Augmentation Explained](https://www.analyticsvidhya.com/blog/2023/12/mosaic-data-augmentation/)
- [MMYOLO Training Tricks](https://mmyolo.readthedocs.io/en/latest/recommended_topics/training_testing_tricks.html)

### Other Competitions
- [NM i AI 2026 Docs](https://app.ainm.no/docs)
- [NM i AI 2025 (NORA)](https://www.nora.ai/competitions/ai-championship-2025/ai-championship.html)
- [State of ML Competitions 2024](https://mlcontests.com/state-of-machine-learning-competitions-2024/)
- [WBF GitHub Repository](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
- [Dense Object Detection on SKU-110K](https://github.com/suryanshgupta9933/Dense-Object-Detection)
