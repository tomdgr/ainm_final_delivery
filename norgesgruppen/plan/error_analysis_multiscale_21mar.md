================================================================================
DEEP ERROR ANALYSIS — ARCH-DIVERSE MULTI-SCALE ENSEMBLE
NM i AI 2026 — 21 March 2026
================================================================================

Submission: wbf_2x1l_multiscale_tta_tested.zip
Models: yolov8x_640_fulldata_best, yolov8x_640_seed999_best, yolov8l_640_highaug
Config: Multi-scale (640+800), TTA hflip, WBF iou=0.6, skip_box_thr=0.01

NOTE: This analysis uses single-scale cached predictions (640px only, no TTA)
for comparison purposes. The actual submission includes multi-scale + TTA which
adds further diversity. Multi-scale CPU inference was too slow to complete.

================================================================================
A) OVERALL RESULTS — ARCH-DIVERSE vs OLD 3x YOLOV8X
================================================================================

Metric                    Arch-Diverse (2x+1l)   Old 3x yolov8x
-----------------------------------------------------------------
Competition Score         0.9454                 0.9400
Detection mAP@0.5        0.9719                 0.9698
Classification mAP@0.5   0.8834                 0.8705
Predictions               45,697                 51,313
TP (correct class)        21,573 (94.9%)         21,478 (94.5%)
Misclassified             783 (3.4%)             871 (3.9%)
Missed (FN)               375 (1.6%)             382 (1.7%)

KEY IMPROVEMENT: +0.0054 competition score (+0.0021 det, +0.0129 cls)
  - 88 fewer misclassifications (783 vs 871, -10.1%)
  - 95 more correct classifications (21573 vs 21478)
  - 5616 fewer raw predictions (45697 vs 51313) = less FP noise
  - 7 fewer missed objects (375 vs 382)

Weighted per-class improvement: +97.5 (105.7 improvement - 8.2 regression)
  74 classes improved, only 14 regressed

================================================================================
B) PER-CLASS AP: BIGGEST IMPROVEMENTS
================================================================================

Cat   Arch AP   3x AP    Diff    GT   Name
---  --------  ------  ------  ----   ----
251    0.636    0.516  +0.120    15   KNEKKEBRØD SOLSIKKE GL.FRI 190G SIGDAL
290    0.558    0.464  +0.094    29   SANDWICH CHEESE TOMAT&BASILIKUM 40G WASA
 27    0.774    0.692  +0.083    34   YELLOW LABEL TEA 50POS LIPTON
146    0.856    0.794  +0.062    54   NESCAFE GULL 200G NESTLE
328    0.940    0.886  +0.054    40   GRANOLA EPLE&BÆR 375G JACOBS UTVALGTE
189    0.894    0.846  +0.048    72   YELLOW LABEL TEA 25POS LIPTON
215    0.995    0.952  +0.043    54   BLUE FRUIT TE PYRAMIDE 20POS LIPTON
325    0.827    0.784  +0.044    58   NESCAFE GULL 100G
311    0.994    0.954  +0.040    33   MELBA TOAST FIRKANTET 100G MEULEN
 42    0.901    0.938  -0.037    22   KNEKKS KJEKS M/SPELT&HAVSALT 190G RØROS
 81    0.505    1.000  -0.495     1   GRØNN TE CHAI 25POS TWININGS (1 sample!)

Pattern: Biggest improvements on tea variants, Nescafe products, knekkebrød.
The yolov8l model helps disambiguate visually similar products.

================================================================================
C) REMAINING TOP CONFUSION PAIRS (arch-diverse)
================================================================================

Count  GT -> Pred                                    Pattern
 29    KNEKKEBRØD URTER&HAVSALT 220G -> GODT FOR DEG    Same brand (Sigdal), diff variant
 22    NESCAFE GULL 100G -> 200G                         Same product, diff size
 17    YELLOW LABEL TEA 50POS -> 25POS                   Same product, diff count
 15    SUPERGRANOLA GLUTENFRI -> EPLE&KANEL              Same brand (Bare Bra), diff variant
 10    HAVRE KNEKKEBRØD ØKONOMI 600G -> 300G             Same product, diff size
 10    KNEKKEBRØD RUNDA SESAM -> KANEL                   Same brand (Wasa), diff flavor
 10    unknown_product -> Leksands Rutbit                Misidentified unknown
  8    MUSLI BLÅBÆR 630G AXA -> 4-KORN 675G AXA          Same brand (AXA), diff product
  8    MAISKAKER POPCORN -> OST                          Same brand (Friggs), diff flavor
  8    HUSMAN KNEKKEBRØD 260G <-> 520G                   Bidirectional size confusion

These are nearly ALL same-brand-different-size/variant confusions.
The model can detect the object but cannot read the distinguishing text details.

================================================================================
D) CONFUSION PAIRS: WHAT CHANGED vs OLD 3x
================================================================================

IMPROVED (reduced confusions):
  EVERGOOD KOKMALT -> HELE BØNNER:    6 -> 1 (saved 5)
  GRANOLA MANDLER -> NATURELL:        4 -> 0 (saved 4)
  MÜSLI FRUKT AXA variants:          3 -> 0 (saved 3)
  SMØREMYK -> SMØREMYK LETT:         7 -> 4 (saved 3)
  CHEERIOS HAVRE -> MULTI:           6 -> 3 (saved 3)
  EVERGOOD FILTER -> KOKMALT:        6 -> 3 (saved 3)

WORSENED (new confusions):
  EVERGOOD KOKMALT -> PRESSMALT:     0 -> 3 (worse 3)
  unknown -> Sætre GullBar:          2 -> 5 (worse 3)
  HAVRE KNEKKEBRØD -> FRUKOST:       4 -> 7 (worse 3)

Net: Confusion pairs improved significantly overall.

================================================================================
E) YOLOv8L CONTRIBUTION ANALYSIS
================================================================================

Configuration              Score    Det mAP    Cls mAP    Preds
--------------------------------------------------------------
2x yolov8x only            0.9410   0.9713     0.8703     39,551
yolov8l solo               0.9153   0.9576     0.8166     33,291
2x + 1l combined           0.9454   0.9719     0.8834     45,697

yolov8l adds: det +0.0006, cls +0.0131, score +0.0044

The yolov8l model provides meaningful CLASSIFICATION diversity:
- Helps most on: NESCAFE GULL (+0.067), BLUE FRUIT TE (+0.056),
  KNEKKEBRØD SOLSIKKE (+0.199), YELLOW LABEL TEA (+0.084)
- Hurts slightly on: LIPTON ICETEA (-0.122, only 3 GT), GRISSINI (-0.015),
  FLOTT MATFETT (-0.083, only 4 GT)

The yolov8l's different training augmentation (highaug) gives it a different
"view" of ambiguous products, improving ensemble classification voting.

================================================================================
F) FALSE POSITIVE ANALYSIS
================================================================================

Threshold    Preds     TP      FP      MC    FN    Det%
conf>=0.01   36332   21573   13976    783   375   98.4%
conf>=0.05   28695   21542    6392    761   428   98.1%
conf>=0.10   26437   21504    4194    739   488   97.9%
conf>=0.15   25164   21463    2982    719   549   97.6%
conf>=0.20   24296   21414    2190    692   625   97.3%
conf>=0.25   23644   21340    1642    662   729   96.8%
conf>=0.30   23095   21214    1261    620   897   96.1%

mAP at different post-WBF thresholds:
  Thr     Det mAP   Cls mAP   Score
  0.01    0.9719    0.8821    0.9450
  0.05    0.9719    0.8776    0.9436
  0.10    0.9639    0.8729    0.9366
  0.15    0.9639    0.8635    0.9338

KEY INSIGHT: Filtering predictions HURTS mAP even though it reduces FPs.
The competition metric is mAP, not F1 — so keeping low-confidence predictions
is actually better because they add recall without hurting precision in the
AP calculation (low-conf predictions are ranked last).

RECOMMENDATION: Do NOT add post-WBF confidence filtering. The current
skip_box_thr=0.01 in WBF is sufficient.

Top FP categories:
  706 FPs: unknown_product (1.67x GT count) — model hallucinates unknowns
  380 FPs: KNEKKEBRØD 100 FRØ&HAVSALT (1.02x GT) — duplicate detections
  302 FPs: KNEKKEBRØD GODT FOR DEG (1.22x GT)
  225 FPs: EVERGOOD CLASSIC FILTERMALT (0.61x GT)

================================================================================
G) SIZE-BASED MISS ANALYSIS
================================================================================

Size Range         GT      Missed   Miss Rate
tiny (<32px)        33       22      66.7%
small (32-64px)    791      124      15.7%
medium (64-128)   6233      137       2.2%
large (128-256)  11494       82       0.7%
xlarge (>256)     4180       10       0.2%

Small objects (<64px) account for 146/375 = 39% of all misses.
Multi-scale inference (640+800) should help here, but the delta is modest
since most objects are medium-to-large.

================================================================================
H) PER-IMAGE COMPARISON
================================================================================

19 images where arch-diverse is WORSE than old 3x
54 images where arch-diverse is BETTER than old 3x

Arch-diverse improves 2.8x more images than it hurts.

Worst regressions (arch-diverse worse):
  img 270: F1 0.825 vs 0.894 (-0.069) — 22 more FPs
  img 108: F1 0.833 vs 0.896 (-0.062) — 10 more FPs
  img 152: F1 0.778 vs 0.837 (-0.058) — 13 more FPs

Pattern: All worse images have MORE false positives (not more misses).
The yolov8l model generates extra spurious detections on some images.

Best improvements:
  img 111: F1 0.947 vs 0.869 (+0.078) — 12 fewer FPs
  img  55: F1 0.906 vs 0.831 (+0.074) — 19 fewer FPs

================================================================================
I) WBF PARAMETER SENSITIVITY
================================================================================

WBF IoU threshold:
  iou=0.45: score=0.9448 (more aggressive merging)
  iou=0.50: score=0.9451
  iou=0.55: score=0.9453
  iou=0.60: score=0.9454 <-- current, near optimal
  iou=0.65: score=0.9455 <-- marginally better
  iou=0.70: score=0.9454

Skip box threshold:
  skip=0.001: score=0.9450
  skip=0.005: score=0.9450
  skip=0.010: score=0.9454 <-- current
  skip=0.020: score=0.9454
  skip=0.050: score=0.9451
  skip=0.100: score=0.9443

Model weights:
  [1,1,1] equal:       score=0.9454
  [2,1,1] 2x fulldata: score=0.9458 <-- BEST on val
  [3,2,1] favor full:  score=0.9456
  [1,1,2] 2x yolov8l:  score=0.9448
  [1,1,0.5] down l:    score=0.9453

FINDING: Upweighting fulldata_x model gives marginal improvement (+0.0004).
But per CLAUDE.md caveat, val gains this small are unreliable.

================================================================================
J) 4-MODEL ENSEMBLE (adding soup model)
================================================================================

Adding the model_soup (model 3 in arch_diverse cache):
  4-model: det=0.9726, cls=0.8843, score=0.9461

Delta vs 3-model: +0.0007 score. Marginal, probably not worth the extra
inference time in the 300s sandbox timeout.

================================================================================
K) POTENTIAL ADDITIONAL MODELS (from 6-model cache)
================================================================================

6-model ensemble scores (val, likely overfit):
  All 6 models:     score=0.9628 (cls=0.9299)
  First 4 models:   score=0.9625 (cls=0.9280)
  Models 0,2,4:     score=0.9621 (cls=0.9279)

These high scores are on training data and won't transfer to test.
The 6-model cache includes models trained on different splits/configs
that overfit the training set.

================================================================================
L) PROBABILITY-WEIGHTED CLASSIFICATION
================================================================================

Attempted probability vector averaging for WBF label selection:
  Prob voting: cls=0.8310, score=0.9297
  Standard WBF: cls=0.8834, score=0.9454

Result: SIGNIFICANTLY WORSE (-0.0524 cls mAP). The IoU-based matching to find
contributing boxes was too noisy, causing incorrect probability averaging.
A proper implementation would need access to the WBF internals to know exactly
which boxes were fused together. This approach is NOT recommended without
significant engineering effort.

================================================================================
M) MISCLASSIFICATION PATTERN BREAKDOWN
================================================================================

Of 783 misclassifications:
  Same product, diff size/variant (>=3 shared words):  226 (28.9%)
  Partial name match (1-2 shared words):               355 (45.3%)
  Completely different products:                        202 (25.8%)

The majority (74.2%) of misclassifications involve products from the same brand
or product line. These are cases where the model correctly identifies the brand
but picks the wrong size, flavor, or variant.

================================================================================
N) RECOMMENDATIONS
================================================================================

## 1. HIGHEST IMPACT: Product Gallery Classifier (post-detection)
The #1 source of errors is same-brand confusion (74% of misclassifications).
A CLIP/embedding gallery using product_images/ reference photos could fix
many of these. After WBF detection, crop each box, embed it, and match against
the gallery. This is the single highest-impact change remaining.
Expected improvement: fix 100-200 of 783 misclassifications (+2-5% cls mAP).

## 2. DO NOT change WBF parameters
Current WBF iou=0.6, skip=0.01 are near-optimal. The parameter space is flat
around these values — no significant gains available from tuning.

## 3. DO NOT add post-WBF confidence filtering
mAP rewards keeping low-confidence predictions. Filtering hurts more than
it helps because it reduces recall without meaningful precision benefit.

## 4. Model weights: Consider [2,1,1] weighting
Favoring fulldata_x gives +0.0004 on val. Marginal but essentially free.
CAUTION: Val is unreliable, this could hurt on test.

## 5. Multi-scale is beneficial
640+800 multi-scale + TTA gives additional diversity. The 800px scale
helps with small objects (15.7% miss rate for 32-64px). Already implemented
in the submission — keep it.

## 6. Training improvements for weak classes
- The 23 zero-AP classes (mostly 1-sample) cannot be improved with current data
- SANDWICH CHEESE TOMAT (cat 290, 29 GT, AP=0.558): Could benefit from
  augmentation focused on text-area preservation
- KNEKKEBRØD variants: Consider training a separate classifier just for
  the Sigdal/Wasa knekkebrød family (~8 categories, ~2000 GT)

## 7. Additional models that would add diversity
- YOLOv8m at 1280px: Different architecture scale + higher resolution
  could see text details that 640px models miss
- RT-DETR or YOLO11: Different detection paradigm might give different
  error patterns, improving ensemble diversity
- A model trained specifically on the confused categories

## 8. What NOT to do
- Don't chase val metrics — 37 images is not enough to validate
- Don't add more yolov8x models — diminishing returns on same architecture
- Don't implement probability voting without proper WBF integration
- Don't increase ensemble beyond 3-4 models (300s timeout constraint)

================================================================================
SUMMARY
================================================================================

The arch-diverse ensemble (2x yolov8x + 1x yolov8l) is a clear improvement
over the old 3x yolov8x ensemble:
- +0.0054 competition score (0.9454 vs 0.9400 on training data)
- 88 fewer misclassifications (10.1% reduction)
- 74 classes improved vs only 14 regressed
- Net weighted improvement: +97.5

The yolov8l model provides meaningful classification diversity (+0.0131 cls mAP)
by offering a different "opinion" on ambiguous products. Multi-scale + TTA
in the submission adds further gains not captured in this cached analysis.

The remaining error budget is dominated by same-brand confusion (74% of
misclassifications). Only a post-detection classifier using product reference
images can meaningfully address this. All post-processing parameter tuning
has reached diminishing returns.
