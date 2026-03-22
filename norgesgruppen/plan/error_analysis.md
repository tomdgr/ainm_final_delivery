# Error Analysis — What's Going Wrong

**Date**: 2026-03-20 ~12:35
**Model analyzed**: yolov8x_640_fulldata (our best single model, basis of 0.9091 ensemble)
**Evaluated on**: Full training set (248 images, 22,731 GT boxes)

---

## Summary

| Metric | Value | Impact |
|--------|-------|--------|
| Detection recall (IoU≥0.5) | **98.3%** | Near-perfect — NOT the problem |
| Missed detections (FN) | 384 / 22,731 (1.7%) | Minor |
| Classification accuracy (of matched) | **96.7%** | THE BOTTLENECK |
| Misclassifications | 730 boxes (3.3%) | Main source of error |
| False positives | 11,328 | Not penalized in mAP |

**Detection is solved. Classification of visually similar products is the bottleneck.**

---

## Misclassification Patterns

Almost all errors are between **visually near-identical products**:

### Pattern 1: Same brand, different size/weight (biggest source)
- NESCAFE GULL 100G → 200G (18x)
- HUSMAN KNEKKEBRØD 260G → 520G (8x)
- HAVRE KNEKKEBRØD ØKONOMI 600G → 300G (11x)
- YELLOW LABEL TEA 50POS → 25POS (16x)

### Pattern 2: Same brand, different variant
- KNEKKEBRØD URTER&HAVSALT → GODT FOR DEG (29x) — most common error
- CHEERIOS MULTI → HAVRE (6x)
- MAISKAKER POPCORN → OST (7x)
- SUPERGRØT SKOGSBÆR → KANEL&PUFFET QUINOA (8x)

### Pattern 3: Same product type, different preparation
- ALI ORIGINAL KOKMALT → FILTERMALT (6x)
- EVERGOOD CLASSIC KOKMALT → FILTERMALT (5x)

### Pattern 4: Similar packaging across sub-brands
- KNEKKEBRØD SOLSIKKE GL.FRI → misclassified 73% of the time
- SUPERGRANOLA GLUTENFRI → GRANOLA EPLE&KANEL (9x)

---

## Worst Classes (classification accuracy, min 5 GT)

| Class | Accuracy | Correct | Wrong | Missed | Total |
|-------|----------|---------|-------|--------|-------|
| KNEKKEBRØD SOLSIKKE GL.FRI 190G SIGDAL | 27% | 4 | 11 | 0 | 15 |
| KNEKKEBRØD GODT FOR DEG OST 190G SIGDAL | 33% | 1 | 2 | 2 | 5 |
| SJOKOLADEDRIKK 512G RETT I KOPPEN | 40% | 2 | 3 | 0 | 5 |
| SUPERGRANOLA GLUTENFRI 350G BARE BRA | 50% | 12 | 12 | 0 | 24 |
| YELLOW LABEL TEA 50POS LIPTON | 53% | 18 | 16 | 0 | 34 |
| NESCAFE GULL 100G | 57% | 33 | 25 | 0 | 58 |

---

## Implications for Strategy

1. **Ensemble WBF helps** — different models make different classification mistakes on these similar products, WBF voting corrects some
2. **TTA helps** — horizontal flip gives second chance at classification
3. **Higher cls loss weight** may help — our combo Azure jobs use cls=1.0 (2x default)
4. **Two-stage classifier** could help IF it can distinguish size/variant differences (but VinBigData winners found crop classifiers don't work well)
5. **More diverse models in ensemble** — maximum diversity = maximum classification correction
6. **Box expansion** unlikely to help — detection is already 98.3% recall
7. **The size/count differences (100G vs 200G, 50POS vs 25POS)** require reading text on packaging — very hard for detection models at 640px
