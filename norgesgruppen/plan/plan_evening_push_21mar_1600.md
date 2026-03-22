# Plan: Evening Push — Mar 21 16:00

**Current**: 4th place, 0.9210. Gap to 1st: 0.0045
**Submissions**: 2 today + ~5 tomorrow = 7 remaining
**Competition ends**: Mar 22 15:00 CET (~23 hours)

---

## What We Know

### The winning formula (0.9210):
- 2x yolov8x (fulldata + seed999) + 1x yolov8l — architecture diversity
- Multi-scale 640+800px — catches different object sizes
- TTA horizontal flip — free robustness
- WBF with absent_model_aware_avg, iou=0.6

### Marginal changes (≤0.001 delta, not worth pursuing):
- 960px scale, weighted WBF, noflip training, cls-tuning, fulldata seeds
- WBF param tuning, soft-NMS, temperature, neighbor voting
- Gallery re-ranking (33 configs tested, all negative)

### The fundamental problem:
74% of remaining errors are same-brand text confusion (100G vs 200G).
No model or post-processing we've tried can solve this.
The gap to 1st likely comes from something we haven't tried at all.

---

## Novel Things to Try (2 submissions today)

### Idea 1: Model Soup from Noflip Models
We have 4 noflip yolov8x models (s42, s77, s123, s999). Average their weights into ONE model.
Then use: soup + seed999 + yolov8l with multi-scale.

Why it might work:
- Soup finds flatter optima = better generalization (proven in literature)
- Noflip soup = 4 training runs averaged = smoother decision surface
- As one member of 3-model ensemble, it doesn't reduce diversity (the other 2 are different)

Why it might not:
- We tried soup before and it scored lower. But that was with models that overlapped with ensemble members. These noflip models are genuinely different.

### Idea 2: Different Aspect Ratio / Native Resolution Inference
All our inference uses square letterbox (640x640, 800x800). But shelf images are wide (typically 3:2 or wider).
What if we run at 640x480 or 800x640 (matching the actual aspect ratio)?
- Less padding waste
- Model sees more of the actual image content per pixel
- Could improve detection of products at image edges

### Idea 3: YOLO11x or RT-DETR (if Azure completes)
Genuinely different architecture. YOLO11x has C3k2 modules, RT-DETR is a transformer.
Different architectures make fundamentally different errors.

### Idea 4: Two-Model Split Strategy
Use 2 weight file slots for strong detection models, 1 slot for a specialized classifier.
E.g., 2x yolov8x for detection + 1x timm classifier model for post-detection re-ranking.
The classifier would be exported to ONNX and run on crops.

---

## Azure Training (overnight)
- Continue batch jobs (80 already submitted)
- Check YOLO11x and RT-DETR status
- If YOLO11x completes: download, export, build multi-scale submission

---

## Submission Strategy

### Today (2 remaining):
1. **Noflip soup ensemble** — novel, theoretically grounded
2. **Best remaining idea from research agent**

### Tomorrow (5 remaining):
1. YOLO11x/RT-DETR ensemble (if available)
2. Best overnight model combos
3. Final tuning based on signal
4-5. Reserve for last-hour iterations

### Final submission selection:
Choose the submission for hidden private test that is most ROBUST, not highest public score.
Our 0.9210 submission is already strong. Any new submission should be genuinely different, not a marginal tweak.
