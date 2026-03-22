# Task 2: NorgesGruppen Data — Official Details

## Confirmed Facts
- **Task:** Object Detection — "Detect and classify grocery products on store shelves"
- **Metric:** mAP@0.5
- **Response Limit:** 360 seconds (6 minutes per submission evaluation)
- **Submission:** Code upload (ZIP file) — NOT API-based
- **Docs:** Behind auth at app.ainm.no, accessible via MCP server or logged-in browser

## Key Implications

### It's OBJECT DETECTION, not classification!
- mAP@0.5 = mean Average Precision at IoU threshold 0.5
- Need bounding boxes + class labels
- This changes our approach — need detection models, not just classifiers

### Models to prioritize:
1. **YOLO (Ultralytics YOLOv8/v11)** — fastest, easiest to set up for detection
2. **FasterRCNN** (already in our scaffold)
3. **FCOS** (already in our scaffold)
4. **RT-DETR** — real-time detection transformer
5. **Co-DETR** — SOTA detection

### Code upload = need reproducible pipeline
- ZIP with all code
- Must run in their environment (likely Docker/GPU)
- 360 second time limit means inference must be fast
- Need: model weights + inference script

## TODO
- [ ] Get full docs (log into app.ainm.no and read NorgesGruppen task docs)
- [ ] Download training data
- [ ] Check class list and annotation format (COCO? YOLO?)
- [ ] Set up YOLOv8/v11 baseline
- [ ] Understand ZIP submission format requirements
