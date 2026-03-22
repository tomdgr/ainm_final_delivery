"""Generate team knowledge doc — what we're doing, why, and how it works."""
from fpdf import FPDF
from datetime import datetime


class Doc(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150)
            self.cell(0, 8, "NorgesGruppen Detection - Technical Playbook", align="C", new_x="LMARGIN", new_y="NEXT")
            self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128)
        self.cell(0, 10, f"{self.page_no()}", align="C")

    def h1(self, text):
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(30, 30, 30)
        self.ln(6)
        self.cell(0, 12, text, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(40, 40, 40)
        self.line(10, self.get_y() + 1, 200, self.get_y() + 1)
        self.ln(6)

    def h2(self, text):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(50, 50, 50)
        self.ln(4)
        self.cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def h3(self, text):
        self.set_font("Helvetica", "BI", 11)
        self.set_text_color(70, 70, 70)
        self.ln(2)
        self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def p(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.set_x(10)
        self.multi_cell(190, 5.5, text)
        self.ln(2)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.set_x(10)
        self.multi_cell(190, 5.5, f"  - {text}")

    def code(self, text):
        self.set_font("Courier", "", 8.5)
        self.set_fill_color(242, 242, 242)
        self.set_text_color(40, 40, 40)
        self.ln(1)
        for line in text.strip().split("\n"):
            self.set_x(12)
            self.cell(186, 4.5, f" {line}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def key_value(self, key, value):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(40, 40, 40)
        self.set_x(10)
        kw = self.get_string_width(key + ": ") + 2
        self.cell(kw, 5.5, f"{key}: ")
        self.set_font("Helvetica", "", 10)
        self.multi_cell(190 - kw, 5.5, value)


def generate():
    pdf = Doc()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title page
    pdf.ln(30)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 15, "Technical Playbook", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(100)
    pdf.cell(0, 10, "NorgesGruppen Shelf Detection", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "NM i AI 2026 - Team Experis", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 6, f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60)
    pdf.set_x(30)
    pdf.multi_cell(150, 5.5,
        "This document explains what techniques we're using, why we chose them, "
        "and the intuition behind each one. It's meant to get everyone on the team "
        "up to speed on the technical approach.",
        align="C")

    # =====================================================
    pdf.add_page()
    pdf.h1("The Problem")
    pdf.p(
        "We get photos of Norwegian grocery store shelves. Each photo has ~92 products on it. "
        "We need to draw a box around every product and say what it is (356 possible products). "
        "Our code runs on their server with a GPU, 300 seconds, no internet."
    )
    pdf.h2("How We Get Points")
    pdf.p(
        "The score has two parts, weighted differently because finding products is harder than naming them:"
    )
    pdf.key_value("70% Detection mAP", "Did you find the product? If your predicted box overlaps >=50% with the real box, that's a hit. Category doesn't matter.")
    pdf.ln(1)
    pdf.key_value("30% Classification mAP", "Did you also name it correctly? Same overlap rule, but now the category must match too.")
    pdf.ln(2)
    pdf.p(
        "mAP (Mean Average Precision) at IoU 0.5 is the standard metric for object detection. "
        "It measures how well you balance finding everything (recall) vs not making false predictions (precision), "
        "averaged across all classes. IoU 0.5 means your box must overlap at least 50% with the ground truth."
    )
    pdf.h2("Why This Is Hard")
    pdf.bullet("Only 248 training images - extremely small dataset for deep learning")
    pdf.bullet("356 product classes - many classes, few examples per class")
    pdf.bullet("74 classes have fewer than 5 training examples - the model barely sees these")
    pdf.bullet("Images vary wildly in size (480px to 5700px) - need to handle multi-scale")
    pdf.bullet("~92 products per image - very dense, lots of small overlapping objects")

    # =====================================================
    pdf.add_page()
    pdf.h1("YOLO: Our Core Detector")
    pdf.h2("What is YOLO?")
    pdf.p(
        "YOLO (You Only Look Once) is a single-stage object detector. Unlike two-stage detectors "
        "(like Faster R-CNN which first proposes regions then classifies them), YOLO predicts bounding "
        "boxes and class probabilities in one pass through the network. This makes it fast."
    )
    pdf.p(
        "The input image gets divided into a grid. Each grid cell predicts multiple bounding boxes "
        "with confidence scores and class probabilities. Non-Maximum Suppression (NMS) then removes "
        "duplicate detections, keeping only the best box for each object."
    )
    pdf.h2("Why YOLOv8?")
    pdf.bullet("The competition sandbox has ultralytics==8.1.0 pre-installed - native .pt weights just work")
    pdf.bullet("Proven architecture on Kaggle detection competitions")
    pdf.bullet("Easy to train, good out-of-the-box performance")
    pdf.bullet("Built-in augmentation pipeline (mosaic, mixup, copy-paste)")
    pdf.bullet("Supports multiple model sizes: n/s/m/l/x (speed vs accuracy tradeoff)")

    pdf.h2("Why YOLO11 Too?")
    pdf.p(
        "YOLO11 is a newer architecture with improved feature extraction. It's not in the sandbox "
        "(only 8.1.0 is installed), so we train with the latest ultralytics and export to ONNX format. "
        "The sandbox has onnxruntime-gpu which can run any ONNX model on the L4 GPU."
    )
    pdf.p(
        "ONNX (Open Neural Network Exchange) is a universal model format. You train in any framework, "
        "export to .onnx, and run it anywhere. The tradeoff: you lose ultralytics' built-in NMS and "
        "post-processing, so we implement that ourselves."
    )

    pdf.h2("Model Sizes We're Training")
    pdf.p("We're training 5 different YOLO variants. The reason: ensemble diversity (explained later).")
    pdf.bullet("YOLOv8x (640px) - largest v8 model, highest single-model accuracy, batch 8")
    pdf.bullet("YOLOv8l (1280px) - high resolution to catch small products, batch 2")
    pdf.bullet("YOLOv8l (640px, heavy aug) - aggressive augmentation to fight overfitting")
    pdf.bullet("YOLOv8m (640px, SGD) - medium model, different optimizer for diversity")
    pdf.bullet("YOLOv8s (640px) - small/fast model, can train 300 epochs without overfitting")
    pdf.bullet("YOLO11x (640px) - latest architecture, exported to ONNX")

    # =====================================================
    pdf.add_page()
    pdf.h1("Augmentation: Fighting Overfitting")
    pdf.h2("The Problem")
    pdf.p(
        "248 images is tiny. A deep neural network with millions of parameters will memorize "
        "the training data rather than learning general patterns. This is overfitting - high accuracy "
        "on training data, poor accuracy on new images."
    )
    pdf.h2("The Solution: Data Augmentation")
    pdf.p(
        "We artificially create more training data by applying random transformations to each image "
        "during training. The model sees a different version every epoch, so it can't memorize."
    )
    pdf.h3("Mosaic (our most important augmentation)")
    pdf.p(
        "Takes 4 random training images and stitches them into one. The model sees 4x more context "
        "per training step, learns to handle partial objects at edges, and gets much more variety. "
        "Invented for YOLO and is a big part of why it works so well on small datasets."
    )
    pdf.h3("Mixup")
    pdf.p(
        "Blends two images together with a random ratio (e.g., 70/30). The loss is also blended. "
        "This regularizes the model - it can't be too confident about any one prediction because "
        "the target is a mixture. We use mixup=0.15 (baseline) to 0.3 (heavy aug)."
    )
    pdf.h3("Copy-Paste")
    pdf.p(
        "Takes objects from one image and pastes them onto another. Particularly useful for rare "
        "classes - if a product only appears 3 times in training, copy-paste can spread it across "
        "many more images. We use copy_paste=0.1 to 0.2."
    )
    pdf.h3("Other augmentations we use")
    pdf.bullet("HSV jitter - random hue/saturation/brightness changes (lighting variation)")
    pdf.bullet("Flip (horizontal + vertical) - products can face either way")
    pdf.bullet("Scale (0.5) - random zoom in/out, helps with size variation")
    pdf.bullet("Rotation (5-10 degrees) - slight tilt")
    pdf.bullet("Shear (2 degrees) - slight perspective distortion")

    # =====================================================
    pdf.add_page()
    pdf.h1("Ensemble: Combining Multiple Models")
    pdf.h2("Why Ensemble?")
    pdf.p(
        "No single model is perfect. Different models make different mistakes. If you combine "
        "their predictions intelligently, errors cancel out and you get better overall accuracy. "
        "This is the single most reliable technique in Kaggle competitions - nearly every winner uses it."
    )
    pdf.h2("WBF (Weighted Boxes Fusion)")
    pdf.p(
        "Our ensemble method. Unlike regular NMS which just picks the highest-confidence box, "
        "WBF actually merges overlapping boxes from different models into a single better box."
    )
    pdf.p("How it works:")
    pdf.bullet("Run N models on the same image, each produces a set of boxes")
    pdf.bullet("Normalize all boxes to [0,1] coordinates")
    pdf.bullet("Cluster overlapping boxes (IoU > threshold)")
    pdf.bullet("For each cluster, compute a weighted average box position and confidence")
    pdf.bullet("Result: more accurate box positions and more reliable confidence scores")
    pdf.p(
        "The ensemble-boxes library is pre-installed in the sandbox, so WBF costs us nothing extra. "
        "We can include up to 3 weight files in the submission ZIP (420MB limit)."
    )
    pdf.h2("Why Different Configs?")
    pdf.p(
        "Ensemble only works if models are diverse - they need to make different errors. "
        "We create diversity through:"
    )
    pdf.bullet("Different model sizes (x/l/m/s) - different capacity, different features")
    pdf.bullet("Different optimizers (AdamW vs SGD) - converge to different local minima")
    pdf.bullet("Different random seeds - different train/val splits, different initialization")
    pdf.bullet("Different augmentation strength - different regularization")
    pdf.bullet("Different architectures (YOLOv8 vs YOLO11) - fundamentally different feature extraction")

    # =====================================================
    pdf.add_page()
    pdf.h1("SAHI: Tiled Inference for Large Images")
    pdf.h2("The Problem")
    pdf.p(
        "Our images are 2000-4000 pixels wide, but YOLO processes them at 640px. When you resize "
        "a 4000px image to 640px, small products (maybe 50px in the original) become just 8 pixels - "
        "way too small to detect reliably."
    )
    pdf.h2("The Solution: Slice and Detect")
    pdf.p(
        "SAHI (Slicing Aided Hyper Inference) cuts the image into overlapping tiles, "
        "runs the detector on each tile at full resolution, then merges all detections. "
        "It's like using a magnifying glass to scan the entire shelf."
    )
    pdf.p("Our implementation:")
    pdf.bullet("Slice the image into 640x640 tiles with 25% overlap")
    pdf.bullet("Run YOLO on each tile (objects are now bigger relative to tile size)")
    pdf.bullet("Map detected boxes back to original image coordinates")
    pdf.bullet("Also run on the full resized image (catches large objects that span tiles)")
    pdf.bullet("Merge all detections with per-class NMS to remove duplicates")
    pdf.p(
        "The tradeoff: more tiles = more GPU time. A 4000x3000 image at 25% overlap produces "
        "~35 tiles. At ~10ms per tile, that's 350ms per image. With 300s timeout and ~50 test "
        "images, we have budget for this."
    )

    # =====================================================
    pdf.add_page()
    pdf.h1("Two-Stage: Detect Then Classify")
    pdf.h2("The Insight")
    pdf.p(
        "The score is 70% detection + 30% classification. These are fundamentally different problems:"
    )
    pdf.bullet("Detection (finding boxes) is about spatial patterns - edges, shapes, shelf structure")
    pdf.bullet("Classification (naming products) is about visual identity - logos, colors, text on packaging")
    pdf.p(
        "YOLO does both at once, which works great when you have lots of training data. But with "
        "74 classes having <5 examples, YOLO can find the box but can't learn what the product looks like."
    )
    pdf.h2("Our Two-Stage Pipeline")
    pdf.p("Stage 1: YOLO finds all product boxes (targets the 70% detection score)")
    pdf.p("Stage 2: For each box, crop the product, extract a feature embedding, "
           "and match it against a gallery of known products (targets the 30% classification score)")

    pdf.h3("The Gallery")
    pdf.p(
        "We have 327 products with multi-angle reference photos (front, back, left, right, top, bottom). "
        "We also crop products from training images. For each product, we extract a ResNet50 feature "
        "vector (2048 dimensions) and average all vectors for that class. This gives us a 'signature' "
        "for each of the 356 classes."
    )
    pdf.p("Gallery stats: 1,577 reference image embeddings + 2,836 training crop embeddings = 356/356 classes covered.")

    pdf.h3("Matching at Inference")
    pdf.p(
        "For each detected crop, extract its ResNet50 embedding. Compute cosine similarity against "
        "all 356 gallery signatures. The highest match is the predicted class. If confidence is high "
        "enough (>0.7), we override YOLO's class prediction with the gallery match."
    )
    pdf.p(
        "ResNet50 is pre-installed in the sandbox via torchvision, so it doesn't count as a weight file. "
        "The gallery embeddings file is only ~3MB. This is essentially free."
    )

    # =====================================================
    pdf.add_page()
    pdf.h1("The Submission")
    pdf.h2("What Goes in the ZIP")
    pdf.p(
        "run.py at the root + up to 3 weight files (420MB total). That's it. No pip install, "
        "no network, no docker. Our code must work with what's pre-installed."
    )
    pdf.h2("Planned Submissions (3 per day)")
    pdf.h3("Submission 1: Conservative")
    pdf.p("Best single YOLOv8 model with TTA (Test-Time Augmentation - run inference on original + flipped image, average results). Low risk, establishes baseline score.")
    pdf.h3("Submission 2: Ensemble")
    pdf.p("WBF of top 3 models. More accurate boxes, better confidence calibration. Moderate complexity.")
    pdf.h3("Submission 3: Full Pipeline")
    pdf.p("Ensemble + SAHI tiling + two-stage classifier. Maximum accuracy but more things that can go wrong. Uses all 300 seconds of compute budget.")

    pdf.h2("Key Sandbox Constraints")
    pdf.bullet("300 second timeout - need to budget GPU time across all images")
    pdf.bullet("Max 3 weight files, 420MB total - limits ensemble size")
    pdf.bullet("No os, sys, subprocess imports - use pathlib for file operations")
    pdf.bullet("ultralytics==8.1.0 pinned - newer YOLO versions need ONNX export")
    pdf.bullet("L4 GPU (24GB VRAM) - can run large models, FP16 recommended")

    # =====================================================
    pdf.add_page()
    pdf.h1("What We Learned from Kaggle Winners")
    pdf.h2("Key Patterns Across Winning Solutions")
    pdf.bullet("Ensemble always wins - every 1st place uses 3-8 model ensemble")
    pdf.bullet("Augmentation > Architecture - heavy aug on small data beats fancy models")
    pdf.bullet("MixUp is critical for small datasets - prevents overfitting on limited data")
    pdf.bullet("Two-stage works for many-class problems - detect first, classify second")
    pdf.bullet("WBF > NMS for ensembles - weighted fusion beats simple suppression")
    pdf.bullet("Different seeds/splits matter - even same architecture with different random seeds adds value")
    pdf.h2("What Didn't Work (in similar competitions)")
    pdf.bullet("Simulated/external data - rarely helps as much as expected")
    pdf.bullet("Very large models - overfit faster on small datasets, smaller models with more augmentation win")
    pdf.bullet("Complex losses (Tversky, Dice) - standard BCE/CE usually sufficient for detection")
    pdf.bullet("Vision Transformers on small data - CNNs (like YOLO) still dominate with <1K images")

    pdf.output("docs/playbook.pdf")
    print("Generated docs/playbook.pdf")


if __name__ == "__main__":
    generate()
