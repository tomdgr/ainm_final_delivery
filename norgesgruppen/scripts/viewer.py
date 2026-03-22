"""Streamlit app to inspect predictions vs ground truth.

Usage:
  uv run streamlit run scripts/viewer.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# Paths
COCO_JSON = "data/raw/coco_dataset/train/annotations.json"
IMAGE_DIR = "data/raw/coco_dataset/train/images"
PREDICTIONS_FILE = "outputs/predictions_for_viewer.json"


@st.cache_data
def load_coco():
    with open(COCO_JSON) as f:
        coco = json.load(f)
    categories = {c["id"]: c["name"] for c in coco["categories"]}
    images = {img["id"]: img for img in coco["images"]}
    gt_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        gt_by_image[ann["image_id"]].append(ann)
    return coco, categories, images, gt_by_image


@st.cache_data
def load_predictions():
    if not Path(PREDICTIONS_FILE).exists():
        return {}
    with open(PREDICTIONS_FILE) as f:
        preds = json.load(f)
    pred_by_image = defaultdict(list)
    for p in preds:
        pred_by_image[p["image_id"]].append(p)
    return pred_by_image


def compute_iou(box_a, box_b):
    """IoU between two [x, y, w, h] boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_predictions(gt_list, pred_list, iou_thr=0.5):
    """Match predictions to ground truths. Returns matched, missed, false_pos."""
    matched = []  # (gt, pred, iou, correct_class)
    missed = []  # gt with no matching pred
    false_pos = []  # pred with no matching gt

    gt_matched = [False] * len(gt_list)
    pred_matched = [False] * len(pred_list)

    # Sort preds by score descending
    sorted_preds = sorted(enumerate(pred_list), key=lambda x: x[1]["score"], reverse=True)

    for pi, pred in sorted_preds:
        best_iou = 0
        best_gi = -1
        for gi, gt in enumerate(gt_list):
            if gt_matched[gi]:
                continue
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_iou >= iou_thr and best_gi >= 0:
            gt_matched[best_gi] = True
            pred_matched[pi] = True
            correct_cls = pred["category_id"] == gt_list[best_gi]["category_id"]
            matched.append((gt_list[best_gi], pred, best_iou, correct_cls))
        else:
            false_pos.append(pred)

    for gi, gt in enumerate(gt_list):
        if not gt_matched[gi]:
            missed.append(gt)

    return matched, missed, false_pos


def draw_boxes(img, boxes, color, categories, label_key="category_id", score_key=None, alpha=180):
    """Draw boxes on image with labels."""
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except Exception:
        font = ImageFont.load_default()

    for box_info in boxes:
        x, y, w, h = box_info["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        cat_name = categories.get(box_info[label_key], str(box_info[label_key]))
        label = cat_name[:25]
        if score_key and score_key in box_info:
            label += f" {box_info[score_key]:.2f}"
        # Background for text
        bbox = draw.textbbox((x, y - 14), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x, y - 14), label, fill="white", font=font)
    return overlay


def main():
    st.set_page_config(layout="wide", page_title="Detection Viewer")
    st.title("Detection Viewer — GT vs Predictions")

    coco, categories, images, gt_by_image = load_coco()
    pred_by_image = load_predictions()

    if not pred_by_image:
        st.warning("No predictions found. Run: `uv run python scripts/generate_predictions.py --onnx weights/yolov8x_640_fulldata_best.onnx`")

    # Image list
    image_list = sorted(images.keys())
    image_names = {img_id: images[img_id]["file_name"] for img_id in image_list}

    # Sidebar controls
    st.sidebar.header("Controls")
    img_idx = st.sidebar.slider("Image index", 0, len(image_list) - 1, 0)
    img_id = image_list[img_idx]
    st.sidebar.write(f"**Image ID**: {img_id}")
    st.sidebar.write(f"**File**: {image_names[img_id]}")

    conf_thr = st.sidebar.slider("Min confidence", 0.0, 1.0, 0.01, 0.01)
    show_gt = st.sidebar.checkbox("Show ground truth", True)
    show_pred = st.sidebar.checkbox("Show predictions", True)
    show_analysis = st.sidebar.checkbox("Show match analysis", True)
    view_mode = st.sidebar.radio("View mode", ["Side by side", "Overlay", "Analysis only"])

    # Load image
    img_info = images[img_id]
    img_path = Path(IMAGE_DIR) / img_info["file_name"]
    img = Image.open(img_path).convert("RGB")

    gt_list = gt_by_image.get(img_id, [])
    pred_list = [p for p in pred_by_image.get(img_id, []) if p["score"] >= conf_thr]

    # Stats
    st.sidebar.markdown("---")
    st.sidebar.write(f"**GT boxes**: {len(gt_list)}")
    st.sidebar.write(f"**Pred boxes** (conf>={conf_thr:.2f}): {len(pred_list)}")

    if view_mode == "Side by side":
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Ground Truth")
            if show_gt:
                gt_img = draw_boxes(img.copy(), gt_list, "lime", categories)
            else:
                gt_img = img.copy()
            st.image(gt_img, use_container_width=True)

        with col2:
            st.subheader("Predictions")
            if show_pred and pred_list:
                pred_img = draw_boxes(img.copy(), pred_list, "cyan", categories, score_key="score")
            else:
                pred_img = img.copy()
            st.image(pred_img, use_container_width=True)

    elif view_mode == "Overlay":
        overlay_img = img.copy()
        if show_gt:
            overlay_img = draw_boxes(overlay_img, gt_list, "lime", categories)
        if show_pred and pred_list:
            overlay_img = draw_boxes(overlay_img, pred_list, "cyan", categories, score_key="score")
        st.image(overlay_img, use_container_width=True)
        st.caption("Green = GT, Cyan = Predictions")

    # Match analysis
    if show_analysis and pred_list:
        matched, missed, false_pos = match_predictions(gt_list, pred_list)

        st.markdown("---")
        st.subheader("Match Analysis")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("GT boxes", len(gt_list))
        col2.metric("Matched", len(matched))
        col3.metric("Missed (FN)", len(missed), delta=-len(missed) if missed else None, delta_color="inverse")
        col4.metric("False Pos (FP)", len(false_pos), delta=-len(false_pos) if false_pos else None, delta_color="inverse")

        # Classification accuracy among matched
        if matched:
            correct = sum(1 for _, _, _, cc in matched if cc)
            wrong = len(matched) - correct
            st.write(f"**Classification**: {correct}/{len(matched)} correct ({100*correct/len(matched):.0f}%), {wrong} misclassified")

            # Show misclassifications
            misclassified = [(gt, pred, iou) for gt, pred, iou, cc in matched if not cc]
            if misclassified:
                st.subheader(f"Misclassifications ({len(misclassified)})")
                for gt, pred, iou in misclassified[:20]:
                    gt_name = categories.get(gt["category_id"], str(gt["category_id"]))
                    pred_name = categories.get(pred["category_id"], str(pred["category_id"]))
                    st.write(f"- GT: **{gt_name}** → Pred: **{pred_name}** (conf={pred['score']:.3f}, IoU={iou:.2f})")

        # Show missed detections
        if missed:
            st.subheader(f"Missed Detections ({len(missed)})")
            # Draw missed on image
            missed_img = img.copy()
            missed_img = draw_boxes(missed_img, missed, "red", categories)
            st.image(missed_img, caption="Red = Missed GT boxes", use_container_width=True)

            missed_cats = defaultdict(int)
            for m in missed:
                missed_cats[categories.get(m["category_id"], str(m["category_id"]))] += 1
            st.write("By class:", dict(sorted(missed_cats.items(), key=lambda x: -x[1])[:15]))

        # Show false positives
        if false_pos:
            st.subheader(f"False Positives ({len(false_pos)})")
            fp_img = img.copy()
            fp_img = draw_boxes(fp_img, false_pos, "red", categories, score_key="score")
            st.image(fp_img, caption="Red = False positive predictions", use_container_width=True)

            fp_cats = defaultdict(int)
            for fp in false_pos:
                fp_cats[categories.get(fp["category_id"], str(fp["category_id"]))] += 1
            st.write("By class:", dict(sorted(fp_cats.items(), key=lambda x: -x[1])[:15]))

    # Global stats
    st.sidebar.markdown("---")
    st.sidebar.subheader("Global Stats")
    if pred_by_image:
        total_preds = sum(len(v) for v in pred_by_image.values())
        total_gt = sum(len(v) for v in gt_by_image.values())
        st.sidebar.write(f"Total predictions: {total_preds}")
        st.sidebar.write(f"Total GT: {total_gt}")
        st.sidebar.write(f"Images with preds: {len(pred_by_image)}")

        # Per-class stats
        if st.sidebar.button("Show worst classes"):
            all_matched, all_missed, all_fp = [], [], []
            for iid in image_list:
                gl = gt_by_image.get(iid, [])
                pl = [p for p in pred_by_image.get(iid, []) if p["score"] >= conf_thr]
                m, mi, fp = match_predictions(gl, pl)
                all_matched.extend(m)
                all_missed.extend(mi)
                all_fp.extend(fp)

            # Classes with most misses
            miss_by_class = defaultdict(int)
            for m in all_missed:
                miss_by_class[categories.get(m["category_id"], str(m["category_id"]))] += 1
            st.sidebar.write("**Most missed classes:**")
            for cat, count in sorted(miss_by_class.items(), key=lambda x: -x[1])[:15]:
                st.sidebar.write(f"  {cat}: {count}")

            # Misclassification pairs
            miscls = defaultdict(int)
            for gt, pred, iou, cc in all_matched:
                if not cc:
                    gt_n = categories.get(gt["category_id"], "?")
                    pr_n = categories.get(pred["category_id"], "?")
                    miscls[(gt_n, pr_n)] += 1
            if miscls:
                st.sidebar.write("**Top confusion pairs:**")
                for (g, p), count in sorted(miscls.items(), key=lambda x: -x[1])[:15]:
                    st.sidebar.write(f"  {g} → {p}: {count}")


if __name__ == "__main__":
    main()
