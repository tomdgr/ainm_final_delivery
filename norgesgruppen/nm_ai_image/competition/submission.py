import logging

import numpy as np

logger = logging.getLogger(__name__)


def format_classification_submission(
    predictions: np.ndarray,
    image_ids: list[str],
) -> dict:
    if predictions.ndim == 1:
        # Already class indices
        labels = predictions.tolist()
    else:
        # Probability matrix -> argmax
        labels = np.argmax(predictions, axis=1).tolist()

    return {
        "predictions": [
            {"image_id": img_id, "label": int(label)}
            for img_id, label in zip(image_ids, labels)
        ]
    }


def format_detection_submission(
    boxes: list[np.ndarray],
    labels: list[np.ndarray],
    scores: list[np.ndarray],
    image_ids: list[str],
) -> dict:
    predictions = []
    for img_id, img_boxes, img_labels, img_scores in zip(image_ids, boxes, labels, scores):
        for box, label, score in zip(img_boxes, img_labels, img_scores):
            predictions.append({
                "image_id": img_id,
                "bbox": box.tolist(),
                "label": int(label),
                "score": float(score),
            })
    return {"predictions": predictions}


def format_segmentation_submission(
    masks: list[np.ndarray],
    image_ids: list[str],
) -> dict:
    predictions = []
    for img_id, mask in zip(image_ids, masks):
        # RLE encode the mask
        predictions.append({
            "image_id": img_id,
            "mask": _rle_encode(mask),
        })
    return {"predictions": predictions}


def _rle_encode(mask: np.ndarray) -> dict:
    """Run-length encode a binary mask."""
    pixels = mask.flatten()
    runs = []
    prev = -1
    start = 0

    for i, p in enumerate(pixels):
        if p != prev:
            if prev > 0:
                runs.extend([start + 1, i - start])
            start = i
            prev = p

    if prev > 0:
        runs.extend([start + 1, len(pixels) - start])

    return {"counts": runs, "size": list(mask.shape)}
