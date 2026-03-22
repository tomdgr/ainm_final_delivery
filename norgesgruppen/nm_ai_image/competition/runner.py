import logging

from nm_ai_image.competition.client import CompetitionClient
from nm_ai_image.competition.submission import (
    format_classification_submission,
    format_detection_submission,
    format_segmentation_submission,
)

logger = logging.getLogger(__name__)


class CompetitionRunner:
    """Orchestrates full competition pipeline: train -> predict -> ensemble -> submit."""

    def __init__(self, client: CompetitionClient | None = None):
        self.client = client or CompetitionClient()
        self.submissions: list[dict] = []

    def submit_classification(
        self, predictions, image_ids: list[str], task_id: str | None = None
    ) -> dict:
        import numpy as np
        formatted = format_classification_submission(np.array(predictions), image_ids)
        result = self.client.submit(formatted, task_id=task_id)
        self.submissions.append({"task_id": task_id, "result": result, "type": "classification"})
        logger.info("Classification submission %d — result: %s", len(self.submissions), result)
        return result

    def submit_detection(
        self, boxes, labels, scores, image_ids: list[str], task_id: str | None = None
    ) -> dict:
        formatted = format_detection_submission(boxes, labels, scores, image_ids)
        result = self.client.submit(formatted, task_id=task_id)
        self.submissions.append({"task_id": task_id, "result": result, "type": "detection"})
        logger.info("Detection submission %d — result: %s", len(self.submissions), result)
        return result

    def submit_segmentation(
        self, masks, image_ids: list[str], task_id: str | None = None
    ) -> dict:
        formatted = format_segmentation_submission(masks, image_ids)
        result = self.client.submit(formatted, task_id=task_id)
        self.submissions.append({"task_id": task_id, "result": result, "type": "segmentation"})
        logger.info("Segmentation submission %d — result: %s", len(self.submissions), result)
        return result

    def get_submission_history(self) -> list[dict]:
        return self.submissions.copy()
