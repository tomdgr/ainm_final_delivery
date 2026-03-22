import logging

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier

logger = logging.getLogger(__name__)


def blend_predictions(model_outputs: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    weights = np.array(weights) / np.sum(weights)
    return sum(w * o for w, o in zip(weights, model_outputs))


class BlendingEnsemble:
    def __init__(self, method: str = "linear"):
        self.method = method
        self.weights: np.ndarray | None = None
        self.meta_model = None

    def fit(self, model_outputs: list[np.ndarray], y_true: np.ndarray):
        n_models = len(model_outputs)

        if self.method == "linear":
            self.weights = self._optimize_weights(model_outputs, y_true)
        elif self.method == "stacking":
            stacked = np.column_stack([o for o in model_outputs])
            self.meta_model = LogisticRegression(max_iter=1000)
            self.meta_model.fit(stacked, y_true)
        else:
            self.weights = np.ones(n_models) / n_models

    def predict(self, model_outputs: list[np.ndarray]) -> np.ndarray:
        if self.method == "stacking" and self.meta_model is not None:
            stacked = np.column_stack([o for o in model_outputs])
            return self.meta_model.predict(stacked)

        if self.weights is None:
            self.weights = np.ones(len(model_outputs)) / len(model_outputs)
        return blend_predictions(model_outputs, self.weights)

    def _optimize_weights(self, model_outputs: list[np.ndarray], y_true: np.ndarray) -> np.ndarray:
        n_models = len(model_outputs)
        best_weights = np.ones(n_models) / n_models
        best_score = -np.inf

        # Grid search over weight simplex
        from itertools import product as iter_product

        steps = 11
        grid = np.linspace(0, 1, steps)

        if n_models == 2:
            for w in grid:
                weights = np.array([w, 1 - w])
                blended = blend_predictions(model_outputs, weights)
                preds = np.argmax(blended, axis=1) if blended.ndim > 1 else (blended > 0.5).astype(int)
                score = np.mean(preds == y_true)
                if score > best_score:
                    best_score = score
                    best_weights = weights
        else:
            # For >2 models, use uniform weights as a simple baseline
            best_weights = np.ones(n_models) / n_models

        logger.info("Optimized blend weights: %s (score: %.4f)", best_weights, best_score)
        return best_weights
