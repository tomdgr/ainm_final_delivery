import logging

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


class StackingEnsemble:
    """Two-level stacking: base models -> meta-model.

    Uses K-fold out-of-fold predictions for training the meta-model.
    """

    def __init__(self, n_folds: int = 5, meta_model=None):
        self.n_folds = n_folds
        self.meta_model = meta_model or RidgeClassifier(alpha=1.0)
        self._fitted = False

    def fit(
        self,
        base_model_predictions: list[np.ndarray],
        y_true: np.ndarray,
    ):
        """Fit meta-model on stacked base model predictions.

        Args:
            base_model_predictions: List of (N, C) prediction arrays from each base model.
            y_true: True labels (N,).
        """
        stacked_features = np.column_stack(base_model_predictions)
        self.meta_model.fit(stacked_features, y_true)
        self._fitted = True

        train_preds = self.meta_model.predict(stacked_features) if hasattr(self.meta_model, 'predict') else None
        if train_preds is not None:
            acc = np.mean(train_preds == y_true)
            logger.info("Stacking meta-model trained — train accuracy: %.4f", acc)

    def fit_oof(
        self,
        train_fn,
        predict_fn,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_base_models: int,
    ):
        """Fit using K-fold out-of-fold predictions.

        Args:
            train_fn: Function(X_train, y_train, fold_idx) -> trained model
            predict_fn: Function(model, X_val) -> predictions (N, C)
            X_train: Training features
            y_train: Training labels
            n_base_models: Number of base models to train per fold
        """
        n_samples = len(X_train)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        oof_predictions = [np.zeros((n_samples,)) for _ in range(n_base_models)]

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            logger.info("Stacking fold %d/%d", fold_idx + 1, self.n_folds)
            X_fold_train, y_fold_train = X_train[train_idx], y_train[train_idx]
            X_fold_val = X_train[val_idx]

            for model_idx in range(n_base_models):
                model = train_fn(X_fold_train, y_fold_train, model_idx)
                preds = predict_fn(model, X_fold_val)
                oof_predictions[model_idx][val_idx] = preds

        self.fit(oof_predictions, y_train)

    def predict(self, base_model_predictions: list[np.ndarray]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Meta-model not fitted")
        stacked_features = np.column_stack(base_model_predictions)
        return self.meta_model.predict(stacked_features)
