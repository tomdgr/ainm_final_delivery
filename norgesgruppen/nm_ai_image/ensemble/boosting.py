import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BoostingConfig:
    n_estimators: int = 100
    learning_rate: float = 0.1
    feature_extraction_backbone: str = "resnet50"


class GradientBoostingEnsemble:
    """Gradient boosting on backbone features + model predictions."""

    def __init__(self, config: BoostingConfig | None = None):
        self.config = config or BoostingConfig()
        self.model = None

    def fit(self, features: np.ndarray, labels: np.ndarray):
        try:
            from sklearn.ensemble import GradientBoostingClassifier

            self.model = GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                random_state=42,
            )
            self.model.fit(features, labels)
            train_acc = self.model.score(features, labels)
            logger.info("Boosting ensemble trained — train accuracy: %.4f", train_acc)
        except ImportError:
            logger.warning("scikit-learn GradientBoostingClassifier not available")

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.predict_proba(features)
