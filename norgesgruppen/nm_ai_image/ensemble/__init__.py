"""Ensemble methods for boosting competition scores."""

from nm_ai_image.ensemble.blending import BlendingEnsemble, blend_predictions
from nm_ai_image.ensemble.voting import VotingEnsemble, WeightedVotingEnsemble
from nm_ai_image.ensemble.tta import TTAPredictor

__all__ = [
    "BlendingEnsemble",
    "blend_predictions",
    "VotingEnsemble",
    "WeightedVotingEnsemble",
    "TTAPredictor",
]
