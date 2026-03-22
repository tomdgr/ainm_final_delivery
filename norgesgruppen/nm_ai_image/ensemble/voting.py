import numpy as np


class VotingEnsemble:
    """Hard or soft voting ensemble."""

    def __init__(self, voting: str = "soft"):
        self.voting = voting

    def predict(self, model_outputs: list[np.ndarray]) -> np.ndarray:
        if self.voting == "hard":
            # Each model output: (N, C) probabilities -> argmax -> majority vote
            votes = np.stack([np.argmax(o, axis=1) for o in model_outputs])  # (M, N)
            from scipy import stats
            result, _ = stats.mode(votes, axis=0, keepdims=False)
            return result
        else:
            # Soft voting: average probabilities
            avg = np.mean(model_outputs, axis=0)
            return np.argmax(avg, axis=1)

    def predict_proba(self, model_outputs: list[np.ndarray]) -> np.ndarray:
        return np.mean(model_outputs, axis=0)


class WeightedVotingEnsemble:
    """Weighted soft voting ensemble."""

    def __init__(self, weights: list[float] | None = None):
        self.weights = weights

    def predict(self, model_outputs: list[np.ndarray]) -> np.ndarray:
        proba = self.predict_proba(model_outputs)
        return np.argmax(proba, axis=1)

    def predict_proba(self, model_outputs: list[np.ndarray]) -> np.ndarray:
        if self.weights is None:
            weights = np.ones(len(model_outputs)) / len(model_outputs)
        else:
            weights = np.array(self.weights) / np.sum(self.weights)

        return sum(w * o for w, o in zip(weights, model_outputs))
