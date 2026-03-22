import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TTAPredictor:
    """Test-time augmentation for improved predictions."""

    def __init__(self, model: nn.Module, transforms: list[str] | None = None):
        self.model = model
        self.transforms = transforms or ["original", "hflip", "vflip"]

    @torch.no_grad()
    def predict_classification(self, images: torch.Tensor) -> torch.Tensor:
        """Average logits over augmented versions of the input."""
        self.model.eval()
        all_logits = []

        for t in self.transforms:
            augmented = self._apply_transform(images, t)
            logits = self.model(augmented)
            all_logits.append(logits)

        return torch.stack(all_logits).mean(dim=0)

    @torch.no_grad()
    def predict_segmentation(self, images: torch.Tensor) -> torch.Tensor:
        """Average probability maps over augmented versions."""
        self.model.eval()
        all_probs = []

        for t in self.transforms:
            augmented = self._apply_transform(images, t)
            logits = self.model(augmented)
            probs = torch.softmax(logits, dim=1)
            probs = self._reverse_transform(probs, t)
            all_probs.append(probs)

        return torch.stack(all_probs).mean(dim=0)

    def _apply_transform(self, images: torch.Tensor, transform: str) -> torch.Tensor:
        if transform == "original":
            return images
        elif transform == "hflip":
            return torch.flip(images, dims=[3])
        elif transform == "vflip":
            return torch.flip(images, dims=[2])
        elif transform == "rotate90":
            return torch.rot90(images, k=1, dims=[2, 3])
        elif transform == "rotate180":
            return torch.rot90(images, k=2, dims=[2, 3])
        elif transform == "rotate270":
            return torch.rot90(images, k=3, dims=[2, 3])
        else:
            return images

    def _reverse_transform(self, output: torch.Tensor, transform: str) -> torch.Tensor:
        if transform == "original":
            return output
        elif transform == "hflip":
            return torch.flip(output, dims=[3])
        elif transform == "vflip":
            return torch.flip(output, dims=[2])
        elif transform == "rotate90":
            return torch.rot90(output, k=3, dims=[2, 3])
        elif transform == "rotate180":
            return torch.rot90(output, k=2, dims=[2, 3])
        elif transform == "rotate270":
            return torch.rot90(output, k=1, dims=[2, 3])
        else:
            return output
