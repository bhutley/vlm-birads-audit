"""Abstract base class for all model wrappers."""

from abc import ABC, abstractmethod

import torch


class BaseModel(ABC):
    """Common interface for all models in the pipeline."""

    @abstractmethod
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into feature vectors."""

    @abstractmethod
    def classify(self, images: torch.Tensor) -> torch.Tensor:
        """Return class probabilities for a batch of images."""

    @abstractmethod
    def get_device(self) -> torch.device:
        """Return the device this model is on."""

    def to(self, device: torch.device) -> "BaseModel":
        """Move model to device. Subclasses should override."""
        return self
