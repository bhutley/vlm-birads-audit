"""CLIP model wrapper for zero-shot and linear probe classification."""

import open_clip
import torch
import torch.nn.functional as F

from .base import BaseModel


class CLIPZeroShot(BaseModel):
    """CLIP zero-shot classifier using text prompt similarity."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        prompts: dict[str, str] | None = None,
        device: torch.device | None = None,
    ):
        if device is None:
            device = _get_default_device()
        self._device = device

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

        self._text_features: torch.Tensor | None = None
        self._class_names: list[str] = []
        if prompts is not None:
            self.set_prompts(prompts)

    def set_prompts(self, prompts: dict[str, str]) -> None:
        """Set text prompts for zero-shot classification.

        Args:
            prompts: Mapping of class_name -> text prompt.
        """
        self._class_names = list(prompts.keys())
        texts = list(prompts.values())
        tokens = self.tokenizer(texts).to(self._device)
        with torch.no_grad():
            self._text_features = self.model.encode_text(tokens)
            self._text_features = F.normalize(self._text_features, dim=-1)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self._device)
        with torch.no_grad():
            features = self.model.encode_image(images)
            features = F.normalize(features, dim=-1)
        return features

    def classify(self, images: torch.Tensor) -> torch.Tensor:
        """Return class probabilities via cosine similarity with text prompts."""
        if self._text_features is None:
            raise RuntimeError("Call set_prompts() before classify().")
        image_features = self.encode_image(images)
        similarity = image_features @ self._text_features.T
        # Scale by CLIP's learned temperature
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * similarity
        return F.softmax(logits, dim=-1)

    def get_device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device) -> "CLIPZeroShot":
        self._device = device
        self.model = self.model.to(device)
        if self._text_features is not None:
            self._text_features = self._text_features.to(device)
        return self

    @property
    def class_names(self) -> list[str]:
        return self._class_names


class BiomedCLIPZeroShot(CLIPZeroShot):
    """BiomedCLIP zero-shot classifier.

    Uses the microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 model
    from HuggingFace via open_clip.
    """

    def __init__(
        self,
        prompts: dict[str, str] | None = None,
        device: torch.device | None = None,
    ):
        if device is None:
            device = _get_default_device()
        self._device = device

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            device=device,
        )
        self.tokenizer = open_clip.get_tokenizer(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        self.model.eval()

        self._text_features = None
        self._class_names = []
        if prompts is not None:
            self.set_prompts(prompts)


class PMCClipZeroShot(CLIPZeroShot):
    """PMC-CLIP zero-shot classifier (open_clip ViT-L/14 reproduction).

    A biomedical *vision* encoder trained on PubMed Central figure-caption pairs
    (PMC-OA; cf. Lin et al. 2023), in open_clip-native format. Used as a third
    biomedical-vision backbone (alongside BiomedCLIP and UniMed-CLIP) to test
    whether BI-RADS grounding is a property of the category, not one checkpoint.
    Embedding dim 768 (vs 512 for BiomedCLIP/UniMed); the per-backbone null is
    computed in each model's own dimension, so excesses stay comparable.
    """

    HF_REPO = "hf-hub:ryanyip7777/pmc_vit_l_14"

    def __init__(
        self,
        prompts: dict[str, str] | None = None,
        device: torch.device | None = None,
    ):
        if device is None:
            device = _get_default_device()
        self._device = device

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.HF_REPO, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(self.HF_REPO)
        self.model.eval()

        self._text_features = None
        self._class_names = []
        if prompts is not None:
            self.set_prompts(prompts)


def _get_default_device() -> torch.device:
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
