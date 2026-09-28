"""Additional VLM wrappers used by the vlm_expansion experiment.

PubMedCLIP weights are stored in standard HuggingFace CLIP format, so they must
be loaded via transformers, not open_clip. SigLIP loads through open_clip's
hf-hub mechanism but uses sigmoid pretraining — we still expose probabilities
via softmax-over-classes so the interface matches CLIPZeroShot, which is
sufficient for argmax accuracy and rank-preserving AUC.
"""

import open_clip
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from transformers import CLIPModel, CLIPProcessor

from .base import BaseModel


def _get_default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class PubMedCLIPZeroShot(BaseModel):
    """PubMedCLIP zero-shot classifier via HuggingFace transformers."""

    DEFAULT_HF_REPO = "flaviagiammarino/pubmed-clip-vit-base-patch32"

    # PubMedCLIP has no learnable logit_scale; this fixed temperature mirrors
    # the value used in the original PubMedCLIP zero-shot evaluation code.
    LOGIT_SCALE = 100.0

    def __init__(
        self,
        hf_repo: str | None = None,
        prompts: dict[str, str] | None = None,
        device: torch.device | None = None,
    ):
        if device is None:
            device = _get_default_device()
        self._device = device
        self._hf_repo = hf_repo or self.DEFAULT_HF_REPO

        self.processor = CLIPProcessor.from_pretrained(self._hf_repo)
        self.model = CLIPModel.from_pretrained(self._hf_repo).to(device)
        self.model.eval()

        self.preprocess = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

        self._text_features: torch.Tensor | None = None
        self._class_names: list[str] = []
        if prompts is not None:
            self.set_prompts(prompts)

    def set_prompts(self, prompts: dict[str, str]) -> None:
        self._class_names = list(prompts.keys())
        texts = list(prompts.values())
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            text_out = self.model.text_model(**inputs)
            feats = self.model.text_projection(text_out.pooler_output)
            self._text_features = F.normalize(feats, dim=-1)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self._device)
        with torch.no_grad():
            vision_out = self.model.vision_model(pixel_values=images)
            feats = self.model.visual_projection(vision_out.pooler_output)
            return F.normalize(feats, dim=-1)

    def classify(self, images: torch.Tensor) -> torch.Tensor:
        if self._text_features is None:
            raise RuntimeError("Call set_prompts() before classify().")
        image_features = self.encode_image(images)
        similarity = image_features @ self._text_features.T
        return F.softmax(self.LOGIT_SCALE * similarity, dim=-1)

    def get_device(self) -> torch.device:
        return self._device

    @property
    def class_names(self) -> list[str]:
        return self._class_names


class SigLIPZeroShot(BaseModel):
    """SigLIP zero-shot classifier via open_clip's hf-hub mechanism.

    SigLIP is trained with a sigmoid pair-loss; for single-label classification
    we still take softmax over class scores so the wrapper's interface matches
    CLIPZeroShot. Argmax is unchanged and ranks are preserved, so accuracy/F1
    and AUC remain meaningful.
    """

    DEFAULT_HUB_NAME = "hf-hub:timm/ViT-B-16-SigLIP-256"

    def __init__(
        self,
        hub_name: str | None = None,
        prompts: dict[str, str] | None = None,
        device: torch.device | None = None,
    ):
        if device is None:
            device = _get_default_device()
        self._device = device
        self._hub_name = hub_name or self.DEFAULT_HUB_NAME

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self._hub_name, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(self._hub_name)
        self.model.eval()

        self._text_features: torch.Tensor | None = None
        self._class_names: list[str] = []
        if prompts is not None:
            self.set_prompts(prompts)

    def set_prompts(self, prompts: dict[str, str]) -> None:
        self._class_names = list(prompts.keys())
        texts = list(prompts.values())
        tokens = self.tokenizer(texts).to(self._device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
            self._text_features = F.normalize(feats, dim=-1)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self._device)
        with torch.no_grad():
            feats = self.model.encode_image(images)
            return F.normalize(feats, dim=-1)

    def classify(self, images: torch.Tensor) -> torch.Tensor:
        if self._text_features is None:
            raise RuntimeError("Call set_prompts() before classify().")
        image_features = self.encode_image(images)
        similarity = image_features @ self._text_features.T
        logit_scale = self.model.logit_scale.exp()
        return F.softmax(logit_scale * similarity, dim=-1)

    def get_device(self) -> torch.device:
        return self._device

    @property
    def class_names(self) -> list[str]:
        return self._class_names
