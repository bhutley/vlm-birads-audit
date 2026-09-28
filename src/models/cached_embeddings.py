"""Serve precomputed VLM embeddings to the geometry pipeline as a drop-in "model".

Some backbones (e.g. UniMed-CLIP) ship a forked ``open_clip`` / older dependency
stack that does not run in this repo's Python environment. We embed them once in an
isolated environment (``scripts/embed_unimed.py``), cache L2-normalised image and
prompt embeddings, and expose them here behind the minimal slice of the model
interface the experiments actually use:

  * ``set_prompts`` / ``_text_features`` — consumed by
    :func:`concept_directions.encode_texts`; text lookup is by exact prompt string,
    so the cache must be regenerated if the concept bank changes.
  * per-site image embeddings — consumed by
    :func:`compute_rho_geometry._embed_site` (benign+malignant filtering happens
    there, exactly as for the live backbones).

Because the geometry is closed-form on cached embeddings, swapping a live encoder
for this cache changes nothing downstream (probe fit, rho, nulls, decomposition).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class CachedEmbeddingModel:
    """Minimal model-interface shim backed by cached embeddings (see module doc)."""

    def __init__(self, text_map: dict[str, np.ndarray],
                 site_embs: dict[str, tuple[np.ndarray, np.ndarray]],
                 site_ids: dict[str, np.ndarray] | None = None):
        self._text_map = text_map        # {prompt str: (d,) float64, L2-normalised}
        self._site_embs = site_embs      # {site: (Z (n,d) float64, labels (n,) int)}
        self._site_ids = site_ids or {}  # {site: (n,) canonical sample-id str}
        self._text_features: torch.Tensor | None = None
        self.preprocess = None           # never used: image encoding is offline

    @classmethod
    def from_cache(cls, cache_dir: str | Path,
                   prefix: str = "unimed_clip") -> "CachedEmbeddingModel":
        cache_dir = Path(cache_dir)
        text_npz = cache_dir / f"{prefix}_text.npz"
        if not text_npz.exists():
            raise FileNotFoundError(
                f"missing {text_npz} — run scripts/embed_unimed.py (isolated env) first")
        td = np.load(text_npz, allow_pickle=True)
        prompts = [str(p) for p in td["prompts"]]
        embs = td["embeddings"].astype(np.float64)
        text_map = {p: embs[i] for i, p in enumerate(prompts)}

        site_embs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        site_ids: dict[str, np.ndarray] = {}
        for f in sorted(cache_dir.glob(f"{prefix}_image_*.npz")):
            site = f.name[len(f"{prefix}_image_"):-len(".npz")]
            d = np.load(f, allow_pickle=True)
            site_embs[site] = (d["embeddings"].astype(np.float64), d["labels"].astype(int))
            if "sample_ids" in d.files:   # caches written after WP2; else sidecar
                site_ids[site] = np.array([str(s) for s in d["sample_ids"]])
        if not site_embs:
            raise FileNotFoundError(
                f"no {prefix}_image_*.npz caches in {cache_dir} — run scripts/embed_unimed.py")
        return cls(text_map, site_embs, site_ids)

    # -- text path (concept_directions.encode_texts) ------------------------- #
    def set_prompts(self, prompts: dict[str, str]) -> None:
        texts = list(prompts.values())
        try:
            vecs = np.stack([self._text_map[t] for t in texts], axis=0)
        except KeyError as e:
            raise KeyError(
                f"prompt absent from the cached UniMed-CLIP text embeddings: {e!s}. "
                "Regenerate the cache (scripts/embed_unimed.py) after editing the "
                "concept bank.") from e
        self._text_features = torch.from_numpy(vecs)

    # -- image path (compute_rho_geometry._embed_site) ----------------------- #
    def site_embeddings(self, site: str) -> tuple[np.ndarray, np.ndarray]:
        if site not in self._site_embs:
            raise FileNotFoundError(
                f"no cached UniMed-CLIP embeddings for site {site!r} "
                f"(have: {sorted(self._site_embs)})")
        return self._site_embs[site]

    def site_sample_ids(self, site: str) -> np.ndarray | None:
        """Canonical sample ids for a site if the cache stored them, else None.

        ``None`` signals the consumer (src/data/cohort.py) to fall back to the
        validated sidecar re-enumeration for pre-WP2 caches.
        """
        return self._site_ids.get(site)

    def get_device(self) -> torch.device:
        return torch.device("cpu")

    def encode_image(self, images):  # pragma: no cover - cache path bypasses this
        raise NotImplementedError(
            "CachedEmbeddingModel serves cached embeddings; image encoding is done "
            "offline in scripts/embed_unimed.py.")

    def classify(self, images):  # pragma: no cover
        raise NotImplementedError
