"""Produce UniMed-CLIP embedding caches for the geometry pipeline (E2/E3).

UniMed-CLIP ships a forked ``open_clip`` + an older dependency stack that does not
run in this repo's Python environment, so we embed it ONCE in an isolated env and
cache the result. The closed-form geometry then consumes the cache via
:class:`src.models.cached_embeddings.CachedEmbeddingModel` (backbone ``unimed_clip``),
so nothing downstream (probe, rho, nulls) changes.

This script must be run in an isolated environment with UniMed-CLIP installed, NOT
in the project env:

    python3.10 -m venv /tmp/unimed && source /tmp/unimed/bin/activate
    git clone https://github.com/mbzuai-oryx/UniMed-CLIP && pip install -e UniMed-CLIP
    pip install torch torchvision transformers huggingface_hub pyyaml openpyxl pillow
    # weights: hf UzairK/unimed-clip-vit-b16 -> unimed-clip-vit-b16.pt
    UNIMED_CKPT=/path/to/unimed-clip-vit-b16.pt python scripts/embed_unimed.py

It encodes, for the SAME concept bank and datasets the repo uses, every BI-RADS
concept / placebo / malignancy-reference prompt string (so they match what
``encode_texts`` requests in-pipeline) and every site's lesion images, writing
L2-normalised float32 caches to ``results/cross_modal_probe/embeddings/``:
    unimed_clip_text.npz         (prompts, embeddings)
    unimed_clip_image_<site>.npz (embeddings, labels)
"""
from __future__ import annotations

import functools
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# torch>=2.6 defaults weights_only=True; the official-mirror checkpoint holds a
# numpy scalar, so force False before any open_clip import triggers torch.load.
torch.load = functools.partial(torch.load, weights_only=False)

from torch.utils.data import DataLoader
from open_clip import create_model_and_transforms, get_mean_std, HFTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.data.datasets import build_bus_dataset            # noqa: E402
from src.evaluation import concept_directions as cd        # noqa: E402

CKPT = os.environ.get("UNIMED_CKPT", str(REPO / "unimed-clip-vit-b16.pt"))
MODEL_NAME = "ViT-B-16-quickgelu"
TEXT_ENC = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
OUT = REPO / "results/cross_modal_probe/embeddings"
SITES = ["busi", "bus_uclm", "bus_bra", "busi_whu", "udiat", "breast"]


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main() -> None:
    device = _device()
    print(f"device: {device}  ckpt: {CKPT}")
    OUT.mkdir(parents=True, exist_ok=True)
    default = yaml.safe_load((REPO / "configs/default.yaml").read_text())
    bank, paths = default["birads_concept_bank"], default["bus_datasets"]

    mean, std = get_mean_std()
    model, _, preprocess = create_model_and_transforms(
        MODEL_NAME, CKPT, precision="fp32", device=device,
        force_quick_gelu=True, mean=mean, std=std, inmem=True,
        text_encoder_name=TEXT_ENC,
    )
    model.eval()
    tokenizer = HFTokenizer(TEXT_ENC, context_length=256)

    @torch.no_grad()
    def encode_text(prompts: list[str]) -> np.ndarray:
        toks = torch.cat([tokenizer(p).to(device) for p in prompts], dim=0)
        f = model.encode_text(toks)
        return (f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy()

    @torch.no_grad()
    def encode_images(loader) -> tuple[np.ndarray, np.ndarray]:
        feats, labels = [], []
        for batch in loader:
            f = model.encode_image(batch["image"].to(device))
            feats.append((f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy())
            labels.append(batch["label"].numpy())
        return np.concatenate(feats), np.concatenate(labels)

    # text: enumerate the exact prompt strings the pipeline will request
    groups = (cd.concept_prompts_from_config(bank),
              cd.reference_prompts_from_config(bank),
              cd.placebo_prompts_from_config(bank),
              cd.malignancy_synonym_prompts_from_config(bank))
    seen: list[str] = []
    for group in groups:
        for pn in group.values():
            for s in list(pn["pos"]) + list(pn["neg"]):
                if s not in seen:
                    seen.append(s)
    print(f"encoding {len(seen)} unique prompts ...")
    np.savez(OUT / "unimed_clip_text.npz",
             prompts=np.array(seen, dtype=object),
             embeddings=encode_text(seen).astype(np.float32))

    # images: one cache per site
    for site in SITES:
        out_f = OUT / f"unimed_clip_image_{site}.npz"
        if out_f.exists():
            print(f"  {site:<10} (cached, skip)")
            continue
        ds = build_bus_dataset(site, Path(paths[site]).expanduser(), transform=preprocess)
        emb, lab = encode_images(DataLoader(ds, batch_size=32, shuffle=False, num_workers=0))
        # Canonical sample ids (loader order == embedding-row order, shuffle=False)
        # so consumers can join to the WP1 cohort manifest without a sidecar
        # re-enumeration (src/data/cohort.py). Older caches without this key fall
        # back to the validated sidecar path.
        sids = np.array([f"{site}/{Path(s[0]).name}" for s in ds.samples], dtype=object)
        np.savez(out_f, embeddings=emb.astype(np.float32), labels=lab.astype(np.int64),
                 sample_ids=sids)
        print(f"  {site:<10} {emb.shape}  benign+malignant={int(np.isin(lab,[0,1]).sum())}")
    print("DONE")


if __name__ == "__main__":
    main()
