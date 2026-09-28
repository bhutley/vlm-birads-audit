"""Experiment E1: BI-RADS concept directions — sanity & modality-gap report.

Author: Brett Hutley
Hypothesis: Text-derived BI-RADS directions are (a) only mildly entangled, (b)
    partly aligned with the learned benign/malignant decision direction, and
    (c) sufficiently inside the image-embedding manifold to be usable despite the
    modality gap. rho of the clinical subspace should exceed the random-subspace
    null even if the absolute value is small.

What it does (all on the frozen backbone, no fine-tuning):
    1. Build a unit direction per BI-RADS concept from paired prompt sets,
       averaged over carrier templates x descriptors (src.evaluation.concept_directions).
    2. Direction sanity: pairwise cosine matrix (entanglement preview), basis rank.
    3. Modality gap: image-centroid vs concept-text-centroid offset + cosine.
    4. Text-direction validity in image space: fraction of each direction retained
       when projected onto the image-embedding PCA subspace (the §8.1 mitigation).
    5. Grounding preview: train a quick benign-vs-malignant logistic probe on the
       image embeddings -> decision vector w; report rho(w, B) vs a random-subspace
       null, and the cosine between the text "malignancy" reference axis and w.

Usage:
    PYTHONHASHSEED=0 python -m src.experiments.run_concept_directions

Note: image-derived CAVs and their alignment with the text directions need
per-feature BI-RADS lexicon labels (BrEaST); those are wired in E2/E3.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import BUSIDataset
from src.evaluation import concept_directions as cd
from src.evaluation import decomposition as dec
from src.models.clip_model import BiomedCLIPZeroShot, CLIPZeroShot
from src.utils.config import load_experiment_config
from src.utils.reproducibility import set_all_seeds
from src.utils.results import save_results

EXPERIMENT_NAME = "concept_directions"

BENIGN, MALIGNANT = 0, 1  # BUSI label scheme (normal=2 is excluded here)


# --------------------------------------------------------------------------- #
# Model / embeddings
# --------------------------------------------------------------------------- #
def _load_model(backbone: str, config: dict):
    if backbone == "biomedclip":
        return BiomedCLIPZeroShot()
    if backbone == "clip":
        m = config["models"]["clip"]
        return CLIPZeroShot(model_name=m["name"], pretrained=m["pretrained"])
    raise ValueError(f"unknown backbone: {backbone}")


def _busi_embeddings(model, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (embeddings, labels) for BUSI, reusing the cache if present."""
    cache = PROJECT_ROOT / config["concept_directions"]["embedding_cache"] / "image_busi.npz"
    backbone = config["concept_directions"]["backbone"]
    if cache.exists() and backbone == "biomedclip":
        print(f"  reusing cached embeddings: {cache}")
        d = np.load(cache)
        return d["embeddings"].astype(np.float64), d["labels"]

    print("  no cache (or non-biomedclip backbone) — embedding BUSI on the fly")
    busi_root = Path(config["data"]["busi_root"]).expanduser()
    ds = BUSIDataset(busi_root, transform=model.preprocess)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    feats, labels = [], []
    for batch in loader:
        feats.append(model.encode_image(batch["image"]).detach().cpu().numpy())
        labels.append(batch["label"].numpy())
    return np.concatenate(feats).astype(np.float64), np.concatenate(labels)


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def run_experiment(config: dict) -> dict:
    repro = config["reproducibility"]
    set_all_seeds(repro["global_seed"], deterministic=repro["deterministic_algorithms"])
    cfg = config["concept_directions"]
    bank = config["birads_concept_bank"]

    print(f"Loading backbone: {cfg['backbone']}")
    model = _load_model(cfg["backbone"], config)
    print(f"  device: {model.get_device()}")

    # --- image embeddings (benign + malignant only) ---
    emb, labels = _busi_embeddings(model, config)
    keep = np.isin(labels, [BENIGN, MALIGNANT])
    Z, y = emb[keep], labels[keep].astype(int)
    print(f"  BUSI benign+malignant: {Z.shape[0]} images, dim={Z.shape[1]}")

    # --- concept directions (text-derived) ---
    embed_fn = lambda texts: cd.encode_texts(model, texts)
    concept_prompts = cd.concept_prompts_from_config(bank)
    directions = cd.build_text_directions(concept_prompts, embed_fn)
    B, names = cd.stack_basis(directions, orthonormalize=cfg["orthonormalize_basis"])
    B_raw = np.stack([directions[n] for n in names], axis=1)  # un-orthonormalised
    print(f"  built {len(names)} concept directions: {names}")
    # placebo (text-direction) null pool: same machinery, non-diagnostic axes
    placebo_dirs = cd.build_text_directions(cd.placebo_prompts_from_config(bank), embed_fn)
    placebo_pool = np.stack(list(placebo_dirs.values()), axis=0)  # (P, d)

    # --- (2) direction sanity: pairwise cosine of raw directions ---
    cos = B_raw.T @ B_raw
    offdiag = cos[~np.eye(len(names), dtype=bool)]
    rank = int(np.linalg.matrix_rank(B_raw))

    # --- (3) modality gap (image embeddings vs all concept-prompt embeddings) ---
    all_prompts = [p for pn in concept_prompts.values() for p in (pn["pos"] + pn["neg"])]
    text_embs = embed_fn(all_prompts)
    mgap = cd.modality_gap(Z, text_embs)

    # --- (4) text-direction validity in image space (PCA-subspace retention) ---
    k = int(cfg["pca_components"])
    retained = {n: cd.project_to_image_subspace(directions[n], Z, k)[1] for n in names}

    # --- (5) grounding preview: quick benign/malignant probe -> w ---
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    clf.fit(Z, y)
    w = clf.coef_.ravel().astype(np.float64)
    b = float(clf.intercept_[0])
    ground = dec.grounding_excess(w, B, n_trials=int(cfg["null_trials"]),
                                  seed=repro["global_seed"])
    placebo = dec.placebo_excess(w, B, placebo_pool, n_trials=int(cfg["null_trials"]),
                                 seed=repro["global_seed"])

    # alignment of the text "malignancy" reference axis with the learned w
    ref_align = {}
    for rname, rp in cd.reference_prompts_from_config(bank).items():
        rdir = cd.paired_text_direction(embed_fn(rp["pos"]), embed_fn(rp["neg"]))
        ref_align[rname] = float(rdir @ (w / (np.linalg.norm(w) + 1e-12)))

    return {
        "backbone": cfg["backbone"],
        "n_images": int(Z.shape[0]),
        "embed_dim": int(Z.shape[1]),
        "concepts": names,
        "direction_sanity": {
            "pairwise_cosine_abs_mean": float(np.abs(offdiag).mean()),
            "pairwise_cosine_abs_max": float(np.abs(offdiag).max()),
            "basis_rank": rank,
            "cosine_matrix": cos.tolist(),
        },
        "modality_gap": {
            "gap_norm": mgap["gap_norm"],
            "centroid_cosine": mgap["centroid_cosine"],
        },
        "image_subspace_retention": {n: float(v) for n, v in retained.items()},
        "grounding": ground,                      # rho, null_mean, excess, excess_z, m
        "grounding_placebo": placebo,             # rho vs placebo text-direction null
        "reference_axis_alignment": ref_align,    # cos(text malignancy axis, w)
    }


def format_summary(results: dict, config: dict) -> list[str]:
    r = results
    L = [
        "=" * 64,
        f"E1 — concept directions & modality gap  [{r['backbone']}]",
        "=" * 64,
        f"images (benign+malignant): {r['n_images']}   dim: {r['embed_dim']}",
        f"concepts ({len(r['concepts'])}): {', '.join(r['concepts'])}",
        "",
        "Direction sanity (entanglement preview):",
        f"  |cos| between concept axes: mean={r['direction_sanity']['pairwise_cosine_abs_mean']:.3f}"
        f"  max={r['direction_sanity']['pairwise_cosine_abs_max']:.3f}"
        f"  rank={r['direction_sanity']['basis_rank']}/{len(r['concepts'])}",
        "",
        "Modality gap (image vs concept-text centroids):",
        f"  gap_norm={r['modality_gap']['gap_norm']:.3f}   centroid_cos={r['modality_gap']['centroid_cosine']:.3f}",
        "",
        "Text-direction retention in image PCA subspace (higher = more usable):",
    ]
    for n, v in r["image_subspace_retention"].items():
        L.append(f"  {n:<14} {v:.3f}")
    g = r["grounding"]
    pl = r.get("grounding_placebo", {})
    L += [
        "",
        "Grounding preview (benign/malignant probe w vs BI-RADS subspace):",
        f"  rho={g['rho']:.4f}   null={g['null_mean']:.4f}+-{g['null_std']:.4f}"
        f"   excess={g['excess']:+.4f}  (z={g['excess_z']:+.1f})",
    ]
    if pl:
        L.append(
            f"  vs placebo ({pl.get('n_placebo', '?')} non-diagnostic axes): "
            f"null={pl['placebo_mean']:.4f}+-{pl['placebo_std']:.4f}"
            f"   excess={pl['excess']:+.4f}  (z={pl['excess_z']:+.1f})")
    L.append("  reference-axis alignment cos(text malignancy axis, w):")
    for n, v in r["reference_axis_alignment"].items():
        L.append(f"    {n:<14} {v:+.3f}")
    L += [
        "",
        "Reading: excess>0 (z>>0) over BOTH the random subspace AND the placebo",
        "text-direction null => the decision is BI-RADS-grounded in content, not just",
        "in prompt-difference geometry; low retention/centroid_cos warns the modality",
        "gap is degrading text directions (fall back to image CAVs).",
    ]
    return L


def main():
    config = load_experiment_config(EXPERIMENT_NAME)
    results = run_experiment(config)
    summary = format_summary(results, config)
    for line in summary:
        print(line)
    save_results(EXPERIMENT_NAME, results, config, summary_lines=summary)


if __name__ == "__main__":
    main()
