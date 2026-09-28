"""Construct BI-RADS concept directions in a frozen VLM's embedding space.

Two ways to build a unit direction for a clinical concept (e.g. "spiculated vs
circumscribed margin") in the shared CLIP/BiomedCLIP space:

  * **text-derived** (cheap, no labels; cf. TextCAVs): the normalised difference
    between the mean embeddings of positive and negative *prompt* sets, averaged
    over many phrasings for robustness. See :func:`build_text_directions`.

  * **image-derived (CAV)** (needs concept-labelled images, e.g. BrEaST's
    BI-RADS lexicon; cf. TCAV): a linear concept classifier on *image*
    embeddings. See :func:`cav_mean_difference` (NumPy) and :func:`cav_logistic`
    (scikit-learn).

The text route depends on text-difference vectors being meaningful directions on
the *image* manifold despite the modality gap (proposal §8.1); this module
therefore also provides :func:`modality_gap` (a diagnostic) and
:func:`project_to_image_subspace` (a mitigation), and the whole pipeline can fall
back to image-derived CAVs, which live in image space by construction.

Embedding-only helpers are pure NumPy and unit-testable without a model. The one
function that touches the encoder, :func:`encode_texts`, imports torch lazily.
"""

from __future__ import annotations

import itertools

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _row_normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


# --------------------------------------------------------------------------- #
# Encoder access (lazy torch import)
# --------------------------------------------------------------------------- #
def encode_texts(model, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """L2-normalised text embeddings for a list of prompts (backbone-agnostic).

    Routes through the wrapper's own ``set_prompts`` so it works for every model
    family in this repo — open_clip wrappers (CLIP, BiomedCLIP, SigLIP) *and*
    the HuggingFace PubMedCLIP wrapper, whose text path differs. Returns a
    (len(texts), d) NumPy array in input order.
    """
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        # unique keys per chunk; insertion order is preserved by set_prompts
        model.set_prompts({str(j): tx for j, tx in enumerate(chunk)})
        feats = model._text_features.detach().cpu().numpy()
        out.append(feats)
    return np.concatenate(out, axis=0).astype(np.float64)


# --------------------------------------------------------------------------- #
# Prompt assembly from a shared concept bank (configs default.yaml:birads_concept_bank)
# --------------------------------------------------------------------------- #
def expand_prompts(templates: list[str], descriptors: list[str]) -> list[str]:
    """Cross every carrier template with every descriptor."""
    return [tpl.format(d=d) for tpl, d in itertools.product(templates, descriptors)]


def concept_prompts_from_config(bank: dict) -> dict:
    """{name: {"pos": [...], "neg": [...]}} from a concept-bank config block.

    ``bank`` has ``carrier_templates`` and ``concepts: {name: {pos, neg}}``.
    """
    tpl = bank["carrier_templates"]
    return {
        name: {"pos": expand_prompts(tpl, pn["pos"]), "neg": expand_prompts(tpl, pn["neg"])}
        for name, pn in bank["concepts"].items()
    }


def reference_prompts_from_config(bank: dict) -> dict:
    """Same shape as :func:`concept_prompts_from_config` for ``reference_axes``."""
    tpl = bank["carrier_templates"]
    return {
        name: {"pos": expand_prompts(tpl, pn["pos"]), "neg": expand_prompts(tpl, pn["neg"])}
        for name, pn in bank.get("reference_axes", {}).items()
    }


def placebo_prompts_from_config(bank: dict) -> dict:
    """Same shape as :func:`concept_prompts_from_config` for ``placebo_concepts``.

    These are non-diagnostic axes built with the identical machinery; they form the
    text-direction null that controls for "this is just a prompt-difference vector"
    when judging whether the decision direction is BI-RADS-grounded.
    """
    tpl = bank["carrier_templates"]
    return {
        name: {"pos": expand_prompts(tpl, pn["pos"]), "neg": expand_prompts(tpl, pn["neg"])}
        for name, pn in bank.get("placebo_concepts", {}).items()
    }


def malignancy_synonym_prompts_from_config(bank: dict) -> dict:
    """Same shape as :func:`concept_prompts_from_config` for ``malignancy_synonym_concepts``.

    A pool of malignancy-*loaded* but non-BI-RADS-descriptor axes. Where the placebo
    null controls for "any text direction," this null controls for the sharper rival
    "any malignancy-correlated text direction": if the BI-RADS subspace's rho also
    exceeds this null, the grounding is specific to BI-RADS *feature* content rather
    than to generic malignancy semantics that any suspicion-loaded phrase would carry.
    """
    tpl = bank["carrier_templates"]
    return {
        name: {"pos": expand_prompts(tpl, pn["pos"]), "neg": expand_prompts(tpl, pn["neg"])}
        for name, pn in bank.get("malignancy_synonym_concepts", {}).items()
    }


# --------------------------------------------------------------------------- #
# Text-derived directions
# --------------------------------------------------------------------------- #
def paired_text_direction(pos_embs: np.ndarray, neg_embs: np.ndarray) -> np.ndarray:
    """Unit direction = norm( mean(norm(pos)) - mean(norm(neg)) ).

    ``pos_embs`` / ``neg_embs`` are (n_p, d) / (n_n, d) text embeddings for the
    positive and negative phrasings of one concept (e.g. spiculated vs
    circumscribed). Averaging over phrasings reduces prompt-wording variance.
    """
    pos = _row_normalize(pos_embs).mean(axis=0)
    neg = _row_normalize(neg_embs).mean(axis=0)
    return _unit(pos - neg)


def subspace_principal_angles(B1: np.ndarray, B2: np.ndarray) -> np.ndarray:
    """Principal angles (radians, ascending) between the spans of ``B1`` and ``B2``.

    Columns need not be orthonormal -- both are re-orthonormalised via QR first, so
    the result depends only on the two *subspaces*, not on the bases chosen for
    them. Returns ``min(k1, k2)`` angles.

    Small angles are computed from **sines**, not from ``arccos``. Taking
    ``arccos`` of a singular value near 1 amplifies round-off to ``sqrt(eps)``
    (~1.5e-8 rad), so two identical subspaces would report a spurious non-zero
    angle -- and small angles are exactly the regime this is used in (a stable
    perturbed basis sits *close* to the primary one). We therefore use the
    standard hybrid: sines of the residual ``Q_small - Q_big Q_big^T Q_small``
    below 45 degrees, cosines of ``Q1^T Q2`` above it.

    Used to quantify how far a prompt- or template-perturbed clinical basis has
    moved from the primary one (ROB-03): the *largest* angle is the worst-case
    direction the perturbation failed to reproduce.
    """
    Q1, _ = np.linalg.qr(np.asarray(B1, dtype=np.float64))
    Q2, _ = np.linalg.qr(np.asarray(B2, dtype=np.float64))
    k1, k2 = Q1.shape[1], Q2.shape[1]

    cos_s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)        # descending
    theta = np.arccos(np.clip(cos_s, -1.0, 1.0))              # ascending

    # residual of the lower-dimensional basis outside the other's span
    if k1 <= k2:
        resid = Q1 - Q2 @ (Q2.T @ Q1)
    else:
        resid = Q2 - Q1 @ (Q1.T @ Q2)
    sin_s = np.linalg.svd(resid, compute_uv=False)            # descending
    theta_sin = np.arcsin(np.clip(sin_s[::-1], -1.0, 1.0))    # -> ascending

    small = np.clip(cos_s, -1.0, 1.0) > np.sqrt(0.5)          # angle < 45 deg
    return np.where(small, theta_sin[:len(theta)], theta)


def build_text_directions(concept_prompts: dict, embed_fn) -> dict:
    """Build a unit direction per concept from paired prompt sets.

    ``concept_prompts`` maps concept name -> {"pos": [str, ...], "neg": [str, ...]}.
    ``embed_fn`` maps list[str] -> (k, d) embeddings (e.g.
    ``functools.partial(encode_texts, model)``, or a stub in tests).
    Returns {name: unit direction (d,)}.
    """
    directions = {}
    for name, pn in concept_prompts.items():
        pos = embed_fn(list(pn["pos"]))
        neg = embed_fn(list(pn["neg"]))
        directions[name] = paired_text_direction(pos, neg)
    return directions


# --------------------------------------------------------------------------- #
# Image-derived directions (CAVs)
# --------------------------------------------------------------------------- #
def cav_mean_difference(embs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Difference-of-class-means CAV (NumPy, no dependencies).

    ``labels`` is a boolean/0-1 array marking images that exhibit the concept.
    Returns the unit direction from negative-class mean to positive-class mean.
    """
    embs = np.asarray(embs, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    if labels.sum() == 0 or (~labels).sum() == 0:
        raise ValueError("cav_mean_difference needs both positive and negative examples")
    return _unit(embs[labels].mean(axis=0) - embs[~labels].mean(axis=0))


def cav_logistic(embs: np.ndarray, labels: np.ndarray, C: float = 1.0) -> np.ndarray:
    """Logistic-regression CAV (classic TCAV); the unit weight vector.

    Requires scikit-learn. Falls back to a clear error if unavailable.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as e:  # pragma: no cover
        raise ImportError("cav_logistic requires scikit-learn") from e
    clf = LogisticRegression(C=C, max_iter=1000)
    clf.fit(np.asarray(embs, dtype=np.float64), np.asarray(labels).astype(int))
    return _unit(clf.coef_.ravel())


# --------------------------------------------------------------------------- #
# Basis assembly
# --------------------------------------------------------------------------- #
def stack_basis(directions: dict, orthonormalize: bool = True):
    """Stack concept directions into a basis matrix ``B`` (d, m).

    With ``orthonormalize=True`` (QR) the columns are orthonormal and span the
    same subspace, which makes the subspace-level quantities (rho, clinical
    share) clean; note that QR mixes the named axes, so for *named* per-axis
    attribution use the raw (un-orthonormalised) directions instead (see
    :func:`decomposition.decompose`'s ``named_contributions``). Returns
    ``(B, names)``.
    """
    names = list(directions.keys())
    B = np.stack([np.asarray(directions[n], dtype=np.float64) for n in names], axis=1)
    if orthonormalize:
        from src.evaluation.decomposition import orthonormal_basis
        B = orthonormal_basis(B, "concept basis")
    return B, names


# --------------------------------------------------------------------------- #
# Modality-gap diagnostics & mitigation (proposal §8.1)
# --------------------------------------------------------------------------- #
def modality_gap(image_embs: np.ndarray, text_embs: np.ndarray) -> dict:
    """Diagnose the image/text modality gap (Liang et al., 2022).

    Returns the centroid offset vector, its norm, and the cosine between the
    (L2-normalised) image and text centroids. A large offset / low cosine warns
    that text-derived directions may not transfer cleanly to image space.
    """
    img_c = _row_normalize(image_embs).mean(axis=0)
    txt_c = _row_normalize(text_embs).mean(axis=0)
    gap = img_c - txt_c
    cos = float(_unit(img_c) @ _unit(txt_c))
    return {"gap_vector": gap, "gap_norm": float(np.linalg.norm(gap)), "centroid_cosine": cos}


def image_subspace(image_embs: np.ndarray, n_components: int):
    """Top-``n_components`` PCA basis (d, k) of the centred image embeddings."""
    X = np.asarray(image_embs, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    # right singular vectors are the principal axes in feature space
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:n_components].T  # (d, k)


def project_to_image_subspace(direction: np.ndarray, image_embs: np.ndarray, n_components: int):
    """Project a (text-derived) direction onto the image-embedding PCA subspace.

    Modality-gap mitigation: keeps only the part of the direction that lives where
    the image data actually varies. Returns ``(projected_unit_direction,
    retained_fraction)`` where retained_fraction in [0,1] is the squared-norm
    kept by the projection (low values flag a direction that is largely off the
    image manifold).
    """
    direction = np.asarray(direction, dtype=np.float64)
    U = image_subspace(image_embs, n_components)  # (d, k), orthonormal columns
    proj = U @ (U.T @ direction)
    retained = float((proj @ proj) / ((direction @ direction) + 1e-12))
    return _unit(proj), retained
