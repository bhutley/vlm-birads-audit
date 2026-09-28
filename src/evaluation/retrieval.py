"""Prototype-anchored counterfactuals and nearest-neighbour plausibility (E5).

The minimum-norm counterfactual is degenerate for a linear probe (every case gets
the same edit direction; see decomposition.py / proposal §5). So the per-case
counterfactual here is *prototype-anchored*: a lesion's counterfactual is its
nearest opposite-class REAL lesion. The "edit" z' - z then has a genuine per-case
direction, which we read in BI-RADS terms, and the retrieval distance is an honest,
generation-free plausibility gate (every shown image is real).

Functions are pure NumPy on L2-normalised embeddings and unit-testable without a
model or data.
"""

from __future__ import annotations

import numpy as np


def _np(x):
    return np.asarray(x, dtype=np.float64)


def nearest_opposite(Z: np.ndarray, y: np.ndarray):
    """For each case, the nearest opposite-class case by cosine distance.

    Returns ``(idx, dist)`` where ``idx[i]`` is the row of the nearest case with a
    different label and ``dist[i] = 1 - cos`` is the cosine distance to it.
    """
    Z = _np(Z)
    y = np.asarray(y)
    sim = Z @ Z.T
    idx = np.empty(len(y), dtype=int)
    dist = np.empty(len(y), dtype=float)
    for i in range(len(y)):
        opp = np.where(y != y[i])[0]
        j = opp[np.argmax(sim[i, opp])]
        idx[i] = j
        dist[i] = 1.0 - float(sim[i, j])
    return idx, dist


def nearest_any(queries: np.ndarray, pool: np.ndarray, exclude: np.ndarray | None = None):
    """Nearest pool row per query (cosine). ``exclude[i]`` is masked out (e.g. self).

    Returns ``(idx, dist)`` with cosine distance ``1 - cos``.
    """
    queries = _np(queries)
    pool = _np(pool)
    sim = queries @ pool.T
    if exclude is not None:
        sim[np.arange(len(queries)), np.asarray(exclude)] = -np.inf
    idx = np.argmax(sim, axis=1)
    dist = 1.0 - sim[np.arange(len(queries)), idx]
    return idx, dist


def edit_concept_fraction(edit: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Fraction of each edit vector that lies in the clinical subspace col(B).

    ``rho_edit[i] = ||proj_C edit_i||^2 / ||edit_i||^2`` — how much of what
    distinguishes a lesion from its opposite-class prototype is BI-RADS-nameable.
    Assumes ``B`` has orthonormal columns (use stack_basis(orthonormalize=True)).
    """
    edit = _np(edit)
    B = _np(B)
    eb = edit @ B                       # coords in the orthonormal basis
    num = (eb ** 2).sum(axis=1)
    den = (edit ** 2).sum(axis=1) + 1e-12
    return num / den


def edit_named_profile(edit: np.ndarray, B_unit: np.ndarray) -> np.ndarray:
    """Per-named-axis component of each edit: ``(edit_i . c_k)`` for unit c_k.

    ``B_unit`` has the *raw* unit concept directions as columns (interpretable but
    not orthogonal). Returns (n, m). Positive = the edit moves toward the
    suspicious pole of that axis.
    """
    return _np(edit) @ _np(B_unit)


def birads_shift(y: np.ndarray, birads: np.ndarray, idx: np.ndarray, benign_max: int = 3) -> dict:
    """Radiologist BI-RADS category shift from each query to its opposite-class prototype.

    An external, label-grounded faithfulness check: a faithful benign->malignant
    counterfactual should retrieve a prototype a radiologist rated *more* suspicious
    (higher BI-RADS), and a malignant->benign one *less* suspicious. ``y`` in
    {0=benign, 1=malignant}; ``birads`` the per-case radiologist category; ``idx[i]``
    the retrieved opposite-class prototype for query i (e.g. from
    :func:`nearest_opposite`, or a random-opposite baseline). ``benign_max`` is the
    benign/suspicious boundary (BI-RADS <= benign_max is benign-range).

    Returns the oriented per-query delta (positive = the prototype moved toward the
    intended class on the BI-RADS scale) plus summary rates: ``concordance_rate`` is
    the fraction whose prototype lands on the intended side of the boundary.
    """
    y = np.asarray(y)
    idx = np.asarray(idx)
    birads = _np(birads)
    sign = np.where(y == 0, 1.0, -1.0)          # benign->mal expects +, mal->ben expects -
    delta = birads[idx] - birads                # raw radiologist-category change
    oriented = sign * delta                      # >0 when prototype moved toward target class
    on_target = np.where(y == 0, birads[idx] > benign_max, birads[idx] <= benign_max)
    return {
        "oriented_delta": oriented,
        "mean_oriented_delta": float(oriented.mean()),
        "concordance_rate": float(on_target.mean()),
        "mean_delta_benign_to_malignant": float(delta[y == 0].mean()) if (y == 0).any() else float("nan"),
        "mean_delta_malignant_to_benign": float(delta[y == 1].mean()) if (y == 1).any() else float("nan"),
    }


def directional_counterfactual(
    Z: np.ndarray, w: np.ndarray, b: float, u: np.ndarray,
    max_alpha: float = 2.0, steps: int = 21,
) -> np.ndarray:
    """Constructed counterfactual embeddings for nearest-real-image retrieval.

    For each case, step AWAY from its current side of the decision along the
    malignancy axis ``u`` (toward malignant for benign-side cases and vice versa)
    and stop at the smallest step that flips the *post-renormalisation* probe
    sign; if none flips within ``max_alpha``, take the largest step. Renormalised
    onto the unit sphere. This is the honest, generation-free "what would this
    look like if flipped" construction — paired with :func:`nearest_any` it yields
    a real image to display, never a synthesised one.
    """
    Z = _np(Z); w = _np(w); u = _np(u)
    # Orient the malignancy axis to agree with the probe direction, so stepping
    # "toward the opposite side" below is toward the opposite *probe* side
    # regardless of how the text axis happens to be signed. An anti-aligned u
    # would otherwise step the wrong way and never flip the probe (flip_rate 0).
    if float(u @ w) < 0.0:
        u = -u
    cur = np.sign(Z @ w + b)              # current side per case (+/-)
    cur[cur == 0] = 1.0
    alphas = np.linspace(0.0, max_alpha, steps)[1:]
    out = Z.copy()
    for i in range(len(Z)):
        direction = -cur[i] * u           # push toward the opposite side
        chosen = Z[i] + max_alpha * direction   # fallback: largest step
        for a in alphas:
            zc = Z[i] + a * direction
            # the probe is read on the renormalised embedding (b is not scaled
            # by ||zc||), so the flip must be tested post-renormalisation too
            zc_hat = zc / (np.linalg.norm(zc) + 1e-12)
            if np.sign((zc_hat @ w + b)) != cur[i]:
                chosen = zc
                break
        out[i] = chosen
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)
