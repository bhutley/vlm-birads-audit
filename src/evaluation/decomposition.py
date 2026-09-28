"""Per-case decomposition of a linear probe's decision in a BI-RADS subspace.

This module implements the formalism in the proposal (§5, revised after the
analytical pressure-test). Given

  * frozen image embeddings ``Z`` (n, d), L2-normalised,
  * a linear probe ``g(z) = w·z + b`` (benign/malignant logit), and
  * a clinical subspace ``C = col(B)`` spanned by BI-RADS concept directions,

it computes the quantities the paper is actually about, and *only* those that
survive the degeneracy analysis:

  * ``decompose`` — the per-case additive split
        g(z) - b = (P_C w)·z  +  w_perp·z   =   clin + resid
    and the per-case **clinical share** s(z) = |clin| / (|clin| + |resid|).
    This is the per-case signal (it varies case to case).

  * ``rho`` / ``grounding_excess`` — the **model-level** geometry
        rho = ||P_C w||^2 / ||w||^2 = cos^2(w, C),
    reported as excess over a random-subspace null (E[rho] ~ m/d), because a
    random m-dim subspace already captures ~m/d of any w. rho does NOT depend on
    z and therefore says nothing per case — it compares models/sites/backbones.

  * ``flip_distances`` / ``minimal_flip_edits`` — provided ONLY to demonstrate
    the degeneracy that killed the original idea: the constrained flip-distance
    d_C is a fixed multiple (1/sqrt(rho)) of the margin d_full, so it ranks cases
    identically to the classifier probability; and the minimum-norm counterfactual
    edit direction is identical (collinear with P_C w) for every case. Do not use
    these as a per-case score — see the proposal's "Resolved by analysis" box.

All functions are pure NumPy and accept either NumPy arrays or anything
``np.asarray`` can convert (e.g. detached torch tensors), so they are unit-testable
without a GPU, a model, or any data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ArrayLike = "np.ndarray"


def _np(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def orthonormal_basis(M: np.ndarray, context: str = "basis", tol: float = 1e-8) -> np.ndarray:
    """QR-orthonormalise the columns of ``M`` (d, m), failing fast if rank-deficient.

    Reduced QR returns ``m`` orthonormal columns even when the input columns are
    linearly dependent — silently injecting arbitrary directions outside their
    span, which would inflate any subspace quantity (rho, clinical share). We
    inspect the R-diagonal and raise rather than return a basis for the wrong
    subspace. Use this everywhere a *meaningful* span is orthonormalised (concept
    basis, placebo subspaces); isotropic-Gaussian draws are full-rank a.s. and
    don't need it.
    """
    M = _np(M)
    Q, R = np.linalg.qr(M)
    m = M.shape[1]
    rank = int(np.count_nonzero(np.abs(np.diag(R)) > tol))
    if rank < m:
        raise ValueError(
            f"{context}: directions are rank-deficient (rank {rank} < {m}); "
            "reduced QR would inject arbitrary non-spanning axes. "
            "Check for duplicate or collinear directions."
        )
    return Q


def project_onto_subspace(vec: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Orthogonal projection of ``vec`` (d,) onto ``col(B)`` (B is d x m).

    Uses least squares, so it is correct whether or not B's columns are
    orthonormal. Returns the projected vector P_C @ vec in R^d.
    """
    vec = _np(vec)
    B = _np(B)
    coef, *_ = np.linalg.lstsq(B, vec, rcond=None)
    return B @ coef


def rho(w: np.ndarray, B: np.ndarray) -> float:
    """Fraction of the decision direction lying in the clinical subspace.

    rho = ||P_C w||^2 / ||w||^2 = cos^2(w, C) in [0, 1]. Basis-invariant
    (depends only on col(B)). Must be read against ``random_subspace_rho``.
    """
    w = _np(w)
    pw = project_onto_subspace(w, B)
    wn = float(w @ w)
    if wn == 0.0:
        return float("nan")
    return float((pw @ pw) / wn)


def random_subspace_rho(
    w: np.ndarray, m: int, n_trials: int = 200, seed: int = 0
) -> np.ndarray:
    """Null distribution of rho for a random m-dimensional subspace.

    A random m-dim subspace captures ~m/d of any fixed w in expectation; this
    returns the empirical null so callers can report rho_real - rho_random.
    """
    w = _np(w)
    d = w.shape[0]
    rng = np.random.default_rng(seed)
    out = np.empty(n_trials, dtype=np.float64)
    for i in range(n_trials):
        Br, _ = np.linalg.qr(rng.standard_normal((d, m)))
        out[i] = rho(w, Br)
    return out


def grounding_excess(
    w: np.ndarray, B: np.ndarray, n_trials: int = 200, seed: int = 0
) -> dict:
    """rho relative to the random-subspace null (the publishable quantity).

    Returns rho_real, the null mean/std, the excess, and a z-score of the excess.
    ``B`` should have the same number of columns as the concept set being tested;
    hold m fixed when comparing backbones/sites.
    """
    B = _np(B)
    m = B.shape[1]
    r = rho(w, B)
    null = random_subspace_rho(w, m, n_trials=n_trials, seed=seed)
    mu, sd = float(null.mean()), float(null.std())
    z = float((r - mu) / sd) if sd > 0 else float("nan")
    # one-sided (upper-tail) permutation p-value: how often the null reaches rho
    p = float((1 + int((null >= r).sum())) / (len(null) + 1))
    return {
        "rho": r,
        "null_mean": mu,
        "null_std": sd,
        "excess": r - mu,
        "excess_z": z,
        "p_value": p,
        "m": m,
    }


def subspace_rho_null(
    w: np.ndarray, dirs: np.ndarray, m: int, n_trials: int = 200, seed: int = 0
) -> np.ndarray:
    """Null distribution of rho from random m-subsets of a fixed pool of directions.

    ``dirs`` is (P, d): a pool of P >= m real unit directions (e.g. the placebo
    concept axes). Each trial draws m of them without replacement, orthonormalises
    (QR) and computes rho(w, .). Unlike :func:`random_subspace_rho` (isotropic
    Gaussian), this is the *text-direction* null: it asks whether w aligns with the
    clinical subspace more than with an equally text-derived but NON-diagnostic
    subspace of the same size, controlling for prompt-difference geometry / the
    modality gap rather than only for dimension counting (m/d).
    """
    w = _np(w)
    dirs = _np(dirs)
    P = dirs.shape[0]
    if P < m:
        raise ValueError(f"need at least m={m} placebo directions, got P={P}")
    rng = np.random.default_rng(seed)
    out = np.empty(n_trials, dtype=np.float64)
    for i in range(n_trials):
        sel = rng.choice(P, size=m, replace=False)
        Bsub = orthonormal_basis(dirs[sel].T, "placebo subspace null")  # (d, m)
        out[i] = rho(w, Bsub)
    return out


def placebo_excess(
    w: np.ndarray, B: np.ndarray, placebo_dirs: np.ndarray, n_trials: int = 200, seed: int = 0
) -> dict:
    """rho relative to the placebo (text-direction) null — the stronger control.

    Same dict shape as :func:`grounding_excess`, but the null is m-subsets of real
    non-diagnostic placebo axes (via :func:`subspace_rho_null`) rather than
    isotropic Gaussian subspaces. ``m`` is taken from ``B`` and held fixed. Report
    this ALONGSIDE ``grounding_excess``: clearing the m/d floor shows the subspace
    is non-trivial; clearing the placebo null shows the grounding is specific to
    BI-RADS content, not generic text-direction geometry.
    """
    B = _np(B)
    placebo_dirs = _np(placebo_dirs)
    m = B.shape[1]
    r = rho(w, B)
    null = subspace_rho_null(w, placebo_dirs, m, n_trials=n_trials, seed=seed)
    mu, sd = float(null.mean()), float(null.std())
    z = float((r - mu) / sd) if sd > 0 else float("nan")
    # one-sided (upper-tail) permutation p-value vs the placebo subspace null
    p = float((1 + int((null >= r).sum())) / (len(null) + 1))
    return {
        "rho": r,
        "placebo_mean": mu,
        "placebo_std": sd,
        "excess": r - mu,
        "excess_z": z,
        "p_value": p,
        "m": m,
        "n_placebo": int(placebo_dirs.shape[0]),
    }


def direction_excess_stats(
    directions: np.ndarray,
    B: np.ndarray,
    null_mean: float,
    placebo_dirs: np.ndarray | None = None,
    n_trials: int = 200,
    seed: int = 0,
) -> dict:
    """Grounding excess for several probe directions (e.g. CV-fold directions).

    A robustness check that the model-level grounding (``rho`` excess) is not an
    artefact of fitting ``w`` once on the full site. Pass the decision directions
    from k cross-validation folds — each fit on a disjoint training split — as
    ``directions`` (k, d). Returns the per-direction ``rho`` and excess plus their
    mean/std/min/max, so a reviewer can see whether *every* held-out-fold direction
    still lands in the clinical subspace.

    The two nulls are treated differently, because their means behave differently
    in ``w``:

      * The **random-subspace** null mean is ~w-independent (E[rho_random]≈m/d by
        rotational invariance), so a single ``null_mean`` from the full-site fit is
        reused for every direction.
      * The **placebo** null mean is NOT w-independent: it is an average of
        ``rho(w, .)`` over m-subsets of a fixed, *anisotropic* pool, so it moves
        with the direction. When ``placebo_dirs`` (the P×d placebo pool) is given,
        the placebo mean is recomputed per direction via :func:`subspace_rho_null`.
        Do **not** pass a single scalar placebo mean fit to a different ``w``.

    Pure NumPy. ``n_trials``/``seed`` control the per-direction placebo-null draws
    (only used when ``placebo_dirs`` is supplied).
    """
    W = _np(directions)
    if W.ndim == 1:
        W = W[None, :]
    B = _np(B)
    m = B.shape[1]
    rhos = np.array([rho(w, B) for w in W], dtype=np.float64)
    er = rhos - float(null_mean)
    out = {
        "n_dirs": int(W.shape[0]),
        "rho": rhos.tolist(),
        "rho_mean": float(rhos.mean()),
        "excess_rand": er.tolist(),
        "excess_rand_mean": float(er.mean()),
        "excess_rand_std": float(er.std()),
        "excess_rand_min": float(er.min()),
        "excess_rand_max": float(er.max()),
    }
    if placebo_dirs is not None:
        placebo_dirs = _np(placebo_dirs)
        # w-dependent null: recompute the placebo mean for each direction rather
        # than subtracting one full-site scalar (see docstring; this is the bug the
        # reviewer caught — the placebo mean is not invariant to w like m/d is).
        plac_means = np.array(
            [subspace_rho_null(w, placebo_dirs, m, n_trials=n_trials, seed=seed).mean()
             for w in W],
            dtype=np.float64,
        )
        ep = rhos - plac_means
        out.update({
            "placebo_mean": plac_means.tolist(),
            "excess_plac": ep.tolist(),
            "excess_plac_mean": float(ep.mean()),
            "excess_plac_std": float(ep.std()),
            "excess_plac_min": float(ep.min()),
            "excess_plac_max": float(ep.max()),
        })
    return out


def leave_one_out_placebo_excess(
    w: np.ndarray,
    B: np.ndarray,
    placebo_dirs: np.ndarray,
    names: list | None = None,
    n_trials: int = 200,
    seed: int = 0,
) -> dict:
    """Placebo grounding excess under the full pool and each leave-one-out pool.

    Robustness check (P3): is the placebo-null verdict driven by one particular
    non-diagnostic descriptor? Recomputes :func:`placebo_excess` with the full pool
    of ``P`` axes and with each single axis dropped (``P`` pools of ``P-1`` axes).
    Returns ``full`` (excess on the full pool), ``loo`` (dropped-axis -> excess),
    and the ``loo_min``/``loo_max`` envelope. A collapse (or a grounding) that holds
    across every drop is not an artefact of any single placebo axis. Pure NumPy.
    """
    placebo_dirs = _np(placebo_dirs)
    P = placebo_dirs.shape[0]
    full = placebo_excess(w, B, placebo_dirs, n_trials=n_trials, seed=seed)["excess"]
    loo = {}
    for j in range(P):
        sub = np.delete(placebo_dirs, j, axis=0)
        key = names[j] if names is not None else j
        loo[key] = placebo_excess(w, B, sub, n_trials=n_trials, seed=seed)["excess"]
    vals = list(loo.values())
    return {
        "full": full,
        "loo": loo,
        "loo_min": float(min(vals)),
        "loo_max": float(max(vals)),
    }


def magnitude_gate(excess: float, m: int, d: int, k: float = 2.0) -> dict:
    """Pre-registered magnitude gate, applied ALONGSIDE the permutation/Holm test.

    A significant excess is not yet a *meaningful* one: when n is large or the null
    happens to be tightly concentrated (small SD), an excess of trivial magnitude can
    still clear the permutation test (e.g. a +0.006 placebo excess against an SD of
    0.0006 gives z~10). To stop significance-alone from qualifying a cell, we require
    the excess to exceed ``k`` times the dimension-matched random-null expectation
    ``E[rho_random] = m/d``. The factor m/d is the per-backbone null floor; gating on a
    multiple of it is power-independent (it does not look at the null SD) and stays
    comparable across embedding dimensions d. Default ``k=2``.

    Returns ``delta_min`` (the threshold) and ``passed`` (excess > delta_min).
    """
    delta_min = float(k) * m / d
    return {"delta_min": float(delta_min), "passed": bool(excess > delta_min)}


@dataclass
class Decomposition:
    """Result of :func:`decompose` (all arrays are length-n unless noted)."""

    clin: np.ndarray  # (n,) clinically-nameable logit content (P_C w)·z
    resid: np.ndarray  # (n,) un-nameable residual w_perp·z
    clinical_share: np.ndarray  # (n,) s(z) = |clin| / (|clin| + |resid|)
    logit_centered: np.ndarray  # (n,) g(z) - b = clin + resid
    named_contributions: np.ndarray  # (n, m) per raw concept axis (non-additive)
    names: list  # length-m concept names (or indices)
    entanglement_gap: np.ndarray  # (n,) clin - sum_k named_contributions[:, k]


def decompose(
    Z: np.ndarray,
    w: np.ndarray,
    b: float,
    B: np.ndarray,
    names: list | None = None,
    B_named: np.ndarray | None = None,
) -> Decomposition:
    """Per-case additive split of the decision into clinical vs residual content.

    The basis-invariant parts (``clin``, ``resid``, ``clinical_share``) use the
    orthogonal projection onto col(B) and are well defined regardless of how the
    concept directions overlap. ``named_contributions[:, k] = (z·c_k)(w·c_k)``
    is interpretable per named axis, but does NOT sum to ``clin`` when the axes
    are correlated — the gap is returned as ``entanglement_gap`` to make that
    explicit (proposal §8.2).

    ``B`` defines the *subspace* (may be QR-orthonormalised). The named columns,
    however, must be the *raw* BI-RADS directions to carry their names: pass them
    as ``B_named`` when ``B`` is orthonormalised (otherwise the named columns are
    QR-mixed and no longer correspond to margin/shape/… ). ``B_named`` defaults to
    ``B`` for the common case where the caller passes raw unit axes.
    """
    Z = _np(Z)
    w = _np(w)
    B = _np(B)
    m = B.shape[1]
    if names is None:
        names = [f"c{k}" for k in range(m)]

    proj_w = project_onto_subspace(w, B)  # P_C w
    clin = Z @ proj_w
    logit_centered = Z @ w  # g(z) - b
    resid = logit_centered - clin

    denom = np.abs(clin) + np.abs(resid)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(denom > 0, np.abs(clin) / denom, np.nan)

    named_basis = B if B_named is None else _np(B_named)  # raw axes for names
    Bu = named_basis / (np.linalg.norm(named_basis, axis=0, keepdims=True) + 1e-12)
    named = (Z @ Bu) * (w @ Bu)  # (n, m)
    gap = clin - named.sum(axis=1)

    return Decomposition(
        clin=clin,
        resid=resid,
        clinical_share=share,
        logit_centered=logit_centered,
        named_contributions=named,
        names=list(names),
        entanglement_gap=gap,
    )


def flip_distances(Z: np.ndarray, w: np.ndarray, b: float, B: np.ndarray) -> dict:
    """Margin vs clinical-subspace flip-distance (degeneracy demonstration).

    Returns ``d_full = |g|/||w||``, ``d_C = |g|/||P_C w||`` and their ratio.
    The ratio is constant across cases (= 1/sqrt(rho)), proving d_C carries no
    per-case information beyond the classifier margin. For demonstration/figures,
    not as a score.
    """
    Z = _np(Z)
    w = _np(w)
    g = np.abs(Z @ w + b)
    proj_w = project_onto_subspace(w, B)
    d_full = g / (np.linalg.norm(w) + 1e-12)
    d_C = g / (np.linalg.norm(proj_w) + 1e-12)
    return {"d_full": d_full, "d_C": d_C, "ratio": d_C / (d_full + 1e-12)}


def minimal_flip_edits(Z: np.ndarray, w: np.ndarray, b: float, B: np.ndarray) -> np.ndarray:
    """Minimum-norm edit within col(B) that reaches the boundary, per case.

    delta_i = -(w·z_i + b) * P_C w / ||P_C w||^2. Returns (n, d). NOTE: every
    row is collinear with ``P_C w`` — i.e. the same explanation for every case
    (degenerate). Kept only to substantiate that claim in the paper.
    """
    Z = _np(Z)
    w = _np(w)
    g = Z @ w + b
    proj_w = project_onto_subspace(w, B)
    scale = (proj_w @ proj_w) + 1e-12
    return (-g[:, None]) * proj_w[None, :] / scale
