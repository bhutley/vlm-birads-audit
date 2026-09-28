"""Pure-NumPy statistics for the E6 concept-direction validity checks.

Kept separate from ``run_concept_validity`` so the diagnosis-control statistics
are unit-testable without torch / sklearn / data. Every function takes 1-D NumPy
arrays and follows the E6 convention that a **higher score means the malignant
pole** (feature label ``1``); feature labels use ``-1`` for unlabelled lesions.

The reviewer's worry is that BI-RADS features co-vary with diagnosis, so a
direction that merely tracks benign-vs-malignant could predict a feature it does
not encode. The statistics here separate feature content from diagnosis:

  * :func:`rank_auc` — AUC = P(score[pos] > score[neg]) with midrank ties
    (identical to ``sklearn.metrics.roc_auc_score`` for binary labels).
  * :func:`conditioned_auc` — AUC restricted to **same-diagnosis** pairs
    (within-benign and within-malignant strata pooled by discordant-pair count):
    the axis's feature signal with the diagnosis confound removed.
  * :func:`bootstrap_stat` / :func:`ci_and_p` — case-level bootstrap of any
    index-taking statistic, with a percentile CI and a one-sided tail p.
  * :func:`paired_diff_ci` — paired bootstrap of ``a(idx) - b(idx)`` on the same
    resample (for diagonal-minus-off-diagonal and diagonal-minus-malignancy-axis).
  * :func:`within_diag_perm_p` — one-sided p for the conditioned AUC under
    label permutation *within* each diagnosis stratum (null: no within-diagnosis
    feature signal).
"""

from __future__ import annotations

import numpy as np


def _midranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks of ``x`` with ties assigned their average rank."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    r = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        r[order[i : j + 1]] = (i + j) / 2.0 + 1.0  # average of 1-based ranks i..j
        i = j + 1
    return r


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC of ``scores`` predicting binary ``labels`` (1 = malignant pole).

    Uses the Mann-Whitney rank identity with midrank tie handling, matching
    ``roc_auc_score``. Entries whose label is neither 0 nor 1 (e.g. the ``-1``
    unlabelled sentinel) are ignored, so full-length arrays can be passed. Returns
    NaN if either class is absent.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    pos = labels == 1
    neg = labels == 0
    npos, nneg = int(pos.sum()), int(neg.sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    keep = pos | neg
    ranks = _midranks(scores[keep])  # rank within the labelled subset only
    pos_in_keep = labels[keep] == 1
    return float((ranks[pos_in_keep].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def stratum_aucs(scores: np.ndarray, feat: np.ndarray, diag: np.ndarray) -> dict:
    """Per-diagnosis-stratum AUC of ``scores`` predicting the feature pole.

    ``feat`` uses -1 for unlabelled (dropped). Returns ``{diagnosis: auc}`` with
    NaN where a stratum lacks both feature poles.
    """
    scores = np.asarray(scores, dtype=np.float64)
    feat = np.asarray(feat)
    diag = np.asarray(diag)
    m = feat >= 0
    scores, feat, diag = scores[m], feat[m], diag[m]
    return {int(g): rank_auc(scores[diag == g], feat[diag == g]) for g in np.unique(diag)}


def conditioned_auc(scores: np.ndarray, feat: np.ndarray, diag: np.ndarray) -> float:
    """Diagnosis-conditioned AUC: AUC over same-diagnosis, discordant-feature pairs.

    Each stratum's AUC is weighted by its number of discordant pairs
    (n_malignant_pole * n_benign_pole) and pooled — the probability that, among
    lesions sharing a diagnosis, the malignant-pole one has the higher projection.
    ``feat`` uses -1 for unlabelled (dropped). NaN if no stratum has both poles.
    """
    scores = np.asarray(scores, dtype=np.float64)
    feat = np.asarray(feat)
    diag = np.asarray(diag)
    m = feat >= 0
    scores, feat, diag = scores[m], feat[m], diag[m]
    num = den = 0.0
    for g in np.unique(diag):
        sub = diag == g
        f = feat[sub]
        npos, nneg = int((f == 1).sum()), int((f == 0).sum())
        if npos == 0 or nneg == 0:
            continue
        a = rank_auc(scores[sub], f)
        num += a * npos * nneg
        den += npos * nneg
    return float(num / den) if den > 0 else float("nan")


def bootstrap_stat(fn, n: int, rng, n_boot: int) -> np.ndarray:
    """Case-level bootstrap: array of ``fn(idx)`` over ``n_boot`` resamples.

    ``fn`` takes an index array (length ``n``, with replacement) and returns a
    float; non-finite replicates (e.g. a resample missing a class) are dropped.
    """
    reps = []
    for _ in range(n_boot):
        v = fn(rng.integers(0, n, n))
        if np.isfinite(v):
            reps.append(v)
    return np.asarray(reps, dtype=np.float64)


def ci_and_p(reps: np.ndarray, null: float = 0.5) -> tuple[list, float]:
    """95% percentile CI and a one-sided tail p (fraction of reps <= ``null``).

    The p is Laplace-smoothed: ``(#{reps <= null} + 1) / (len + 1)``. Both NaN if
    no finite replicate survived.
    """
    reps = np.asarray(reps, dtype=np.float64)
    if reps.size == 0:
        return [float("nan"), float("nan")], float("nan")
    ci = [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]
    p = float((np.sum(reps <= null) + 1) / (reps.size + 1))
    return ci, p


def paired_diff_ci(fa, fb, n: int, rng, n_boot: int, null: float = 0.0) -> tuple[list, float]:
    """Paired bootstrap CI and one-sided p for ``a - b`` on the same resample.

    ``fa``/``fb`` each take the shared index array and return a float; a replicate
    is dropped if either side is non-finite. The one-sided p tests
    ``a - b <= null`` (i.e. how often the difference fails to be positive).
    """
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a, b = fa(idx), fb(idx)
        if np.isfinite(a) and np.isfinite(b):
            diffs.append(a - b)
    return ci_and_p(np.asarray(diffs, dtype=np.float64), null=null)


def within_diag_perm_p(
    scores: np.ndarray, feat: np.ndarray, diag: np.ndarray, n_perm: int, rng
) -> float:
    """One-sided p for the conditioned AUC under within-diagnosis label permutation.

    Null: the projection carries no feature signal *beyond* diagnosis. Feature
    labels are shuffled within each diagnosis stratum (preserving the per-stratum
    feature marginal, breaking only the within-stratum score->feature link), and
    the conditioned AUC is recomputed. Returns the Laplace-smoothed fraction of
    permutations whose conditioned AUC is >= the observed one. NaN if unobservable.
    """
    scores = np.asarray(scores, dtype=np.float64)
    feat = np.asarray(feat)
    diag = np.asarray(diag)
    m = feat >= 0
    scores, feat, diag = scores[m], feat[m], diag[m]
    obs = conditioned_auc(scores, feat, diag)
    if not np.isfinite(obs):
        return float("nan")
    strata = [np.where(diag == g)[0] for g in np.unique(diag)]
    ge = valid = 0
    for _ in range(n_perm):
        fp = feat.copy()
        for sel in strata:
            fp[sel] = feat[sel][rng.permutation(sel.shape[0])]
        a = conditioned_auc(scores, fp, diag)
        if np.isfinite(a):
            valid += 1
            ge += int(a >= obs)
    return float((ge + 1) / (valid + 1)) if valid else float("nan")
