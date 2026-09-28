"""Group-disjoint out-of-fold predictive audit of the decision direction (PERF-01).

E3 audits *where* the probe's decision direction ``w`` points (rho, placebo
excess) but has never reported how well ``w`` actually predicts. A direction from
a near-chance probe is not a meaningful clinical decision to audit, so every
backbone x site cell now carries out-of-fold discrimination alongside its
geometry.

The estimand is deliberately the same cohort/fold/weight construction the
geometry uses (``src/data/cohort``):

* the probe is refit on each 4-of-5 canonical **group-disjoint** fold union and
  read on the held-out fold, so no image is scored by a probe that trained on its
  own audit group;
* metrics are **group-weighted** (``group_class_balanced_weights``) so repeated
  images / multi-image patients cannot dominate;
* uncertainty is a **cluster bootstrap over groups** of the stored out-of-fold
  predictions -- no refits, so the CI costs arithmetic, not model fits.

Metrics are implemented here in pure NumPy rather than pulled from sklearn so the
weighting semantics are explicit and testable without data or a GPU
(``tests/test_probe_performance.py``); the AUC is cross-checked against
``sklearn.metrics.roc_auc_score`` in those tests.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

BENIGN, MALIGNANT = 0, 1


class FoldError(RuntimeError):
    """Raised when the canonical folds cannot support an out-of-fold estimate."""


# --- weighted metrics (pure) -------------------------------------------------
def weighted_auc(y, score, weight=None) -> float:
    """Weighted ROC AUC as the normalised Mann-Whitney statistic, ties at 0.5.

    ``AUC = sum_{i in pos, j in neg} w_i w_j [s_i > s_j] + 0.5 [s_i == s_j]``,
    divided by ``(sum_pos w)(sum_neg w)``. Computed via mid-ranks in O(n log n)
    rather than the O(n^2) double sum.

    Invariant to rescaling either class's weights as a whole, so the
    class-balancing step of :func:`~src.data.cohort.group_class_balanced_weights`
    does not move the value -- what remains is inverse-group weighting. Returns
    NaN when either class is absent or carries no weight.
    """
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=np.float64)
    w = (np.ones(len(y), dtype=np.float64) if weight is None
         else np.asarray(weight, dtype=np.float64))
    if not (len(y) == len(score) == len(w)):
        raise ValueError("y, score and weight must be the same length")
    pos, neg = y == MALIGNANT, y == BENIGN
    wp, wn = w[pos].sum(), w[neg].sum()
    if wp <= 0 or wn <= 0:
        return float("nan")

    # Mid-rank of every point within the weighted pooled sample: the total weight
    # strictly below it, plus half the weight tied with it. Summing that over the
    # positives and subtracting the positives' own internal contribution leaves
    # exactly the weighted concordance count.
    order = np.argsort(score, kind="mergesort")
    s_sorted, w_sorted = score[order], w[order]
    cum = np.concatenate([[0.0], np.cumsum(w_sorted)])
    # group tied runs
    starts = np.concatenate([[True], s_sorted[1:] != s_sorted[:-1]])
    grp = np.cumsum(starts) - 1                      # tie-group index per sorted row
    grp_lo = cum[np.searchsorted(grp, np.arange(grp[-1] + 1), side="left")]
    grp_hi = cum[np.searchsorted(grp, np.arange(grp[-1] + 1), side="right")]
    mid = grp_lo[grp] + 0.5 * (grp_hi[grp] - grp_lo[grp])   # weight below + half tied
    midrank = np.empty(len(y), dtype=np.float64)
    midrank[order] = mid

    # Mid-ranks mix both classes, so subtract the positives' contribution to each
    # other. That term is exactly wp^2 / 2 whatever the tie structure: the strictly
    # -below pairs contribute (wp^2 - T)/2 and the tied pairs T/2, where T is the
    # positive-positive tied weight.
    concordant = float((w[pos] * midrank[pos]).sum())
    return (concordant - 0.5 * wp ** 2) / (wp * wn)


def weighted_balanced_accuracy(y, prob, weight=None, threshold: float = 0.5) -> float:
    """Weighted balanced accuracy = mean of the weighted per-class recalls.

    Thresholds ``prob`` at a **fixed** ``threshold`` (0.5); no cutoff is tuned.
    Returns NaN when either class is absent or carries no weight.
    """
    y = np.asarray(y, dtype=int)
    pred = (np.asarray(prob, dtype=np.float64) >= threshold).astype(int)
    w = (np.ones(len(y), dtype=np.float64) if weight is None
         else np.asarray(weight, dtype=np.float64))
    if not (len(y) == len(pred) == len(w)):
        raise ValueError("y, prob and weight must be the same length")
    recalls = []
    for c in (BENIGN, MALIGNANT):
        m = y == c
        tot = w[m].sum()
        if tot <= 0:
            return float("nan")
        recalls.append(float((w[m] * (pred[m] == c)).sum() / tot))
    return float(np.mean(recalls))


# --- out-of-fold prediction --------------------------------------------------
def check_folds_group_disjoint(group_ids, folds) -> None:
    """Fail closed unless every group lies entirely within one fold."""
    group_ids = np.asarray(group_ids)
    folds = np.asarray(folds, dtype=int)
    spans = {g: set(folds[group_ids == g].tolist()) for g in set(group_ids.tolist())}
    bad = {g: sorted(f) for g, f in spans.items() if len(f) > 1}
    if bad:
        first = next(iter(bad.items()))
        raise FoldError(
            f"{len(bad)} group(s) span multiple folds, e.g. {first[0]} -> {first[1]}")


def out_of_fold_scores(Z, y, group_ids, folds, fit_fn) -> dict:
    """Cross-fit the probe over the canonical folds; return one OOF score per row.

    ``fit_fn(Z_tr, y_tr, weight_tr) -> (w, b)`` is injected so this helper stays
    free of sklearn and of the caller's regularisation choice; training weights
    are recomputed *within* each training split (group-inverse, class-balanced),
    matching how the fold-refit direction stability is fitted.

    Returns ``{"logit", "prob", "n_folds", "folds_used"}`` with ``logit``/``prob``
    row-aligned to the inputs. Raises :class:`FoldError` if a group spans folds or
    if no fold yields a usable train/test split.
    """
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    group_ids = np.asarray(group_ids)
    folds = np.asarray(folds, dtype=int)
    check_folds_group_disjoint(group_ids, folds)

    from src.data.cohort import group_class_balanced_weights

    logit = np.full(len(y), np.nan, dtype=np.float64)
    used: list[int] = []
    for f in sorted(set(folds.tolist())):
        te = folds == f
        tr = ~te
        if not te.any() or len(np.unique(y[tr])) < 2:
            continue                      # single-class training split: no estimate
        wgt = group_class_balanced_weights(group_ids[tr], y[tr])
        w, b = fit_fn(Z[tr], y[tr], wgt)
        logit[te] = Z[te] @ np.asarray(w, dtype=np.float64) + float(b)
        used.append(int(f))
    if not used:
        raise FoldError("no fold produced a two-class training split")
    return {
        "logit": logit,
        "prob": 1.0 / (1.0 + np.exp(-logit)),
        "n_folds": len(used),
        "folds_used": used,
    }


# --- cluster bootstrap over the stored predictions ---------------------------
def cluster_bootstrap_metrics(y, logit, prob, group_ids, selections) -> dict:
    """Percentile CIs for OOF AUC / balanced accuracy over group resamples.

    ``selections`` is a list of group-id arrays (drawn with replacement); pass the
    caller's existing per-site selections so these intervals are **paired** to the
    same cohort perturbations as the geometry CIs. Weights are recomputed on each
    resampled index, so a group drawn twice contributes twice its share of groups
    but not twice its share within a class beyond that.

    Replicates missing a class (or with all-NaN predictions) are skipped and
    counted in ``invalid_frac``.
    """
    y = np.asarray(y, dtype=int)
    group_ids = np.asarray(group_ids)
    from src.data.cohort import group_class_balanced_weights

    rows_by_group: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(group_ids):
        rows_by_group[g].append(i)

    aucs, baccs = [], []
    n_invalid = 0
    for sel in selections:
        idx = np.fromiter((i for g in sel for i in rows_by_group[g]), dtype=int)
        yb = y[idx]
        ok = np.isfinite(logit[idx])
        if len(np.unique(yb[ok])) < 2:
            n_invalid += 1
            continue
        idx, yb = idx[ok], yb[ok]
        wgt = group_class_balanced_weights(group_ids[idx], yb)
        a = weighted_auc(yb, logit[idx], wgt)
        b = weighted_balanced_accuracy(yb, prob[idx], wgt)
        if not (np.isfinite(a) and np.isfinite(b)):
            n_invalid += 1
            continue
        aucs.append(a)
        baccs.append(b)

    def _ci(a):
        return ([float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
                if a else None)
    n = len(selections)
    return {
        "auc_ci": _ci(aucs),
        "balanced_accuracy_ci": _ci(baccs),
        "invalid_frac": n_invalid / n if n else 0.0,
    }


def oof_performance(Z, y, group_ids, folds, fit_fn, selections=None) -> dict:
    """Full PERF-01 cell: group-disjoint OOF AUC + balanced accuracy (+ CIs).

    ``weak_probe`` flags a cell whose AUC 95% CI lower bound fails to clear 0.50 --
    the cheap part of the weak-probe sensitivity, so the geometry claim can be read
    conditional on probe quality without tuning any cutoff.
    """
    y = np.asarray(y, dtype=int)
    group_ids = np.asarray(group_ids)
    oof = out_of_fold_scores(Z, y, group_ids, folds, fit_fn)
    logit, prob = oof["logit"], oof["prob"]
    scored = np.isfinite(logit)

    wgt = None
    if scored.any():
        from src.data.cohort import group_class_balanced_weights
        wgt = group_class_balanced_weights(group_ids[scored], y[scored])
    auc = (weighted_auc(y[scored], logit[scored], wgt)
           if scored.any() else float("nan"))
    bacc = (weighted_balanced_accuracy(y[scored], prob[scored], wgt)
            if scored.any() else float("nan"))

    out = {
        "auc": float(auc),
        "balanced_accuracy": float(bacc),
        "auc_ci": None,
        "balanced_accuracy_ci": None,
        "n_scored": int(scored.sum()),
        "n_unscored": int((~scored).sum()),
        "n_groups": len(set(group_ids[scored].tolist())) if scored.any() else 0,
        "n_pos": int((y[scored] == MALIGNANT).sum()),
        "n_neg": int((y[scored] == BENIGN).sum()),
        "n_folds": oof["n_folds"],
        "folds_used": oof["folds_used"],
        "boot_invalid_frac": None,
        "weak_probe": None,
    }
    if selections:
        ci = cluster_bootstrap_metrics(y, logit, prob, group_ids, selections)
        out["auc_ci"] = ci["auc_ci"]
        out["balanced_accuracy_ci"] = ci["balanced_accuracy_ci"]
        out["boot_invalid_frac"] = ci["invalid_frac"]
        if ci["auc_ci"] is not None:
            out["weak_probe"] = bool(ci["auc_ci"][0] <= 0.5)
    return out
