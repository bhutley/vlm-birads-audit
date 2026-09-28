"""Data-free tests for the PERF-01 out-of-fold probe audit.

Synthetic arrays only — no data, no GPU. Guards the estimand the reviewer asked
for: predictions are group-disjoint out-of-fold, metrics are group-weighted so
duplicate images cannot dominate, the weighted AUC agrees with sklearn, and the
cluster bootstrap is seed-reproducible and fails closed on degenerate folds.

Run:  python tests/test_probe_performance.py   (or: python -m pytest tests/)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cohort import group_class_balanced_weights
from src.evaluation.probe_performance import (
    FoldError,
    check_folds_group_disjoint,
    cluster_bootstrap_metrics,
    oof_performance,
    out_of_fold_scores,
    weighted_auc,
    weighted_balanced_accuracy,
)


def _lsq_fit(Z, y, weight):
    """Deterministic stand-in for the logistic probe: weighted least squares.

    Keeps the tests free of sklearn's optimiser while exercising the same
    ``fit_fn(Z, y, weight) -> (w, b)`` contract the experiment injects.
    """
    X = np.column_stack([Z, np.ones(len(Z))])
    W = np.diag(np.asarray(weight, dtype=np.float64))
    beta, *_ = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ (2.0 * y - 1.0), rcond=None)
    return beta[:-1], float(beta[-1])


# --- weighted AUC ------------------------------------------------------------
def test_weighted_auc_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(10, 80))
        y = rng.integers(0, 2, size=n)
        if len(np.unique(y)) < 2:
            continue
        # round the scores so ties actually occur
        s = np.round(rng.normal(size=n), 1)
        w = rng.uniform(0.2, 3.0, size=n)
        assert abs(weighted_auc(y, s, w)
                   - roc_auc_score(y, s, sample_weight=w)) < 1e-9
        assert abs(weighted_auc(y, s) - roc_auc_score(y, s)) < 1e-9


def test_weighted_auc_separation_extremes_and_ties():
    y = np.array([0, 0, 1, 1])
    assert weighted_auc(y, np.array([0.0, 0.1, 0.9, 1.0])) == 1.0
    assert weighted_auc(y, np.array([1.0, 0.9, 0.1, 0.0])) == 0.0
    assert weighted_auc(y, np.zeros(4)) == 0.5            # all tied -> chance
    assert np.isnan(weighted_auc(np.zeros(4, dtype=int), np.arange(4.0)))


def test_weighted_auc_invariant_to_class_rescaling():
    """Class balancing must not move the AUC — only inverse-group weighting does."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=40)
    s, w = rng.normal(size=40), rng.uniform(0.5, 2.0, size=40)
    w2 = w.copy()
    w2[y == 1] *= 7.3                                      # rescale one class
    assert abs(weighted_auc(y, s, w) - weighted_auc(y, s, w2)) < 1e-12


def test_group_weighting_neutralises_duplicated_images():
    """Copying one lesion's images must not change the group-weighted metrics."""
    y = np.array([0, 0, 1, 1, 1])
    s = np.array([0.1, 0.4, 0.6, 0.8, 0.9])
    g = np.array(["g0", "g1", "g2", "g3", "g4"])
    a0 = weighted_auc(y, s, group_class_balanced_weights(g, y))

    dup = np.array([0, 1, 1, 2, 3, 4])                     # image 1 appears twice
    yd, sd = y[dup], s[dup]
    gd = np.array(["g0", "g1", "g1", "g2", "g3", "g4"])    # ...within its own group
    a1 = weighted_auc(yd, sd, group_class_balanced_weights(gd, yd))
    assert abs(a0 - a1) < 1e-12


# --- balanced accuracy -------------------------------------------------------
def test_weighted_balanced_accuracy_hand_case():
    y = np.array([0, 0, 0, 1])
    prob = np.array([0.1, 0.2, 0.9, 0.8])                  # benign recall 2/3, mal 1/1
    assert abs(weighted_balanced_accuracy(y, prob) - (2 / 3 + 1.0) / 2) < 1e-12
    # a fixed 0.5 threshold, not a tuned cutoff: everything predicted malignant
    assert abs(weighted_balanced_accuracy(y, np.ones(4)) - 0.5) < 1e-12
    assert np.isnan(weighted_balanced_accuracy(np.zeros(4, dtype=int), prob))


# --- fold hygiene ------------------------------------------------------------
def test_group_spanning_folds_fails_closed():
    check_folds_group_disjoint(np.array(["a", "a", "b"]), np.array([0, 0, 1]))
    try:
        check_folds_group_disjoint(np.array(["a", "a", "b"]), np.array([0, 1, 1]))
        assert False, "expected FoldError"
    except FoldError:
        pass


def _toy(n_per_fold=6, n_folds=5, seed=0):
    rng = np.random.default_rng(seed)
    n = n_per_fold * n_folds
    y = np.tile([0, 1], n // 2)
    Z = rng.normal(size=(n, 4)) + y[:, None] * 0.8         # separable-ish
    groups = np.array([f"g{i}" for i in range(n)])
    folds = np.repeat(np.arange(n_folds), n_per_fold)
    return Z, y, groups, folds


def test_every_row_scored_exactly_once_and_out_of_fold():
    Z, y, g, f = _toy()
    oof = out_of_fold_scores(Z, y, g, f, _lsq_fit)
    assert np.isfinite(oof["logit"]).all()                 # one prediction per row
    assert oof["n_folds"] == 5

    # a row's score must not depend on its own fold's rows: refit without fold 0
    # and confirm fold 0's OOF scores reproduce exactly
    tr = f != 0
    w, b = _lsq_fit(Z[tr], y[tr], group_class_balanced_weights(g[tr], y[tr]))
    assert np.allclose(oof["logit"][f == 0], Z[f == 0] @ w + b)


def test_row_order_invariance():
    Z, y, g, f = _toy()
    a = oof_performance(Z, y, g, f, _lsq_fit)
    perm = np.random.default_rng(7).permutation(len(y))
    b = oof_performance(Z[perm], y[perm], g[perm], f[perm], _lsq_fit)
    assert abs(a["auc"] - b["auc"]) < 1e-12
    assert abs(a["balanced_accuracy"] - b["balanced_accuracy"]) < 1e-12


def test_single_class_training_split_is_skipped_not_faked():
    """A fold whose complement is single-class yields no estimate for that fold."""
    Z, y, g, f = _toy(n_per_fold=6, n_folds=2)
    y = y.copy()
    y[f == 0] = 0                              # fold 0 all benign, so fold 1's
    oof = out_of_fold_scores(Z, y, g, f, _lsq_fit)   # training split is single-class
    assert oof["folds_used"] == [0]
    assert np.isfinite(oof["logit"][f == 0]).all()
    assert np.isnan(oof["logit"][f == 1]).all()


def test_no_usable_fold_fails_closed():
    Z, y, g, f = _toy(n_per_fold=4, n_folds=2)
    try:
        out_of_fold_scores(Z, np.zeros(len(y), dtype=int), g, f, _lsq_fit)
        assert False, "expected FoldError"
    except FoldError:
        pass


# --- cluster bootstrap -------------------------------------------------------
def _selections(groups, n_boot, seed):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups.tolist())))
    return [rng.choice(uniq, size=len(uniq), replace=True) for _ in range(n_boot)]


def test_bootstrap_ci_brackets_point_estimate_and_is_reproducible():
    Z, y, g, f = _toy(n_per_fold=10, n_folds=5, seed=3)
    sels = _selections(g, 200, seed=42)
    a = oof_performance(Z, y, g, f, _lsq_fit, selections=sels)
    b = oof_performance(Z, y, g, f, _lsq_fit, selections=_selections(g, 200, seed=42))
    assert a["auc_ci"] == b["auc_ci"]                       # same seed -> same CI
    lo, hi = a["auc_ci"]
    assert lo <= a["auc"] <= hi
    assert a["boot_invalid_frac"] == 0.0
    assert a["weak_probe"] is (lo <= 0.5)


def test_weak_probe_flag_fires_on_a_chance_direction():
    rng = np.random.default_rng(11)
    n = 60
    y = np.tile([0, 1], n // 2)
    Z = rng.normal(size=(n, 4))                            # label-independent
    g = np.array([f"g{i}" for i in range(n)])
    f = np.repeat(np.arange(5), n // 5)
    out = oof_performance(Z, y, g, f, _lsq_fit, selections=_selections(g, 300, 42))
    assert out["weak_probe"] is True                       # CI must include chance


def test_bootstrap_counts_invalid_replicates():
    y = np.array([0, 1, 1, 1])
    logit = np.array([0.1, 0.2, 0.3, 0.4])
    g = np.array(["g0", "g1", "g2", "g3"])
    sels = [np.array(["g1", "g2", "g3", "g3"]),            # malignant only -> invalid
            np.array(["g0", "g1", "g2", "g3"])]
    out = cluster_bootstrap_metrics(y, logit, 1 / (1 + np.exp(-logit)), g, sels)
    assert out["invalid_frac"] == 0.5


def test_counts_are_reported():
    Z, y, g, f = _toy()
    out = oof_performance(Z, y, g, f, _lsq_fit)
    assert out["n_scored"] == len(y) and out["n_unscored"] == 0
    assert out["n_pos"] + out["n_neg"] == len(y)
    assert out["n_groups"] == len(set(g.tolist()))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
