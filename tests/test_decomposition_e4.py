"""Data-free tests for the WP5 E4 repair (src/experiments/run_decomposition.py).

Synthetic arrays only — no data, no GPU. Guards the WP5 acceptance invariants
(plan §11.2, §15.4): the likelihood-ratio test uses a VALID unpenalized MLE fit
(rejects a planted s-signal, does not reject a null s, flags separation as
invalid), and audit-unit aggregation is single-label / fold-indivisible.

Run:  python tests/test_decomposition_e4.py   (or: python -m pytest tests/)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cohort import SiteEmbeddings
from src.experiments.run_decomposition import _aggregate_audit_units, _lr_test


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def test_lr_test_rejects_planted_s_signal():
    rng = np.random.default_rng(0)
    n = 800
    conf = rng.uniform(0, 0.5, n)
    s = rng.standard_normal(n)
    # error depends on BOTH confidence and s -> adding s should be significant
    z = lambda x: (x - x.mean()) / (x.std() + 1e-12)
    p = _sigmoid(-1.2 * z(conf) + 1.3 * z(s))
    err = (rng.uniform(size=n) < p).astype(int)
    r = _lr_test(conf, s, err)
    assert r["valid"] and r["p"] < 0.05 and r["stat"] > 0
    assert r["ll1"] > r["ll0"]     # alternative fits at least as well


def test_lr_test_does_not_reject_null_s():
    rng = np.random.default_rng(3)
    n = 800
    conf = rng.uniform(0, 0.5, n)
    s = rng.standard_normal(n)         # s is pure noise, unrelated to error
    z = lambda x: (x - x.mean()) / (x.std() + 1e-12)
    err = (rng.uniform(size=n) < _sigmoid(-1.0 * z(conf))).astype(int)
    r = _lr_test(conf, s, err)
    assert r["valid"] and r["p"] > 0.05


def test_lr_test_flags_separation_invalid():
    rng = np.random.default_rng(1)
    n = 300
    conf = rng.uniform(0, 0.5, n)
    s = rng.standard_normal(n)
    err = (s > 0).astype(int)          # error perfectly separated by s
    r = _lr_test(conf, s, err)
    assert not r["valid"]              # must NOT trust a chi-square here


def test_lr_test_single_class_invalid():
    r = _lr_test(np.linspace(0, 1, 20), np.random.default_rng(0).normal(size=20),
                 np.zeros(20, dtype=int))
    assert not r["valid"]


def test_aggregate_audit_units_means_and_fold_indivisible():
    # 3 groups: g0 (2 benign, fold 0), g1 (1 malignant, fold 1), g2 (3 benign, fold 2)
    Z = np.array([[1., 0.], [3., 0.],          # g0
                  [0., 2.],                     # g1
                  [1., 1.], [1., 1.], [1., 1.]], dtype=float)  # g2
    y = np.array([0, 0, 1, 0, 0, 0])
    groups = np.array(["g0", "g0", "g1", "g2", "g2", "g2"])
    folds = np.array([0, 0, 1, 2, 2, 2])
    se = SiteEmbeddings("busi", Z, y, np.array([f"busi/{i}" for i in range(6)]),
                        groups, folds, np.ones(6))
    Zu, yu, foldu = _aggregate_audit_units(se)
    assert len(yu) == 3
    # groups in sorted order g0,g1,g2 -> labels 0,1,0; folds 0,1,2
    assert list(yu) == [0, 1, 0]
    assert list(foldu) == [0, 1, 2]
    assert np.allclose(np.linalg.norm(Zu, axis=1), 1.0)   # each unit L2-normalised


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
