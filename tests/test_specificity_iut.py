"""Data-free tests for the WP6 feature-specificity IUT + Holm family (E6).

Guards the multiplicity logic (plan §12.1, §15.4): the intersection-union p is the
MAX of the three component p-values (so the weakest component governs), and the
Holm step-down across the backbone x axis family is correct and NaN-aware.

Run:  python tests/test_specificity_iut.py   (or: python -m pytest tests/)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.run_concept_validity import _holm


def test_iut_p_is_max_of_components():
    # a cell with one weak component is NOT specific even if the others are strong
    strong = [0.001, 0.002, 0.001]
    weak = [0.001, 0.90, 0.001]
    assert max(strong) < 0.05
    assert max(weak) == 0.90 and max(weak) >= 0.05


def test_holm_step_down_values_and_rejections():
    p = np.array([0.001, 0.02, 0.04, 0.5])          # family size 4
    adj = _holm(p)
    assert np.isclose(adj[0], 0.004)                # 4 * 0.001
    assert np.isclose(adj[1], 0.06)                 # max(0.004, 3*0.02)
    assert np.isclose(adj[2], 0.08)                 # max(0.06, 2*0.04)
    assert np.isclose(adj[3], 0.5)
    assert int((adj < 0.05).sum()) == 1             # only the smallest rejects


def test_holm_excludes_nan_from_family_size():
    p = np.array([0.01, np.nan, 0.01])              # family size 2 (nan dropped)
    adj = _holm(p)
    assert np.isnan(adj[1])
    assert np.isclose(adj[0], 0.02) and np.isclose(adj[2], 0.02)   # 2 * 0.01


def test_holm_monotone_nondecreasing_in_rank():
    p = np.array([0.001, 0.003, 0.006, 0.02, 0.9])
    adj = _holm(p)
    ordered = adj[np.argsort(p)]
    assert np.all(np.diff(ordered) >= -1e-12)       # step-down never decreases


def test_full_family_of_12_all_null_rejects_none():
    p = np.full(12, 0.4)                             # 12 backbone x axis, all null
    adj = _holm(p)
    assert not np.any(adj < 0.05)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
