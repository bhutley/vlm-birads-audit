"""Data-free tests for the WP2 cohort join / alignment (src/data/cohort.py).

Synthetic manifests + arrays only — no data, no GPU. Guards the WP2 acceptance
gate (plan §8): every kept row has one sample/group/fold; alignment fails closed
on count/label/duplicate/missing errors; weighting is deterministic, order
invariant, class-balanced, and not proportional to a group's image count.

Run:  python tests/test_cohort.py   (or: python -m pytest tests/)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cohort import (
    AlignmentError,
    CohortManifest,
    align_site,
    check_label_alignment,
    group_class_balanced_weights,
    sample_id,
)


def _manifest(site="s"):
    # 3 groups: g0 (2 benign imgs), g1 (1 benign), g2 (1 malignant). Plus one
    # excluded conflict image and one normal, both absent from the manifest rows.
    rows = {
        sample_id(site, "a.png"): {"dataset": site, "group_id": "g0", "fold": 0,
                                   "label_idx": 0, "binary_eligible": True},
        sample_id(site, "b.png"): {"dataset": site, "group_id": "g0", "fold": 0,
                                   "label_idx": 0, "binary_eligible": True},
        sample_id(site, "c.png"): {"dataset": site, "group_id": "g1", "fold": 1,
                                   "label_idx": 0, "binary_eligible": True},
        sample_id(site, "d.png"): {"dataset": site, "group_id": "g2", "fold": 2,
                                   "label_idx": 1, "binary_eligible": True},
    }
    excluded = {sample_id(site, "x.png")}
    return CohortManifest(rows, excluded)


# --- weighting ---------------------------------------------------------------
def test_weights_sum_to_n_and_classes_are_balanced():
    g = np.array(["g0", "g0", "g1", "g2"])
    y = np.array([0, 0, 0, 1])
    w = group_class_balanced_weights(g, y)
    assert np.isclose(w.sum(), 4.0)
    assert np.isclose(w[y == 0].sum(), w[y == 1].sum())  # equal class mass


def test_repeated_group_not_weighted_by_image_count():
    # g0 has 2 benign images, g1 has 1 benign image -> the two groups carry equal
    # TOTAL weight; the 2-image group's members are each downweighted.
    g = np.array(["g0", "g0", "g1"])
    y = np.array([0, 0, 0])
    w = group_class_balanced_weights(g, y)
    assert np.isclose(w[0] + w[1], w[2])       # group totals equal
    assert np.isclose(w[0], w[1])              # within-group equal


def test_weights_are_order_invariant():
    g = np.array(["g0", "g0", "g1", "g2"])
    y = np.array([0, 0, 0, 1])
    w = group_class_balanced_weights(g, y)
    perm = [3, 1, 2, 0]
    wp = group_class_balanced_weights(g[perm], y[perm])
    assert np.allclose(wp, w[perm])


# --- alignment ---------------------------------------------------------------
def _emb(site, names, labels):
    ids = [sample_id(site, n) for n in names]
    Z = np.random.default_rng(0).normal(size=(len(names), 4))
    return Z, np.array(labels), ids


def test_align_keeps_binary_drops_normal_and_excluded():
    m = _manifest()
    # embeddings enumerate ALL loader rows incl a normal and the excluded conflict.
    Z, y, ids = _emb("s", ["a.png", "b.png", "c.png", "d.png", "n.png", "x.png"],
                     [0, 0, 0, 1, 2, 1])
    se = align_site("s", Z, y, ids, m)
    assert len(se.y) == 4 and se.n_groups == 3
    assert set(se.sample_ids) == m.site_binary_ids("s")
    assert getattr(se, "_dropped") == {"normal": 1, "excluded": 1}


def test_align_fails_on_unexpected_binary_not_in_manifest():
    m = _manifest()
    Z, y, ids = _emb("s", ["a.png", "b.png", "c.png", "d.png", "ghost.png"],
                     [0, 0, 0, 1, 1])  # ghost is malignant, not excluded, not in manifest
    try:
        align_site("s", Z, y, ids, m)
        assert False, "expected AlignmentError"
    except AlignmentError:
        pass


def test_align_fails_on_missing_manifest_sample():
    m = _manifest()
    # d.png (a manifest binary sample) is absent from the embeddings.
    Z, y, ids = _emb("s", ["a.png", "b.png", "c.png"], [0, 0, 0])
    try:
        align_site("s", Z, y, ids, m)
        assert False, "expected AlignmentError"
    except AlignmentError:
        pass


def test_align_fails_on_duplicate_ids():
    m = _manifest()
    Z, y, ids = _emb("s", ["a.png", "a.png", "b.png", "c.png", "d.png"],
                     [0, 0, 0, 0, 1])
    try:
        align_site("s", Z, y, ids, m)
        assert False, "expected AlignmentError"
    except AlignmentError:
        pass


# --- cache sidecar label guard ----------------------------------------------
def test_cache_label_alignment_passes_and_fails():
    check_label_alignment("s", [0, 1, 2, 0], [0, 1, 2, 0])   # ok
    for bad in ([0, 1, 2], [0, 1, 1, 0]):                    # count / label
        try:
            check_label_alignment("s", [0, 1, 2, 0], bad)
            assert False, "expected AlignmentError"
        except AlignmentError:
            pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
