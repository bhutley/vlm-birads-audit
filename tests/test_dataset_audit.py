"""Data-free tests for the WP1 dataset-audit pure logic (src/data/audit.py).

No images, no GPU: every test builds synthetic record dicts and exercises the
clustering / conflict / grouping / fold logic. Guards the load-bearing WP0
invariants (plan §3.2, §7.4):

* label-conflict clusters excluded in full; consistent-label clusters retained;
* native ids ∪ near-dup clusters merged into one fold-indivisible group;
* fold assignment deterministic and invariant to input row order;
* no group ever spans more than one fold.

Run:  python tests/test_dataset_audit.py   (or: python -m pytest tests/)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.audit import (
    assign_groups,
    build_clusters,
    build_dataset_folds,
    exclusions_from_conflicts,
    label_conflicts,
)


def _rec(ds, fn, label, *, sha="s", phash="0000000000000000",
         patient="", near="", path=None):
    return {
        "dataset": ds, "filename": fn, "label_name": label,
        "label_idx": {"benign": 0, "malignant": 1, "normal": 2}[label],
        "sha256": sha, "phash_hex": phash,
        "patient_group": patient, "near_group": near,
        "path": path or f"/{ds}/{fn}",
    }


def test_exact_duplicate_same_label_is_not_a_conflict():
    recs = [_rec("busi", "a.png", "benign", sha="X"),
            _rec("busi", "b.png", "benign", sha="X")]
    exact, near = build_clusters(recs, hamming_threshold=6)
    assert len(exact) == 1
    assert label_conflicts(exact, near, recs) == []
    assert exclusions_from_conflicts(label_conflicts(exact, near, recs)) == set()


def test_label_conflict_cluster_excluded_in_full():
    # Two byte-identical images with DISAGREEING labels -> whole cluster dropped.
    recs = [_rec("busi", "a.png", "benign", sha="X", phash="0000000000000000"),
            _rec("busi", "b.png", "malignant", sha="X", phash="0000000000000000"),
            # unrelated singleton: distinct sha AND distant phash
            _rec("busi", "c.png", "benign", sha="Y", phash="ffffffffffffffff")]
    exact, near = build_clusters(recs, hamming_threshold=6)
    conflicts = label_conflicts(exact, near, recs)
    excluded = exclusions_from_conflicts(conflicts)
    assert excluded == {0, 1}          # both conflict members, no winner picked
    assert 2 not in excluded           # unrelated image untouched


def test_near_duplicate_within_threshold_clusters():
    # phashes differ by 1 bit -> near-dup at threshold 6; a distant one does not.
    recs = [_rec("busi", "a.png", "benign", phash="0000000000000000"),
            _rec("busi", "b.png", "benign", phash="0000000000000001"),
            _rec("busi", "c.png", "benign", phash="ffffffffffffffff")]
    _, near = build_clusters(recs, hamming_threshold=6)
    assert len(near) == 1
    members = next(iter(near.values()))
    assert set(members) == {0, 1}


def test_consistent_label_neardup_is_retained():
    recs = [_rec("busi", "a.png", "benign", phash="0000000000000000"),
            _rec("busi", "b.png", "benign", phash="0000000000000001")]
    exact, near = build_clusters(recs, hamming_threshold=6)
    assert exclusions_from_conflicts(label_conflicts(exact, near, recs)) == set()


def test_group_merges_native_id_and_neardup_cluster():
    # a~b share a patient id; b~c share a near-dup cluster -> {a,b,c} one group.
    recs = [_rec("busi", "a.png", "benign", patient="P1"),
            _rec("busi", "b.png", "benign", patient="P1", near="near_0000"),
            _rec("busi", "c.png", "benign", near="near_0000"),
            _rec("busi", "d.png", "benign")]  # singleton
    g = assign_groups(recs)
    assert g[0] == g[1] == g[2]        # transitive merge through b
    assert g[3] != g[0]                # singleton stays separate


def _fold_dataset():
    # 10 single-class groups (5 benign, 5 malignant), 2 images each -> feasible
    # for StratifiedGroupKFold(5).
    recs = []
    for gi in range(10):
        label = "benign" if gi < 5 else "malignant"
        for k in range(2):
            recs.append(_rec("busi", f"g{gi:02d}_{k}.png", label,
                             patient=f"P{gi}"))
    return recs


def test_folds_no_group_crosses_a_fold():
    rows = build_dataset_folds(_fold_dataset(), n_folds=5, seed=42)
    group_folds: dict[str, set[int]] = {}
    for r in rows:
        group_folds.setdefault(r["group_id"], set()).add(r["fold"])
    assert all(len(fs) == 1 for fs in group_folds.values())


def test_fold_assignment_is_input_order_invariant():
    base = build_dataset_folds(_fold_dataset(), n_folds=5, seed=42)
    shuffled = list(reversed(_fold_dataset()))
    other = build_dataset_folds(shuffled, n_folds=5, seed=42)
    map_a = {r["filename"]: r["fold"] for r in base}
    map_b = {r["filename"]: r["fold"] for r in other}
    assert map_a == map_b


def test_every_image_gets_exactly_one_fold():
    rows = build_dataset_folds(_fold_dataset(), n_folds=5, seed=42)
    assert len(rows) == 20
    assert all(0 <= r["fold"] < 5 for r in rows)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
