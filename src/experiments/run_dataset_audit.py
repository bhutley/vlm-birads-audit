"""Experiment: dataset_audit (WP1)

Six-site BUS dataset audit: inventory + duplicate/overlap screen + label-conflict
exclusions + group-aware canonical 5-fold manifest. Foundation for the
patient/audit-unit-aware reanalysis (E3/E4/E6).

Private per-image artefacts (paths/filenames) are written under
``results/dataset_audit/`` and git-ignored; only aggregate counts, grouping
sources, exclusion counts, and manifest SHA-256s enter ``results.json``.

Usage:
    PYTHONHASHSEED=0 python -m src.experiments.run_dataset_audit
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.audit import (
    TASK_CLASSES,
    build_clusters,
    build_dataset_folds,
    enumerate_dataset,
    exclusions_from_conflicts,
    label_conflicts,
    sha256_of_text,
)
from src.utils.config import load_experiment_config
from src.utils.results import save_results

EXPERIMENT_NAME = "dataset_audit"

# Regression checks: SII 2027 audit aggregates (5 sites, pHash threshold 6).
# Differences must be explained before the new cohort is accepted (plan §7.5).
SII_EXPECTED = {
    "busi": {"raw": 780, "conflict_excluded": 18, "cleaned": 762, "binary_eligible": 631},
    "bus_uclm": {"raw": 683, "binary_eligible": 264, "binary_groups_approx": 35},
    "bus_bra": {"raw": 1875, "native_cases": 1064},
    "busi_whu": {"raw": 927, "conflict_excluded": 2, "cleaned": 925},
    "udiat": {"raw": 163, "audit_groups": 158},
    "_cross_dataset_overlap": 0,
}


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> str:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    return sha256_of_text(path.read_text())


def run_experiment(config: dict) -> dict:
    cfg = config["dataset_audit"]
    sites = cfg["sites"]
    ph = cfg["phash"]
    ham = ph["hamming_threshold"]
    fold_cfg = cfg["folds"]
    out_dir = PROJECT_ROOT / cfg["outputs"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Inventory + hashing ---------------------------------------------
    all_records: list[dict] = []
    per_dataset: dict[str, dict] = {}
    for name in sites:
        root = config["bus_datasets"][name]
        print(f"[audit] enumerating {name} ...", flush=True)
        recs = enumerate_dataset(name, root, side=ph["side"], lowfreq=ph["lowfreq"])
        cc: dict[str, int] = defaultdict(int)
        for r in recs:
            cc[r["label_name"]] += 1
        per_dataset[name] = {
            "n_images": len(recs),
            "class_counts": dict(cc),
            "group_source": recs[0]["group_source"] if recs else "",
            "n_with_group_id": sum(1 for r in recs if r["patient_group"]),
        }
        all_records.extend(recs)
        print(f"          {len(recs)} images, classes={dict(cc)}", flush=True)

    # --- 2. Duplicate clusters + conflicts + exclusions ---------------------
    print(f"[audit] clustering {len(all_records)} images (Hamming ≤ {ham}) ...",
          flush=True)
    exact, near = build_clusters(all_records, hamming_threshold=ham)
    exact_of = {i: g for g, idxs in exact.items() for i in idxs}
    near_of = {i: g for g, idxs in near.items() for i in idxs}
    for i, r in enumerate(all_records):
        r["exact_group"] = exact_of.get(i, "")
        r["near_group"] = near_of.get(i, "")

    conflicts = label_conflicts(exact, near, all_records)
    excluded_idx = exclusions_from_conflicts(conflicts)
    excluded_paths = {all_records[i]["path"] for i in excluded_idx}

    cross = {g: idxs for g, idxs in near.items()
             if len({all_records[i]["dataset"] for i in idxs}) > 1}

    # --- 3. Group-aware canonical folds (per dataset, in-task, non-excluded)-
    dropped_out_of_task: dict[str, int] = defaultdict(int)
    manifest: list[dict] = []
    fold_stats: dict[str, dict] = {}
    for name in sites:
        recs = [r for r in all_records
                if r["dataset"] == name and r["path"] not in excluded_paths]
        keep = []
        for r in recs:
            if r["label_name"] in TASK_CLASSES[name]:
                keep.append(r)
            else:
                dropped_out_of_task[name] += 1
        rows = build_dataset_folds(keep, n_folds=fold_cfg["n_folds"],
                                   seed=fold_cfg["seed"])
        manifest.extend(rows)
        n_groups = len({r["group_id"] for r in rows})
        n_bin_groups = len({r["group_id"] for r in rows if r["binary_eligible"]})
        per_fold: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for r in rows:
            per_fold[r["fold"]][r["label_name"]] += 1
        fold_stats[name] = {
            "n_manifest_images": len(rows),
            "n_binary_eligible": sum(r["binary_eligible"] for r in rows),
            "n_groups": n_groups,
            "n_binary_groups": n_bin_groups,
            "per_fold": {int(f): dict(c) for f, c in sorted(per_fold.items())},
        }

    # --- 4. Write private artefacts (git-ignored) + hash --------------------
    manifest.sort(key=lambda r: (r["dataset"], r["filename"]))
    inv_hash = _write_csv(
        out_dir / "image_inventory.csv",
        ["dataset", "filename", "label_idx", "label_name", "width", "height",
         "mode", "sha256", "phash_hex", "patient_group", "group_source",
         "vendor", "exact_group", "near_group", "path"],
        all_records,
    )
    excl_rows = [{
        "dataset": all_records[i]["dataset"], "filename": all_records[i]["filename"],
        "label_name": all_records[i]["label_name"], "reason": "label_conflict",
        "path": all_records[i]["path"],
    } for i in sorted(excluded_idx)]
    excl_hash = _write_csv(
        out_dir / "exclusions.csv",
        ["dataset", "filename", "label_name", "reason", "path"], excl_rows,
    )
    man_hash = _write_csv(
        out_dir / "fold_manifest.csv",
        ["dataset", "filename", "path", "label_idx", "label_name",
         "label_binary", "binary_eligible", "group_id", "fold"], manifest,
    )

    # --- 5. Regression checks vs SII ----------------------------------------
    regression: dict[str, dict] = {}
    for name in sites:
        exp = SII_EXPECTED.get(name)
        if not exp:
            regression[name] = {"note": "new site (no SII baseline)"}
            continue
        got = {
            "raw": per_dataset[name]["n_images"],
            "binary_eligible": fold_stats[name]["n_binary_eligible"],
            "conflict_excluded": sum(1 for i in excluded_idx
                                     if all_records[i]["dataset"] == name),
        }
        regression[name] = {"expected": exp, "got": got,
                            "match": all(got.get(k) == v for k, v in exp.items()
                                         if k in got)}

    # --- 6. Aggregate results (path-free) -----------------------------------
    for name in sites:
        per_dataset[name].update({
            "n_conflict_excluded": sum(1 for i in excluded_idx
                                       if all_records[i]["dataset"] == name),
            "n_dropped_out_of_task": dropped_out_of_task[name],
            **fold_stats[name],
        })

    return {
        "config": {"phash": ph, "hamming_threshold": ham,
                   "n_folds": fold_cfg["n_folds"], "seed": fold_cfg["seed"]},
        "datasets": per_dataset,
        "totals": {
            "n_images": len(all_records),
            "n_exact_clusters": len(exact),
            "n_near_clusters": len(near),
            "n_near_images": sum(len(v) for v in near.values()),
            "n_cross_dataset_clusters": len(cross),
            "n_label_conflict_clusters": len(conflicts),
            "n_excluded": len(excluded_idx),
            "n_manifest_images": len(manifest),
        },
        "conflict_clusters": [
            {"kind": k, "group": g,
             "datasets": sorted({all_records[i]["dataset"] for i in idxs}),
             "labels": sorted({all_records[i]["label_name"] for i in idxs}),
             "n": len(idxs)} for k, g, idxs in conflicts],
        "cross_dataset_clusters": [
            {"group": g, "datasets": sorted({all_records[i]["dataset"] for i in idxs}),
             "n": len(idxs)} for g, idxs in cross.items()],
        "manifest_hashes": {"image_inventory": inv_hash, "exclusions": excl_hash,
                            "fold_manifest": man_hash},
        "regression_vs_sii": regression,
    }


def format_summary(results: dict, config: dict) -> list[str]:
    t = results["totals"]
    lines = [
        "=" * 68,
        "WP1 Dataset Audit — six BUS sites",
        "=" * 68,
        f"Total images: {t['n_images']}   exact-dup clusters: {t['n_exact_clusters']}"
        f"   near-dup clusters: {t['n_near_clusters']}",
        f"Cross-dataset overlap clusters: {t['n_cross_dataset_clusters']}"
        f"   label-conflict clusters: {t['n_label_conflict_clusters']}"
        f"   excluded images: {t['n_excluded']}",
        f"Manifest images (in-task, cleaned): {t['n_manifest_images']}",
        "",
        f"{'site':<10}{'imgs':>6}{'bin-elig':>9}{'groups':>8}{'bin-grp':>8}"
        f"{'excl':>6}{'src':>22}",
        "-" * 68,
    ]
    for name, d in results["datasets"].items():
        lines.append(
            f"{name:<10}{d['n_images']:>6}{d.get('n_binary_eligible', 0):>9}"
            f"{d.get('n_groups', 0):>8}{d.get('n_binary_groups', 0):>8}"
            f"{d.get('n_conflict_excluded', 0):>6}{d['group_source']:>22}"
        )
    lines += ["", "Regression vs SII audit:"]
    for name, reg in results["regression_vs_sii"].items():
        if "note" in reg:
            lines.append(f"  {name:<10} {reg['note']}")
        else:
            flag = "OK" if reg["match"] else "*** MISMATCH ***"
            lines.append(f"  {name:<10} {flag}  expected={reg['expected']} got={reg['got']}")
    if results["conflict_clusters"]:
        lines += ["", "Label-conflict clusters (excluded in full):"]
        for c in results["conflict_clusters"]:
            lines.append(f"  {c['kind']} {c['group']} {c['datasets']} "
                         f"labels={c['labels']} n={c['n']}")
    lines += ["", f"Manifest SHA-256 (fold_manifest): "
              f"{results['manifest_hashes']['fold_manifest'][:16]}…"]
    return lines


def main():
    config = load_experiment_config(EXPERIMENT_NAME)
    results = run_experiment(config)
    summary = format_summary(results, config)
    for line in summary:
        print(line)
    save_results(EXPERIMENT_NAME, results, config, summary_lines=summary)


if __name__ == "__main__":
    main()
