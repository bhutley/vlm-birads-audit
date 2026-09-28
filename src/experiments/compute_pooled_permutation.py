"""Pooled-label permutation test — SECONDARY sensitivity for STAT-01.

Reviewer 2 blocked the fixed-bank placebo rank on exchangeability: the 5 clinical
axes are hand-selected from a closed clinical lexicon while the 4368 reference
statistics are subsets of a *separately curated* 16-axis placebo pool, so nothing
makes the observed statistic exchangeable with the reference set.

The obvious repair — pool clinical + placebo and enumerate C(21,5) — does not fix
it: the supplement already states the placebo pool is matched on construction but
**not** on semantic richness, which is exactly why the anatomy pool was built. A
pooled-label randomisation over {clinical, placebo} would assume an exchangeability
the manuscript itself documents as false.

So we pool the clinical axes with the **richness-matched anatomy** pool
(``compute_reviewer_addenda.ANATOMY_CONCEPTS``: rich noun phrases on both poles,
identical carrier templates and mean-difference encoding, no BI-RADS descriptor and
no malignancy semantics) and enumerate every full-rank 5-subset of the 15-axis
union. The observed clinical subset is one of the 3003 and is scored in the same
loop, so under the null "the clinical/anatomy label is uninformative about
alignment with w" the rank is an *exact* randomisation p-value. Validity is
demonstrated by simulation in ``tests/test_global_inference.py``
(``test_pooled_permutation_p_is_uniform_under_exchangeability``).

DECLARED 2026-07-23 BEFORE COMPUTATION — see
``docs/20260723-pooled-permutation-declaration.md`` for the specification, the
pre-committed interpretation of every outcome, and the recorded prediction. This is
a **secondary** sensitivity: it does not replace the primary verdict, its Holm
family is descriptive, and it must not be promoted to primary after the fact.

Usage:
    PYTHONHASHSEED=0 python -m src.experiments.compute_pooled_permutation
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cohort import CohortManifest
from src.evaluation import concept_directions as cd
from src.evaluation import global_inference as gi
from src.evaluation.metrics import holm_bonferroni
from src.experiments.compute_reviewer_addenda import ANATOMY_CONCEPTS, _anatomy_prompts
from src.experiments.compute_rho_geometry import (
    _load_model,
    _load_site_embeddings,
    _probe_w_grouped,
)
from src.utils.config import load_experiment_config
from src.utils.reproducibility import set_all_seeds
from src.utils.results import save_results

EXPERIMENT_NAME = "pooled_permutation"


def run_experiment(config: dict) -> dict:
    repro = config["reproducibility"]
    set_all_seeds(repro["global_seed"], deterministic=repro["deterministic_algorithms"])
    cfg = config[EXPERIMENT_NAME]
    bank = config["birads_concept_bank"]

    manifest = CohortManifest.load(PROJECT_ROOT / cfg["cohort_manifest"])
    concept_prompts = cd.concept_prompts_from_config(bank)
    anatomy_prompts = _anatomy_prompts(bank)
    C = cfg["probe_C"]

    results: dict = {
        "declaration": "docs/20260723-pooled-permutation-declaration.md",
        "status": "SECONDARY sensitivity — not the confirmatory family",
        "control_pool": cfg["control_pool"],
        "clinical_axes": list(concept_prompts.keys()),
        "control_axes": list(ANATOMY_CONCEPTS.keys()),
        "backbones_run": list(cfg["backbones"]),
        "by_backbone": {},
    }

    for backbone in cfg["backbones"]:
        print(f"\n=== backbone: {backbone} ===")
        model = _load_model(backbone, config)
        embed_fn = lambda texts: cd.encode_texts(model, texts)

        try:
            clin_dirs = cd.build_text_directions(concept_prompts, embed_fn)
            anat_dirs = cd.build_text_directions(anatomy_prompts, embed_fn)
        except KeyError as e:
            # Fail-closed, outcome-independent exclusion: a cache-served backbone
            # whose text cache predates the control pool cannot be scored at all,
            # so this is decided by cache contents, never by a statistic
            # (declaration §2 amendment, 23 July 2026).
            print(f"  [exclude] {backbone}: {e}")
            results["by_backbone"][backbone] = {
                "error": "control-pool prompts absent from the cached text embeddings",
                "detail": str(e),
                "outcome_independent": True,
            }
            continue
        clin_names = list(clin_dirs.keys())
        m = len(clin_names)

        # Pool rows: clinical first, then the richness-matched controls. The
        # observed subset is therefore rows 0..m-1 and is enumerated with the rest.
        pool = np.stack([clin_dirs[n] for n in clin_names]
                        + list(anat_dirs.values()), axis=0)
        clinical_idx = tuple(range(m))

        site_dirs, site_order = [], []
        for site in cfg["sites"]:
            try:
                se = _load_site_embeddings(model, backbone, site, config, manifest,
                                           batch_size=cfg["batch_size"])
            except (KeyError, FileNotFoundError) as e:
                print(f"  [skip] {site}: {e}")
                continue
            if len(np.unique(se.y)) < 2:
                print(f"  [skip] {site}: only one class present")
                continue
            site_dirs.append(_probe_w_grouped(se.Z, se.y, se.sample_weights, C))
            site_order.append(site)
            print(f"  {site:<10} n={se.Z.shape[0]:<5} g={se.n_groups}")

        if not site_dirs:
            results["by_backbone"][backbone] = {"error": "no usable sites"}
            continue

        r = gi.pooled_label_permutation(np.stack(site_dirs, axis=0), pool,
                                        clinical_idx, m)
        r["sites"] = site_order
        r["m"] = m
        results["by_backbone"][backbone] = r
        print(f"  -> T_clin={r['T_clin']:.4f}  rank={r['rank']}/{r['n_reference']}"
              f"  p_pooled={r['p_pooled']:.4f}"
              f"  pool_mean={r['null_mean']:.4f}  max={r['null_max']:.4f}")

    # Holm across the six backbones — DESCRIPTIVE (declaration §2); the
    # confirmatory family remains the one in compute_rho_geometry.
    ps = {b: blob["p_pooled"] for b, blob in results["by_backbone"].items()
          if "p_pooled" in blob}
    holm = holm_bonferroni(ps) if ps else {}
    results["holm_descriptive"] = {
        b: {"p_pooled": ps[b],
            "holm_reject": bool((holm.get(b) or {}).get("reject")),
            "holm_threshold": (holm.get(b) or {}).get("holm_threshold")}
        for b in ps
    }
    return results


def format_summary(results: dict, config: dict) -> list[str]:
    n_ctrl = len(results["control_axes"])
    m = len(results["clinical_axes"])
    L = [
        "=" * 78,
        "Pooled-label permutation test (SECONDARY sensitivity — STAT-01)",
        "=" * 78,
        f"declaration : {results['declaration']}",
        f"status      : {results['status']}",
        "",
        f"pool        : {m} clinical + {n_ctrl} {results['control_pool']} (richness-matched)",
        f"clinical    : {', '.join(results['clinical_axes'])}",
        f"control     : {', '.join(results['control_axes'])}",
        "",
        "H0: the clinical/control label is uninformative about alignment with w.",
        "The observed clinical subset is one of the enumerated subsets, so",
        "p = #{T_S >= T_clin} / n_reference is EXACT (no +1 smoothing).",
        "",
        f"{'backbone':<12}{'T_clin':>9}{'pool_mean':>11}{'excess':>9}"
        f"{'rank':>12}{'p_pooled':>10}{'holm':>7}",
        "-" * 78,
    ]
    hd = results.get("holm_descriptive", {})
    for b, r in results["by_backbone"].items():
        if "p_pooled" not in r:
            why = r.get("error", "not run")
            tag = "EXCLUDED (outcome-independent)" if r.get("outcome_independent") else "SKIPPED"
            L.append(f"{b:<12}  {tag}: {why}")
            continue
        rej = "rej" if (hd.get(b) or {}).get("holm_reject") else "--"
        rank_s = f"{r['rank']}/{r['n_reference']}"
        L.append(
            f"{b:<12}{r['T_clin']:>9.4f}{r['null_mean']:>11.4f}"
            f"{r['excess_over_pool_mean']:>+9.4f}{rank_s:>12}"
            f"{r['p_pooled']:>10.4f}{rej:>7}"
        )
    L += [
        "",
        "Composition of the subsets that OUTRANK the clinical set, by how many of the",
        f"{m} clinical axes they retain. k=0 means a PURE control subset wins (controls",
        "explain the alignment); mass at high k means the labelled set is simply not",
        "the best subset of the pooled bank (one weak clinical axis).",
        "",
        f"{'backbone':<12}" + "".join(f"{'k='+str(k):>9}" for k in range(m + 1))
        + f"{'outrank':>9}",
        f"{'(reference)':<12}",
    ]
    for b, r in results["by_backbone"].items():
        if "outrank_by_n_clinical" not in r:
            continue
        tot = r["reference_by_n_clinical"]
        L[-1] = (f"{'(reference)':<12}"
                 + "".join(f"{tot[str(k)] if str(k) in tot else tot[k]:>9}"
                           for k in range(m + 1)) + f"{r['n_reference']:>9}")
        break
    for b, r in results["by_backbone"].items():
        h = r.get("outrank_by_n_clinical")
        if not h:
            continue
        L.append(f"{b:<12}"
                 + "".join(f"{h[str(k)] if str(k) in h else h[k]:>9}"
                           for k in range(m + 1))
                 + f"{r['n_outranking']:>9}")

    L += [
        "",
        "rank 1 = the clinical subspace beats every richness-matched alternative.",
        "Read against the pre-committed interpretation table in the declaration:",
        "an unfavourable result is reported, not demoted, and the primary verdict",
        "is NOT revised on this secondary test alone.",
    ]
    return L


def main():
    config = load_experiment_config(EXPERIMENT_NAME)
    results = run_experiment(config)
    summary = format_summary(results, config)
    for line in summary:
        print(line)
    save_results(EXPERIMENT_NAME, results, config, summary_lines=summary)


if __name__ == "__main__":
    main()
