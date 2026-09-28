"""ROB-01/02: does the expressibility verdict survive dropping any clinical axis?

The five BI-RADS axes were fixed a priori, but two reviewer concerns bear on them:
the placebo pool received a drop-one analysis while the *clinical* bank never did
(ROB-01), and **orientation** carries no external per-feature label on BrEaST, so a
verdict resting on it is unvalidated (ROB-02).

For each axis we drop it, rebuild the basis at ``m=4``, and re-run the whole global
test against **dimension-matched** references: random subspaces at ``m=4``, all
``C(16,4)=1820`` full-rank placebo 4-subsets, and the gate at ``2m/d`` with ``m=4``.
Holding ``m`` fixed between statistic and null is the invariant in ``CLAUDE.md`` --
scoring an ``m=4`` subspace against an ``m=5`` null would manufacture a difference
out of dimension counting alone.

Efficiency note: the cluster bootstrap refits the group-weighted probe once per
replicate and scores **every** variant basis off that single refit, so all five
leave-one-axis CIs cost one bootstrap pass rather than five. Group selections reuse
E3's site-seeded draws, so these intervals are paired to the same cohort
perturbations as the primary analysis.

SENSITIVITY ANALYSIS -- see ``docs/20260724-axis-robustness-declaration.md``. It does
not replace the five-axis primary verdict; its Holm families are descriptive.

Usage:
    PYTHONHASHSEED=0 python -m src.experiments.compute_axis_robustness
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cohort import CohortManifest, group_class_balanced_weights
from src.evaluation import concept_directions as cd
from src.evaluation import decomposition as dec
from src.evaluation import global_inference as gi
from src.evaluation.metrics import holm_bonferroni
from src.experiments.compute_rho_geometry import (
    _boot_group_selections,
    _load_model,
    _load_site_embeddings,
    _probe_w_grouped,
)
from src.utils.config import load_experiment_config
from src.utils.reproducibility import set_all_seeds
from src.utils.results import save_results

EXPERIMENT_NAME = "axis_robustness"
FULL = "all5"


def _variant_bases(directions: dict, orthonormalize: bool) -> dict:
    """``{variant: (B, kept_names)}`` for the full bank and each leave-one-out."""
    names = list(directions)
    out = {}
    for dropped in [None] + names:
        kept = [n for n in names if n != dropped]
        sub = {n: directions[n] for n in kept}
        B, _ = cd.stack_basis(sub, orthonormalize=orthonormalize)
        out[FULL if dropped is None else f"drop_{dropped}"] = (B, kept)
    return out


def _bootstrap_all_variants(se, variants, projectors, C, sels):
    """One refit per replicate; score every variant basis off it.

    Returns ``{variant: [excess_placebo per valid replicate]}``. Replicates missing
    a class are skipped (identically for every variant, so the CIs stay paired).
    """
    rows_by_group: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(se.group_ids):
        rows_by_group[g].append(i)

    acc = {v: [] for v in variants}
    n_invalid = 0
    for sel_groups in sels:
        idx = np.fromiter((i for g in sel_groups for i in rows_by_group[g]), dtype=int)
        yb = se.y[idx]
        if len(np.unique(yb)) < 2:
            n_invalid += 1
            continue
        wgt = group_class_balanced_weights(se.group_ids[idx], yb)
        wb = _probe_w_grouped(se.Z[idx], yb, wgt, C)     # the one expensive step
        for v, (B, _) in variants.items():
            acc[v].append(float(dec.rho(wb, B))
                          - gi.rho_from_projector(wb, projectors[v]))
    return acc, (n_invalid / len(sels) if sels else 0.0)


def run_experiment(config: dict) -> dict:
    repro = config["reproducibility"]
    set_all_seeds(repro["global_seed"], deterministic=repro["deterministic_algorithms"])
    cfg = config[EXPERIMENT_NAME]
    bank = config["birads_concept_bank"]
    seed = repro["global_seed"]

    manifest = CohortManifest.load(PROJECT_ROOT / cfg["cohort_manifest"])
    concept_prompts = cd.concept_prompts_from_config(bank)
    placebo_prompts = cd.placebo_prompts_from_config(bank)
    C = cfg["probe_C"]
    k_gate = float(cfg["gate_k_primary"])

    results: dict = {
        "status": "SENSITIVITY — does not replace the five-axis primary verdict",
        "declaration": "docs/20260724-axis-robustness-declaration.md",
        "clinical_axes": list(concept_prompts),
        "gate_k": k_gate,
        "by_backbone": {},
    }

    for backbone in cfg["backbones"]:
        print(f"\n=== backbone: {backbone} ===")
        model = _load_model(backbone, config)
        embed_fn = lambda texts: cd.encode_texts(model, texts)

        directions = cd.build_text_directions(concept_prompts, embed_fn)
        placebo_pool = np.stack(
            list(cd.build_text_directions(placebo_prompts, embed_fn).values()), axis=0)
        variants = _variant_bases(directions, cfg["orthonormalize_basis"])
        # dimension-matched mean placebo projector per variant (m differs by variant)
        projectors = {v: gi.mean_projector(placebo_pool, B.shape[1])
                      for v, (B, _) in variants.items()}

        site_dirs, site_order, boot_by_site = [], [], {}
        for site in cfg["sites"]:
            try:
                se = _load_site_embeddings(model, backbone, site, config, manifest,
                                           batch_size=cfg["batch_size"])
            except (KeyError, FileNotFoundError) as e:
                print(f"  [skip] {site}: {e}")
                continue
            if len(np.unique(se.y)) < 2:
                continue
            site_dirs.append(_probe_w_grouped(se.Z, se.y, se.sample_weights, C))
            site_order.append(site)
            sels = _boot_group_selections(se.group_ids, cfg["cluster_bootstrap_trials"],
                                          seed, site)
            acc, invalid = _bootstrap_all_variants(se, variants, projectors, C, sels)
            boot_by_site[site] = acc
            print(f"  {site:<10} n={se.Z.shape[0]:<5} g={se.n_groups:<5} "
                  f"(boot invalid {invalid:.1%})")

        if not site_dirs:
            results["by_backbone"][backbone] = {"error": "no usable sites"}
            continue
        W = np.stack(site_dirs, axis=0)

        by_variant = {}
        for v, (B, kept) in variants.items():
            gc = gi.global_core_test(W, B, placebo_pool,
                                     n_random=int(cfg["global_random_trials"]),
                                     seed=seed)
            # equal-site mean excess per replicate -> paired CI, as in E3
            reps = np.array([boot_by_site[s][v] for s in site_order])   # (S, R)
            mean_over_sites = reps.mean(axis=0) if reps.size else np.array([])
            ci = ([float(np.percentile(mean_over_sites, 2.5)),
                   float(np.percentile(mean_over_sites, 97.5))]
                  if mean_over_sites.size else None)
            by_variant[v] = {
                "kept_axes": kept,
                "m": int(B.shape[1]),
                "T_clin": gc["T_clin"],
                "p_random": gc["p_random"],
                "placebo_tail_fraction": gc["placebo_tail_fraction"],
                "placebo_rank": gc["placebo_rank"],
                "n_placebo_reference": gc["n_placebo_reference"],
                "mean_excess_placebo": gc["mean_excess_placebo"],
                "mean_excess_placebo_ci": ci,
            }
        results["by_backbone"][backbone] = {
            "embed_dim": int(placebo_pool.shape[1]),
            "sites": site_order,
            "by_variant": by_variant,
        }

    # Holm across backbones WITHIN each variant (descriptive), then the verdict.
    variant_names = [FULL] + [f"drop_{a}" for a in results["clinical_axes"]]
    for v in variant_names:
        ps = {b: blob["by_variant"][v]["p_random"]
              for b, blob in results["by_backbone"].items() if "by_variant" in blob}
        holm = holm_bonferroni(ps) if ps else {}
        for b, blob in results["by_backbone"].items():
            if "by_variant" not in blob:
                continue
            r = blob["by_variant"][v]
            hb = holm.get(b) or {}
            ci = r["mean_excess_placebo_ci"]
            r["holm_adjusted_p_random"] = hb.get("holm_adjusted_p")
            r["verdict"] = gi.audit_verdict(
                bool(hb.get("reject")), r["placebo_tail_fraction"],
                r["mean_excess_placebo"], ci[0] if ci else None,
                r["m"], blob["embed_dim"], k=k_gate)
    return results


def format_summary(results: dict, config: dict) -> list[str]:
    axes = results["clinical_axes"]
    variants = [FULL] + [f"drop_{a}" for a in axes]
    L = [
        "=" * 78,
        "ROB-01/02 — leave-one-clinical-axis robustness (SENSITIVITY)",
        "=" * 78,
        f"declaration : {results['declaration']}",
        f"status      : {results['status']}",
        "",
        "Each variant drops one axis and re-runs the global test against",
        "DIMENSION-MATCHED references (random + all C(16,4)=1820 placebo 4-subsets)",
        "with the gate at 2m/d for m=4. 'drop_orientation' is the named clinically",
        "important row: orientation has no external per-feature label (ROB-02).",
        "",
        "EXPRESSIBLE per variant (E = all four criteria hold):",
        "",
        f"{'backbone':<12}" + "".join(f"{v.replace('drop_','-'):>13}" for v in variants),
        "-" * (12 + 13 * len(variants)),
    ]
    for b, blob in results["by_backbone"].items():
        if "by_variant" not in blob:
            L.append(f"{b:<12}  (no sites)")
            continue
        row = f"{b:<12}"
        for v in variants:
            r = blob["by_variant"][v]
            row += f"{('E' if r['verdict']['expressible'] else '.'):>13}"
        L.append(row)

    L += ["", "Equal-site mean placebo excess [95% CI] per variant:", ""]
    for b, blob in results["by_backbone"].items():
        if "by_variant" not in blob:
            continue
        L.append(f"  {b}")
        base = blob["by_variant"][FULL]["mean_excess_placebo"]
        for v in variants:
            r = blob["by_variant"][v]
            ci = r["mean_excess_placebo_ci"]
            ci_s = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if ci else "--"
            delta = "" if v == FULL else f"  d={r['mean_excess_placebo']-base:+.3f}"
            fails = [k for k in ("holm_reject_random", "placebo_extreme",
                                 "ci_excludes_zero", "passes_gate")
                     if not r["verdict"][k]]
            L.append(f"    {v:<20} m={r['m']} exc={r['mean_excess_placebo']:+.4f} "
                     f"{ci_s:>18} r_plac={r['placebo_tail_fraction']:.4f}{delta}"
                     f"{'  fails=' + ','.join(fails) if fails else ''}")
    L += [
        "",
        "Read as sensitivity, not a new verdict: the five-axis analysis remains",
        "primary. A variant that flips identifies an axis the verdict leans on.",
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
