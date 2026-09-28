"""ROB-04: is the expressibility verdict an artefact of the probe's regularisation?

E3 fits every probe at ``C=1``. The reviewer's point is precise -- *fold stability
does not establish regularisation stability* -- so we sweep a fixed grid
``C in {0.01, 0.1, 1, 10, 100}`` and re-run the whole audit at each grid point.

``C=1`` **remains primary**. This sweep exists to show the story is not a
regularisation artefact; it must never be used to pick a new primary ``C``, which
would be selecting an analysis choice on its outcome.

Per backbone x site x grid point we report the probe's group-disjoint out-of-fold
discrimination (so predictive quality is visible alongside geometry, as in PERF-01)
and ``cos(w_C, w_{C=1})``, the direct measure of whether regularisation moves the
decision *direction* at all -- which is what rho depends on. Per backbone x grid
point we re-run the global test.

Unlike ROB-03, both null distributions **do** move with ``C`` (they depend on the
fitted site directions), so they are recomputed at every grid point rather than
shared. The cluster-bootstrap CI is the one primary criterion not recomputed: it
needs 2000 refits per grid point per site, so the verdict here covers the three
remaining criteria and the CI is held at its ``C=1`` value. This is stated in the
artefact.

SENSITIVITY -- see ``docs/20260724-probe-regularisation-declaration.md``.

Usage:
    PYTHONHASHSEED=0 python -m src.experiments.compute_probe_regularisation
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
from src.evaluation import probe_performance as pp
from src.evaluation.metrics import holm_bonferroni
from src.experiments.compute_rho_geometry import (
    _load_model,
    _load_site_embeddings,
    _probe_w_grouped,
    _probe_wb_grouped,
)
from src.utils.config import load_experiment_config
from src.utils.reproducibility import set_all_seeds
from src.utils.results import save_results

EXPERIMENT_NAME = "probe_regularisation"


def run_experiment(config: dict) -> dict:
    repro = config["reproducibility"]
    set_all_seeds(repro["global_seed"], deterministic=repro["deterministic_algorithms"])
    cfg = config[EXPERIMENT_NAME]
    bank = config["birads_concept_bank"]
    seed = repro["global_seed"]

    manifest = CohortManifest.load(PROJECT_ROOT / cfg["cohort_manifest"])
    concept_prompts = cd.concept_prompts_from_config(bank)
    placebo_prompts = cd.placebo_prompts_from_config(bank)
    grid = [float(c) for c in cfg["C_grid"]]
    C_primary = float(cfg["C_primary"])
    k_gate = float(cfg["gate_k_primary"])

    results: dict = {
        "status": "SENSITIVITY — C=1 remains primary; do not select a new C from this",
        "declaration": "docs/20260724-probe-regularisation-declaration.md",
        "C_grid": grid,
        "C_primary": C_primary,
        "criteria_covered": ("Holm(p_random), placebo tail<0.05, gate; the cohort "
                             "bootstrap CI is held at its C=1 value (2000 refits per "
                             "grid point is prohibitive)"),
        "by_backbone": {},
    }

    for backbone in cfg["backbones"]:
        print(f"\n=== backbone: {backbone} ===")
        model = _load_model(backbone, config)
        embed_fn = lambda texts: cd.encode_texts(model, texts)

        B, _ = cd.stack_basis(cd.build_text_directions(concept_prompts, embed_fn),
                              orthonormalize=cfg["orthonormalize_basis"])
        placebo_pool = np.stack(
            list(cd.build_text_directions(placebo_prompts, embed_fn).values()), axis=0)
        m, d_emb = int(B.shape[1]), int(B.shape[0])

        # load each site once; refit at every grid point
        sites = {}
        for site in cfg["sites"]:
            try:
                se = _load_site_embeddings(model, backbone, site, config, manifest,
                                           batch_size=cfg["batch_size"])
            except (KeyError, FileNotFoundError) as e:
                print(f"  [skip] {site}: {e}")
                continue
            if len(np.unique(se.y)) < 2:
                continue
            sites[site] = se

        if not sites:
            results["by_backbone"][backbone] = {"error": "no usable sites"}
            continue

        w_primary = {s: _probe_w_grouped(se.Z, se.y, se.sample_weights, C_primary)
                     for s, se in sites.items()}

        by_C = {}
        for C in grid:
            per_site, dirs = {}, []
            for s, se in sites.items():
                w = _probe_w_grouped(se.Z, se.y, se.sample_weights, C)
                wp = w_primary[s]
                cos = float(w @ wp / (np.linalg.norm(w) * np.linalg.norm(wp) + 1e-12))
                oof = pp.oof_performance(
                    se.Z, se.y, se.group_ids, se.folds,
                    lambda Zt, yt, wt: _probe_wb_grouped(Zt, yt, wt, C))
                per_site[s] = {
                    "cos_with_primary": cos,
                    "oof_auc": oof["auc"],
                    "oof_balanced_accuracy": oof["balanced_accuracy"],
                }
                dirs.append(w)
            W = np.stack(dirs, axis=0)
            gc = gi.global_core_test(W, B, placebo_pool,
                                     n_random=int(cfg["global_random_trials"]), seed=seed)
            by_C[f"C={C:g}"] = {
                "C": C,
                "by_site": per_site,
                "min_cos_with_primary": float(min(v["cos_with_primary"]
                                                  for v in per_site.values())),
                "median_oof_auc": float(np.median([v["oof_auc"] for v in per_site.values()])),
                "T_clin": gc["T_clin"],
                "p_random": gc["p_random"],
                "placebo_tail_fraction": gc["placebo_tail_fraction"],
                "mean_excess_placebo": gc["mean_excess_placebo"],
                "passes_gate": bool(gc["mean_excess_placebo"] > k_gate * m / d_emb),
            }
            print(f"  C={C:<7g} exc={gc['mean_excess_placebo']:+.4f} "
                  f"r_plac={gc['placebo_tail_fraction']:.4f} "
                  f"min_cos={by_C[f'C={C:g}']['min_cos_with_primary']:.4f} "
                  f"medAUC={by_C[f'C={C:g}']['median_oof_auc']:.3f}")

        results["by_backbone"][backbone] = {
            "m": m, "embed_dim": d_emb, "sites": list(sites),
            "by_C": by_C,
        }

    # Holm across the six backbones WITHIN each grid point (descriptive), then the
    # three-criterion verdict.
    for C in grid:
        key = f"C={C:g}"
        ps = {b: blob["by_C"][key]["p_random"]
              for b, blob in results["by_backbone"].items() if "by_C" in blob}
        holm = holm_bonferroni(ps) if ps else {}
        for b, blob in results["by_backbone"].items():
            if "by_C" not in blob:
                continue
            r = blob["by_C"][key]
            hb = holm.get(b) or {}
            r["holm_adjusted_p_random"] = hb.get("holm_adjusted_p")
            r["holm_reject"] = bool(hb.get("reject"))
            r["expressible_3crit"] = bool(
                r["holm_reject"] and r["placebo_tail_fraction"] < 0.05
                and r["passes_gate"])
    return results


def format_summary(results: dict, config: dict) -> list[str]:
    grid = results["C_grid"]
    keys = [f"C={c:g}" for c in grid]
    L = [
        "=" * 78,
        "ROB-04 — logistic regularisation sensitivity (SENSITIVITY)",
        "=" * 78,
        f"declaration : {results['declaration']}",
        f"status      : {results['status']}",
        f"criteria    : {results['criteria_covered']}",
        "",
        "EXPRESSIBLE (3 criteria) across the C grid:",
        "",
        f"{'backbone':<12}" + "".join(f"{k:>12}" for k in keys),
        "-" * (12 + 12 * len(keys)),
    ]
    for b, blob in results["by_backbone"].items():
        if "by_C" not in blob:
            L.append(f"{b:<12}  (no sites)")
            continue
        L.append(f"{b:<12}" + "".join(
            f"{('E' if blob['by_C'][k]['expressible_3crit'] else '.'):>12}" for k in keys))

    L += ["", "Equal-site mean placebo excess / min cos(w_C, w_1) / median OOF AUC:", ""]
    for b, blob in results["by_backbone"].items():
        if "by_C" not in blob:
            continue
        L.append(f"  {b}")
        for k in keys:
            r = blob["by_C"][k]
            L.append(f"    {k:<10} exc={r['mean_excess_placebo']:+.4f}  "
                     f"r_plac={r['placebo_tail_fraction']:.4f}  "
                     f"min_cos={r['min_cos_with_primary']:+.4f}  "
                     f"medAUC={r['median_oof_auc']:.3f}")
    L += [
        "",
        "min_cos near 1 means regularisation barely moves the decision DIRECTION,",
        "which is all rho depends on. C=1 remains primary by declaration.",
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
