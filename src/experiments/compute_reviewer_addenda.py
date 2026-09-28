"""Reviewer addenda (2026-07): three post-review analyses (P0-1 / P1-4 / P1-6).

Standalone. Reuses the E3 (``compute_rho_geometry``) embedding + group-weighted
probe path verbatim, so ``w``, ``B`` and the placebo pool are identical to the
confirmatory run; nothing here mutates the locked E3 outputs. Adds, per backbone:

  P0-1  anatomy null      -- rho excess over a semantically RICH but non-diagnostic
                            ANATOMY text-direction pool (the richness-matched control
                            the supplement defers to future work). Core-test style:
                            equal-site mean excess, exhaustive-subset permutation p
                            (C(10,5)=252), and the 2m/d magnitude gate.
  P1-4  completeness      -- AUC of the probe restricted to the clinical subspace
                            (P_C w) vs full w, vs the mean placebo / anatomy subspace:
                            does C carry DECISION content, not only direction alignment?
  P1-6  modality retention -- mean image-subspace (PCA-50, matching E1) norm retention
                            of the clinical vs placebo vs anatomy pools per backbone:
                            is the clinical excess a manifold-proximity artefact?

Usage:
    PYTHONHASHSEED=0 python -m src.experiments.compute_reviewer_addenda [backbones...]
    (default backbones: biomedclip pmc_clip clip; ADDENDA_SITES=busi to smoke one site)
"""

from __future__ import annotations

import json
import math
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cohort import CohortManifest
from src.evaluation import concept_directions as cd
from src.evaluation import decomposition as dec
from src.evaluation import global_inference as gi
from src.experiments.compute_rho_geometry import (
    _load_model,
    _load_site_embeddings,
    _probe_w_grouped,
)
from src.utils.config import load_experiment_config
from src.utils.reproducibility import set_all_seeds

EXPERIMENT_NAME = "reviewer_addenda"
PCA_K = 50  # matches E1 (configs/concept_directions.yaml: pca_components)

# --- P0-1: richness-matched, non-diagnostic ANATOMY null ---------------------
# Contrastive pairs of clinically RICH but non-diagnostic anatomical structures,
# built with the identical carrier templates + mean-difference machinery as the
# BI-RADS and placebo axes. 10 axes -> C(10,5)=252 full-rank subsets (matching the
# malignancy-synonym pool's resolution). Deliberately avoids the five BI-RADS mass
# descriptors and any malignancy semantics; both poles are rich noun phrases, so the
# pool is matched to the clinical axes on semantic richness (unlike the terse placebo
# pool) -- this is the control the supplement's residual-confound note asks for.
ANATOMY_CONCEPTS = {
    "rib_cooper":          {"pos": ["a rib shadow"],           "neg": ["a Cooper's ligament"]},
    "pectoralis_fat":      {"pos": ["the pectoralis muscle"],  "neg": ["subcutaneous fat"]},
    "glandular_adipose":   {"pos": ["glandular parenchyma"],   "neg": ["adipose tissue"]},
    "skin_chestwall":      {"pos": ["the skin line"],          "neg": ["the chest wall"]},
    "duct_vessel":         {"pos": ["a lactiferous duct"],     "neg": ["a blood vessel"]},
    "fascia_muscle":       {"pos": ["a fascial plane"],        "neg": ["a muscle layer"]},
    "retro_premammary":    {"pos": ["retromammary fat"],       "neg": ["premammary fat"]},
    "node_vessel":         {"pos": ["a normal lymph node"],    "neg": ["a vascular structure"]},
    "fibroglandular_fatty":{"pos": ["fibroglandular tissue"],  "neg": ["a fatty lobule"]},
    "nipple_areola":       {"pos": ["the nipple shadow"],      "neg": ["the areolar region"]},
}


def _anatomy_prompts(bank: dict) -> dict:
    tpl = bank["carrier_templates"]
    return {
        n: {"pos": cd.expand_prompts(tpl, pn["pos"]), "neg": cd.expand_prompts(tpl, pn["neg"])}
        for n, pn in ANATOMY_CONCEPTS.items()
    }


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score))


def _pool_retention(pool_dirs, Z: np.ndarray, k: int) -> float:
    """Mean image-subspace (PCA-k) squared-norm retention over a pool of unit dirs."""
    dirs = pool_dirs.values() if isinstance(pool_dirs, dict) else pool_dirs
    return float(np.mean([cd.project_to_image_subspace(d, Z, k)[1] for d in dirs]))


def _subspace_auc_mean(Z, y, w, pool, m, n_sample, seed):
    """Dimension-matched null AUC: mean/max AUC of the probe restricted to a random
    ``m``-subspace of ``pool``, so the baseline spans the SAME m dims as the clinical
    subspace (a mean projector over the full 16-dim pool would not be comparable).
    """
    P = pool.shape[0]
    total = math.comb(P, m)
    if total <= n_sample:
        subsets = list(combinations(range(P), m))
    else:
        rng = np.random.default_rng(seed)
        subsets = [tuple(rng.choice(P, size=m, replace=False)) for _ in range(n_sample)]
    aucs = []
    for sel in subsets:
        Q, R = np.linalg.qr(pool[list(sel)].T)
        if np.count_nonzero(np.abs(np.diag(R)) > 1e-8) < m:
            continue
        aucs.append(roc_auc_score(y, Z @ (Q @ (Q.T @ w))))
    return float(np.mean(aucs)), float(np.max(aucs))


def _skill(auc_sub: float, auc_full: float) -> float:
    """Fraction of full-probe AUC skill (above chance) retained by a subspace."""
    denom = auc_full - 0.5
    return float((auc_sub - 0.5) / denom) if denom > 1e-9 else float("nan")


def run_experiment(config: dict, backbones: list[str]) -> dict:
    repro = config["reproducibility"]
    set_all_seeds(repro["global_seed"], deterministic=repro["deterministic_algorithms"])
    cfg = config["rho_geometry"]
    bank = config["birads_concept_bank"]
    seed = repro["global_seed"]
    C = cfg["probe_C"]
    manifest = CohortManifest.load(PROJECT_ROOT / cfg["cohort_manifest"])
    sites = (os.environ.get("ADDENDA_SITES", "").split(",")
             if os.environ.get("ADDENDA_SITES") else cfg["sites"])

    concept_prompts = cd.concept_prompts_from_config(bank)
    placebo_prompts = cd.placebo_prompts_from_config(bank)
    anatomy_prompts = _anatomy_prompts(bank)

    out: dict = {
        "pca_k": PCA_K,
        "anatomy_axes": list(ANATOMY_CONCEPTS.keys()),
        "sites_requested": list(sites),
        "by_backbone": {},
    }

    for backbone in backbones:
        print(f"\n=== backbone: {backbone} ===")
        model = _load_model(backbone, config)
        embed_fn = lambda texts: cd.encode_texts(model, texts)  # noqa: E731

        directions = cd.build_text_directions(concept_prompts, embed_fn)
        names = list(directions.keys())
        B, _ = cd.stack_basis(directions, orthonormalize=cfg["orthonormalize_basis"])
        placebo_dirs = cd.build_text_directions(placebo_prompts, embed_fn)
        placebo_pool = np.stack(list(placebo_dirs.values()), axis=0)
        anatomy_dirs = cd.build_text_directions(anatomy_prompts, embed_fn)
        anatomy_pool = np.stack(list(anatomy_dirs.values()), axis=0)
        m = len(names)
        d = int(B.shape[0])
        P_bar_plac = gi.mean_projector(placebo_pool, m)
        P_bar_anat = gi.mean_projector(anatomy_pool, m)

        site_w: list[np.ndarray] = []
        site_order: list[str] = []
        per_site: dict = {}
        for site in sites:
            try:
                se = _load_site_embeddings(model, backbone, site, config, manifest,
                                           batch_size=cfg["batch_size"])
            except (KeyError, FileNotFoundError) as e:
                print(f"  [skip] {site}: {e}")
                continue
            if len(np.unique(se.y)) < 2:
                print(f"  [skip] {site}: only one class present")
                continue

            w = _probe_w_grouped(se.Z, se.y, se.sample_weights, C)
            Pc_w = dec.project_onto_subspace(w, B)          # P_C w
            rho_w = float(dec.rho(w, B))

            # P1-4 completeness: rank-based AUC (intercept-invariant) of the probe
            # restricted to the 5-dim clinical subspace vs the full probe, and vs
            # DIMENSION-MATCHED (5-dim) placebo / anatomy null subspaces.
            auc_full = _auc(se.y, se.Z @ w)
            auc_clin = _auc(se.y, se.Z @ Pc_w)
            auc_plac, auc_plac_max = _subspace_auc_mean(se.Z, se.y, w, placebo_pool, m, 500, seed)
            auc_anat, auc_anat_max = _subspace_auc_mean(se.Z, se.y, w, anatomy_pool, m, 500, seed)

            # P1-6 modality retention: PCA-50 image-subspace norm retention per pool.
            ret_clin = _pool_retention(directions, se.Z, PCA_K)
            ret_plac = _pool_retention(placebo_dirs, se.Z, PCA_K)
            ret_anat = _pool_retention(anatomy_dirs, se.Z, PCA_K)

            # P0-1 anatomy null: per-site rho excess over the anatomy pool.
            exc_plac = rho_w - gi.rho_from_projector(w, P_bar_plac)
            exc_anat = rho_w - gi.rho_from_projector(w, P_bar_anat)

            per_site[site] = {
                "n": int(se.Z.shape[0]), "groups": int(se.n_groups),
                "rho": rho_w,
                "excess_placebo": exc_plac, "excess_anatomy": exc_anat,
                "auc_full": auc_full, "auc_clin": auc_clin,
                "auc_plac": auc_plac, "auc_plac_max": auc_plac_max,
                "auc_anat": auc_anat, "auc_anat_max": auc_anat_max,
                "skill_clin": _skill(auc_clin, auc_full),
                "skill_plac": _skill(auc_plac, auc_full),
                "retention_clin": ret_clin, "retention_plac": ret_plac,
                "retention_anat": ret_anat,
            }
            site_w.append(w)
            site_order.append(site)
            print(f"  {site:<10} rho={rho_w:.3f} exc_plac={exc_plac:+.3f} "
                  f"exc_anat={exc_anat:+.3f} | AUC full/clin/plac/anat="
                  f"{auc_full:.3f}/{auc_clin:.3f}/{auc_plac:.3f}/{auc_anat:.3f} | "
                  f"ret c/p/a={ret_clin:.2f}/{ret_plac:.2f}/{ret_anat:.2f}")

        if not site_w:
            print(f"  [no sites for {backbone}]")
            continue

        # P0-1 core test against the anatomy null (reuse global_core_test: pass the
        # anatomy pool where it expects the placebo pool -> p_placebo == p_anatomy).
        W = np.stack(site_w, axis=0)
        gc_anat = gi.global_core_test(W, B, anatomy_pool,
                                      n_random=int(cfg["null_trials"]), seed=seed)
        esm = lambda key: float(np.mean([per_site[s][key] for s in site_order]))  # noqa: E731
        delta_min = 2.0 * m / d
        mean_exc_anat = esm("excess_anatomy")
        out["by_backbone"][backbone] = {
            "m": m, "d": d, "sites": site_order,
            "T_clin": gc_anat["T_clin"],
            # P0-1
            "mean_excess_placebo": esm("excess_placebo"),
            "mean_excess_anatomy": mean_exc_anat,
            "p_anatomy": gc_anat["p_placebo"],
            "anatomy_null_subsets": gc_anat["placebo_meta"]["n_full_rank"],
            "anatomy_gate_delta_min": delta_min,
            "anatomy_clears_gate": bool(mean_exc_anat > delta_min),
            # P1-4
            "mean_auc_full": esm("auc_full"), "mean_auc_clin": esm("auc_clin"),
            "mean_auc_plac": esm("auc_plac"), "mean_auc_anat": esm("auc_anat"),
            "mean_skill_clin": esm("skill_clin"), "mean_skill_plac": esm("skill_plac"),
            # P1-6
            "mean_retention_clin": esm("retention_clin"),
            "mean_retention_plac": esm("retention_plac"),
            "mean_retention_anat": esm("retention_anat"),
            "by_site": per_site,
        }
    return out


def format_summary(out: dict) -> list[str]:
    L = ["=" * 92, "Reviewer addenda: anatomy null (P0-1), completeness (P1-4), retention (P1-6)",
         "=" * 92]
    L.append(f"{'backbone':<12}{'exc_plac':>9}{'exc_anat':>9}{'p_anat':>8}{'gate':>6}"
             f"{'aucF':>7}{'aucClin':>8}{'aucPlac':>8}{'sklClin':>8}{'sklPlac':>8}"
             f"{'retC':>6}{'retP':>6}{'retA':>6}")
    L.append("-" * 100)
    for bk, b in out["by_backbone"].items():
        L.append(
            f"{bk:<12}{b['mean_excess_placebo']:>+9.3f}{b['mean_excess_anatomy']:>+9.3f}"
            f"{b['p_anatomy']:>8.4f}{('Y' if b['anatomy_clears_gate'] else 'n'):>6}"
            f"{b['mean_auc_full']:>7.3f}{b['mean_auc_clin']:>8.3f}{b['mean_auc_plac']:>8.3f}"
            f"{b['mean_skill_clin']:>8.2f}{b['mean_skill_plac']:>8.2f}"
            f"{b['mean_retention_clin']:>6.2f}{b['mean_retention_plac']:>6.2f}"
            f"{b['mean_retention_anat']:>6.2f}"
        )
    L += ["",
          "aucPlac/aucAnat are DIMENSION-MATCHED (mean over random 5-dim null subsets).",
          "P0-1: exc_anat>0, small p_anat, gate=Y => clinical excess survives a RICH,",
          "      non-diagnostic anatomy null (not just the terse placebo pool).",
          "P1-4: sklClin = fraction of full-probe AUC skill kept by the 5-dim clinical C;",
          "      sklClin > sklPlac => C carries more decision content than a matched null.",
          "P1-6: retC vs retP/retA => whether clinical dirs sit closer to the image manifold."]
    return L


def main():
    config = load_experiment_config("rho_geometry")
    argv = [a for a in sys.argv[1:] if not a.startswith("-")]
    backbones = argv or ["biomedclip", "pmc_clip", "clip"]
    out = run_experiment(config, backbones)
    summary = format_summary(out)
    for line in summary:
        print(line)
    outdir = PROJECT_ROOT / "results" / EXPERIMENT_NAME
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(json.dumps(out, indent=2))
    (outdir / "summary.txt").write_text("\n".join(summary))
    print(f"\n[saved] {outdir}/results.json")


if __name__ == "__main__":
    main()
