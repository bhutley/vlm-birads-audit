"""Experiment E3: grounding geometry — rho-excess across backbones and sites.

WP4 (2026-07-22) patient/audit-unit-aware reanalysis:

  * embeddings are joined to the WP1 cohort manifest (``src/data/cohort``) so the
    decision direction ``w`` is fitted with **group-weighted, class-balanced**
    weights (repeated images/patients do not dominate); the image-weighted probe
    becomes a named sensitivity analysis;
  * uncertainty resamples **groups**, not images (cluster bootstrap), with the
    exact mean placebo rho from a precomputed projector (no nested Monte Carlo);
  * the headline verdict is a single **global backbone-level core test**
    (``src/evaluation/global_inference``): equal-site mean rho against synchronized
    random and exhaustive-placebo nulls, ``p_core = max(p_random, p_placebo)``,
    Holm-corrected across the six backbones, gated by ``2m/d``. Site cells are
    descriptive (cluster-bootstrap intervals), not significance-starred.

RQ3 is observational heterogeneity across six frozen checkpoints; RQ4 is recurrence
across independently fitted, group-disjoint site probes (not transfer).

Usage:
    PYTHONHASHSEED=0 python -m src.experiments.compute_rho_geometry
"""

from __future__ import annotations

import math
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cohort import (
    CohortManifest,
    align_site,
    build_cache_sidecar,
    group_class_balanced_weights,
    sample_id,
)
from src.data.datasets import build_bus_dataset
from src.evaluation import concept_directions as cd
from src.evaluation import decomposition as dec
from src.evaluation import global_inference as gi
from src.evaluation import probe_performance as pp
from src.evaluation.metrics import holm_bonferroni
from src.models.clip_model import BiomedCLIPZeroShot, CLIPZeroShot, PMCClipZeroShot
from src.utils.config import load_experiment_config
from src.utils.reproducibility import set_all_seeds
from src.utils.results import save_results

EXPERIMENT_NAME = "rho_geometry"
BENIGN, MALIGNANT = 0, 1


def _load_model(backbone: str, config: dict):
    if backbone == "biomedclip":
        return BiomedCLIPZeroShot()
    if backbone == "clip":
        m = config["models"]["clip"]
        return CLIPZeroShot(model_name=m["name"], pretrained=m["pretrained"])
    if backbone == "pubmedclip":
        from src.models.extra_vlms import PubMedCLIPZeroShot
        return PubMedCLIPZeroShot()
    if backbone == "siglip":
        from src.models.extra_vlms import SigLIPZeroShot
        return SigLIPZeroShot()
    if backbone == "pmc_clip":
        return PMCClipZeroShot()
    if backbone == "unimed_clip":
        # UniMed-CLIP runs in an isolated env (forked open_clip); we serve its
        # cached image+prompt embeddings. See scripts/embed_unimed.py.
        from src.models.cached_embeddings import CachedEmbeddingModel
        cache = (config.get("rho_geometry", {}).get("embedding_cache")
                 or config.get("concept_grounding", {}).get("embedding_cache")
                 or "results/cross_modal_probe/embeddings")
        return CachedEmbeddingModel.from_cache(PROJECT_ROOT / cache)
    raise ValueError(f"unknown backbone: {backbone}")


def _embed_site(model, backbone: str, site: str, config: dict, *,
                batch_size: int = 32,
                embedding_cache: str = "results/cross_modal_probe/embeddings"):
    """Return (Z, y) for one site, benign+malignant only. Reuses the BUSI cache
    only for the biomedclip backbone.

    ``batch_size`` / ``embedding_cache`` are explicit parameters (not read from a
    fixed config block) so this helper is reusable by E2/E4, whose merged configs
    have no ``rho_geometry`` section. Callers pass their own experiment's values."""
    if backbone == "unimed_clip":
        Z, lab = model.site_embeddings(site)
        keep = np.isin(lab, [BENIGN, MALIGNANT])
        return Z[keep].astype(np.float64), lab[keep].astype(int)
    if site == "busi" and backbone == "biomedclip":
        cache = PROJECT_ROOT / embedding_cache / "image_busi.npz"
        if cache.exists():
            d = np.load(cache)
            emb, labels = d["embeddings"].astype(np.float64), d["labels"]
            keep = np.isin(labels, [BENIGN, MALIGNANT])
            return emb[keep], labels[keep].astype(int)

    root = Path(config["bus_datasets"][site]).expanduser()
    ds = build_bus_dataset(site, root, transform=model.preprocess)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    feats, labels = [], []
    for batch in loader:
        feats.append(model.encode_image(batch["image"]).detach().cpu().numpy())
        labels.append(batch["label"].numpy())
    emb = np.concatenate(feats).astype(np.float64)
    lab = np.concatenate(labels)
    keep = np.isin(lab, [BENIGN, MALIGNANT])
    return emb[keep], lab[keep].astype(int)


def _probe_wb(Z: np.ndarray, y: np.ndarray, C: float):
    """Fit the benign/malignant logistic probe; return ``(w, b)``.

    The intercept ``b`` matters for any probability-space readout (e.g. E2's
    dose-response endpoint deltas, where the sigmoid is nonlinear so ``b`` does
    *not* cancel). It is irrelevant to the direction-only geometry (rho), so the
    rho path uses :func:`_probe_w`.
    """
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    clf.fit(Z, y)
    return clf.coef_.ravel().astype(np.float64), float(clf.intercept_[0])


def _probe_w(Z: np.ndarray, y: np.ndarray, C: float) -> np.ndarray:
    w, _ = _probe_wb(Z, y, C)
    return w


# --- WP4: manifest-aware loading + group-weighted probe ---------------------
def _load_site_embeddings(model, backbone: str, site: str, config: dict,
                          manifest: CohortManifest, *, batch_size: int = 32):
    """Return manifest-aligned :class:`SiteEmbeddings` (group ids, folds, weights).

    UniMed uses its cache (with a validated sidecar for pre-WP2 caches); every
    other backbone embeds live and reads ``batch["path"]`` for the canonical id.
    Alignment fails closed on any mismatch (``src/data/cohort.align_site``).
    """
    if backbone == "unimed_clip":
        Z_all, lab = model.site_embeddings(site)
        ids = model.site_sample_ids(site)
        if ids is None:
            root = Path(config["bus_datasets"][site]).expanduser()
            ids = build_cache_sidecar(site, root, lab)   # fail-closed label match
        else:
            ids = list(ids)
        return align_site(site, Z_all, lab, ids, manifest)

    root = Path(config["bus_datasets"][site]).expanduser()
    ds = build_bus_dataset(site, root, transform=model.preprocess)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    feats, labs, ids = [], [], []
    for batch in loader:
        feats.append(model.encode_image(batch["image"]).detach().cpu().numpy())
        labs.append(batch["label"].numpy())
        ids.extend(sample_id(site, Path(p).name) for p in batch["path"])
    Z_all = np.concatenate(feats).astype(np.float64)
    lab = np.concatenate(labs)
    return align_site(site, Z_all, lab, ids, manifest)


def _probe_wb_grouped(Z, y, sample_weight, C: float):
    """Group-weighted, class-balanced logistic probe; return ``(w, b)``.

    ``sample_weight`` already carries the inverse-group × class balance, so
    ``class_weight="balanced"`` is disabled to avoid double balancing. The
    geometry needs only the direction, but the PERF-01 out-of-fold audit reads
    probabilities at a fixed 0.5 threshold, so the intercept is kept.
    """
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit(Z, y, sample_weight=sample_weight)
    return clf.coef_.ravel().astype(np.float64), float(clf.intercept_[0])


def _probe_w_grouped(Z, y, sample_weight, C: float) -> np.ndarray:
    """Direction-only view of :func:`_probe_wb_grouped` (primary E3 probe)."""
    return _probe_wb_grouped(Z, y, sample_weight, C)[0]


def _boot_group_selections(group_ids: np.ndarray, n_boot: int, seed: int, site: str):
    """Backbone-independent bootstrap group resamples (seeded by site only).

    Because group membership is identical across backbones, seeding by site gives
    the *same* group selections for every backbone, so the equal-site global
    placebo-excess CI is paired to one cohort perturbation (plan §10.3).
    """
    rng = np.random.default_rng([seed, zlib.crc32(site.encode())])
    uniq = np.array(sorted(set(group_ids.tolist())))
    return [rng.choice(uniq, size=len(uniq), replace=True) for _ in range(n_boot)]


def _cluster_bootstrap(se, B, C, null_mean_rand, P_bar_plac, P_bar_malig, sels):
    """Cluster (group) bootstrap of the group-weighted probe direction.

    For each resample of groups, refit the group-weighted probe and recompute
    rho and the three excesses (random via the reused w-independent ``null_mean``;
    placebo/malignancy via the exact mean projectors). Returns percentile CIs, the
    per-replicate placebo-excess array (NaN where a replicate lacks a class, kept
    for the paired global CI), and the invalid-replicate fraction.
    """
    rows_by_group: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(se.group_ids):
        rows_by_group[g].append(i)

    rho_b, er_r, er_p, er_m = [], [], [], []
    plac_per_rep = np.full(len(sels), np.nan)
    n_invalid = 0
    for j, sel_groups in enumerate(sels):
        idx = np.fromiter((i for g in sel_groups for i in rows_by_group[g]),
                          dtype=int)
        yb = se.y[idx]
        if len(np.unique(yb)) < 2:
            n_invalid += 1
            continue
        wgt = group_class_balanced_weights(se.group_ids[idx], yb)
        wb = _probe_w_grouped(se.Z[idx], yb, wgt, C)
        rb = float(dec.rho(wb, B))
        ep = rb - gi.rho_from_projector(wb, P_bar_plac)
        rho_b.append(rb)
        er_r.append(rb - null_mean_rand)
        er_p.append(ep)
        er_m.append(rb - gi.rho_from_projector(wb, P_bar_malig))
        plac_per_rep[j] = ep

    def _ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if a else None
    return {
        "rho_ci": _ci(rho_b),
        "excess_random_ci": _ci(er_r),
        "excess_placebo_ci": _ci(er_p),
        "excess_malignancy_ci": _ci(er_m),
        "plac_per_rep": plac_per_rep,
        "invalid_frac": n_invalid / len(sels) if sels else 0.0,
    }


def _fold_refit_stability(se, B, C, P_bar_plac):
    """Group-disjoint fold-refit direction stability (RQ4; not held-out rho).

    Refit the group-weighted probe on each 4-of-5 canonical fold union (folds are
    group-disjoint by construction) and recompute rho + placebo excess, showing the
    direction is not an artefact of fitting once on the full site.
    """
    rhos, plac = [], []
    for f in sorted(set(se.folds.tolist())):
        tr = se.folds != f
        if len(np.unique(se.y[tr])) < 2:
            continue
        wgt = group_class_balanced_weights(se.group_ids[tr], se.y[tr])
        wf = _probe_w_grouped(se.Z[tr], se.y[tr], wgt, C)
        rhos.append(float(dec.rho(wf, B)))
        plac.append(rhos[-1] - gi.rho_from_projector(wf, P_bar_plac))
    if not rhos:
        return None
    return {
        "n_folds": len(rhos),
        "rho_mean": float(np.mean(rhos)), "rho_min": float(np.min(rhos)),
        "plac_excess_mean": float(np.mean(plac)), "plac_excess_min": float(np.min(plac)),
        "plac_excess_all": [float(v) for v in plac],   # CONS-01: no "every fold" claim
    }


def _oof_probe_performance(se, C, sels):
    """PERF-01: group-disjoint out-of-fold discrimination of this site's probe.

    The reviewer's point is that a direction from a near-chance probe is not a
    meaningful decision to audit, so every geometry cell is reported next to the
    predictive quality of the probe that produced it. Same cohort, folds and
    group weighting as the geometry (:mod:`src.evaluation.probe_performance`);
    the CIs reuse the caller's group selections, so they are *paired* to the same
    cohort perturbations as the placebo-excess CIs.
    """
    try:
        return pp.oof_performance(
            se.Z, se.y, se.group_ids, se.folds,
            lambda Zt, yt, wt: _probe_wb_grouped(Zt, yt, wt, C),
            selections=sels,
        )
    except pp.FoldError as e:            # documented failure, never a merged fold
        return {"auc": float("nan"), "balanced_accuracy": float("nan"),
                "auc_ci": None, "balanced_accuracy_ci": None,
                "weak_probe": None, "error": str(e)}


def run_experiment(config: dict) -> dict:
    repro = config["reproducibility"]
    set_all_seeds(repro["global_seed"], deterministic=repro["deterministic_algorithms"])
    cfg = config["rho_geometry"]
    bank = config["birads_concept_bank"]
    seed = repro["global_seed"]

    manifest = CohortManifest.load(PROJECT_ROOT / cfg["cohort_manifest"])
    backbones = cfg["pilot_backbones"] or cfg["backbones"]
    n_boot = int(cfg["cluster_bootstrap_trials"])
    n_global = int(cfg["global_random_trials"])
    C = cfg["probe_C"]

    concept_prompts = cd.concept_prompts_from_config(bank)
    ref_prompts = cd.reference_prompts_from_config(bank)
    placebo_prompts = cd.placebo_prompts_from_config(bank)
    malignancy_prompts = cd.malignancy_synonym_prompts_from_config(bank)

    results: dict = {
        "concepts": list(concept_prompts.keys()),
        "placebo_axes": list(placebo_prompts.keys()),
        "malignancy_synonym_axes": list(malignancy_prompts.keys()),
        "backbones_run": list(backbones),
        "by_backbone": {},
    }

    for backbone in backbones:
        print(f"\n=== backbone: {backbone} ===")
        model = _load_model(backbone, config)
        embed_fn = lambda texts: cd.encode_texts(model, texts)

        directions = cd.build_text_directions(concept_prompts, embed_fn)
        names = list(directions.keys())
        B, _ = cd.stack_basis(directions, orthonormalize=cfg["orthonormalize_basis"])
        B_raw = np.stack([directions[n] for n in names], axis=1)
        placebo_dirs = cd.build_text_directions(placebo_prompts, embed_fn)
        placebo_pool = np.stack(list(placebo_dirs.values()), axis=0)   # (P, d)
        malig_dirs = cd.build_text_directions(malignancy_prompts, embed_fn)
        malig_pool = np.stack(list(malig_dirs.values()), axis=0)       # (Q, d)
        mal_dir = cd.paired_text_direction(
            embed_fn(ref_prompts["malignancy"]["pos"]),
            embed_fn(ref_prompts["malignancy"]["neg"]),
        ) if "malignancy" in ref_prompts else None

        m_axes = len(names)
        d_emb = int(B_raw.shape[0])
        # exact mean projectors (site-independent) for placebo/malignancy excess
        P_bar_plac = gi.mean_projector(placebo_pool, m_axes)
        P_bar_malig = gi.mean_projector(malig_pool, m_axes)

        site_results: dict = {}
        site_dirs: list[np.ndarray] = []
        site_order: list[str] = []
        plac_per_rep_by_site: dict[str, np.ndarray] = {}
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

            w = _probe_w_grouped(se.Z, se.y, se.sample_weights, C)       # primary
            w_img = _probe_w(se.Z, se.y, C)                              # sensitivity
            g = dec.grounding_excess(w, B, n_trials=cfg["null_trials"], seed=seed)
            null_mean_rand = g["null_mean"]
            rho_w = g["rho"]
            excess_plac = rho_w - gi.rho_from_projector(w, P_bar_plac)
            excess_malig = rho_w - gi.rho_from_projector(w, P_bar_malig)
            excess_plac_img = (float(dec.rho(w_img, B))
                               - gi.rho_from_projector(w_img, P_bar_plac))

            sels = _boot_group_selections(se.group_ids, n_boot, seed, site)
            boot = _cluster_bootstrap(se, B, C, null_mean_rand,
                                      P_bar_plac, P_bar_malig, sels)
            plac_per_rep_by_site[site] = boot["plac_per_rep"]
            fold = _fold_refit_stability(se, B, C, P_bar_plac)
            oof = _oof_probe_performance(se, C, sels)

            w_hat = w / (np.linalg.norm(w) + 1e-12)
            axis_cos = {n: float(B_raw[:, i] @ w_hat) for i, n in enumerate(names)}
            mal_cos = float(mal_dir @ w_hat) if mal_dir is not None else None

            site_dirs.append(w)
            site_order.append(site)
            site_results[site] = {
                "n_images": int(se.Z.shape[0]),
                "n_groups": int(se.n_groups),
                "rho": rho_w,
                "excess_random": g["excess"],
                "excess_random_ci": boot["excess_random_ci"],
                "excess_placebo": excess_plac,
                "excess_placebo_ci": boot["excess_placebo_ci"],
                "excess_malignancy": excess_malig,
                "excess_malignancy_ci": boot["excess_malignancy_ci"],
                "excess_placebo_image_weighted": excess_plac_img,
                "excess_placebo_group_minus_image": excess_plac - excess_plac_img,
                "fold_refit": fold,
                "oof_performance": oof,          # PERF-01
                "boot_invalid_frac": boot["invalid_frac"],
                "axis_cosine": axis_cos,
                "malignancy_axis_cosine": mal_cos,
            }
            auc_ci = oof.get("auc_ci")
            auc_s = (f"auc={oof['auc']:.3f}" if np.isfinite(oof["auc"]) else "auc=  na")
            if auc_ci:
                auc_s += f"[{auc_ci[0]:.2f},{auc_ci[1]:.2f}]"
            print(f"  {site:<10} n={se.Z.shape[0]:<4} g={se.n_groups:<4} "
                  f"rho={rho_w:.4f} exc_rand={g['excess']:+.4f} "
                  f"exc_plac={excess_plac:+.4f} exc_mal={excess_malig:+.4f} "
                  f"{auc_s}{' WEAK' if oof.get('weak_probe') else ''} "
                  f"(boot invalid {boot['invalid_frac']:.1%})")

        # --- global backbone-level core test (equal site weight) ---
        global_blob = None
        if len(site_dirs) >= 1:
            W = np.stack(site_dirs, axis=0)
            gc = gi.global_core_test(W, B, placebo_pool, n_random=n_global, seed=seed)
            # paired cluster-bootstrap CI for the equal-site mean placebo excess:
            # per replicate, average the per-site placebo excess (nan-skipping),
            # then subtract the placebo null mean already folded into per-site excess.
            reps = np.vstack([plac_per_rep_by_site[s] for s in site_order])  # (S, n_boot)
            with np.errstate(invalid="ignore"):
                mean_over_sites = np.nanmean(reps, axis=0)                   # (n_boot,)
            valid = mean_over_sites[~np.isnan(mean_over_sites)]
            gc_ci = ([float(np.percentile(valid, 2.5)),
                      float(np.percentile(valid, 97.5))] if valid.size else None)
            mean_excess_malig = float(np.mean(
                [site_results[s]["excess_malignancy"] for s in site_order]))
            global_blob = {
                "sites": site_order,
                "T_clin": gc["T_clin"],
                "p_random": gc["p_random"],
                "placebo_tail_fraction": gc["placebo_tail_fraction"],
                "placebo_rank": gc["placebo_rank"],
                "n_placebo_reference": gc["n_placebo_reference"],
                "mean_excess_random": gc["mean_excess_random"],
                "mean_excess_placebo": gc["mean_excess_placebo"],
                "mean_excess_placebo_ci": gc_ci,       # paired cluster bootstrap
                "mean_excess_malignancy": mean_excess_malig,  # descriptive secondary
                "placebo_meta": gc["placebo_meta"],
            }

        # PERF-01 backbone roll-up: is this checkpoint's geometry read off probes
        # that actually discriminate? Descriptive; it gates no verdict.
        aucs = {s: c["oof_performance"]["auc"] for s, c in site_results.items()
                if np.isfinite(c["oof_performance"].get("auc", np.nan))}
        baccs = [c["oof_performance"]["balanced_accuracy"] for c in site_results.values()
                 if np.isfinite(c["oof_performance"].get("balanced_accuracy", np.nan))]
        oof_summary = {
            "n_sites_scored": len(aucs),
            "auc_min": float(min(aucs.values())) if aucs else None,
            "auc_median": float(np.median(list(aucs.values()))) if aucs else None,
            "auc_max": float(max(aucs.values())) if aucs else None,
            "bacc_min": float(min(baccs)) if baccs else None,
            "bacc_median": float(np.median(baccs)) if baccs else None,
            "bacc_max": float(max(baccs)) if baccs else None,
            "weak_sites": sorted(s for s, c in site_results.items()
                                 if c["oof_performance"].get("weak_probe")),
        }

        results["by_backbone"][backbone] = {
            "m": m_axes,
            "embed_dim": d_emb,
            "oof_summary": oof_summary,
            "n_placebo": int(placebo_pool.shape[0]),
            "n_malignancy": int(malig_pool.shape[0]),
            "n_distinct_placebo_subsets": int(math.comb(int(placebo_pool.shape[0]), m_axes)),
            "n_distinct_malignancy_subsets": int(math.comb(int(malig_pool.shape[0]), m_axes)),
            "by_site": site_results,
            "global": global_blob,
        }

    # --- confirmatory family: Holm across the six backbones on p_random ---
    # STAT-01: only the sampled random null yields a calibrated p-value, so it is
    # the only quantity Holm corrects. The fixed placebo bank stays a *required*
    # criterion of the verdict, entering as an empirical reference rank.
    p_rand = {b: blob["global"]["p_random"]
              for b, blob in results["by_backbone"].items()
              if blob.get("global")}
    holm = holm_bonferroni(p_rand) if p_rand else {}
    gate_ks = cfg["gate_k_sweep"]
    family = {}
    for b, blob in results["by_backbone"].items():
        gl = blob.get("global")
        if not gl:
            continue
        hb = holm.get(b) or {}
        reject = bool(hb.get("reject"))
        ci = gl.get("mean_excess_placebo_ci")
        verdicts = {
            f"k={k}": gi.audit_verdict(reject, gl["placebo_tail_fraction"],
                                       gl["mean_excess_placebo"],
                                       ci[0] if ci else None,
                                       blob["m"], blob["embed_dim"], k=k)
            for k in gate_ks
        }
        family[b] = {
            "p_random": gl["p_random"],
            "holm_adjusted_p_random": hb.get("holm_adjusted_p"),
            "holm_reject": reject,
            "holm_threshold": hb.get("holm_threshold"),
            "placebo_tail_fraction": gl["placebo_tail_fraction"],
            "placebo_rank": gl["placebo_rank"],
            "verdict_by_k": verdicts,
            "expressible": verdicts["k=2.0"]["expressible"],  # primary gate
        }
    results["global_family"] = {
        "holm_family": "six_backbone_p_random",
        "alpha": 0.05,
        "gate_k_primary": 2.0,
        "verdict_rule": ("Holm-adjusted p_random < alpha AND placebo tail "
                         "fraction < 0.05 AND bootstrap CI low > 0 AND "
                         "mean excess > 2m/d"),
        "backbones": family,
    }
    return results


def format_summary(results: dict, config: dict) -> list[str]:
    names = results["concepts"]
    sites = config["rho_geometry"]["sites"]
    L = [
        "=" * 76,
        "E3 (WP4) — group-weighted rho-excess + global backbone core test",
        "=" * 76,
        f"concepts (m={len(names)} held fixed): {', '.join(names)}",
        f"backbones run: {', '.join(results.get('backbones_run', []))}",
        "",
        "Per-site PLACEBO excess (group-weighted probe; [95% cluster-bootstrap CI]).",
        "Site cells are descriptive — the verdict is the global core test below.",
        "",
        f"{'backbone':<12}" + "".join(f"{s:>14}" for s in sites),
        "-" * (12 + 14 * len(sites)),
    ]
    for bk, blob in results["by_backbone"].items():
        row = f"{bk:<12}"
        for s in sites:
            cell = blob["by_site"].get(s)
            if cell:
                ci = cell.get("excess_placebo_ci")
                lo = f"{ci[0]:+.2f}" if ci else "  na"
                row += f"{cell['excess_placebo']:+.3f}>{lo:>6}".rjust(14)
            else:
                row += f"{'--':>14}"
        L.append(row)

    L += [
        "",
        "PERF-01 — group-disjoint OUT-OF-FOLD probe discrimination (AUC [95% CI]),",
        "group-weighted, fixed C; the direction each geometry cell above is read from.",
        "'*' = AUC CI lower bound does not clear 0.50 (weak probe).",
        "",
        f"{'backbone':<12}" + "".join(f"{s:>16}" for s in sites),
        "-" * (12 + 16 * len(sites)),
    ]
    for bk, blob in results["by_backbone"].items():
        row = f"{bk:<12}"
        for s in sites:
            cell = blob["by_site"].get(s)
            o = cell.get("oof_performance") if cell else None
            if o and np.isfinite(o.get("auc", np.nan)):
                ci = o.get("auc_ci")
                ci_s = f"[{ci[0]:.2f},{ci[1]:.2f}]" if ci else ""
                star = "*" if o.get("weak_probe") else " "
                row += f"{o['auc']:.2f}{ci_s}{star}".rjust(16)
            else:
                row += f"{'--':>16}"
        L.append(row)

    L.append("")
    for bk, blob in results["by_backbone"].items():
        o = blob.get("oof_summary") or {}
        if not o.get("n_sites_scored"):
            continue
        weak = ", ".join(o["weak_sites"]) if o["weak_sites"] else "none"
        L.append(f"  {bk:<12} AUC min/med/max = "
                 f"{o['auc_min']:.2f}/{o['auc_median']:.2f}/{o['auc_max']:.2f}"
                 f"   bal.acc = "
                 f"{o['bacc_min']:.2f}/{o['bacc_median']:.2f}/{o['bacc_max']:.2f}"
                 f"   weak: {weak}")

    L += [
        "",
        "GLOBAL backbone test: T_clin = equal-site mean rho.",
        "  p_rand  = Monte-Carlo random-subspace null (20k) -> a calibrated p-value;",
        "            Holm-corrected across the six backbones (the ONLY such family).",
        "  plac    = position in the FIXED placebo bank (C(16,5)=4368) -> an empirical",
        "            reference RANK, not a p-value (the +1 is rank smoothing).",
        "EXPRESSIBLE = Holm(p_rand)<.05 AND plac<.05 AND CI>0 AND excess>2m/d.",
        "",
        f"{'backbone':<12}{'T_clin':>8}{'exc_plac':>9}{'[95% CI]':>17}"
        f"{'p_rand':>8}{'holmP':>8}{'plac':>7}{'express':>9}",
        "-" * 78,
    ]
    fam = results.get("global_family", {}).get("backbones", {})
    for bk, blob in results["by_backbone"].items():
        gl = blob.get("global")
        f = fam.get(bk, {})
        if not gl:
            L.append(f"{bk:<12}{'(no sites)':>9}")
            continue
        ci = gl.get("mean_excess_placebo_ci")
        ci_s = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if ci else "        --"
        hp = f.get("holm_adjusted_p_random")
        L.append(
            f"{bk:<12}{gl['T_clin']:>8.3f}{gl['mean_excess_placebo']:>+9.3f}"
            f"{ci_s:>17}{gl['p_random']:>8.4f}"
            f"{(f'{hp:.4f}' if hp is not None else '--'):>8}"
            f"{gl['placebo_tail_fraction']:>7.3f}"
            f"{('YES' if f.get('expressible') else 'no'):>9}"
        )

    L += [
        "",
        "Gate-k sensitivity (expressible? at k in the sweep):",
    ]
    for bk, f in fam.items():
        vk = f.get("verdict_by_k", {})
        cells = "  ".join(f"{k}:{'Y' if v['expressible'] else 'n'}"
                          for k, v in vk.items())
        L.append(f"  {bk:<12} {cells}")

    L += [
        "",
        "Which axes reconstruct w (cos with decision dir), site-averaged:",
    ]
    for bk, blob in results["by_backbone"].items():
        sb = blob["by_site"]
        if not sb:
            continue
        agg = {n: np.mean([c["axis_cosine"][n] for c in sb.values()]) for n in names}
        ordered = sorted(agg.items(), key=lambda kv: -abs(kv[1]))
        L.append(f"  {bk:<12} " + "  ".join(f"{n}={v:+.2f}" for n, v in ordered))

    L += [
        "",
        "RQ2/RQ3: read the global table — which checkpoints clear BOTH nulls after",
        "  Holm + the 2m/d gate. RQ4: fold-refit stability + site CIs (per-site JSON).",
        "  Site significance stars are deliberately NOT shown (no per-site family).",
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
