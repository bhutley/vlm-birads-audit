"""Experiment E6: concept-direction validity against radiologist BI-RADS labels.

Author: Brett Hutley
Question (P2 / Domain reviewer):
    Do the text-derived BI-RADS concept directions actually encode the named
    clinical features, or are the axis names labels of convenience? We test this
    externally on BrEaST, the one site with radiologist *per-feature* BI-RADS
    lexicon labels (shape, margin, echogenicity, posterior features; BrEaST does
    not annotate orientation, so 4 of the 5 axes are validatable).

For each backbone and axis k:
    1. build the raw unit concept direction c_k from text (same machinery as E3);
    2. embed the BrEaST lesions and project each onto c_k: proj = <z, c_k>;
    3. binarise the human descriptor to the axis's benign/malignant pole;
    4. report AUC(proj -> human pole). proj is oriented benign->malignant, so a
       grounded direction gives AUC > 0.5. Case-level bootstrap CI over lesions.

We also report the full backbone x (direction, feature) AUC matrix: a direction
that encodes its *own* feature should predict it better than it predicts the
other features (diagonal dominance), and the general-domain control (CLIP) should
sit near 0.5 throughout.

    PYTHONHASHSEED=0 python -m src.experiments.run_concept_validity
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import BrEaSTDataset, build_bus_dataset
from src.evaluation import concept_directions as cd
from src.evaluation import validity_stats as vs
from src.experiments.compute_rho_geometry import _load_model
from src.utils.config import load_experiment_config
from src.utils.reproducibility import set_all_seeds
from src.utils.results import save_results

EXPERIMENT_NAME = "concept_validity"
BENIGN, MALIGNANT = 0, 1
AXES = list(BrEaSTDataset.AXES)  # shape, margin, echogenicity, posterior


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC of ``scores`` predicting binary ``labels`` (1 = malignant pole).

    Higher score -> label 1. Returns NaN if a class is absent.
    """
    from sklearn.metrics import roc_auc_score
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _boot_auc(scores: np.ndarray, labels: np.ndarray, n_boot: int, rng):
    """Case-level bootstrap of the AUC.

    Returns ``(ci, p_one_sided)`` where ``ci`` is the ``[2.5, 97.5]`` percentile
    interval and ``p_one_sided`` tests H0: AUC <= 0.5 against AUC > 0.5 as the
    (Laplace-smoothed) fraction of resampled AUCs at or below 0.5. Both are NaN
    if no resample retained two classes.
    """
    n = len(labels)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(labels[idx])) < 2:
            continue
        out.append(_auc(scores[idx], labels[idx]))
    if not out:
        return [float("nan"), float("nan")], float("nan")
    reps = np.asarray(out)
    ci = [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]
    p = float((np.sum(reps <= 0.5) + 1) / (len(reps) + 1))
    return ci, p


def _holm(pvals: np.ndarray) -> np.ndarray:
    """Holm--Bonferroni step-down adjusted p-values, returned in input order.

    NaN p-values (axes that could not be tested) are excluded from the family
    size and passed through as NaN.
    """
    pvals = np.asarray(pvals, dtype=float)
    finite = np.where(np.isfinite(pvals))[0]
    adj = np.full(pvals.shape, np.nan)
    m = len(finite)
    if m == 0:
        return adj
    order = finite[np.argsort(pvals[finite])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj


def _embed_breast(model, dataset: str, config: dict, batch_size: int):
    """Return (Z, feats, diag) for BrEaST benign+malignant lesions (L2-normalised).

    ``diag`` is the radiologist diagnosis per lesion (0=benign, 1=malignant),
    needed to condition the per-feature validation on diagnosis.
    """
    root = Path(config["bus_datasets"][dataset]).expanduser()
    ds = build_bus_dataset(dataset, root, transform=model.preprocess)
    assert isinstance(ds, BrEaSTDataset)  # only BrEaST carries per-feature labels
    keep = [i for i, (_, lab, _) in enumerate(ds.samples) if lab in (BENIGN, MALIGNANT)]
    feats_by_path = ds._features_by_path
    diag_by_path = {str(p): int(lab) for p, lab, _ in ds.samples}

    Z, paths = [], []
    for start in range(0, len(keep), batch_size):
        chunk = keep[start:start + batch_size]
        imgs = []
        for i in chunk:
            sample = ds[i]
            imgs.append(sample["image"])
            paths.append(sample["path"])
        import torch
        batch = torch.stack(imgs)
        Z.append(model.encode_image(batch).detach().cpu().numpy())
    Z = np.concatenate(Z).astype(np.float64)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    feats = [feats_by_path[p] for p in paths]
    diag = np.array([diag_by_path[p] for p in paths], dtype=int)
    return Z, feats, diag


def run_experiment(config: dict) -> dict:
    repro = config["reproducibility"]
    set_all_seeds(repro["global_seed"], deterministic=repro["deterministic_algorithms"])
    cfg = config["concept_validity"]
    bank = config["birads_concept_bank"]
    concept_prompts = cd.concept_prompts_from_config(bank)
    ref_prompts = cd.reference_prompts_from_config(bank)
    n_perm = int(cfg.get("n_perm", 2000))

    results: dict = {
        "dataset": cfg["dataset"],
        "axes": AXES,
        "note": "BrEaST has no orientation label; 4 of 5 axes validatable.",
        "by_backbone": {},
    }

    for backbone in cfg["backbones"]:
        print(f"\n=== backbone: {backbone} ===")
        model = _load_model(backbone, config)
        embed_fn = lambda texts: cd.encode_texts(model, texts)
        directions = cd.build_text_directions(concept_prompts, embed_fn)  # name -> unit c_k
        # generic malignancy reference axis (same machinery as E3), for the
        # diagonal-vs-malignancy-axis control: does the named direction predict its
        # feature better than a plain benign->malignant direction does?
        mal_dir = cd.paired_text_direction(
            embed_fn(ref_prompts["malignancy"]["pos"]),
            embed_fn(ref_prompts["malignancy"]["neg"]),
        ) if "malignancy" in ref_prompts else None

        Z, feats, diag = _embed_breast(model, cfg["dataset"], config, cfg["batch_size"])
        rng = np.random.default_rng([repro["global_seed"], zlib.crc32(backbone.encode())])
        # separate stream for the diagnosis-control stats so adding them does not
        # perturb the marginal per-axis bootstrap CIs / Holm p-values above.
        rng_dx = np.random.default_rng([repro["global_seed"], zlib.crc32((backbone + "_dx").encode())])

        # projection of every lesion onto every (raw, unit) concept axis
        proj = {ax: Z @ directions[ax] for ax in AXES if ax in directions}
        proj_mal = Z @ mal_dir if mal_dir is not None else None
        # per-axis feature-pole label vectors (-1 = unlabelled), aligned to Z
        lab_of = {ax: np.array([f[ax] if f[ax] is not None else -1 for f in feats])
                  for ax in AXES}

        per_axis = {}
        auc_matrix = {}  # direction -> {feature -> AUC}
        for d_ax in AXES:
            if d_ax not in proj:
                continue
            auc_matrix[d_ax] = {}
            for f_ax in AXES:
                lab = np.array([f[f_ax] if f[f_ax] is not None else -1 for f in feats])
                mask = lab >= 0
                if mask.sum() < 2:
                    auc_matrix[d_ax][f_ax] = None
                    continue
                auc_matrix[d_ax][f_ax] = _auc(proj[d_ax][mask], lab[mask])

            # diagonal: does direction d_ax predict its OWN human feature?
            n = int(Z.shape[0])
            proj_d, feat_d = proj[d_ax], lab_of[d_ax]
            mask = feat_d >= 0
            s, y = proj_d[mask], feat_d[mask]
            nb = int(cfg["n_boot"])
            ci, p_one_sided = _boot_auc(s, y, nb, rng)

            # (1) diagnosis-conditioned AUC: the feature signal with the diagnosis
            # confound removed (same-diagnosis concordance), plus a within-diagnosis
            # permutation p and a case-level bootstrap CI. This is the primary
            # feature-specificity control (reviewer: BI-RADS features co-vary with
            # diagnosis, so a plain malignancy direction could predict them).
            cond = vs.conditioned_auc(proj_d, feat_d, diag)
            cond_ci, _ = vs.ci_and_p(
                vs.bootstrap_stat(
                    lambda idx: vs.conditioned_auc(proj_d[idx], feat_d[idx], diag[idx]),
                    n, rng_dx, nb),
                null=0.5)
            cond_perm_p = vs.within_diag_perm_p(proj_d, feat_d, diag, n_perm, rng_dx)

            # (2) paired diagonal-minus-off-diagonal: does d_ax predict its OWN
            # feature more than it predicts the OTHER features? (feature-specificity)
            per_axis_auc = _auc(s, y)
            others = [f for f in AXES if f != d_ax and f in proj]
            off_vals = [auc_matrix[d_ax][f] for f in others if auc_matrix[d_ax][f] is not None]
            doff_point = per_axis_auc - float(np.mean(off_vals)) if off_vals else float("nan")
            doff_ci, doff_p = vs.paired_diff_ci(
                lambda idx: vs.rank_auc(proj_d[idx], feat_d[idx]),
                lambda idx: float(np.nanmean([vs.rank_auc(proj_d[idx], lab_of[f][idx]) for f in others]))
                            if others else float("nan"),
                n, rng_dx, nb, null=0.0)

            # (3) paired diagonal-minus-malignancy-axis: does the NAMED direction
            # beat a generic benign->malignant direction at predicting feature d_ax?
            # (feature content beyond malignancy semantics)
            dmal_point = dmal_ci = dmal_p = None
            if proj_mal is not None:
                dmal_point = per_axis_auc - vs.rank_auc(proj_mal, feat_d)
                dmal_ci, dmal_p = vs.paired_diff_ci(
                    lambda idx: vs.rank_auc(proj_d[idx], feat_d[idx]),
                    lambda idx: vs.rank_auc(proj_mal[idx], feat_d[idx]),
                    n, rng_dx, nb, null=0.0)

            per_axis[d_ax] = {
                "n_labelled": int(mask.sum()),
                "n_malignant_pole": int((y == 1).sum()),
                "auc": per_axis_auc,
                "auc_ci": ci,
                "p_one_sided": p_one_sided,  # H0: AUC <= 0.5
                # diagnosis-conditioned feature signal
                "conditioned_auc": cond,
                "conditioned_auc_ci": cond_ci,
                "conditioned_stratum_auc": vs.stratum_aucs(proj_d, feat_d, diag),
                "conditioned_perm_p": cond_perm_p,  # H0: no within-diagnosis signal
                # paired specificity contrasts (lower CI bound > 0 => specific)
                "diag_minus_off": doff_point,
                "diag_minus_off_ci": doff_ci,
                "diag_minus_off_p": doff_p,
                "diag_minus_malig": dmal_point,
                "diag_minus_malig_ci": dmal_ci,
                "diag_minus_malig_p": dmal_p,
            }
            # Feature-specificity intersection-union test (plan §12.1): "feature-
            # specific" is the CONJUNCTION of (1) within-diagnosis signal, (2) own
            # feature beaten only by itself vs the others, and (3) own feature above
            # the generic malignancy axis. The IUT p is the MAX of the three
            # component p-values; all three effect directions must also be positive.
            comp_p = [c if c is not None else float("nan")
                      for c in (cond_perm_p, doff_p, dmal_p)]
            per_axis[d_ax]["p_conditioned"] = cond_perm_p
            per_axis[d_ax]["p_own_vs_other"] = doff_p
            per_axis[d_ax]["p_own_vs_malignancy"] = dmal_p
            per_axis[d_ax]["p_specificity"] = (
                float(np.max(comp_p)) if all(np.isfinite(c) for c in comp_p) else float("nan"))
            per_axis[d_ax]["directions_positive"] = bool(
                np.isfinite(cond) and cond > 0.5
                and np.isfinite(doff_point) and doff_point > 0
                and dmal_point is not None and np.isfinite(dmal_point) and dmal_point > 0)
            print(f"  {d_ax:13} n={per_axis[d_ax]['n_labelled']:<4} "
                  f"AUC={per_axis_auc:.3f} CI={ci}  cond={cond:.3f} permp={cond_perm_p:.3g}  "
                  f"d-off={doff_point:+.3f} p={doff_p:.3g}  "
                  f"d-mal={dmal_point if dmal_point is None else f'{dmal_point:+.3f}'} "
                  f"p={dmal_p if dmal_p is None else f'{dmal_p:.3g}'}  "
                  f"-> p_spec={per_axis[d_ax]['p_specificity']:.3g} "
                  f"dir+={per_axis[d_ax]['directions_positive']}")

        diag = [per_axis[a]["auc"] for a in per_axis if np.isfinite(per_axis[a]["auc"])]
        offdiag = [auc_matrix[d][f] for d in auc_matrix for f in auc_matrix[d]
                   if d != f and auc_matrix[d][f] is not None]
        results["by_backbone"][backbone] = {
            "n_lesions": int(Z.shape[0]),
            "per_axis": per_axis,
            "auc_matrix": auc_matrix,
            "mean_diagonal_auc": float(np.mean(diag)) if diag else float("nan"),
            "mean_offdiagonal_auc": float(np.mean(offdiag)) if offdiag else float("nan"),
        }

    # --- primary confirmatory family: 12 backbone x axis specificity IUT p-values,
    #     Holm-Bonferroni across the whole family (plan §12.1, review M-6). This
    #     directly tests the phrase "feature-specific"; the marginal AUCs and the
    #     individual component contrasts above remain descriptive. ---
    cells = [(bk, ax) for bk in cfg["backbones"] for ax in AXES]
    pvec = np.array([
        results["by_backbone"].get(bk, {}).get("per_axis", {}).get(ax, {})
        .get("p_specificity", float("nan"))
        for bk, ax in cells
    ])
    padj = _holm(pvec)
    hypotheses = {}
    for (bk, ax), p, pa in zip(cells, pvec, padj):
        cell = results["by_backbone"].get(bk, {}).get("per_axis", {}).get(ax)
        dir_pos = bool(cell.get("directions_positive")) if cell else False
        specific = bool(np.isfinite(pa) and pa < 0.05 and dir_pos)
        hypotheses[f"{bk}:{ax}"] = {
            "p_specificity": float(p) if np.isfinite(p) else None,
            "p_specificity_holm": float(pa) if np.isfinite(pa) else None,
            "directions_positive": dir_pos,
            "feature_specific": specific,
        }
        if cell is not None:
            cell["p_specificity_holm"] = float(pa) if np.isfinite(pa) else None
            cell["feature_specific"] = specific
    results["specificity_family"] = {
        "description": "IUT p=max(p_conditioned,p_own_vs_other,p_own_vs_malignancy); "
                       "Holm across backbone x axis; feature_specific = Holm p<0.05 "
                       "AND all effect directions positive.",
        "family_size": int(np.isfinite(pvec).sum()),
        "n_cells": len(cells),
        "alpha": 0.05,
        "hypotheses": hypotheses,
        "feature_specific_axes": [k for k, v in hypotheses.items() if v["feature_specific"]],
    }
    return results


def format_summary(results: dict, config: dict) -> list[str]:
    axes = results["axes"]
    L = [
        "=" * 72,
        "E6 — concept-direction validity vs radiologist BI-RADS (BrEaST)",
        "=" * 72,
        "AUC of the per-axis projection <z, c_k> predicting the human BI-RADS",
        "feature pole (1 = malignant pole). Direction oriented benign->malignant,",
        "so a grounded direction gives AUC > 0.5. " + results["note"],
        "",
        f"{'backbone':<12}" + "".join(f"{a:>14}" for a in axes) + f"{'diag':>8}{'off':>8}",
        "-" * (12 + 14 * len(axes) + 16),
    ]
    for bk, blob in results["by_backbone"].items():
        row = f"{bk:<12}"
        for a in axes:
            cell = blob["per_axis"].get(a)
            if cell and np.isfinite(cell["auc"]):
                lo, _ = cell["auc_ci"]
                star = "*" if lo > 0.5 else " "
                row += f"{cell['auc']:.2f}{star}".rjust(14)
            else:
                row += f"{'--':>14}"
        row += f"{blob['mean_diagonal_auc']:>8.2f}{blob['mean_offdiagonal_auc']:>8.2f}"
        L.append(row)
    L += [
        "",
        "* = bootstrap 95% CI lower bound > 0.5 (direction predicts its feature).",
        "diag = mean AUC of each direction on its OWN feature; off = mean AUC on the",
        "OTHER features. diag >> off and diag >> 0.5 => directions are feature-specific,",
        "not labels of convenience. CLIP (general control) should sit near 0.5.",
        "",
        "PRIMARY: feature-SPECIFICITY intersection-union test, Holm-corrected across",
        f"the {results.get('specificity_family', {}).get('family_size', 0)} backbone x axis "
        f"hypotheses. p_spec = max(p_conditioned, p_own-vs-other, p_own-vs-malignancy);",
        "a cell is feature-SPECIFIC only if Holm-adjusted p_spec < 0.05 AND all directions +.",
        f"{'backbone':<12}" + "".join(f"{a:>16}" for a in axes),
        "-" * (12 + 16 * len(axes)),
    ]
    for bk, blob in results["by_backbone"].items():
        row = f"{bk:<12}"
        for a in axes:
            cell = blob["per_axis"].get(a)
            ph = cell.get("p_specificity_holm") if cell else None
            if ph is not None:
                mark = ("SPEC" if cell.get("feature_specific")
                        else ("+" if cell.get("directions_positive") else "-"))
                row += f"{ph:.3f}{mark}".rjust(16)
            else:
                row += f"{'--':>16}"
        L.append(row)
    fs = results.get("specificity_family", {}).get("feature_specific_axes", [])
    L += [
        f">>> FEATURE-SPECIFIC axes (Holm-adjusted IUT): {fs if fs else 'NONE'}",
        "    (SPEC=passes; += directions positive but not Holm-sig; -=a direction non-positive.",
        "     Marginal/conditioned AUCs above remain descriptive, not full specificity.)",
    ]

    # Diagnosis-controlled feature-specificity (reviewer: BI-RADS features co-vary
    # with diagnosis, so a plain malignancy direction could predict them).
    def _ci(c):
        if c is None or not np.isfinite(c[0]):
            return "[--,--]"
        return f"[{c[0]:+.2f},{c[1]:+.2f}]"

    L += [
        "",
        "Diagnosis-controlled checks (own-feature axis; CI = 95% case bootstrap):",
        "  cond  = within-diagnosis AUC (feature signal beyond diagnosis; >0.5 = grounded)",
        "  permp = within-diagnosis permutation p for cond (H0: no signal beyond diagnosis)",
        "  d-off = AUC(own) - mean AUC(other features); d-mal = AUC(own) - AUC(malignancy axis)",
        "  a paired CI excluding 0 (d-off / d-mal) => the axis is feature-SPECIFIC.",
        "",
        f"{'backbone':<11}{'axis':<13}{'cond':>6}{'cond_CI':>15}{'permp':>7}"
        f"{'d-off':>7}{'d-off_CI':>15}{'d-mal':>7}{'d-mal_CI':>15}",
        "-" * 91,
    ]
    for bk, blob in results["by_backbone"].items():
        for a in axes:
            cell = blob["per_axis"].get(a)
            if not cell:
                continue
            cond = cell.get("conditioned_auc", float("nan"))
            permp = cell.get("conditioned_perm_p", float("nan"))
            doff = cell.get("diag_minus_off", float("nan"))
            dmal = cell.get("diag_minus_malig")
            dmal_s = f"{dmal:+.3f}" if isinstance(dmal, (int, float)) and np.isfinite(dmal) else "  --  "
            L.append(
                f"{bk:<11}{a:<13}{cond:>6.3f}{_ci(cell.get('conditioned_auc_ci')):>15}"
                f"{permp:>7.3f}{doff:>+7.3f}{_ci(cell.get('diag_minus_off_ci')):>15}"
                f"{dmal_s:>7}{_ci(cell.get('diag_minus_malig_ci')):>15}"
            )
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
