"""ROB-03: is the clinical subspace a property of BI-RADS, or of our phrasings?

ROB-01/02 varied *which axes* are in the bank. This varies *the text that builds
each axis* -- the more basic question, because the concept directions **are**
prompts, and prompt-difference geometry under the modality gap is exactly the
threat the placebo null exists to control.

Three checks, all on the same fitted site directions as E3:

* **carrier-template jackknife** -- rebuild the five-axis bank three times, dropping
  each carrier template in turn;
* **prompt bootstrap** -- resample carrier templates and, within each polarity,
  descriptor phrasings (with replacement, seed 42), rebuild the basis, rescore;
* **principal angles** -- how far each perturbed basis has moved from the primary
  one (``concept_directions.subspace_principal_angles``), per subspace and per axis.

Efficiency: both null distributions -- the random one and the enumerated placebo
bank -- depend only on the site directions and ``m``, **never on the clinical
basis**. They are therefore computed once per backbone and every draw is ranked
against the same fixed references, so 1000 draws cost basis rebuilds and lookups
rather than 1000 nulls. Text embeddings are cached per (template, descriptor) pair,
so resampling never re-encodes.

Bootstrap fractions here are **sensitivity summaries, not p-values**: the resampling
is over our own prompt bank, not over any null model of the data.

SENSITIVITY -- see ``docs/20260724-prompt-robustness-declaration.md``. Does not
revise the primary verdict.

Usage:
    PYTHONHASHSEED=0 python -m src.experiments.compute_prompt_robustness
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cohort import CohortManifest
from src.evaluation import concept_directions as cd
from src.evaluation import decomposition as dec
from src.evaluation import global_inference as gi
from src.evaluation.metrics import holm_bonferroni
from src.experiments.compute_rho_geometry import (
    _load_model,
    _load_site_embeddings,
    _probe_w_grouped,
)
from src.utils.config import load_experiment_config
from src.utils.reproducibility import set_all_seeds
from src.utils.results import save_results

EXPERIMENT_NAME = "prompt_robustness"


def _encode_prompt_grid(bank: dict, embed_fn) -> dict:
    """Row-normalised embedding per (axis, polarity, template, descriptor).

    Returns ``{axis: {polarity: (n_templates, n_descriptors, d)}}`` so a draw can
    index resampled template/descriptor pairs without re-encoding anything.
    """
    tpl = bank["carrier_templates"]
    grid = {}
    for axis, pn in bank["concepts"].items():
        grid[axis] = {}
        for pol in ("pos", "neg"):
            descs = pn[pol]
            flat = [t.format(d=d) for t in tpl for d in descs]
            E = np.asarray(embed_fn(flat), dtype=np.float64)
            E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
            grid[axis][pol] = E.reshape(len(tpl), len(descs), -1)
    return grid


def _direction(grid_axis: dict, t_idx, d_idx_pos, d_idx_neg) -> np.ndarray:
    """One axis direction from selected template/descriptor indices.

    Mirrors :func:`concept_directions.paired_text_direction` exactly: rows are
    already normalised, so mean the selected pos and neg rows, subtract, normalise.
    """
    pos = grid_axis["pos"][np.ix_(t_idx, d_idx_pos)].reshape(-1, grid_axis["pos"].shape[-1])
    neg = grid_axis["neg"][np.ix_(t_idx, d_idx_neg)].reshape(-1, grid_axis["neg"].shape[-1])
    v = pos.mean(axis=0) - neg.mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-12)


def _basis(grid: dict, axes: list, t_idx, d_sel: dict, orthonormalize: bool):
    dirs = {a: _direction(grid[a], t_idx, d_sel[a]["pos"], d_sel[a]["neg"]) for a in axes}
    B, _ = cd.stack_basis(dirs, orthonormalize=orthonormalize)
    return B, dirs


def _score(W, B, rand_null, plac_null, P_bar, m, d, k_gate) -> dict:
    """Rank a perturbed basis against the backbone's FIXED reference distributions."""
    T = float(np.mean([dec.rho(w, B) for w in W]))
    excess = T - float(np.mean([gi.rho_from_projector(w, P_bar) for w in W]))
    return {
        "T_clin": T,
        "p_random": float((1 + int((rand_null >= T).sum())) / (len(rand_null) + 1)),
        "placebo_tail_fraction": float((1 + int((plac_null >= T).sum())) / (len(plac_null) + 1)),
        "mean_excess_placebo": excess,
        "passes_gate": bool(excess > k_gate * m / d),
    }


def run_experiment(config: dict) -> dict:
    repro = config["reproducibility"]
    set_all_seeds(repro["global_seed"], deterministic=repro["deterministic_algorithms"])
    cfg = config[EXPERIMENT_NAME]
    bank = config["birads_concept_bank"]
    seed = repro["global_seed"]

    manifest = CohortManifest.load(PROJECT_ROOT / cfg["cohort_manifest"])
    placebo_prompts = cd.placebo_prompts_from_config(bank)
    axes = list(bank["concepts"])
    templates = bank["carrier_templates"]
    n_tpl = len(templates)
    k_gate = float(cfg["gate_k_primary"])
    n_boot = int(cfg["n_prompt_bootstrap"])

    # axes whose polarity has a single phrasing cannot be perturbed there
    fixed_pol = {a: [p for p in ("pos", "neg") if len(bank["concepts"][a][p]) < 2]
                 for a in axes}

    results: dict = {
        "status": "SENSITIVITY — bootstrap fractions are not p-values",
        "declaration": "docs/20260724-prompt-robustness-declaration.md",
        "clinical_axes": axes,
        "n_carrier_templates": n_tpl,
        "n_prompt_bootstrap": n_boot,
        "polarities_with_single_phrasing": {a: v for a, v in fixed_pol.items() if v},
        "by_backbone": {},
    }

    for backbone in cfg["backbones"]:
        print(f"\n=== backbone: {backbone} ===")
        model = _load_model(backbone, config)
        embed_fn = lambda texts: cd.encode_texts(model, texts)

        grid = _encode_prompt_grid(bank, embed_fn)
        placebo_pool = np.stack(
            list(cd.build_text_directions(placebo_prompts, embed_fn).values()), axis=0)
        d_emb = int(placebo_pool.shape[1])
        m = len(axes)

        all_t = np.arange(n_tpl)
        d_all = {a: {p: np.arange(len(bank["concepts"][a][p])) for p in ("pos", "neg")}
                 for a in axes}
        B_full, dirs_full = _basis(grid, axes, all_t, d_all, cfg["orthonormalize_basis"])

        site_dirs, site_order = [], []
        for site in cfg["sites"]:
            try:
                se = _load_site_embeddings(model, backbone, site, config, manifest,
                                           batch_size=cfg["batch_size"])
            except (KeyError, FileNotFoundError) as e:
                print(f"  [skip] {site}: {e}")
                continue
            if len(np.unique(se.y)) < 2:
                continue
            site_dirs.append(_probe_w_grouped(se.Z, se.y, se.sample_weights, cfg["probe_C"]))
            site_order.append(site)
        if not site_dirs:
            results["by_backbone"][backbone] = {"error": "no usable sites"}
            continue
        W = np.stack(site_dirs, axis=0)

        # FIXED references: independent of the clinical basis, so computed once.
        rand_null = gi.global_random_null(W, m, n_trials=int(cfg["global_random_trials"]),
                                          seed=seed)
        plac_null = gi.placebo_bank(W, placebo_pool, m)["null_means"]
        P_bar = gi.mean_projector(placebo_pool, m)
        base = _score(W, B_full, rand_null, plac_null, P_bar, m, d_emb, k_gate)
        print(f"  full bank: T={base['T_clin']:.4f} exc={base['mean_excess_placebo']:+.4f}")

        # --- carrier-template jackknife ---
        jack = {}
        for i in range(n_tpl):
            keep = np.array([j for j in range(n_tpl) if j != i])
            B_j, dirs_j = _basis(grid, axes, keep, d_all, cfg["orthonormalize_basis"])
            s = _score(W, B_j, rand_null, plac_null, P_bar, m, d_emb, k_gate)
            s["max_principal_angle"] = float(
                cd.subspace_principal_angles(B_full, B_j).max())
            s["per_axis_angle"] = {
                a: float(np.arccos(np.clip(abs(float(dirs_full[a] @ dirs_j[a])), -1, 1)))
                for a in axes}
            jack[f"drop_template_{i}"] = s
            print(f"  jackknife -{i}: exc={s['mean_excess_placebo']:+.4f} "
                  f"angle={s['max_principal_angle']:.4f} rad")

        # --- prompt bootstrap ---
        rng = np.random.default_rng([seed, 3])       # backbone-independent draws
        draws, angles = [], []
        per_axis_ang = {a: [] for a in axes}
        for _ in range(n_boot):
            t_idx = rng.choice(all_t, size=n_tpl, replace=True)
            d_sel = {a: {p: rng.choice(d_all[a][p], size=len(d_all[a][p]), replace=True)
                         for p in ("pos", "neg")} for a in axes}
            B_b, dirs_b = _basis(grid, axes, t_idx, d_sel, cfg["orthonormalize_basis"])
            s = _score(W, B_b, rand_null, plac_null, P_bar, m, d_emb, k_gate)
            draws.append(s)
            angles.append(float(cd.subspace_principal_angles(B_full, B_b).max()))
            for a in axes:
                per_axis_ang[a].append(
                    float(np.arccos(np.clip(abs(float(dirs_full[a] @ dirs_b[a])), -1, 1))))

        def _q(vals, qs=(2.5, 50, 97.5)):
            return {f"p{q}": float(np.percentile(vals, q)) for q in qs}

        results["by_backbone"][backbone] = {
            "sites": site_order, "m": m, "embed_dim": d_emb,
            "full_bank": base,
            "template_jackknife": jack,
            "prompt_bootstrap": {
                "T_clin": _q([x["T_clin"] for x in draws]),
                "mean_excess_placebo": _q([x["mean_excess_placebo"] for x in draws]),
                "max_principal_angle_rad": {**_q(angles, (50, 95, 97.5)),
                                            "max": float(np.max(angles))},
                "frac_placebo_extreme": float(np.mean(
                    [x["placebo_tail_fraction"] < 0.05 for x in draws])),
                "frac_passes_gate": float(np.mean([x["passes_gate"] for x in draws])),
                "frac_random_significant": float(np.mean(
                    [x["p_random"] < 0.05 for x in draws])),
                "per_axis_angle_median_rad": {a: float(np.median(per_axis_ang[a]))
                                              for a in axes},
                "per_axis_angle_p95_rad": {a: float(np.percentile(per_axis_ang[a], 95))
                                           for a in axes},
            },
            "_draws": [{k: x[k] for k in
                        ("p_random", "placebo_tail_fraction", "passes_gate")}
                       for x in draws],
        }

    # Per draw, apply Holm across the six backbones (as the primary does) and check
    # the three basis-dependent criteria together. The fourth criterion -- the
    # cohort bootstrap CI -- is not recomputed here: it depends on refitting w per
    # replicate, not on the prompt bank, so it is held at its primary value and the
    # fraction below is explicitly over three of the four criteria.
    runs = [b for b, v in results["by_backbone"].items() if "_draws" in v]
    if runs:
        n = min(len(results["by_backbone"][b]["_draws"]) for b in runs)
        flags = {b: 0 for b in runs}
        for i in range(n):
            ps = {b: results["by_backbone"][b]["_draws"][i]["p_random"] for b in runs}
            holm = holm_bonferroni(ps)
            for b in runs:
                dr = results["by_backbone"][b]["_draws"][i]
                if (holm[b]["reject"] and dr["placebo_tail_fraction"] < 0.05
                        and dr["passes_gate"]):
                    flags[b] += 1
        for b in runs:
            pb = results["by_backbone"][b]["prompt_bootstrap"]
            pb["frac_holm_placebo_gate"] = flags[b] / n if n else None
            pb["criteria_covered"] = ("Holm(p_random), placebo tail<0.05, gate; "
                                      "the cohort bootstrap CI is prompt-independent "
                                      "and held at its primary value")
            results["by_backbone"][b].pop("_draws", None)
    return results


def format_summary(results: dict, config: dict) -> list[str]:
    axes = results["clinical_axes"]
    L = [
        "=" * 78,
        "ROB-03 — carrier-template and prompt-wording stability (SENSITIVITY)",
        "=" * 78,
        f"declaration : {results['declaration']}",
        f"status      : {results['status']}",
        f"draws       : {results['n_prompt_bootstrap']} prompt bootstraps; "
        f"{results['n_carrier_templates']} carrier templates jackknifed",
        "",
        "Polarities with a single phrasing CANNOT be perturbed by the bootstrap:",
        f"  {results['polarities_with_single_phrasing']}",
        "",
        "Fractions below are sensitivity summaries over OUR prompt bank, NOT p-values.",
        "",
    ]
    for b, v in results["by_backbone"].items():
        if "prompt_bootstrap" not in v:
            L.append(f"{b:<12} (not run)")
            continue
        pb, base = v["prompt_bootstrap"], v["full_bank"]
        ang = pb["max_principal_angle_rad"]
        L += [
            f"{b}",
            f"    full bank      exc={base['mean_excess_placebo']:+.4f}  "
            f"r_plac={base['placebo_tail_fraction']:.4f}",
            f"    jackknife      exc range ["
            + ", ".join(f"{s['mean_excess_placebo']:+.4f}"
                        for s in v['template_jackknife'].values()) + "]",
            f"    bootstrap exc  median {pb['mean_excess_placebo']['p50']:+.4f}  "
            f"[{pb['mean_excess_placebo']['p2.5']:+.4f},"
            f"{pb['mean_excess_placebo']['p97.5']:+.4f}]",
            f"    max princ.angle median {ang['p50']:.4f} rad  p95 {ang['p95']:.4f}  "
            f"max {ang['max']:.4f}",
            f"    frac of draws  placebo-extreme {pb['frac_placebo_extreme']:.3f}  "
            f"gate {pb['frac_passes_gate']:.3f}  random-sig {pb['frac_random_significant']:.3f}"
            f"  ALL-3 {pb.get('frac_holm_placebo_gate', float('nan')):.3f}",
            "    per-axis angle (median rad): "
            + "  ".join(f"{a[:5]}={pb['per_axis_angle_median_rad'][a]:.3f}" for a in axes),
        ]
    L += [
        "",
        "A stable subspace = small principal angles AND fractions near 1 for the",
        "checkpoints the primary analysis calls expressible.",
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
