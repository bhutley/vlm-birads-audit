"""P3 sensitivity analyses for the grounding result (peer-review robustness).

WP4 (2026-07-22): both parts now consume the group-aware WP4 layer.

Part A — magnitude-gate k-sweep, read from the frozen E3 artefact's *global*
  verdict family (``global_family.backbones[*].verdict_by_k``). Shows the primary
  gate k=2 is not cherry-picked: the expressible verdicts are k-robust. No
  recompute (the gate sweep is computed inside E3 at the global level).

Part B — leave-one-descriptor-out placebo pool (BiomedCLIP), refit with the
  **group-weighted** probe on the manifest-aligned cohort. Drops each placebo axis
  in turn and recomputes the placebo excess per site, showing the grounding (and
  the BUS-UCLM fragility) is not driven by any single placebo descriptor.

    PYTHONHASHSEED=0 python scripts/run_p3_sensitivity.py

Reads results/rho_geometry/results.json for Part A; loads BiomedCLIP for Part B.
Writes results/rho_geometry/p3_sensitivity.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cohort import CohortManifest
from src.evaluation import concept_directions as cd
from src.evaluation import decomposition as dec
from src.experiments.compute_rho_geometry import (
    EXPERIMENT_NAME, _load_model, _load_site_embeddings, _probe_w_grouped,
)
from src.utils.config import load_experiment_config
from src.utils.reproducibility import set_all_seeds


def part_a_gate_sweep(res: dict) -> dict:
    """Report the global expressibility verdict at each gate k (from E3)."""
    fam = res.get("global_family", {}).get("backbones", {})
    out = {"by_backbone": {}, "flips": []}
    for bk, f in fam.items():
        vk = {k: v["expressible"] for k, v in f.get("verdict_by_k", {}).items()}
        out["by_backbone"][bk] = vk
        if len(set(vk.values())) > 1:      # verdict changes across the swept k
            out["flips"].append({"backbone": bk, "expressible_at_k": vk})
    return out


def part_b_loo_placebo(config: dict, backbone: str = "biomedclip") -> dict:
    repro = config["reproducibility"]
    set_all_seeds(repro["global_seed"], deterministic=repro["deterministic_algorithms"])
    cfg = config["rho_geometry"]
    bank = config["birads_concept_bank"]
    seed = repro["global_seed"]
    manifest = CohortManifest.load(PROJECT_ROOT / cfg["cohort_manifest"])

    model = _load_model(backbone, config)
    embed_fn = lambda texts: cd.encode_texts(model, texts)
    directions = cd.build_text_directions(cd.concept_prompts_from_config(bank), embed_fn)
    B, _ = cd.stack_basis(directions, orthonormalize=cfg["orthonormalize_basis"])
    placebo_dirs = cd.build_text_directions(cd.placebo_prompts_from_config(bank), embed_fn)
    plac_names = list(placebo_dirs.keys())
    placebo_pool = np.stack(list(placebo_dirs.values()), axis=0)
    d, m = B.shape[0], B.shape[1]
    delta_min = dec.magnitude_gate(0.0, m, d)["delta_min"]

    by_site = {}
    for site in cfg["sites"]:
        try:
            se = _load_site_embeddings(model, backbone, site, config, manifest)
        except (KeyError, FileNotFoundError):
            continue
        if len(np.unique(se.y)) < 2:
            continue
        w = _probe_w_grouped(se.Z, se.y, se.sample_weights, cfg["probe_C"])
        loo = dec.leave_one_out_placebo_excess(
            w, B, placebo_pool, names=plac_names,
            n_trials=cfg["null_trials"], seed=seed,
        )
        by_site[site] = {**loo, "delta_min": delta_min,
                         "grounded_full": bool(loo["full"] > delta_min),
                         "grounded_all_loo": bool(all(v > delta_min
                                                      for v in loo["loo"].values()))}
        print(f"  {site:10} full={loo['full']:+.4f}  LOO[min,max]="
              f"[{loo['loo_min']:+.4f},{loo['loo_max']:+.4f}]  "
              f"full/all-LOO>gate={by_site[site]['grounded_full']}/"
              f"{by_site[site]['grounded_all_loo']}")
    return {"backbone": backbone, "m": m, "embed_dim": d, "P": len(plac_names),
            "placebo_axes": plac_names, "by_site": by_site,
            "note": "group-weighted probe on the manifest-aligned cohort (WP4)"}


def main():
    config = load_experiment_config(EXPERIMENT_NAME)
    data = json.loads((PROJECT_ROOT / "results/rho_geometry/results.json").read_text())
    res = data.get("results", data)

    print("=== Part A: global gate-k sweep (expressible? per backbone) ===")
    a = part_a_gate_sweep(res)
    for bk, vk in a["by_backbone"].items():
        print(f"  {bk:12} " + "  ".join(f"{k}:{'Y' if v else 'n'}" for k, v in vk.items()))
    print(f"  verdict flips across k: {len(a['flips'])}")

    print("\n=== Part B: leave-one-descriptor-out placebo (BiomedCLIP, group-weighted) ===")
    b = part_b_loo_placebo(config)

    out = {"gate_k_sweep": a, "loo_placebo": b}
    (PROJECT_ROOT / "results/rho_geometry/p3_sensitivity.json").write_text(json.dumps(out, indent=2))
    print("\nwrote results/rho_geometry/p3_sensitivity.json")


if __name__ == "__main__":
    main()
