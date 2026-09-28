"""Fold-refit direction-stability report (RQ4 robustness), from the E3 artefact.

WP4 (2026-07-22): the group-disjoint fold refit is now computed *inside* E3
(``compute_rho_geometry._fold_refit_stability`` -> ``by_site[*]["fold_refit"]``),
using the canonical group-disjoint folds and the group-weighted probe. This script
no longer refits an image-level probe — it simply reads the frozen E3 result and
reports the fold-refit stability so a reader can see the grounding is not an
artefact of fitting ``w`` once on the full site.

This is **fold-refit direction stability**, NOT held-out predictive rho: no
held-out image prediction enters rho (plan §10.1, review M-5).

    PYTHONHASHSEED=0 python scripts/run_heldout_rho.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

ARTEFACT = PROJECT_ROOT / "results" / "rho_geometry" / "results.json"


def main():
    if not ARTEFACT.exists():
        raise FileNotFoundError(
            f"{ARTEFACT} missing — run `python -m src.experiments.compute_rho_geometry` first")
    data = json.loads(ARTEFACT.read_text())
    res = data.get("results", data)

    out = {"source": "results/rho_geometry/results.json",
           "analysis_commit": data.get("metadata", {}).get("git_commit"),
           "by_backbone": {}}
    hdr = (f"{'backbone':<12}{'site':<10}{'n':>6}{'gid':>6}"
           f"{'full_exc_plac':>14}{'fold_rho_mean':>14}{'fold_rho_min':>13}"
           f"{'fold_plac_min':>14}")
    print(hdr)
    print("-" * len(hdr))
    for bk, blob in res["by_backbone"].items():
        rows = {}
        for s, c in blob["by_site"].items():
            fr = c.get("fold_refit")
            if not fr:
                continue
            rows[s] = {"n_images": c["n_images"], "n_groups": c["n_groups"],
                       "full_excess_placebo": c["excess_placebo"], "fold_refit": fr}
            print(f"{bk:<12}{s:<10}{c['n_images']:>6}{c['n_groups']:>6}"
                  f"{c['excess_placebo']:>+14.4f}{fr['rho_mean']:>14.4f}"
                  f"{fr['rho_min']:>13.4f}{fr['plac_excess_min']:>+14.4f}")
        out["by_backbone"][bk] = rows

    outpath = PROJECT_ROOT / "results" / "rho_geometry" / "fold_refit_stability.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
