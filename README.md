# Auditing BI-RADS Expressibility of Breast-Ultrasound Decisions in Frozen Vision-Language Models

Code for the research paper "Auditing BI-RADS Expressibility of Breast-Ultrasound Decisions in Frozen Vision-Language Models" (AJCAI 2026). 

The object of study is a frozen image-text encoder whose benign/malignant probe direction `w` is audited against a BI-RADS concept subspace `C` (m = 5 text-derived directions: margin, shape, orientation, echogenicity, posterior). The headline quantity is ρ = ‖P_C w‖² / ‖w‖², reported as an excess over three nulls (random subspace, placebo bank, malignancy-synonym bank) and combined into a four-criterion audit rule per checkpoint. Nothing in any backbone is fine-tuned.

---

## What the experiments answer

All scripts live in `src/experiments/` unless a path is given.

| Paper section | Script | Question | Output |
|---|---|---|---|
| §4, Table 1, Supp. S1 | `run_dataset_audit` | Duplicate/leakage audit of the six sites; group-aware canonical 5-fold manifest | `results/dataset_audit/` |
| §5.1, Fig. 2, Supp. S4–S6 | `compute_rho_geometry` | Is `w` in `C` above chance, across 6 backbones × 6 sites? Random/placebo/synonym nulls, Holm, cluster bootstrap, fold refits, out-of-fold AUCs, gate sweep | `results/rho_geometry/` |
| §5 (per-case), Supp. S10 | `run_decomposition` | Does the clinical share `s_clin` predict error beyond confidence? (negative) | `results/decomposition/` |
| Supp. S2 | `run_concept_directions` | Concept-axis Gram matrix and the modality gap | `results/concept_directions/` |
| §5.3, Supp. S11 | `run_concept_validity` | Do the axes predict their named feature on BrEaST (Holm-corrected intersection-union test)? | `results/concept_validity/` |
| Supp. S6 | `compute_prompt_robustness` | Template jackknife + 1000 prompt-bank bootstrap | `results/prompt_robustness/` |
| Supp. S6, App. B | `compute_axis_robustness` | Leave-one-clinical-axis at dimension-matched m = 4 | `results/axis_robustness/` |
| Supp. S6, App. B | `compute_probe_regularisation` | C_reg sweep {0.01 … 100} | `results/probe_regularisation/` |
| Supp. S7 | `compute_reviewer_addenda` | Richness-matched anatomy null, completeness, manifold retention | `results/reviewer_addenda/` |
| Supp. S8 | `compute_pooled_permutation` | Pooled clinical+anatomy label permutation test | `results/pooled_permutation/` |
| Supp. S6, App. B | `scripts/run_p3_sensitivity.py` | Gate k-sweep; drop-one placebo descriptor (BiomedCLIP) | `results/rho_geometry/p3_sensitivity.json` |
| Supp. S6 (fold-refit table) | `scripts/run_heldout_rho.py` | Fold-refit direction stability report (reads the `rho_geometry` artefact) | `results/rho_geometry/fold_refit_stability.json` |
| §5.2, Supp. S9 | `ajcai/figures/make_fig3_shortcut.py` | BUS-UCLM placebo-null warning (group-aware separability) | `ajcai/figures/fig3_placebo_warning.json` |

---

## Setup

```bash
# from the project root
pip install -e .            # installs deps from pyproject.toml
#   (torch, open-clip-torch, transformers, scikit-learn, scikit-image,
#    scipy, matplotlib, seaborn, pyyaml, pillow, numpy)
```

Device is auto-selected MPS → CUDA → CPU. Always run from the project root and set `PYTHONHASHSEED=0`.

### Data

Datasets live outside the repo and are never committed. Point `configs/default.yaml` at local copies:

- `data.busi_root`: BUSI in class-folder layout: `<root>/{benign,malignant,normal}/*.png`.
- `bus_datasets.{busi,bus_uclm,bus_bra,busi_whu,udiat,breast}`: the six-site collection. BrEaST (`breast`) is required for the per-feature validation.

### UniMed-CLIP embeddings

UniMed-CLIP needs a forked `open_clip` that does not run in the project environment. Embed it once in an isolated environment (instructions in the docstring of `scripts/embed_unimed.py`); the result is cached to `results/cross_modal_probe/embeddings/` and consumed by the `unimed_clip` backbone. The other five backbones load directly.

---

## Run order

```bash
# 0. Unit tests: pure NumPy, no data or GPU needed (verifies the geometry and statistics)
python -m pytest tests/

# 1. Cohort audit: writes fold_manifest.csv + exclusions.csv, which every later step reads
PYTHONHASHSEED=0 python -m src.experiments.run_dataset_audit

# 2. UniMed-CLIP cache (isolated env; see above)
UNIMED_CKPT=/path/to/unimed-clip-vit-b16.pt python scripts/embed_unimed.py

# 3. Main audit (Fig. 2)
PYTHONHASHSEED=0 python -m src.experiments.compute_rho_geometry

# 4. Remaining analyses (independent of each other; need step 1)
PYTHONHASHSEED=0 python -m src.experiments.run_decomposition
PYTHONHASHSEED=0 python -m src.experiments.run_concept_directions
PYTHONHASHSEED=0 python -m src.experiments.run_concept_validity
PYTHONHASHSEED=0 python -m src.experiments.compute_prompt_robustness
PYTHONHASHSEED=0 python -m src.experiments.compute_axis_robustness
PYTHONHASHSEED=0 python -m src.experiments.compute_probe_regularisation
PYTHONHASHSEED=0 python -m src.experiments.compute_reviewer_addenda
PYTHONHASHSEED=0 python -m src.experiments.compute_pooled_permutation

# 5. Reports that read the step-3 artefact
PYTHONHASHSEED=0 python scripts/run_p3_sensitivity.py
PYTHONHASHSEED=0 python scripts/run_heldout_rho.py
```

Each experiment writes `results/<name>/results.json` (with metadata, git commit and a full config snapshot) and `results/<name>/summary.txt` (the table printed to the console).

---

## What to look for

- rho_geometry: the global audit table. A checkpoint is *expressible* only if all four criteria hold: (i) Holm-adjusted random-null p < 0.05, (ii) placebo reference rank < 0.05, (iii) cluster-bootstrap CI of the equal-site mean placebo excess above zero, (iv) that excess above the gate 2m/d. The paper's result: BiomedCLIP and PMC-CLIP pass; UniMed-CLIP fails (ii), SigLIP fails (iv), CLIP and PubMedCLIP fail everything.
- decomposition: the VERDICT line. YES only if the bootstrap ΔAUC CI lower bound > 0 *and* the likelihood-ratio test is significant; the paper reports NO.
- concept_validity: the Holm-adjusted `p_spec` column over the 12 backbone × axis cells; only PMC-CLIP shape passes.
- Robustness scripts: verdicts should be unchanged across prompt resamples, dropped axes, dropped placebo descriptors and gate multipliers, and should fail only at C_reg = 100.

---

## Configuration

`configs/default.yaml` holds shared settings and the single source of truth for the concept bank (`birads_concept_bank`: carrier templates, the five clinical concepts, the malignancy reference axis, and the placebo, malignancy-synonym and anatomy pools). Each experiment has a `configs/<name>.yaml` overlay, deep-merged over the default; `compute_reviewer_addenda` and `scripts/run_p3_sensitivity.py` reuse `configs/rho_geometry.yaml`. 

---

## Layout

```
src/
  data/         audit.py (dedup/leakage/folds), cohort.py (manifest, group weights), datasets.py
  evaluation/   concept_directions.py, decomposition.py (ρ, s_clin), global_inference.py (audit rule),
                probe_performance.py, validity_stats.py, metrics.py,
                retrieval.py (exercised by tests/test_concept_geometry.py only)
  models/       clip_model.py (BiomedCLIP, CLIP, PMC-CLIP), extra_vlms.py (PubMedCLIP, SigLIP),
                cached_embeddings.py (UniMed-CLIP cache), base.py
  utils/        config.py, reproducibility.py, results.py
  experiments/  the entry points above
scripts/        UniMed embedding, sensitivity and fold-refit reports
tests/          pure-NumPy unit tests
```

## Reproducibility and caveats

- `reproducibility.global_seed: 42`, `data.split_seed: 42`, plus `PYTHONHASHSEED=0`.
- `results/MANIFEST.yaml` links every number reported in the paper to its result artefact and JSON Pointer (or derivation), with each artefact's producing script and git commit.
- BUSI and UDIAT have no native patient IDs, so their groups are duplicate-safe audit units; BUS-UCLM's subject prefix is a heuristic.
- Per-image audit artefacts under `results/dataset_audit/` are git-ignored; only aggregate counts enter `results.json`.
