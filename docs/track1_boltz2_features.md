# Track 1: Boltz-2 pose-derived features

**Goal**: test whether Boltz-2 predicted pose + affinity outputs add signal
to the Track 1 pEC50 regressor, on top of the canonical 2D-descriptor
baselines (Mordred, rdkit_desc_full).

**Date**: 2026-04-18. Source PR: feature/boltz2-track1-features.

Summary of the literature context: `docs/papers/boltz2_affinity_notes.md`.

## TL;DR

- **Mordred + Tier-0 + Tier-1 Boltz-2 features** wins the LGBM bake-off
  (OOF MAE 0.5204 / RAE 0.5719 under canonical UMAP split, default params).
  Optuna (20 trials) brings this to **OOF MAE 0.5113 / RAE 0.5619**.
- `affinity_pred_value` (Boltz-2 affinity regression head) dominates
  feature importance: rank 1 with ~23.8% of total model gain from only
  19 Tier-0 scalars (26x efficiency vs Mordred per feature).
- **Tier-2 ProLIF interaction fingerprints did not help** — residue-level
  H-bond / hydrophobic / VdW bits are redundant with Mordred's 2D
  encoding of H-bond donor/acceptor/aromatic counts + Boltz-2 scalar
  `ligand_to_pocket_distance_a`.
- The tuned model correlates r=0.97 with `lgbm_mordred_jazzy_umap`; if
  added to the ensemble it should **replace** that member (same family),
  not be added alongside.

## Feature tiers

### Tier-0 — DB-resident scalars (19 features)

Already populated in `compound_boltz2` by
`track2_structure/scripts/boltz2_postprocess.py`. Pulled via SQL;
see `track1_activity/scripts/eda_boltz2_feature_bakeoff.py` for the
column list. Includes:

- `affinity_pred_value` + 2 ensemble members
- `affinity_probability_binary` + 2 ensemble members
- `iptm`, `ligand_iptm`, `protein_iptm`, `ptm`, `confidence_score`
- `complex_plddt`, `complex_iplddt`, `complex_pde`, `complex_ipde`
- `ligand_atom_count`, `ligand_to_pocket_distance_a`
- Derived: `ensemble_diff_affinity`, `ensemble_diff_prob`

### Tier-1 — Confidence-map aggregates (44 features)

Re-aggregates the per-token pLDDT and per-token-pair PAE/PDE matrices
from Boltz-2's `.npz` outputs separately for protein / ligand / pocket /
cross-blocks. Extractor:
`track1_activity/scripts/extract_boltz2_confidence_features.py` →
`data/boltz2_confidence_features.parquet` (gitignored).

Key finding: `pde_pocket_ligand_min` is the strongest Tier-1 feature
(rank 11 in the combined model). It answers *"is there a
high-confidence pocket residue—ligand atom distance prediction?"* — a
physically meaningful proxy for "the model placed at least one specific
contact with confidence".

### Tier-2 — ProLIF interaction fingerprints (114 features; NOT ADOPTED)

Extracts per-residue / per-interaction-type bits from the predicted
complex via ProLIF. Extractor:
`track1_activity/scripts/extract_boltz2_ifp_features.py` →
`data/boltz2_ifp_features.parquet` (gitignored, regenerable in ~6 min).

Requires `prolif` + `pdbfixer` (added to `pyproject.toml`).

Pipeline detail: gemmi reads the predicted CIF, PDBFixer adds H's,
`RDKit.MolFromPDBFile(..., sanitize=False, proximityBonding=True)`
loads the protein, the post-processed ligand pkl provides bond-accurate
ligand structure.

**Result**: adding Tier-2 on top of Mordred + Tier-0 + Tier-1 gave
**zero additional OOF MAE/RAE improvement**. Only 43/114 features had
non-zero LGBM gain, and their total contributed 0.3% of the model's
total gain. PXR's plastic binding pocket means most drug-like
compounds engage a handful of the same core residues (S247, Q285,
H407, W299, L209) regardless of chemistry, so IFP bits carry little
compound-differentiating information once Mordred's H-bond/aromatic
counts + Boltz-2's global pocket-distance scalar are in the model.

The pipeline is kept for future use (e.g. as a standalone orthogonal
member) but not added to the final feature set.

## Bake-off results (canonical UMAP split, 5-fold, default LGBM)

| Feature set | #feat | OOF MAE | OOF RAE | Pearson |
|---|---:|---:|---:|---:|
| mordred | 1531 | 0.5369 | 0.5900 | 0.7483 |
| rdkit_desc_full | 217 | 0.5468 | 0.6009 | 0.7384 |
| boltz2_tier0 | 19 | 0.6055 | 0.6654 | 0.6845 |
| boltz2_tier0+tier1 | 63 | 0.5909 | 0.6494 | 0.7011 |
| boltz2_tier0+tier1+tier2 | 177 | 0.5894 | 0.6478 | 0.7012 |
| **mordred + tier0** | **1550** | **0.5246** | **0.5765** | **0.7657** |
| **mordred + tier0 + tier1** | **1594** | **0.5204** | **0.5719** | **0.7648** |
| mordred + tier0 + tier1 + tier2 | 1708 | 0.5204 | 0.5719 | 0.7659 |

Tier-2 contribution is strictly zero on MAE/RAE. Keep Tier-0 + Tier-1,
drop Tier-2.

## Optuna tuning (20 trials)

Script: `track1_activity/scripts/tune_mordred_boltz2_optuna.py`.
Objective: minimize OOF MAE (per `project_oof_lb_gap_is_rae_denominator`
memo — MAE transfers more cleanly than RAE).

| | MAE | RAE | Pearson |
|---|---:|---:|---:|
| baseline (default params) | 0.5204 | 0.5719 | 0.7648 |
| **Optuna best (trial 11)** | **0.5113** | **0.5619** | **0.7739** |
| Δ | -0.0091 | -0.0100 | +0.0091 |

**Best hyperparameters**: `num_leaves=252, learning_rate=0.010,
max_depth=4, feature_fraction=0.64, bagging_fraction=0.75, bagging_freq=6,
min_child_samples=42, reg_alpha=1.1e-4, reg_lambda=0.023`.

Pattern: low learning-rate + shallow `max_depth=4` + moderate
regularization. "Wide but shallow" fits this feature mix better than
the deeper default. Top-5 trials all converged to this region, so the
optimum is robust.

## Feature-importance breakdown (final tuned model)

Gain share by tier:

| Tier | #features | utilized | gain share | gain/feature vs Mordred |
|---|---:|---:|---:|---:|
| mordred | 1531 | 1228 | 72.1% | 1× (baseline) |
| boltz2 tier-0 | 19 | 18 | **23.8%** | **26.6×** |
| boltz2 tier-1 | 44 | 44 | 3.8% | 1.8× |
| boltz2 tier-2 | 114 | 43 | 0.3% | 0.05× |

Top 10 features overall:

| rank | feature | tier | gain |
|---:|---|---|---:|
| 1 | affinity_pred_value | tier0 | 27644 |
| 2 | SLogP | mordred | 21173 |
| 3 | affinity_probability_binary_2 | tier0 | 8208 |
| 4 | MINdO | mordred | 3480 |
| 5 | ABCGG | mordred | 3398 |
| 6 | affinity_pred_value_2 | tier0 | 3111 |
| 7 | affinity_probability_binary | tier0 | 2690 |
| 8 | ATSC0d | mordred | 2445 |
| 9 | VE1_Dzi | mordred | 2335 |
| 10 | VR3_DzZ | mordred | 2257 |

Note the ensemble-member-2 dominance: `affinity_probability_binary_2`
at rank 3 (gain 8208) vs member 1 at rank 42 (gain 543). Boltz-2
paper notes member 2 is trained with 4 PairFormer layers and
λ_focal=0.6, early-stopped at 12.5M samples — the lighter-regularized
variant calibrates better for PXR than member 1 (8 layers, 55M samples,
λ_focal=0.8).

## Orthogonality check (OOF vs existing ensemble members)

Pearson correlation of the tuned model's OOF with current ensemble
members:

| member | OOF RAE | corr with new model |
|---|---:|---:|
| lgbm_mordred_jazzy_umap | 0.5784 | **0.969** |
| tabpfn_mordred_jazzy_umap | 0.5453 | **0.952** |
| lgbm_rdkit_desc_full_umap | 0.5887 | 0.943 |
| residual_physprop+mordred_umap | 0.5861 | 0.943 |
| tabpfn_chemeleon_umap | 0.5625 | 0.914 |
| lgbm_chemeleon_umap | — | 0.897 |
| chemprop_multitask5_umap_aux0.0_tuned | 0.5817 | 0.876 |
| chemprop_optuna_umap | 0.5785 | 0.872 |

**Implication**: despite Boltz-2 contributing 23.8% of LGBM gain, the
final predictions still track `lgbm_mordred_jazzy_umap` at r=0.97.
Boltz-2 corrects Mordred's predictions locally but does not reshape the
overall prediction manifold. Per the existing "mordred family collapse"
rule in `track1_activity/scripts/run_ensemble.py`, this model would
**replace** `lgbm_mordred_jazzy_umap` rather than be added alongside it.

## Scripts overview

- `eda_boltz2_affinity_vs_pec50.py` — diagnostic scatter + linear fit
  between Boltz-2 `affinity_pred_value` and experimental pEC50. Finds
  Pearson r = -0.54 (negative because affinity is log10 IC50, pEC50 is
  -log10 EC50). Slope -0.928 matches the naive -1 from theory; intercept
  needs assay-specific calibration.
- `eda_boltz2_residual_profile.py` / `eda_boltz2_residual_compounds.py`
  — profile which compounds Boltz-2 over/under-predicts. Boltz-2
  systematically **over-predicts weak binders** (pEC50 3-4) and
  **under-predicts strong binders** (pEC50 5-6). Scaffold-level
  effects are weak; sulfonanilide and aryl-piperazine are slightly
  worse than indole, but within noise.
- `extract_boltz2_confidence_features.py` — Tier-1 extractor.
- `extract_boltz2_ifp_features.py` — Tier-2 (ProLIF) extractor.
- `eda_boltz2_feature_bakeoff.py` — Mordred / rdkit_desc_full +
  Tier-0 LGBM bake-off (PoseBusters dropped after first pass — all
  booleans gain=0).
- `eda_boltz2_confidence_bakeoff.py` — adds Tier-1 to the bake-off.
- `eda_boltz2_ifp_bakeoff.py` — adds Tier-2, confirms no incremental
  value.
- `eda_boltz2_importance_summary.py` — tier-level gain share and
  per-tier top-N importance.
- `tune_mordred_boltz2_optuna.py` — Optuna (20 trials) on
  mordred + tier0 + tier1.

Reports (CSVs, PNGs, OOF) live under `track1_activity/reports/`.

## Next steps (not in this PR)

- Decide whether to **replace** `lgbm_mordred_jazzy_umap` in the
  ensemble with the tuned mordred + tier0 + tier1 model. Requires
  running full train → test predict, registering as an experiment,
  and re-running `run_ensemble.py`. This was deferred pending a
  broader architecture review of the descriptor/model zoo
  (potentially collapsing mordred/rdkit into a single canonical
  descriptor feature and pruning LGBM variants in favour of TabPFN).
- Investigate adding a **Boltz-2-only** orthogonal ensemble member
  (tier0 + tier1 without Mordred; OOF RAE ~0.65 untuned, likely
  lower after Optuna). Correlation with Mordred-based members would
  be much lower than the current r=0.97, adding genuine diversity.
- Tier-2 (ProLIF IFP) extractor stays in the repo; useful as a
  standalone member or for future PXR-specific engineered features
  (e.g., GLN285 HBAcceptor as a single feature — this interaction
  occurs in 30.6% of compounds and is the canonical PXR H-bond).
