# Pseudo-Public Holdout Retrain Battery

Date: 2026-05-16

Goal: test whether pseudo-public holdouts change model/feature rankings when
fast models are retrained, rather than only rescoring existing OOF predictions.

## Setup

- Script:
  `track1_activity/analysis/oof_reliability_audit/retrain_pseudo_public_holdouts.py`
- Outputs:
  `track1_activity/analysis/oof_reliability_audit/outputs/pseudo_public_retrain/`
- Splits:
  - `umap_canonical`
  - `public_adv_top513`
  - `public_testnn_top513`
  - `public_log2fc_top513`
  - `public_hybrid_nolabel_top513`
  - `public_hybrid_with_y_top513`
  - `public_chembl_ext_nn_ge025`
- Leakage-free retrains:
  LGBM on RDKit-41, RDKit-full, Mordred, Morgan2048,
  RDKit/Mordred plus predicted log2fc, and RDKit/Mordred plus observed
  single-concentration + Jazzy features.
- Diagnostics only:
  existing pool OOF simple mean, Ridge stack, and Caruana stack. These are not
  split-specific base-model retrains.

## Main Results

Leakage-free retrain winners by MAE:

| split | best leakage-free model | MAE | Spearman | bias |
|---|---|---:|---:|---:|
| `umap_canonical` | `rdkit_full_lf_pred_lgbm` | 0.3883 | 0.8510 | +0.0258 |
| `public_adv_top513` | `rdkit_full_lf_pred_lgbm` | 0.3746 | 0.8204 | -0.0902 |
| `public_testnn_top513` | `mordred_lf_pred_lgbm` | 0.3597 | 0.8660 | -0.0616 |
| `public_log2fc_top513` | `rdkit_full_lf_pred_lgbm` | 0.2749 | 0.6378 | -0.0077 |
| `public_hybrid_nolabel_top513` | `rdkit_full_lf_pred_lgbm` | 0.3300 | 0.7584 | -0.0809 |
| `public_hybrid_with_y_top513` | `rdkit_full_lf_pred_lgbm` | 0.3212 | 0.6971 | -0.2112 |
| `public_chembl_ext_nn_ge025` | `rdkit_full_lf_pred_lgbm` | 0.3939 | 0.8447 | +0.0371 |

Predicted log2fc features were consistently the strongest axis. Relative to the
same base descriptors without log2fc prediction:

| split | RDKit-full + log2fc delta MAE | Mordred + log2fc delta MAE |
|---|---:|---:|
| `umap_canonical` | -0.1463 | -0.1335 |
| `public_adv_top513` | -0.1625 | -0.1315 |
| `public_testnn_top513` | -0.1692 | -0.1644 |
| `public_log2fc_top513` | -0.1760 | -0.1588 |
| `public_hybrid_nolabel_top513` | -0.1698 | -0.1347 |
| `public_hybrid_with_y_top513` | -0.2232 | -0.1904 |
| `public_chembl_ext_nn_ge025` | -0.1217 | -0.1149 |

Observed single-concentration + Jazzy features helped against plain 2D features,
but did not beat predicted-log2fc variants on any tested holdout.

## Interpretation

The pseudo-public retrain battery does not reveal a new feature family that
beats the current log2fc-predicted axis. It instead reinforces the existing
story: most transferable signal is still coming through the single-concentration
activity axis, and general-purpose 2D/FP models are much weaker.

The most useful new observation is bias: the `public_hybrid_with_y_top513`
holdout shows strong underprediction even for the best leakage-free retrain
(`bias=-0.2112`). That suggests future experiments should prioritize
conservative test-like/high-activity calibration or gating over adding another
generic feature family.

## Status

No submission candidate was generated from this run.
