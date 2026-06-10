# Submission Preflight: `phase2_as1_aug_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_tabpfnv3_ne8_t0p7_labels_as1.csv`

Verdict: **HOLD**

## Inputs

- candidate: `track1_activity/submissions/phase2_as1_aug_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_tabpfnv3_ne8_t0p7_labels_as1.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.881843
- Spearman vs anchor: 0.911284
- mean shift: -0.056483
- mean abs shift: 0.264122
- p90 abs shift: 0.620348
- max abs shift: 2.875959
- |shift| > 0.05: 426
- |shift| > 0.10: 328
- |shift| > 0.20: 196

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.741868 / 0.895810

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 | -0.043822 |  -0.069379 |              -0.229886 |
| id56_minus_id51 | -0.006321 |  -0.038370 |              -0.018034 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- large_anchor_shift
- extreme_single_compound_shift
- prediction_scale_changed
- rank_order_changed
