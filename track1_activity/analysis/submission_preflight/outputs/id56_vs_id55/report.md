# Submission Preflight: `ens_swap_optuna_t10_top500_calibrated_importance.csv`

Verdict: **HOLD**

## Inputs

- candidate: `track1_activity/submissions/ens_swap_optuna_t10_top500_calibrated_importance.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.995891
- Spearman vs anchor: 0.995475
- mean shift: -0.002784
- mean abs shift: 0.052693
- p90 abs shift: 0.111276
- max abs shift: 0.356443
- |shift| > 0.05: 206
- |shift| > 0.10: 63
- |shift| > 0.20: 10

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.795566 / 0.779627

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  1.000000 |   1.000000 |               1.000000 |
| id56_minus_id51 |  0.936864 |   0.938008 |               1.049973 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- extreme_single_compound_shift
- aligned_with_known_bad_axis
