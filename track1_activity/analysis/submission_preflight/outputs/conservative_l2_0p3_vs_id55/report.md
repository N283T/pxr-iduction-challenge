# Submission Preflight: `ens_conservative_probe_simplex_mae_anchor_l2_0p3.csv`

Verdict: **HOLD**

## Inputs

- candidate: `track1_activity/submissions/ens_conservative_probe_simplex_mae_anchor_l2_0p3.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.994338
- Spearman vs anchor: 0.993776
- mean shift: -0.061497
- mean abs shift: 0.093772
- p90 abs shift: 0.172867
- max abs shift: 0.367013
- |shift| > 0.05: 383
- |shift| > 0.10: 212
- |shift| > 0.20: 30

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.736853 / 0.722708

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.767075 |   0.724415 |               1.039504 |
| id56_minus_id51 |  0.662815 |   0.637813 |               1.000347 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- large_anchor_shift
- extreme_single_compound_shift
- aligned_with_known_bad_axis
- rank_order_changed
