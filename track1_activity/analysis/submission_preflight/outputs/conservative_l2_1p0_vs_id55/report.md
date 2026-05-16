# Submission Preflight: `ens_conservative_probe_simplex_mae_anchor_l2_1p0.csv`

Verdict: **HOLD**

## Inputs

- candidate: `track1_activity/submissions/ens_conservative_probe_simplex_mae_anchor_l2_1p0.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.995388
- Spearman vs anchor: 0.994900
- mean shift: -0.061176
- mean abs shift: 0.092442
- p90 abs shift: 0.166788
- max abs shift: 0.353051
- |shift| > 0.05: 379
- |shift| > 0.10: 206
- |shift| > 0.20: 26

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.737174 / 0.716402

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.711055 |   0.653288 |               0.934881 |
| id56_minus_id51 |  0.577828 |   0.539112 |               0.847059 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- large_anchor_shift
- extreme_single_compound_shift
- aligned_with_known_bad_axis
- rank_order_changed
