# Submission Preflight: `ens_conservative_probe_simplex_mae_anchor_l2_0p03_cap_0p35.csv`

Verdict: **HOLD**

## Inputs

- candidate: `track1_activity/submissions/ens_conservative_probe_simplex_mae_anchor_l2_0p03_cap_0p35.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.993723
- Spearman vs anchor: 0.993485
- mean shift: -0.059963
- mean abs shift: 0.095144
- p90 abs shift: 0.176076
- max abs shift: 0.400399
- |shift| > 0.05: 385
- |shift| > 0.10: 213
- |shift| > 0.20: 35

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.738387 / 0.722854

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.772005 |   0.732294 |               1.083761 |
| id56_minus_id51 |  0.668382 |   0.645884 |               1.045385 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- large_anchor_shift
- extreme_single_compound_shift
- aligned_with_known_bad_axis
- rank_order_changed
