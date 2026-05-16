# Submission Preflight: `ens_id57_high_activity_lift_rank2.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_id57_high_activity_lift_rank2.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g50.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999955
- Spearman vs anchor: 0.999902
- mean shift: +0.007616
- mean abs shift: 0.007616
- p90 abs shift: 0.024373
- max abs shift: 0.030000
- |shift| > 0.05: 0
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798810 / 0.774569
- candidate mean/std: 4.806426 / 0.780945

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.213766 |   0.303873 |               0.025120 |
| id56_minus_id51 |  0.307341 |   0.376742 |               0.044047 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
