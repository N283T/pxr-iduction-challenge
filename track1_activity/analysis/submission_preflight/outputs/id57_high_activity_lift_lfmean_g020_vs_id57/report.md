# Submission Preflight: `ens_id57_high_activity_lift_lfmean_g020.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_id57_high_activity_lift_lfmean_g020.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g50.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999980
- Spearman vs anchor: 0.999951
- mean shift: +0.005077
- mean abs shift: 0.005077
- p90 abs shift: 0.016248
- max abs shift: 0.020000
- |shift| > 0.05: 0
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798810 / 0.774569
- candidate mean/std: 4.803887 / 0.778812

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.213766 |   0.303987 |               0.016747 |
| id56_minus_id51 |  0.307341 |   0.376865 |               0.029365 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
