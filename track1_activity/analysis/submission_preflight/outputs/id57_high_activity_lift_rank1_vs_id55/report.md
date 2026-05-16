# Submission Preflight: `ens_id57_high_activity_lift_rank1.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_id57_high_activity_lift_rank1.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999845
- Spearman vs anchor: 0.999811
- mean shift: +0.009276
- mean abs shift: 0.013478
- p90 abs shift: 0.036893
- max abs shift: 0.067850
- |shift| > 0.05: 9
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.807626 / 0.782452

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 | -0.162524 |   0.014953 |              -0.046034 |
| id56_minus_id51 |  0.087156 |   0.205037 |               0.020603 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
