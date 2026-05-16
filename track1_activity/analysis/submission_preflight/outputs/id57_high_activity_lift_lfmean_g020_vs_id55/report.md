# Submission Preflight: `ens_id57_high_activity_lift_lfmean_g020.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_id57_high_activity_lift_lfmean_g020.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999901
- Spearman vs anchor: 0.999867
- mean shift: +0.005537
- mean abs shift: 0.009909
- p90 abs shift: 0.023786
- max abs shift: 0.067850
- |shift| > 0.05: 3
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.803887 / 0.778812

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 | -0.266582 |  -0.018925 |              -0.053567 |
| id56_minus_id51 |  0.048608 |   0.215110 |               0.007948 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
