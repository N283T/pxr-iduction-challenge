# Submission Preflight: `ens_id57_high_activity_lift_rank2.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_id57_high_activity_lift_rank2.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999879
- Spearman vs anchor: 0.999830
- mean shift: +0.008076
- mean abs shift: 0.012287
- p90 abs shift: 0.030000
- max abs shift: 0.067850
- |shift| > 0.05: 5
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.806426 / 0.780945

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 | -0.184428 |   0.028099 |              -0.045194 |
| id56_minus_id51 |  0.105437 |   0.244802 |               0.022630 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
