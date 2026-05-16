# Submission Preflight: `ens_id57_high_activity_lift_rank3.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_id57_high_activity_lift_rank3.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999846
- Spearman vs anchor: 0.999822
- mean shift: +0.008912
- mean abs shift: 0.013159
- p90 abs shift: 0.036307
- max abs shift: 0.067850
- |shift| > 0.05: 9
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.807263 / 0.782315

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 | -0.170001 |   0.003112 |              -0.047468 |
| id56_minus_id51 |  0.080590 |   0.195336 |               0.018777 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
