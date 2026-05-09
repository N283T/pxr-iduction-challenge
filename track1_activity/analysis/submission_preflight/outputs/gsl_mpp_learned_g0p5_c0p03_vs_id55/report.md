# Submission Preflight: `ens_gsl_mpp_learned_initial_g0p5_c0p03.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_gsl_mpp_learned_initial_g0p5_c0p03.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999613
- Spearman vs anchor: 0.999276
- mean shift: +0.018737
- mean abs shift: 0.027767
- p90 abs shift: 0.030000
- max abs shift: 0.030000
- |shift| > 0.05: 0
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.817087 / 0.771371

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.035729 |   0.024689 |               0.000403 |
| id56_minus_id51 |  0.045667 |   0.060675 |               0.007457 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
