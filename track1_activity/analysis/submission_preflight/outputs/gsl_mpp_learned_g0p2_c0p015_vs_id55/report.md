# Submission Preflight: `ens_gsl_mpp_learned_initial_small_g0p2_c0p015.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_gsl_mpp_learned_initial_small_g0p2_c0p015.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999908
- Spearman vs anchor: 0.999796
- mean shift: +0.009274
- mean abs shift: 0.013551
- p90 abs shift: 0.015000
- max abs shift: 0.015000
- |shift| > 0.05: 0
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.807624 / 0.771113

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.039196 |   0.047320 |               0.000642 |
| id56_minus_id51 |  0.050460 |   0.041176 |               0.004385 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
