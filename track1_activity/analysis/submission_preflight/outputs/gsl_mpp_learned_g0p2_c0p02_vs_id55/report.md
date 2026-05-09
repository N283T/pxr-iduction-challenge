# Submission Preflight: `ens_gsl_mpp_learned_initial_small_g0p2_c0p02.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_gsl_mpp_learned_initial_small_g0p2_c0p02.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999845
- Spearman vs anchor: 0.999701
- mean shift: +0.012151
- mean abs shift: 0.017444
- p90 abs shift: 0.020000
- max abs shift: 0.020000
- |shift| > 0.05: 0
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.810501 / 0.771177

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.039880 |   0.063277 |               0.000887 |
| id56_minus_id51 |  0.052496 |   0.060068 |               0.006060 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
