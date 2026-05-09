# Submission Preflight: `ens_gsl_mpp_learned_initial_g0p5_c0p06.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_gsl_mpp_learned_initial_g0p5_c0p06.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.998668
- Spearman vs anchor: 0.997722
- mean shift: +0.035862
- mean abs shift: 0.050859
- p90 abs shift: 0.060000
- max abs shift: 0.060000
- |shift| > 0.05: 385
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.834212 / 0.772139

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.040392 |   0.067257 |               0.002767 |
| id56_minus_id51 |  0.052606 |   0.063116 |               0.017749 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
