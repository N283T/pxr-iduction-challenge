# Submission Preflight: `ens_gsl_mpp_learned_initial_g0p25_c0p03.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_gsl_mpp_learned_initial_g0p25_c0p03.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999667
- Spearman vs anchor: 0.999364
- mean shift: +0.017931
- mean abs shift: 0.025430
- p90 abs shift: 0.030000
- max abs shift: 0.030000
- |shift| > 0.05: 0
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.816281 / 0.771306

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.040392 |   0.043771 |               0.001383 |
| id56_minus_id51 |  0.052606 |   0.067868 |               0.008875 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
