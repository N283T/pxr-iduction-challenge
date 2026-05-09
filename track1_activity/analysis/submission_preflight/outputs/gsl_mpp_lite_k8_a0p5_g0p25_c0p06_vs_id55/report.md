# Submission Preflight: `ens_gsl_mpp_lite_initial_k8_a0p5_g0p25_c0p06.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_gsl_mpp_lite_initial_k8_a0p5_g0p25_c0p06.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999859
- Spearman vs anchor: 0.999717
- mean shift: +0.013643
- mean abs shift: 0.014815
- p90 abs shift: 0.028525
- max abs shift: 0.060000
- |shift| > 0.05: 11
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.811993 / 0.769321

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 | -0.037332 |  -0.085597 |              -0.014439 |
| id56_minus_id51 | -0.055617 |  -0.084917 |              -0.017310 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
