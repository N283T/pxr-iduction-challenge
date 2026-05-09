# Submission Preflight: `ens_gsl_mpp_learned_initial_small_g0p2_c0p03.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_gsl_mpp_learned_initial_small_g0p2_c0p03.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999688
- Spearman vs anchor: 0.999388
- mean shift: +0.017530
- mean abs shift: 0.024404
- p90 abs shift: 0.030000
- max abs shift: 0.030000
- |shift| > 0.05: 0
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.815881 / 0.771148

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.040476 |   0.040812 |               0.001264 |
| id56_minus_id51 |  0.050826 |   0.059712 |               0.007972 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
