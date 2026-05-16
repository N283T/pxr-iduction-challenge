# Submission Preflight: `ens_id57_bold_lf_calib_rank2.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_id57_bold_lf_calib_rank2.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g50.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999690
- Spearman vs anchor: 0.999476
- mean shift: +0.020309
- mean abs shift: 0.020309
- p90 abs shift: 0.064994
- max abs shift: 0.080000
- |shift| > 0.05: 87
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798810 / 0.774569
- candidate mean/std: 4.819119 / 0.791724

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.213766 |   0.304105 |               0.066988 |
| id56_minus_id51 |  0.307341 |   0.376931 |               0.117459 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
