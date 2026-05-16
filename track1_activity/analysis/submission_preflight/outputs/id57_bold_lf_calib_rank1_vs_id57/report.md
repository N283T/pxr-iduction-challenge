# Submission Preflight: `ens_id57_bold_lf_calib_rank1.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_id57_bold_lf_calib_rank1.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g50.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999783
- Spearman vs anchor: 0.999608
- mean shift: +0.005516
- mean abs shift: 0.031145
- p90 abs shift: 0.050000
- max abs shift: 0.050000
- |shift| > 0.05: 4
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798810 / 0.774569
- candidate mean/std: 4.804326 / 0.805377

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.256899 |   0.306283 |               0.123417 |
| id56_minus_id51 |  0.390109 |   0.396503 |               0.212977 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
