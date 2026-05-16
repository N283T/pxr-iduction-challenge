# Submission Preflight: `ens_id57_bold_lf_calib_rank5.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_id57_bold_lf_calib_rank5.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g50.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999724
- Spearman vs anchor: 0.999725
- mean shift: +0.016824
- mean abs shift: 0.016824
- p90 abs shift: 0.058290
- max abs shift: 0.080000
- |shift| > 0.05: 73
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798810 / 0.774569
- candidate mean/std: 4.815634 / 0.790412

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.174553 |   0.277932 |               0.050381 |
| id56_minus_id51 |  0.267237 |   0.358396 |               0.095307 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
