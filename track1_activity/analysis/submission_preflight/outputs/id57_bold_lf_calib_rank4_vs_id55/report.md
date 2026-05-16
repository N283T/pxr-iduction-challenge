# Submission Preflight: `ens_id57_bold_lf_calib_rank4.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_id57_bold_lf_calib_rank4.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999735
- Spearman vs anchor: 0.999552
- mean shift: +0.005121
- mean abs shift: 0.031920
- p90 abs shift: 0.054296
- max abs shift: 0.107658
- |shift| > 0.05: 73
- |shift| > 0.10: 2
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.803472 / 0.803231

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.080959 |   0.189856 |               0.039381 |
| id56_minus_id51 |  0.285376 |   0.333339 |               0.164684 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
