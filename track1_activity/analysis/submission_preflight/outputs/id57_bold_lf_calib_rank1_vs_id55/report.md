# Submission Preflight: `ens_id57_bold_lf_calib_rank1.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_id57_bold_lf_calib_rank1.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999722
- Spearman vs anchor: 0.999590
- mean shift: +0.005976
- mean abs shift: 0.034011
- p90 abs shift: 0.057324
- max abs shift: 0.107658
- |shift| > 0.05: 90
- |shift| > 0.10: 2
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.804326 / 0.805377

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.102396 |   0.212266 |               0.053103 |
| id56_minus_id51 |  0.314214 |   0.367394 |               0.191560 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
