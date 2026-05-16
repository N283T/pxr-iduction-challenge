# Submission Preflight: `ens_id57_bold_lf_calib_rank5.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_id57_bold_lf_calib_rank5.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999655
- Spearman vs anchor: 0.999684
- mean shift: +0.017284
- mean abs shift: 0.021478
- p90 abs shift: 0.062834
- max abs shift: 0.098637
- |shift| > 0.05: 84
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.815634 / 0.790412

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 | -0.025947 |   0.069780 |              -0.019933 |
| id56_minus_id51 |  0.182102 |   0.245795 |               0.073891 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
