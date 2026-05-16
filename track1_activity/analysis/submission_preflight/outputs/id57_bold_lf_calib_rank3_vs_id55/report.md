# Submission Preflight: `ens_id57_bold_lf_calib_rank3.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_id57_bold_lf_calib_rank3.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999607
- Spearman vs anchor: 0.999338
- mean shift: +0.004766
- mean abs shift: 0.038934
- p90 abs shift: 0.075981
- max abs shift: 0.119265
- |shift| > 0.05: 166
- |shift| > 0.10: 5
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.803117 / 0.811725

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.115368 |   0.226076 |               0.072808 |
| id56_minus_id51 |  0.310027 |   0.364387 |               0.225301 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
