# Submission Preflight: `ens_id57_bold_lf_calib_rank4.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_id57_bold_lf_calib_rank4.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g50.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999782
- Spearman vs anchor: 0.999528
- mean shift: +0.004662
- mean abs shift: 0.029076
- p90 abs shift: 0.050000
- max abs shift: 0.050000
- |shift| > 0.05: 2
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798810 / 0.774569
- candidate mean/std: 4.803472 / 0.803231

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.240922 |   0.288354 |               0.109695 |
| id56_minus_id51 |  0.359971 |   0.359473 |               0.186101 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
