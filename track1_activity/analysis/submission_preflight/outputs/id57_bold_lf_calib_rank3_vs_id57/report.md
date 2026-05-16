# Submission Preflight: `ens_id57_bold_lf_calib_rank3.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_id57_bold_lf_calib_rank3.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g50.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999657
- Spearman vs anchor: 0.999338
- mean shift: +0.004307
- mean abs shift: 0.036311
- p90 abs shift: 0.068270
- max abs shift: 0.080000
- |shift| > 0.05: 148
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798810 / 0.774569
- candidate mean/std: 4.803117 / 0.811725

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.242458 |   0.305266 |               0.143122 |
| id56_minus_id51 |  0.369395 |   0.393892 |               0.246718 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
