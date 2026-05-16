# Submission Preflight: `ens_id57_bold_lf_calib_rank2.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_id57_bold_lf_calib_rank2.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999628
- Spearman vs anchor: 0.999452
- mean shift: +0.020769
- mean abs shift: 0.024771
- p90 abs shift: 0.067917
- max abs shift: 0.098637
- |shift| > 0.05: 97
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.819119 / 0.791724

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.019581 |   0.118681 |              -0.003326 |
| id56_minus_id51 |  0.223252 |   0.284680 |               0.096042 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
