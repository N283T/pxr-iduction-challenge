# Submission Preflight: `ens_id55_shap_numrings_pos_highsoftq20_g50.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_id55_shap_numrings_pos_highsoftq20_g50.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999800
- Spearman vs anchor: 0.999654
- mean shift: +0.009419
- mean abs shift: 0.009419
- p90 abs shift: 0.034066
- max abs shift: 0.103700
- |shift| > 0.05: 20
- |shift| > 0.10: 1
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.807769 / 0.776774

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.286109 |   0.368934 |               0.061398 |
| id56_minus_id51 |  0.425906 |   0.505965 |               0.107168 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
