# Submission Preflight: `ens_id57_high_activity_lift_rank3.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_id57_high_activity_lift_rank3.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g50.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999910
- Spearman vs anchor: 0.999860
- mean shift: +0.008452
- mean abs shift: 0.008452
- p90 abs shift: 0.028799
- max abs shift: 0.050000
- |shift| > 0.05: 0
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798810 / 0.774569
- candidate mean/std: 4.807263 / 0.782315

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.150434 |   0.262594 |               0.022846 |
| id56_minus_id51 |  0.213475 |   0.322991 |               0.040194 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
