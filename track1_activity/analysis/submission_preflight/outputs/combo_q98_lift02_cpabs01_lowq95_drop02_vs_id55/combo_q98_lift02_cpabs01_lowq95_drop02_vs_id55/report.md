# Submission Preflight: `combo_q98_lift02__cpabs01_lowq95_drop02.csv`

Verdict: **HOLD**

## Inputs

- candidate: `track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_train_quantile_gates/combo_q98_lift02__cpabs01_lowq95_drop02.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.998403
- Spearman vs anchor: 0.998212
- mean shift: +0.002339
- mean abs shift: 0.010136
- p90 abs shift: 0.000000
- max abs shift: 0.200000
- |shift| > 0.05: 26
- |shift| > 0.10: 26
- |shift| > 0.20: 25

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.800689 / 0.781094

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.048757 |   0.084749 |               0.029642 |
| id56_minus_id51 |  0.096352 |   0.126810 |               0.067552 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- large_anchor_shift
