# Submission Preflight: `combo_q98_lift03__cpabs01_lowq95_drop03.csv`

Verdict: **HOLD**

## Inputs

- candidate: `track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_train_quantile_gates/combo_q98_lift03__cpabs01_lowq95_drop03.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.996457
- Spearman vs anchor: 0.996499
- mean shift: +0.003509
- mean abs shift: 0.015205
- p90 abs shift: 0.000000
- max abs shift: 0.300000
- |shift| > 0.05: 26
- |shift| > 0.10: 26
- |shift| > 0.20: 26

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.801859 / 0.787064

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.048757 |   0.084981 |               0.044463 |
| id56_minus_id51 |  0.096352 |   0.127095 |               0.101329 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- large_anchor_shift
