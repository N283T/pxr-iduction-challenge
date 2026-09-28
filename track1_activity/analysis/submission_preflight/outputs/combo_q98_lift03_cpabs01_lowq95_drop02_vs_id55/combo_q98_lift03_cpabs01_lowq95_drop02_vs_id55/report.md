# Submission Preflight: `combo_q98_lift03__cpabs01_lowq95_drop02.csv`

Verdict: **HOLD**

## Inputs

- candidate: `track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_train_quantile_gates/combo_q98_lift03__cpabs01_lowq95_drop02.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.997208
- Spearman vs anchor: 0.996875
- mean shift: +0.005458
- mean abs shift: 0.013255
- p90 abs shift: 0.000000
- max abs shift: 0.300000
- |shift| > 0.05: 26
- |shift| > 0.10: 26
- |shift| > 0.20: 25

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.803808 / 0.784632

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.033938 |   0.085168 |               0.025535 |
| id56_minus_id51 |  0.088363 |   0.127207 |               0.081051 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- large_anchor_shift
