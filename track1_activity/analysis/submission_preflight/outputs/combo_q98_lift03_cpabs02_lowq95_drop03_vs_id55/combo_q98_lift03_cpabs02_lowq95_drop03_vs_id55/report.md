# Submission Preflight: `combo_q98_lift03__cpabs02_lowq95_drop03.csv`

Verdict: **HOLD**

## Inputs

- candidate: `track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_train_quantile_gates/combo_q98_lift03__cpabs02_lowq95_drop03.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.996348
- Spearman vs anchor: 0.996501
- mean shift: +0.002924
- mean abs shift: 0.015789
- p90 abs shift: 0.000000
- max abs shift: 0.300000
- |shift| > 0.05: 27
- |shift| > 0.10: 27
- |shift| > 0.20: 27

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.801274 / 0.788254

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.037729 |   0.065323 |               0.034991 |
| id56_minus_id51 |  0.092935 |   0.113608 |               0.099868 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- large_anchor_shift
