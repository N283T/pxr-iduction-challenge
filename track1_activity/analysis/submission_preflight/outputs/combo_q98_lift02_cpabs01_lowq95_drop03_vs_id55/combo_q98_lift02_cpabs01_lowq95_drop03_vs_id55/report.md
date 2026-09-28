# Submission Preflight: `combo_q98_lift02__cpabs01_lowq95_drop03.csv`

Verdict: **HOLD**

## Inputs

- candidate: `track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_train_quantile_gates/combo_q98_lift02__cpabs01_lowq95_drop03.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.997641
- Spearman vs anchor: 0.997836
- mean shift: +0.000390
- mean abs shift: 0.012086
- p90 abs shift: 0.000000
- max abs shift: 0.300000
- |shift| > 0.05: 26
- |shift| > 0.10: 26
- |shift| > 0.20: 26

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.798740 / 0.783529

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.063093 |   0.084562 |               0.048570 |
| id56_minus_id51 |  0.101500 |   0.126699 |               0.087830 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- large_anchor_shift
