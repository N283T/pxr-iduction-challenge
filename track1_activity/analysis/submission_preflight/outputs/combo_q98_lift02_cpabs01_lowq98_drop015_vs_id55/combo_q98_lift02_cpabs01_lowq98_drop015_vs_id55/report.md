# Submission Preflight: `combo_q98_lift02__cp_abs01_lowq98_drop015.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_train_quantile_gates/combo_q98_lift02__cp_abs01_lowq98_drop015.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.998981
- Spearman vs anchor: 0.998605
- mean shift: +0.005945
- mean abs shift: 0.006530
- p90 abs shift: 0.000000
- max abs shift: 0.200000
- |shift| > 0.05: 17
- |shift| > 0.10: 17
- |shift| > 0.20: 16

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.804296 / 0.777127

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.003232 |   0.035716 |              -0.001686 |
| id56_minus_id51 |  0.062803 |   0.088805 |               0.032671 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
