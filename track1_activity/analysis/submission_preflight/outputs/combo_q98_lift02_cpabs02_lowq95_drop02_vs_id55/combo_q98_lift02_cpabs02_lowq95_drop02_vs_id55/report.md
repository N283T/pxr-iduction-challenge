# Submission Preflight: `combo_q98_lift02__cpabs02_lowq95_drop02.csv`

Verdict: **HOLD**

## Inputs

- candidate: `track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_train_quantile_gates/combo_q98_lift02__cpabs02_lowq95_drop02.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.998352
- Spearman vs anchor: 0.998232
- mean shift: +0.001949
- mean abs shift: 0.010526
- p90 abs shift: 0.000000
- max abs shift: 0.200000
- |shift| > 0.05: 27
- |shift| > 0.10: 27
- |shift| > 0.20: 25

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.800300 / 0.781868

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.037729 |   0.065334 |               0.023327 |
| id56_minus_id51 |  0.092935 |   0.113499 |               0.066578 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- large_anchor_shift
