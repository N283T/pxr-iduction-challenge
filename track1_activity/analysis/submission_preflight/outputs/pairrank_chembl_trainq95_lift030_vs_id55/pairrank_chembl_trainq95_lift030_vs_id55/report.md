# Submission Preflight: `pairrank_chembl_q95_lift030.csv`

Verdict: **HOLD**

## Inputs

- candidate: `track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_train_quantile_gates/pairrank_chembl_q95_lift030.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.994583
- Spearman vs anchor: 0.990266
- mean shift: +0.025731
- mean abs shift: 0.025731
- p90 abs shift: 0.000000
- max abs shift: 0.300000
- |shift| > 0.05: 44
- |shift| > 0.10: 44
- |shift| > 0.20: 44

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.824081 / 0.792279

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.021660 |   0.039852 |               0.011389 |
| id56_minus_id51 |  0.070404 |   0.079737 |               0.082556 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- large_anchor_shift
- rank_order_changed
