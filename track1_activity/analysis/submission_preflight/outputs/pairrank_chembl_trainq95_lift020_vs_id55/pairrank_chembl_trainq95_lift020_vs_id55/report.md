# Submission Preflight: `pairrank_chembl_q95_lift020.csv`

Verdict: **HOLD**

## Inputs

- candidate: `track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_train_quantile_gates/pairrank_chembl_q95_lift020.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.997547
- Spearman vs anchor: 0.995230
- mean shift: +0.017154
- mean abs shift: 0.017154
- p90 abs shift: 0.000000
- max abs shift: 0.200000
- |shift| > 0.05: 44
- |shift| > 0.10: 44
- |shift| > 0.20: 44

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.815504 / 0.784244

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.021660 |   0.039844 |               0.007592 |
| id56_minus_id51 |  0.070404 |   0.079869 |               0.055038 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- large_anchor_shift
