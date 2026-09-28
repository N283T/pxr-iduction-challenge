# Submission Preflight: `combo_high_htchem_cp02_q98_lift020.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_train_quantile_gates/combo_high_htchem_cp02_q98_lift020.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999019
- Spearman vs anchor: 0.998624
- mean shift: +0.006238
- mean abs shift: 0.006238
- p90 abs shift: 0.000000
- max abs shift: 0.200000
- |shift| > 0.05: 16
- |shift| > 0.10: 16
- |shift| > 0.20: 16

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.804588 / 0.776940

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 | -0.009680 |   0.020405 |              -0.008214 |
| id56_minus_id51 |  0.053936 |   0.077009 |               0.026997 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
