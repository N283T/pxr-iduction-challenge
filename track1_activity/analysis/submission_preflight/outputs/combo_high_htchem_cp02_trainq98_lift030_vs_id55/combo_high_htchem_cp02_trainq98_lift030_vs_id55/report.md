# Submission Preflight: `combo_high_htchem_cp02_q98_lift030.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_train_quantile_gates/combo_high_htchem_cp02_q98_lift030.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.997811
- Spearman vs anchor: 0.997283
- mean shift: +0.009357
- mean abs shift: 0.009357
- p90 abs shift: 0.000000
- max abs shift: 0.300000
- |shift| > 0.05: 16
- |shift| > 0.10: 16
- |shift| > 0.20: 16

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.807707 / 0.780482

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 | -0.009680 |   0.020926 |              -0.012322 |
| id56_minus_id51 |  0.053936 |   0.077503 |               0.040496 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
