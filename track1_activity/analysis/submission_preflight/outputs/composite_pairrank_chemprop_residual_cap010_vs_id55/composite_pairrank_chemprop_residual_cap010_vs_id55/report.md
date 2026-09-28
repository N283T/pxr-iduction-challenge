# Submission Preflight: `test_residual_candidate.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/analysis/phase2_classifier_gate/outputs/composite_pairrank_chemprop_residual_cap010/test_residual_candidate.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.998554
- Spearman vs anchor: 0.998302
- mean shift: +0.027438
- mean abs shift: 0.046491
- p90 abs shift: 0.100000
- max abs shift: 0.100000
- |shift| > 0.05: 208
- |shift| > 0.10: 13
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.825789 / 0.794902

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 | -0.017417 |   0.055099 |              -0.027132 |
| id56_minus_id51 |  0.058578 |   0.088307 |               0.033075 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
