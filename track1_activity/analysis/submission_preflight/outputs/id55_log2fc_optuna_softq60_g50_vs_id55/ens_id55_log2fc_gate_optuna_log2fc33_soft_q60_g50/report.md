# Submission Preflight: `ens_id55_log2fc_gate_optuna_log2fc33_soft_q60_g50.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_id55_log2fc_gate_optuna_log2fc33_soft_q60_g50.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999743
- Spearman vs anchor: 0.999375
- mean shift: +0.005139
- mean abs shift: 0.008220
- p90 abs shift: 0.028294
- max abs shift: 0.096722
- |shift| > 0.05: 22
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.803489 / 0.775514

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.164208 |   0.279406 |               0.039054 |
| id56_minus_id51 |  0.286471 |   0.384915 |               0.079768 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
