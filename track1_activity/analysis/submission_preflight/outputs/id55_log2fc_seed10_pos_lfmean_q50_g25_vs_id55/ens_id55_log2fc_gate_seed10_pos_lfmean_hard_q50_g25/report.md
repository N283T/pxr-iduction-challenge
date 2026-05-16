# Submission Preflight: `ens_id55_log2fc_gate_seed10_pos_lfmean_hard_q50_g25.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_id55_log2fc_gate_seed10_pos_lfmean_hard_q50_g25.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999828
- Spearman vs anchor: 0.999600
- mean shift: +0.007805
- mean abs shift: 0.007805
- p90 abs shift: 0.031223
- max abs shift: 0.097980
- |shift| > 0.05: 14
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.806155 / 0.777807

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 | -0.112268 |  -0.063922 |              -0.029475 |
| id56_minus_id51 |  0.019091 |   0.092291 |               0.001446 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
