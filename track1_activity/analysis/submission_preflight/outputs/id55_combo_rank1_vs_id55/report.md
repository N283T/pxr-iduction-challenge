# Submission Preflight: `ens_id55_combo_gate_rank1.csv`

Verdict: **PASS**

## Inputs

- candidate: `track1_activity/submissions/ens_id55_combo_gate_rank1.csv`
- anchor: `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## CSV Sanity

- rows: 513 / 513
- SMILES order match: True
- Molecule Name order match: True

## Anchor Shift

- Pearson vs anchor: 0.999780
- Spearman vs anchor: 0.999537
- mean shift: +0.008316
- mean abs shift: 0.010797
- p90 abs shift: 0.032508
- max abs shift: 0.077779
- |shift| > 0.05: 18
- |shift| > 0.10: 0
- |shift| > 0.20: 0

## Prediction Distribution

- anchor mean/std: 4.798350 / 0.770987
- candidate mean/std: 4.806666 / 0.776449

## Known Bad Axis

| label           |   pearson |   spearman |   candidate_projection |
|:----------------|----------:|-----------:|-----------------------:|
| id56_minus_id55 |  0.275546 |   0.394273 |               0.061939 |
| id56_minus_id51 |  0.444627 |   0.548586 |               0.116850 |

## Experiment Metadata

No matching `experiment_summary` row found for this CSV path.

## Reasons

- small_anchor_shift
