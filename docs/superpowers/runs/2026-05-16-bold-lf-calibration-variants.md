# Bold LF Calibration Variants

Date: 2026-05-16

Goal: use cooldown wait time to test bolder log2fc-shaped calibration around
id57. The conservative first-choice candidate remains the small
`lf_mean_soft_q50` lift unless a bolder shape looks clearly better.

## Setup

- Script:
  `track1_activity/analysis/compound_level_lb/probe_bold_lf_calibration_variants.py`
- Outputs:
  `track1_activity/analysis/compound_level_lb/outputs/bold_lf_calibration_variants/`
- Anchor:
  `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g50.csv`
  (id57)
- Candidate families:
  - high log2fc lifts
  - high-minus-low log2fc tilts
  - clipped-z log2fc stretches
  - high log2fc x high anchor prediction gates

## Best Bold-Safe Rows

| rank | candidate | variant | scale | full OOF delta | high-y pseudo delta | p90 shift vs id57 | max shift vs id57 | id56 projection vs id57 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `ens_id57_bold_lf_calib_rank1.csv` | `lf_mean_clipped_z` | 0.05 | -0.002112 | -0.009033 | 0.050000 | 0.050000 | +0.123417 |
| 2 | `ens_id57_bold_lf_calib_rank2.csv` | `lf_mean_high_q50` | 0.08 | -0.001092 | -0.008994 | 0.064994 | 0.080000 | +0.066988 |
| 3 | `ens_id57_bold_lf_calib_rank3.csv` | `lf_mean_tilt_highq50_lowq50` | 0.08 | -0.001980 | -0.008511 | 0.068270 | 0.080000 | +0.143122 |
| 4 | `ens_id57_bold_lf_calib_rank4.csv` | `lf_max_clipped_z` | 0.05 | -0.001798 | -0.008366 | 0.050000 | 0.050000 | +0.109695 |
| 5 | `ens_id57_bold_lf_calib_rank5.csv` | `lf_mean_highq50_x_pred_highq60` | 0.08 | -0.001036 | -0.007671 | 0.058290 | 0.080000 | +0.050381 |

All five pass preflight vs id57 and id55.

## Preflight Summary

Against id57:

| rank | verdict | mean abs shift | p90 shift | max shift | rows >0.05 | rows >0.10 | id56 projection |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | PASS | 0.031145 | 0.050000 | 0.050000 | 4 | 0 | +0.123417 |
| 2 | PASS | 0.020309 | 0.064994 | 0.080000 | 87 | 0 | +0.066988 |
| 3 | PASS | 0.036311 | 0.068270 | 0.080000 | 148 | 0 | +0.143122 |
| 4 | PASS | 0.029076 | 0.050000 | 0.050000 | 2 | 0 | +0.109695 |
| 5 | PASS | 0.016824 | 0.058290 | 0.080000 | 73 | 0 | +0.050381 |

Against id55:

| rank | verdict | mean abs shift | p90 shift | max shift | rows >0.05 | rows >0.10 | id56 projection |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | PASS | 0.034011 | 0.057324 | 0.107658 | 90 | 2 | +0.053103 |
| 2 | PASS | 0.024771 | 0.067917 | 0.098637 | 97 | 0 | -0.003326 |
| 3 | PASS | 0.038934 | 0.075981 | 0.119265 | 166 | 5 | +0.072808 |
| 4 | PASS | 0.031920 | 0.054296 | 0.107658 | 73 | 2 | +0.039381 |
| 5 | PASS | 0.021478 | 0.062834 | 0.098637 | 84 | 0 | -0.019933 |

## Interpretation

The bolder variants are not ridiculous, but most of the locally strongest rows
start to project positively onto the id56 bad axis and move many more compounds
than the conservative rank2 candidate.

The only bold candidate I would seriously consider is rank5:

- stronger proxy evidence than conservative rank2
- still a simple high-log2fc/high-prediction lift
- lowest id56 projection among the bold rows
- anti-aligned with id56 when measured from id55

However, it is still a larger move (p90 0.058, max 0.080 vs id57), so the
recommendation remains:

1. Conservative submit: `ens_id57_high_activity_lift_rank2.csv`
2. Bold submit: `ens_id57_bold_lf_calib_rank5.csv`
3. Ultra-conservative fallback: `ens_id57_high_activity_lift_lfmean_g020.csv`

No submission was made from this run.
