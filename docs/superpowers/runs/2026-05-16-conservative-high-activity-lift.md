# Conservative High-Activity Lift Probe

Date: 2026-05-16

Goal: test a very small calibration-style lift after the pseudo-public retrain
battery showed underprediction on the `public_hybrid_with_y_top513` holdout.

## Setup

- Script:
  `track1_activity/analysis/compound_level_lb/probe_conservative_high_activity_lift.py`
- Outputs:
  `track1_activity/analysis/compound_level_lb/outputs/conservative_high_activity_lift/`
- Anchor CSV:
  `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g50.csv`
  (LB id57, current best among recent probes)
- Candidate formula:

```text
candidate = id57 + amount * gate(log2fc_pred, anchor_pred, optional potent46)
```

This intentionally avoids adding more top500 delta after id58 regressed on LB.

## Result

Best safe-ish candidates:

| candidate | gate | amount | full OOF delta MAE | public stress mean delta | high-y pseudo delta | p90 shift vs id57 | max shift vs id57 | id56 projection vs id57 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| rank1 | `lf_max_soft_q50_AND_pred_soft_q50` | 0.05 | -0.000545 | -0.001913 | -0.004353 | 0.029765 | 0.050000 | +0.024280 |
| rank2 | `lf_mean_soft_q50` | 0.03 | -0.000706 | -0.001868 | -0.004313 | 0.024373 | 0.030000 | +0.025120 |
| rank3 | `lf_max_soft_q50_AND_pred_soft_q60` | 0.05 | -0.000546 | -0.001985 | -0.004178 | 0.028799 | 0.050000 | +0.022846 |

Preflight vs id57:

| candidate | verdict | mean abs shift | p90 shift | max shift | rows >0.05 | id56 projection |
|---|---|---:|---:|---:|---:|---:|
| rank1 | PASS | 0.008816 | 0.029765 | 0.050000 | 0 | +0.024280 |
| rank2 | PASS | 0.007616 | 0.024373 | 0.030000 | 0 | +0.025120 |
| rank3 | PASS | 0.008452 | 0.028799 | 0.050000 | 0 | +0.022846 |

Preflight vs id55:

| candidate | verdict | mean abs shift | p90 shift | max shift | rows >0.05 | id56 projection |
|---|---|---:|---:|---:|---:|---:|
| rank1 | PASS | 0.013478 | 0.036893 | 0.067850 | 9 | -0.046034 |
| rank2 | PASS | 0.012287 | 0.030000 | 0.067850 | 5 | -0.045194 |
| rank3 | PASS | 0.013159 | 0.036307 | 0.067850 | 9 | -0.047468 |

Generated ignored candidate CSVs:

- `track1_activity/submissions/ens_id57_high_activity_lift_rank1.csv`
- `track1_activity/submissions/ens_id57_high_activity_lift_rank2.csv`
- `track1_activity/submissions/ens_id57_high_activity_lift_rank3.csv`

## Interpretation

This is a cleaner follow-up than id58 because it does not rely on another
top500 movement and the shift is tiny. The signal is still only proxy-level:
the full OOF gain is about 0.0005-0.0007 MAE, while the intended high-y
pseudo-public holdout improves by about 0.004.

If spending a cooldown, rank2 is the most conservative choice despite rank1
having the slightly better high-y proxy score. Rank2 is just a 0.03 pEC50 lift
on high predicted log2fc mean, has the smallest max shift vs id57, has the best
full OOF delta among the three, and is anti-aligned with the id56 bad axis when
viewed from id55.

No submission was made from this run.
