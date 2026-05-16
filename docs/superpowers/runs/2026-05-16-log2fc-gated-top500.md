# Log2fc-Gated Top500 Probe

Date: 2026-05-16

Goal: follow up on the OOF proxy failure by asking whether the log2fc-heavy
top500 signal can be used only in selected regions rather than via global
ensemble reweighting.

## Setup

- Anchor for candidate movement: id55
  `ens_id51_top500_potent46_t40_soft_g35`.
- log2fc gate source:
  `data/chemprop_pretrain_log2fc_predictions_optuna_trial10_seed5ens.parquet`.
- Delta sources:
  - `seed10_top500_minus_current_raw`
  - `optuna_top500_minus_current_raw`
- Gate families:
  - hard/soft thresholds on predicted `log2fc_33_pred`, mean log2fc, and
    predicted slope (`log2fc_33_pred - log2fc_8p25_pred`)
  - positive top500 deltas only
  - potent46 soft gate intersections
- Candidate formula:

```text
candidate = id55 + gamma * gate(log2fc_pred, optional_delta_sign, optional_potent46) * (top500 - current_raw)
```

Outputs:

- `track1_activity/analysis/compound_level_lb/outputs/log2fc_gated_top500/log2fc_gated_top500_summary.csv`
- `track1_activity/analysis/compound_level_lb/outputs/log2fc_gated_top500/log2fc_gated_top500_safeish.csv`
- preflight reports under
  `track1_activity/analysis/submission_preflight/outputs/id55_log2fc_*`

## Result

This is more promising than global OOF/proxy reweighting. Several log2fc-gated
variants keep id55 movement small and pass preflight.

Best safe-ish OOF candidates:

| source | gate | gamma | OOF MAE | delta vs raw | id55 p90 shift | id55 max shift | id56 projection |
|---|---|---:|---:|---:|---:|---:|---:|
| optuna_top500_minus_current_raw | log2fc_33_pred_soft_q60_to_q95 | 0.50 | 0.388976 | -0.002468 | 0.028294 | 0.096722 | 0.039054 |
| optuna_top500_minus_current_raw | lf_mean_hard_q50 | 0.15 | 0.389089 | -0.002355 | 0.027459 | 0.054865 | 0.055916 |
| seed10_top500_minus_current_raw | positive_delta_x_lf_mean_hard_q50 | 0.25 | 0.389565 | -0.001879 | 0.031223 | 0.097980 | -0.029475 |

Generated ignored candidate CSVs for preflight:

- `track1_activity/submissions/ens_id55_log2fc_gate_optuna_log2fc33_soft_q60_g50.csv`
- `track1_activity/submissions/ens_id55_log2fc_gate_seed10_pos_lfmean_hard_q50_g25.csv`

Preflight:

| candidate | verdict | mean abs shift | p90 shift | max shift | id56 projection |
|---|---|---:|---:|---:|---:|
| optuna log2fc33 soft q60 gamma 0.50 | PASS | 0.008220 | 0.028294 | 0.096722 | 0.039054 |
| seed10 positive lfmean hard q50 gamma 0.25 | PASS | 0.007805 | 0.031223 | 0.097980 | -0.029475 |

## Interpretation

The log2fc line is plausible. Unlike global reweighting, these candidates do
not strongly project onto id56. The seed10 variant is directionally anti-id56,
but it overlaps more with the id57-id55 axis, which already had a tiny LB
regression. The optuna soft-log2fc variant has better local OOF support and only
weak id56/id57 alignment, so it is the cleaner new hypothesis if spending a
cooldown.

Recommendation: keep both as candidates, with a preference for the optuna
`log2fc_33_pred_soft_q60_to_q95` gate if we decide to submit one. It is still a
small local move from id55, not a new broad ensemble family.
