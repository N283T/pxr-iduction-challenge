# Combined-Gated Top500 Probe

Date: 2026-05-16

Goal: test whether the promising log2fc, `num_rings`, and `family_gap` gates can
be combined to improve local OOF without making the id55 anchor move too much.

## Setup

- Anchor for candidate movement: id55
  `ens_id51_top500_potent46_t40_soft_g35`.
- Delta sources:
  - `seed10_top500_minus_current_raw`
  - `optuna_top500_minus_current_raw`
- Combined gates:
  - log2fc soft gates: `log2fc_33_pred` q50/q60 and `lf_mean` q50
  - SHAP-inspired gates: `num_rings_high_soft_q20`,
    `family_gap_high_soft_q20`
  - combinations: product/AND, average/MEAN, and union-like MAX
  - optional positive top500-delta mask for descriptor gates
- Candidate formula:

```text
candidate = id55 + gamma * combined_gate * (top500 - current_raw)
```

Outputs:

- `track1_activity/analysis/compound_level_lb/outputs/combined_gated_top500/combined_gated_top500_summary.csv`
- `track1_activity/analysis/compound_level_lb/outputs/combined_gated_top500/combined_gated_top500_safeish.csv`
- `track1_activity/analysis/compound_level_lb/outputs/combined_gated_top500/combined_gated_top500_candidates.csv`
- preflight reports under
  `track1_activity/analysis/submission_preflight/outputs/id55_combo_rank*_vs_id55/`

## Result

The broadest MAX gates can drive stronger OOF gains, but they move id55 too much
for a first submission candidate. The best safe-ish combination slightly beats
the standalone `num_rings` gate while reducing the maximum blind-test shift.

Best safe-ish candidates:

| rank | source | gate | gamma | OOF MAE | delta vs raw | id55 p90 shift | id55 max shift | id56 projection | preflight |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | optuna_top500_minus_current_raw | log2fc_33_pred_soft_q60_to_q95_MAX_positive_delta_AND_num_rings_high_soft_q20 | 0.35 | 0.387422 | -0.004022 | 0.032508 | 0.077779 | 0.061939 | PASS |
| 2 | optuna_top500_minus_current_raw | positive_delta_AND_num_rings_high_soft_q20 | 0.50 | 0.387464 | -0.003980 | 0.034066 | 0.103700 | 0.061398 | PASS |
| 3 | optuna_top500_minus_current_raw | log2fc_33_pred_soft_q50_to_q95_MAX_family_gap_high_soft_q20 | 0.25 | 0.387564 | -0.003880 | 0.034483 | 0.086024 | 0.083228 | PASS |

Generated ignored candidate CSVs:

- `track1_activity/submissions/ens_id55_combo_gate_rank1.csv`
- `track1_activity/submissions/ens_id55_combo_gate_rank2.csv`
- `track1_activity/submissions/ens_id55_combo_gate_rank3.csv`

Preflight details:

| candidate | verdict | mean abs shift | p90 shift | max shift | `|shift| > 0.05` | `|shift| > 0.10` |
|---|---|---:|---:|---:|---:|---:|
| rank1 | PASS | 0.010797 | 0.032508 | 0.077779 | 18 | 0 |
| rank2 | PASS | 0.009419 | 0.034066 | 0.103700 | 20 | 1 |
| rank3 | PASS | 0.012900 | 0.034483 | 0.086024 | 21 | 0 |

## Interpretation

Combination helps, but mostly through union-like MAX gates. Pure intersections
are safer but lose too much of the useful movement. Rank1 is the best combined
candidate: it keeps the log2fc signal available, adds the positive `num_rings`
region, improves OOF slightly over standalone `num_rings`, and removes the
single >0.10 pEC50 blind-test shift.

Current submission ordering:

1. `ens_id55_combo_gate_rank1.csv` if we want the best local evidence from this
   round.
2. `ens_id55_log2fc_gate_optuna_log2fc33_soft_q60_g50.csv` if we want the
   cleanest single-axis hypothesis.
3. `ens_id55_shap_numrings_pos_highsoftq20_g50.csv` only as a bolder baseline
   for the combined result.
