# OOF Reliability Audit

Date: 2026-05-16

Goal: after the combined-gated top500 submission regressed slightly on LB
despite improved OOF, test whether alternative OOF scoring definitions are more
LB-aligned than global OOF MAE.

## Setup

Script:

- `track1_activity/analysis/oof_reliability_audit/audit_oof_metric_battery.py`

Outputs:

- `track1_activity/analysis/oof_reliability_audit/outputs/oof_metric_battery/oof_metric_battery_scored_submissions.csv`
- `track1_activity/analysis/oof_reliability_audit/outputs/oof_metric_battery/oof_metric_battery_correlations.csv`
- `track1_activity/analysis/oof_reliability_audit/outputs/oof_metric_battery/oof_metric_battery_diagnostics.csv`
- `track1_activity/analysis/oof_reliability_audit/outputs/oof_metric_battery/oof_metric_battery_skipped.csv`

Metric battery:

- global OOF MAE / RMSE / bias / Spearman
- train-vs-test classifier importance weighting with Morgan LogReg and LGBM
- test-nearest-neighbor slices
- train-vs-test classifier top slices
- high predicted log2fc slices
- high measured pEC50 slices
- potent46-nearest-neighbor slices

The script uses LB-known rows with either direct full OOF predictions in
`experiment_oof_predictions` or reconstructable weighted-blend OOF from
`experiments.hyperparameters.weights`.

## Data Coverage

The usable historical sample is still small:

- matched/reconstructable rows: 22
- deduplicated latest experiments: 7
- full OOF unavailable or partial for several older rows, especially early
  analog-split or overwritten submission paths

The train-vs-test shift is real:

| diagnostic | value |
|---|---:|
| Morgan LogReg train-vs-test AUC | 0.940111 |
| Morgan LGBM train-vs-test AUC | 0.956383 |
| train rows | 4140 |
| potent46 rows | 46 |

## Result

On deduplicated experiments, high-pEC50 slice MAE is the best simple metric, but
the evidence is weak because `n=7`.

| subset | metric | n | Spearman vs LB MAE | Pearson vs LB MAE |
|---|---|---:|---:|---:|
| dedup_latest_experiment | `slice_pec50_top30__mae` | 7 | 0.535714 | 0.712673 |
| dedup_latest_experiment | `slice_pec50_top19__mae` | 7 | 0.535714 | 0.695695 |
| dedup_latest_experiment | `global_mae__mae` | 7 | 0.392857 | 0.679706 |
| dedup_latest_experiment | `weighted_lgbm_testlikeness__mae` | 7 | 0.392857 | 0.678872 |

The all-row table is confounded by repeated submissions with the same OOF but
different LB results, so it is not useful for rank-order claims. It mainly shows
that repeated LB records can dominate correlations.

## Would This Have Stopped id58?

No. Re-scoring the 2026-05-16 candidates by high-pEC50 slices still favors the
submitted combined gate.

| candidate OOF | global MAE | top30 pEC50 MAE | top19 pEC50 MAE | top9 pEC50 MAE |
|---|---:|---:|---:|---:|
| raw current | 0.391444 | 0.358749 | 0.406511 | 0.518599 |
| log2fc q60 gamma 0.50 | 0.388976 | 0.352526 | 0.396731 | 0.501356 |
| num_rings positive gamma 0.50 | 0.387464 | 0.342796 | 0.386036 | 0.485795 |
| combined rank1 gamma 0.35 | 0.387422 | 0.346211 | 0.389638 | 0.490751 |
| optuna top500 | 0.382878 | 0.328417 | 0.356304 | 0.417494 |

Interpretation: high-activity slice scoring might be a useful supplemental
diagnostic, but it does not solve the current OOF/LB mismatch. It would still
have supported the id58 direction that was slightly LB-negative.

## Next Higher-EV OOF Work

The remaining OOF problem is likely not a simple row-weighting problem. Higher
EV directions:

1. Build a public-LB simulator split, not just a metric: hold out train compounds
   selected by test-likeness, high pEC50, log2fc, and analog-neighborhood
   constraints, then retrain/evaluate candidates against that split.
2. Require split stability across several such synthetic Analog Set 1 splits
   before treating small OOF gains as actionable.
3. Reconstruct OOF for recent CSV-only candidates directly into a common
   candidate registry, so future LB feedback can be compared against the exact
   local OOF candidate used for submission.
4. Treat current-pool OOF improvements below about 0.004 as weak unless they
   survive a new split simulator or Phase 2 labels.

Bottom line: simple alternative OOF metrics do not rescue the current setting.
The next serious attempt should change the validation split construction itself.
