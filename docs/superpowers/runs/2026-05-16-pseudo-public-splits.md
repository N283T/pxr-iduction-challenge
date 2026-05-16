# Pseudo-Public Split Audit

Date: 2026-05-16

Goal: after ChEMBL external judging proved too sparse/weak as a hard gate,
start the split-redesign direction without using past LB outcomes.

## Setup

Script:

- `track1_activity/analysis/oof_reliability_audit/audit_pseudo_public_splits.py`

Outputs:

- `track1_activity/analysis/oof_reliability_audit/outputs/pseudo_public_splits/pseudo_public_split_summary.csv`
- `track1_activity/analysis/oof_reliability_audit/outputs/pseudo_public_splits/pseudo_public_split_fold_summary.csv`
- `track1_activity/analysis/oof_reliability_audit/outputs/pseudo_public_splits/pseudo_public_split_report.md`

Signals used for split construction and audit:

- train-vs-test adversarial classifier probability
- Morgan nearest-neighbor similarity to the blinded test set
- potent46 nearest-neighbor similarity
- predicted log2fc
- exact-overlap-excluded ChEMBL PXR activation nearest-neighbor coverage
- optionally pEC50 for label-aware diagnostic holdouts

No LB outcomes are used.

## Finding

Standard full-coverage five-fold splits are not enough for public-LB simulation:
because every fold averages over the whole train distribution, fold means are
nearly identical to global train. That is useful for stable OOF, but it does not
stress the public-analog region.

The more useful candidates are single pseudo-public holdouts:

| split | n | pEC50 mean | pEC50 std | test-NN >= 0.25 | adv top20 frac | log2fc top20 frac | ChEMBL NN >= 0.30 |
|---|---:|---:|---:|---:|---:|---:|---:|
| public_adv_top513 | 513 | 4.8569 | 0.9611 | 0.9571 | 1.0000 | 0.2885 | 26 |
| public_hybrid_nolabel_top513 | 513 | 5.0868 | 0.6941 | 1.0000 | 0.8187 | 0.4250 | 31 |
| public_hybrid_with_y_top513 | 513 | 5.3241 | 0.4795 | 1.0000 | 0.7739 | 0.4951 | 29 |
| public_testnn_top513 | 513 | 4.7176 | 1.0708 | 1.0000 | 0.5712 | 0.2788 | 29 |
| public_log2fc_top513 | 513 | 5.1860 | 0.4651 | 0.7349 | 0.2807 | 1.0000 | 23 |

## Interpretation

Use pseudo-public holdouts as candidate validation stress tests rather than
replacement OOF metrics:

- `public_adv_top513`: best no-label proxy for train/test distribution shift.
- `public_hybrid_nolabel_top513`: stronger active/log2fc/test-neighborhood
  stress split without using pEC50 labels.
- `public_hybrid_with_y_top513`: label-aware diagnostic upper-stress split,
  useful for debugging high-activity behavior but probably too biased for model
  selection alone.
- `public_log2fc_top513`: harsh high-log2fc/high-pEC50 region stress test.

Next implementation step: retrain or re-evaluate small candidate families across
these pseudo-public holdouts and require agreement with canonical UMAP before
trusting sub-0.004 OOF gains.
