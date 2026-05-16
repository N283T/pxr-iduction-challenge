# OOF Proxy Diagnostics

Date: 2026-05-16

Goal: test whether the current ensemble pool can be reweighted using stronger
OOF/proxy signals without repeating the id56 LB-negative direction.

## Setup

- Pool: latest DB `ens_caruana_bag20` weights, 9 members.
- Raw current Caruana OOF MAE: `0.391444`.
- Trusted LB anchor: id55 `ens_id51_top500_potent46_t40_soft_g35`
  (`0.407080` public MAE).
- Checked validation proxies:
  - canonical UMAP split
  - mixed analog/diversity splits at potent46 NN thresholds 0.20, 0.25, 0.30
  - direct train-to-test NN splits at thresholds 0.25, 0.30
- Checked row weights:
  - uniform
  - adversarial importance weights
  - potent46-soft test-likeness weights
  - importance x potent46-soft weights
  - low-disagreement weights
- Optimizer: simplex MAE reweighting, anchored to current Caruana weights with
  L2 penalties `0.1`, `0.3`, `1.0`, `3.0`.

Outputs:

- `track1_activity/analysis/oof_proxy_diagnostics/oof_proxy_diagnostics_summary.csv`
- `track1_activity/analysis/oof_proxy_diagnostics/oof_proxy_diagnostics_weights.csv`
- `track1_activity/analysis/oof_proxy_diagnostics/lb_submission_direction_table.csv`
- `track1_activity/analysis/oof_proxy_diagnostics/simplex_bootstrap_stability.csv`

## Main Result

The stronger OOF/proxy improvements are still the same risky direction. The
best OOF settings concentrate weight into the id56-regressed top500 axis and
move substantially away from id55.

| setting | OOF MAE | delta vs raw | id55 p90 shift | projection on id56-id55 | max weight |
|---|---:|---:|---:|---:|---:|
| mixed_analog_t20 + importance_x_potent46, L2=0.1 | 0.383930 | -0.007514 | 0.214716 | 1.427299 | 0.540626 |
| UMAP + importance_x_potent46, L2=0.1 | 0.383942 | -0.007502 | 0.214716 | 1.427299 | 0.540626 |
| mixed_analog_t20 + importance, L2=0.1 | 0.384949 | -0.006495 | 0.197366 | 1.310500 | 0.483978 |
| UMAP + uniform, L2=0.1 | 0.386345 | -0.005099 | 0.182756 | 1.216583 | 0.431911 |

The least risky OOF-positive variants are heavily regularized. They retain only
tiny OOF gains and are still basically aligned with the known-bad direction:

| setting family | L2 | mean OOF delta | mean id55 p90 shift | mean projection on id56-id55 |
|---|---:|---:|---:|---:|
| low_disagreement | 3.0 | -0.000282 | 0.164955 | 0.891683 |
| uniform | 3.0 | -0.000379 | 0.165132 | 0.897547 |
| importance | 3.0 | -0.000545 | 0.165257 | 0.905850 |
| importance_x_potent46 | 1.0 | -0.001937 | 0.168869 | 0.983941 |

Important caveat: current raw Caruana itself is already far from id55
(`mean_abs=0.0919`, `p90=0.1647`, id56-id55 projection `0.8754`). So a high
id55 shift in these diagnostics partly reflects that the stored Caruana pool has
moved into the id56-like region before any new reweighting.

## Bootstrap Stability

Bootstrap reweighting is numerically stable after anchoring, but stability does
not solve directionality. The stable solutions are stable in the same axis.

| setting | weight L1 mean | weight L1 std | raw-anchor test p95 shift mean | top member weight p90 |
|---|---:|---:|---:|---:|
| uniform, L2=0.3 | 0.143153 | 0.017973 | 0.038926 | 0.370177 |
| importance, L2=0.3 | 0.199234 | 0.016876 | 0.051542 | 0.394959 |
| potent46_soft, L2=0.3 | 0.167581 | 0.018665 | 0.045592 | 0.379643 |
| importance_x_potent46, L2=0.3 | 0.248560 | 0.022599 | 0.063391 | 0.422076 |

## LB-History Proxy

Across existing submitted files, distance from id55 is a much stronger warning
signal than projection on id56 alone:

- `id55_abs_delta_p90` vs LB delta from id55: Pearson `0.879`, Spearman `0.922`.
- `id55_abs_delta_mean` vs LB delta from id55: Pearson `0.891`, Spearman `0.922`.
- `id56_minus_id55_projection` vs LB delta from id55: Pearson `0.130`,
  Spearman `0.217`.

Recent examples:

| id | submission | LB MAE | delta vs id55 | id55 p90 shift | id56 projection |
|---:|---|---:|---:|---:|---:|
| 53 | repooled_trunk_core_only | 0.410564 | +0.003484 | 0.092054 | 0.529309 |
| 54 | id51_plus_potent_noaux | 0.409582 | +0.002502 | 0.047573 | 0.166587 |
| 56 | swap_optuna_t10_top500 | 0.413460 | +0.006380 | 0.111276 | 1.000000 |
| 57 | id51_top500_potent46_g50 | 0.407389 | +0.000309 | 0.016707 | -0.070314 |

## Decision

Do not submit any of these OOF/proxy reweighting candidates. The useful outcome
is negative: changing the OOF lens or row weighting does not uncover a new safe
ensemble direction in the current material pool. It mostly makes the optimizer
more confident about the already LB-negative top500-heavy axis.

Near-term recommendation: keep id55/id57-style tiny, localized movement as the
only plausible submission family until phase 2 labels change the signal. For
offline work, prioritize diagnostics that explain why id55's small local gate
survived while broader OOF-optimal motion failed.
