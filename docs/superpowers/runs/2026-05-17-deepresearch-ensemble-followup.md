# DeepResearch Ensemble Follow-up

Date: 2026-05-17

Goal: lightly test the DeepResearch recommendation to prioritize stable
ensemble-weight estimation over new model families:

- anchor-simplex LAD blending
- bootstrap-bagged anchor blends
- small affine LAD calibration

This was run as a diagnostic only. No new Track 1 submission was selected.

## Inputs

- DeepResearch report: local download
  `C:/Users/kitak/Downloads/deep-research-report (6).md`
- Existing implementation used for the first two recommendations:
  `track1_activity/scripts/run_oof_proxy_diagnostics.py`
- Additional quick affine LAD probe output:
  `track1_activity/analysis/oof_proxy_diagnostics/dr_affine_probe.csv`
- Updated public LB snapshots:
  - `docs/leaderboards/activity/leaderboard_2026-05-16_2123JST.csv`
  - `docs/leaderboards/activity/leaderboard_2026-05-16_2141JST.csv`

## Anchor-simplex LAD Result

The existing OOF proxy diagnostic already implements the core recommendation:
simplex-constrained nonnegative MAE optimization with L2 shrinkage toward the
current Caruana weights, plus bootstrap stability checks.

The large OOF winners still move into the previously LB-negative direction:

| setting | OOF MAE | delta vs anchor | id55 p90 shift | id56-id55 projection |
|---|---:|---:|---:|---:|
| mixed_analog_t20 + importance_x_potent46, L2=0.1 | 0.383930 | -0.007514 | 0.214716 | 1.427299 |
| UMAP + importance_x_potent46, L2=0.1 | 0.383942 | -0.007502 | 0.214716 | 1.427299 |
| mixed_analog_t20 + importance, L2=0.1 | 0.384949 | -0.006495 | 0.197366 | 1.310500 |

The safer heavily anchored variants reduce movement, but their OOF gain is
small enough that the recent id58/id59 LB failures make them poor Phase 1
submission candidates:

| setting | OOF MAE | delta vs anchor | raw-anchor p95 shift | id55 p90 shift | id56-id55 projection |
|---|---:|---:|---:|---:|---:|
| low_disagreement, L2=1.0 | 0.390614 | -0.000830 | 0.010652 | 0.166220 | 0.920160 |
| importance_x_potent46, L2=3.0 | 0.390707 | -0.000737 | 0.009316 | 0.165547 | 0.914618 |
| importance, L2=3.0 | 0.390877 | -0.000567 | 0.006940 | 0.165257 | 0.905850 |
| uniform, L2=3.0 | 0.391039 | -0.000405 | 0.005273 | 0.165132 | 0.897547 |

## Bootstrap Stability

The bootstrap-bagged weight fits are numerically stable when strongly anchored.
That is useful, but it does not change the directional concern.

| setting | weight L1 mean | weight L1 std | test p95 shift mean | top member weight p90 |
|---|---:|---:|---:|---:|
| uniform, L2=3.0 | 0.018592 | 0.002315 | 0.005271 | 0.316712 |
| potent46_soft, L2=3.0 | 0.022188 | 0.002575 | 0.006321 | 0.318150 |
| importance, L2=3.0 | 0.026034 | 0.002333 | 0.007043 | 0.319777 |
| importance_x_potent46, L2=3.0 | 0.034078 | 0.002981 | 0.009323 | 0.323245 |

## Affine LAD Calibration

Affine LAD calibration can produce another local OOF gain, but it moves the test
predictions much more than the conservative anchor-simplex variants:

| setting | OOF MAE | delta vs base | slope | raw-anchor p95 shift | id56-id55 projection |
|---|---:|---:|---:|---:|---:|
| low_disagreement L2=1 + affine lambda=0.1 | 0.389356 | -0.001259 | 1.048049 | 0.081779 | 0.984138 |
| importance_x_potent46 L2=3 + affine lambda=0.1 | 0.389407 | -0.001300 | 1.047471 | 0.081283 | 0.977423 |
| anchor + affine lambda=1.0 | 0.390342 | -0.001102 | 1.022510 | 0.050546 | 0.896654 |

This is a plausible Phase 2 calibration tool once Analog Set 1 labels are
released, especially given the high public R2 of id59, but it is too much
distribution movement for another Phase 1 submission.

## Updated LB Context

After the id59 fetch, the recent public LB sequence is:

| id | submission | rank | MAE | RAE | R2 | Spearman |
|---:|---|---:|---:|---:|---:|---:|
| 59 | `ens_id57_high_activity_lift_rank2` | 5 | 0.407730 | 0.512323 | 0.678512 | 0.844763 |
| 58 | `ens_id55_combo_gate_rank1` | 5 | 0.407520 | 0.512036 | 0.677719 | 0.844657 |
| 57 | `ens_id51_top500_potent46_t40_soft_g50` | 4 | 0.407389 | 0.511885 | 0.678707 | 0.845314 |
| 55 | `ens_id51_top500_potent46_t40_soft_g35` | 3 | 0.407080 | 0.511480 | 0.678253 | 0.845494 |

N283T/id59 was public-rank 1 by R2 among the visible rows at the 2026-05-16
21:41 JST fetch, while ranking 5th by the primary absolute-error metrics. This
supports the current interpretation: the broad variance/ranking structure may
generalize, but Analog Set 1 absolute calibration is not aligned well enough for
MAE/RAE.

## Decision

Do not submit a DeepResearch-inspired blend before Phase 2.

The useful result is negative but actionable:

- DR's recommended family is already largely covered by the existing
  `run_oof_proxy_diagnostics.py` workflow.
- The strongest local improvements still follow dangerous movement patterns.
- The safest variants are too small relative to the observed public-LB noise.
- Affine LAD calibration should be kept for Phase 2, not used blindly in
  Phase 1.

Recommended Phase 2 use:

1. Keep id55/id57/id58/id59 deltas as diagnostic anchors.
2. Use released Analog Set 1 labels to fit very small affine or monotone
   calibration layers.
3. Preserve the high-R2 broad structure unless the unblinded labels show that it
   is public-subset overfit.
