# GSL-MPP-Lite Report

Run name: `initial`
Source OOF ensemble: `ens_caruana_bag20` id `2420`
Anchor test CSV: `/home/nagaet/pxr-iduction-challenge/track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## Anchor OOF

MAE: `0.391444`
Spearman: `0.849031`

## Best candidates by OOF MAE delta

| label                |   k |   alpha |   gamma |   clip |   oof_mae |   oof_delta_mae |   oof_spearman |   oof_delta_spearman |   oof_resid_std |   test_resid_std |   test_mean_shift |   test_mean_abs_shift |   test_p90_abs_shift |   test_max_abs_shift |   test_gt_005 |   test_gt_010 |   test_gt_020 |
|:---------------------|----:|--------:|--------:|-------:|----------:|----------------:|---------------:|---------------------:|----------------:|-----------------:|------------------:|----------------------:|---------------------:|---------------------:|--------------:|--------------:|--------------:|
| k8_a0p5_g0p25_c0p06  |   8 |     0.5 |    0.25 |   0.06 |  0.391225 |    -0.000219415 |       0.848744 |         -0.000286742 |       0.0634331 |        0.053215  |        0.0136427  |             0.0148151 |            0.0285249 |                 0.06 |            11 |             0 |             0 |
| k16_a0p5_g0p25_c0p06 |  16 |     0.5 |    0.25 |   0.06 |  0.391306 |    -0.000137641 |       0.848947 |         -8.40647e-05 |       0.0481203 |        0.0445969 |        0.00934133 |             0.0118788 |            0.0234226 |                 0.06 |             1 |             0 |             0 |
| k8_a0p5_g0p25_c0p03  |   8 |     0.5 |    0.25 |   0.03 |  0.391311 |    -0.000132506 |       0.848767 |         -0.000263683 |       0.0634331 |        0.053215  |        0.012544   |             0.0137106 |            0.0285249 |                 0.03 |             0 |             0 |             0 |

## Preflight

- k8_a0p5_g0p25_c0p06: OK: see /home/nagaet/pxr-iduction-challenge/track1_activity/analysis/gsl_mpp_lite/outputs/initial/preflight_k8_a0p5_g0p25_c0p06.log
- k16_a0p5_g0p25_c0p06: OK: see /home/nagaet/pxr-iduction-challenge/track1_activity/analysis/gsl_mpp_lite/outputs/initial/preflight_k16_a0p5_g0p25_c0p06.log
- k8_a0p5_g0p25_c0p03: OK: see /home/nagaet/pxr-iduction-challenge/track1_activity/analysis/gsl_mpp_lite/outputs/initial/preflight_k8_a0p5_g0p25_c0p03.log

## Read

Treat this as a transductive molecule-graph diagnostic, not an automatic submission decision.
The OOF anchor is reconstructed from `ens_caruana_bag20`; if the test anchor is id55,
OOF and test anchors are intentionally not identical because id55 was a CSV-only perturbation.
## Decision

Do not submit any initial GSL-MPP-lite candidate. The best OOF MAE delta is only `-0.000219`, below the `-0.0015` gate, and it loses Spearman by `-0.000287`. Preflight says the test shifts are small, so the implementation is safe for diagnostics, but the first molecule-graph residual smoothing signal is too weak to spend a cooldown.

Next, if continuing GSL-MPP, use this as the baseline/null and either port the upstream learned graph network or try a stronger node-feature graph learner rather than simple fixed-graph residual propagation.
