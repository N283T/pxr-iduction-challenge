# Learned GSL-MPP Report

Run name: `initial`
Source OOF ensemble: `ens_caruana_bag20` id `2420`
Anchor test CSV: `/home/nagaet/pxr-iduction-challenge/track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## Hyperparameters

- svd_components: `256`
- init_k / learned_k: `32` / `32`
- epochs: `600`
- hidden_dim: `128`
- graph_skip: `0.8`

## Anchor OOF

MAE: `0.391444`
Spearman: `0.849031`

## Best candidates by OOF MAE delta

| label       |   gamma |   clip |   oof_mae |   oof_delta_mae |   oof_spearman |   oof_delta_spearman |   oof_resid_std |   test_resid_std |   final_loss_start |   final_loss_end |   test_mean_shift |   test_mean_abs_shift |   test_p90_abs_shift |   test_max_abs_shift |   test_gt_005 |   test_gt_010 |   test_gt_020 |
|:------------|--------:|-------:|----------:|----------------:|---------------:|---------------------:|----------------:|-----------------:|-------------------:|-----------------:|------------------:|----------------------:|---------------------:|---------------------:|--------------:|--------------:|--------------:|
| g0p5_c0p03  |    0.5  |   0.03 |  0.3908   |    -0.00064396  |       0.849428 |          0.000397269 |        0.304733 |         0.307375 |           0.431852 |        0.0516934 |         0.0187366 |             0.0277668 |                 0.03 |                 0.03 |             0 |             0 |             0 |
| g0p25_c0p03 |    0.25 |   0.03 |  0.390948 |    -0.00049623  |       0.849284 |          0.000253416 |        0.304733 |         0.307375 |           0.431852 |        0.0516934 |         0.0179308 |             0.0254297 |                 0.03 |                 0.03 |             0 |             0 |             0 |
| g0p5_c0p06  |    0.5  |   0.06 |  0.391601 |     0.000157494 |       0.848232 |         -0.000798892 |        0.304733 |         0.307375 |           0.431852 |        0.0516934 |         0.0358617 |             0.0508594 |                 0.06 |                 0.06 |           385 |             0 |             0 |

## Fold losses

|   fold |   n_train |   n_val |   loss_start |   loss_end |   val_resid_std |
|-------:|----------:|--------:|-------------:|-----------:|----------------:|
|      0 |      3309 |     831 |     0.410974 |  0.0479427 |        0.351025 |
|      1 |      3310 |     830 |     0.419752 |  0.050556  |        0.302694 |
|      2 |      3311 |     829 |     0.417275 |  0.0465488 |        0.323233 |
|      3 |      3322 |     818 |     0.423697 |  0.0531185 |        0.271199 |
|      4 |      3308 |     832 |     0.414342 |  0.0465949 |        0.253996 |

## Preflight

- g0p5_c0p03: OK: see /home/nagaet/pxr-iduction-challenge/track1_activity/analysis/gsl_mpp_learned/outputs/initial/preflight_g0p5_c0p03.log
- g0p25_c0p03: OK: see /home/nagaet/pxr-iduction-challenge/track1_activity/analysis/gsl_mpp_learned/outputs/initial/preflight_g0p25_c0p03.log
- g0p5_c0p06: OK: see /home/nagaet/pxr-iduction-challenge/track1_activity/analysis/gsl_mpp_learned/outputs/initial/preflight_g0p5_c0p06.log

## Read

This is a learned molecule-graph residual model. Treat OOF gain, test shift, and preflight together before any cooldown spend.
## Decision

Do not submit from this learned GSL-MPP run as-is. The signal is real but below the project submission gate: best safe candidates are around `-0.0005` to `-0.0007` OOF MAE with small Spearman gains, not the `-0.0015` MAE / `+0.0010` Spearman threshold. Treat as a framework plus weak-positive diagnostic.
