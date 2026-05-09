# Learned GSL-MPP Report

Run name: `initial_small`
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
| g0p2_c0p02  |     0.2 |  0.02  |  0.390966 |    -0.000477866 |       0.84941  |          0.00037952  |        0.304733 |         0.307375 |           0.431852 |        0.0516934 |        0.0121505  |             0.0174438 |                0.02  |                0.02  |             0 |             0 |             0 |
| g0p2_c0p015 |     0.2 |  0.015 |  0.390989 |    -0.000454739 |       0.849405 |          0.000374381 |        0.304733 |         0.307375 |           0.431852 |        0.0516934 |        0.00927365 |             0.0135513 |                0.015 |                0.015 |             0 |             0 |             0 |
| g0p2_c0p03  |     0.2 |  0.03  |  0.391003 |    -0.000441198 |       0.849229 |          0.000198486 |        0.304733 |         0.307375 |           0.431852 |        0.0516934 |        0.0175302  |             0.0244043 |                0.03  |                0.03  |             0 |             0 |             0 |

## Fold losses

|   fold |   n_train |   n_val |   loss_start |   loss_end |   val_resid_std |
|-------:|----------:|--------:|-------------:|-----------:|----------------:|
|      0 |      3309 |     831 |     0.410974 |  0.0479427 |        0.351025 |
|      1 |      3310 |     830 |     0.419752 |  0.050556  |        0.302694 |
|      2 |      3311 |     829 |     0.417275 |  0.0465488 |        0.323233 |
|      3 |      3322 |     818 |     0.423697 |  0.0531185 |        0.271199 |
|      4 |      3308 |     832 |     0.414342 |  0.0465949 |        0.253996 |

## Preflight

- g0p2_c0p02: OK: see /home/nagaet/pxr-iduction-challenge/track1_activity/analysis/gsl_mpp_learned/outputs/initial_small/preflight_g0p2_c0p02.log
- g0p2_c0p015: OK: see /home/nagaet/pxr-iduction-challenge/track1_activity/analysis/gsl_mpp_learned/outputs/initial_small/preflight_g0p2_c0p015.log
- g0p2_c0p03: OK: see /home/nagaet/pxr-iduction-challenge/track1_activity/analysis/gsl_mpp_learned/outputs/initial_small/preflight_g0p2_c0p03.log

## Read

This is a learned molecule-graph residual model. Treat OOF gain, test shift, and preflight together before any cooldown spend.
## Decision

Do not submit from this learned GSL-MPP run as-is. The signal is real but below the project submission gate: best safe candidates are around `-0.0005` to `-0.0007` OOF MAE with small Spearman gains, not the `-0.0015` MAE / `+0.0010` Spearman threshold. Treat as a framework plus weak-positive diagnostic.
