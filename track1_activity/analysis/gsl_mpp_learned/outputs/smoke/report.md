# Learned GSL-MPP Report

Run name: `smoke`
Source OOF ensemble: `ens_caruana_bag20` id `2420`
Anchor test CSV: `/home/nagaet/pxr-iduction-challenge/track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## Hyperparameters

- svd_components: `32`
- init_k / learned_k: `32` / `32`
- epochs: `5`
- hidden_dim: `32`
- graph_skip: `0.8`

## Anchor OOF

MAE: `0.391444`
Spearman: `0.849031`

## Best candidates by OOF MAE delta

| label       |   gamma |   clip |   oof_mae |   oof_delta_mae |   oof_spearman |   oof_delta_spearman |   oof_resid_std |   test_resid_std |   final_loss_start |   final_loss_end |   test_mean_shift |   test_mean_abs_shift |   test_p90_abs_shift |   test_max_abs_shift |   test_gt_005 |   test_gt_010 |   test_gt_020 |
|:------------|--------:|-------:|----------:|----------------:|---------------:|---------------------:|----------------:|-----------------:|-------------------:|-----------------:|------------------:|----------------------:|---------------------:|---------------------:|--------------:|--------------:|--------------:|
| g0p25_c0p06 |    0.25 |   0.06 |  0.390744 |    -0.000700469 |       0.849503 |          0.000471987 |        0.115909 |         0.086419 |           0.421494 |         0.397646 |       -0.00272647 |              0.017105 |            0.0359998 |                 0.06 |            16 |             0 |             0 |

## Fold losses

|   fold |   n_train |   n_val |   loss_start |   loss_end |   val_resid_std |
|-------:|----------:|--------:|-------------:|-----------:|----------------:|
|      0 |      3309 |     831 |     0.470585 |   0.406764 |       0.108625  |
|      1 |      3310 |     830 |     0.484633 |   0.413295 |       0.117953  |
|      2 |      3311 |     829 |     0.416726 |   0.394467 |       0.111519  |
|      3 |      3322 |     818 |     0.449594 |   0.410311 |       0.0969311 |
|      4 |      3308 |     832 |     0.461342 |   0.402748 |       0.0849737 |

## Preflight

- g0p25_c0p06: SKIP

## Read

This is a learned molecule-graph residual model. Treat OOF gain, test shift, and preflight together before any cooldown spend.
## Decision

Do not submit from this learned GSL-MPP run as-is. The signal is real but below the project submission gate: best safe candidates are around `-0.0005` to `-0.0007` OOF MAE with small Spearman gains, not the `-0.0015` MAE / `+0.0010` Spearman threshold. Treat as a framework plus weak-positive diagnostic.
