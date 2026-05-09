# Learned GSL-MPP Report

Run name: `epoch100`
Source OOF ensemble: `ens_caruana_bag20` id `2420`
Anchor test CSV: `/home/nagaet/pxr-iduction-challenge/track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv`

## Hyperparameters

- svd_components: `64`
- init_k / learned_k: `32` / `32`
- epochs: `100`
- hidden_dim: `64`
- graph_skip: `0.8`

## Anchor OOF

MAE: `0.391444`
Spearman: `0.849031`

## Best candidates by OOF MAE delta

| label      |   gamma |   clip |   oof_mae |   oof_delta_mae |   oof_spearman |   oof_delta_spearman |   oof_resid_std |   test_resid_std |   final_loss_start |   final_loss_end |   test_mean_shift |   test_mean_abs_shift |   test_p90_abs_shift |   test_max_abs_shift |   test_gt_005 |   test_gt_010 |   test_gt_020 |
|:-----------|--------:|-------:|----------:|----------------:|---------------:|---------------------:|----------------:|-----------------:|-------------------:|-----------------:|------------------:|----------------------:|---------------------:|---------------------:|--------------:|--------------:|--------------:|
| g0p1_c0p02 |     0.1 |   0.02 |  0.391692 |     0.000247886 |       0.848486 |         -0.000544433 |         0.31363 |         0.295654 |           0.425284 |         0.257662 |        0.00853904 |             0.0151499 |                 0.02 |                 0.02 |             0 |             0 |             0 |
| g0p2_c0p02 |     0.2 |   0.02 |  0.391792 |     0.000347614 |       0.848358 |         -0.000673209 |         0.31363 |         0.295654 |           0.425284 |         0.257662 |        0.00920196 |             0.0174143 |                 0.02 |                 0.02 |             0 |             0 |             0 |

## Fold losses

|   fold |   n_train |   n_val |   loss_start |   loss_end |   val_resid_std |
|-------:|----------:|--------:|-------------:|-----------:|----------------:|
|      0 |      3309 |     831 |     0.403942 |   0.242843 |        0.281673 |
|      1 |      3310 |     830 |     0.42678  |   0.249278 |        0.311992 |
|      2 |      3311 |     829 |     0.413888 |   0.245759 |        0.357733 |
|      3 |      3322 |     818 |     0.433272 |   0.254535 |        0.30238  |
|      4 |      3308 |     832 |     0.416258 |   0.239312 |        0.300606 |

## Preflight

- g0p1_c0p02: SKIP
- g0p2_c0p02: SKIP

## Read

This is a learned molecule-graph residual model. Treat OOF gain, test shift, and preflight together before any cooldown spend.
## Decision

Do not submit from this learned GSL-MPP run as-is. The signal is real but below the project submission gate: best safe candidates are around `-0.0005` to `-0.0007` OOF MAE with small Spearman gains, not the `-0.0015` MAE / `+0.0010` Spearman threshold. Treat as a framework plus weak-positive diagnostic.
