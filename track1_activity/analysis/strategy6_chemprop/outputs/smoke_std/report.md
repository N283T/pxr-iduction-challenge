# Buterez Strategy 6 ChemProp Report

Run name: `smoke_std`
Experiment: `chemprop_strategy6_adaptive_readout_smoke_std_umap`
Pretrain checkpoint: `/home/nagaet/pxr-iduction-challenge/track1_activity/checkpoints/chemprop_pretrain/pretrain.pt`

## OOF Metrics

Coverage: `1661 / 4140`
MAE: `0.625330`
RAE: `0.655877`
Spearman: `0.657604`
Residual r vs ens_caruana_bag20: `nan`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   pretrain_val_loss |   frozen_param_tensors |   encoder_hidden_dim |   target_mean |   target_std |   val_MAE |   val_RAE |   val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|--------------------:|-----------------------:|---------------------:|--------------:|-------------:|----------:|----------:|---------:|-----------------:|------------------:|
|      0 |            2 |       0.600864 |     0.867328 |   0.570686 |            0.364729 |                      4 |                  256 |       4.30623 |      1.11576 |  0.600864 |  0.66287  | 0.486058 |         0.655746 |          0.472613 |
|      1 |            2 |       0.649824 |     0.939611 |   0.649014 |            0.364729 |                      4 |                  256 |       4.34846 |      1.10393 |  0.649824 |  0.654126 | 0.41422  |         0.651895 |          0.482151 |

## Decision Gate

Direct-submit gate: MAE <= 0.48 and no Spearman collapse; otherwise only test via Caruana ADD if decorrelated.

## Final Read

Do not submit directly unless this beats the current ChemProp Strategy 3 sibling or earns non-trivial Caruana weight with low residual correlation.
