# Buterez Strategy 6 ChemProp Report

Run name: `smoke`
Experiment: `chemprop_strategy6_adaptive_readout_smoke_umap`
Pretrain checkpoint: `/home/nagaet/pxr-iduction-challenge/track1_activity/checkpoints/chemprop_pretrain/pretrain.pt`

## OOF Metrics

Coverage: `1661 / 4140`
MAE: `2.401174`
RAE: `2.518474`
Spearman: `0.179190`
Residual r vs ens_caruana_bag20: `nan`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   pretrain_val_loss |   frozen_param_tensors |   encoder_hidden_dim |   val_MAE |   val_RAE |   val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|--------------------:|-----------------------:|---------------------:|----------:|----------:|---------:|-----------------:|------------------:|
|      0 |            2 |        2.28799 |      13.6376 |    7.51046 |            0.364729 |                      4 |                  256 |   2.28799 |   2.5241  | -3.86501 |       0.00850728 |        0.00577402 |
|      1 |            2 |        2.5145  |      15.3185 |    9.88591 |            0.364729 |                      4 |                  256 |   2.5145  |   2.53114 | -4.51634 |       0.478832   |        0.327387   |

## Decision Gate

Direct-submit gate: MAE <= 0.48 and no Spearman collapse; otherwise only test via Caruana ADD if decorrelated.

## Final Read

Do not submit directly unless this beats the current ChemProp Strategy 3 sibling or earns non-trivial Caruana weight with low residual correlation.
