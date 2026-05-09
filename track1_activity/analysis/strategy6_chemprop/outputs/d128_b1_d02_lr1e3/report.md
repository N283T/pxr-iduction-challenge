# Buterez Strategy 6 ChemProp Report

Run name: `d128_b1_d02_lr1e3`
Experiment: `chemprop_strategy6_adaptive_readout_d128_b1_d02_lr1e3_umap`
Pretrain checkpoint: `/home/nagaet/pxr-iduction-challenge/track1_activity/checkpoints/chemprop_pretrain/pretrain.pt`

## OOF Metrics

Coverage: `4140 / 4140`
MAE: `0.487709`
RAE: `0.535985`
Spearman: `0.756535`
Residual r vs ens_caruana_bag20: `0.858011`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   pretrain_val_loss |   frozen_param_tensors |   encoder_hidden_dim |   target_mean |   target_std |   val_MAE |   val_RAE |   val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|--------------------:|-----------------------:|---------------------:|--------------:|-------------:|----------:|----------:|---------:|-----------------:|------------------:|
|      0 |           54 |       0.526447 |      1.00626 |   0.22325  |            0.364729 |                      4 |                  256 |       4.30623 |      1.11576 |  0.526447 |  0.580773 | 0.590633 |         0.714735 |          0.524625 |
|      1 |           51 |       0.465165 |      1.14663 |   0.241329 |            0.364729 |                      4 |                  256 |       4.34846 |      1.10393 |  0.465165 |  0.468245 | 0.7059   |         0.797093 |          0.600414 |
|      2 |           60 |       0.49618  |      1.17123 |   0.222364 |            0.364729 |                      4 |                  256 |       4.3286  |      1.14126 |  0.49618  |  0.584213 | 0.575555 |         0.748986 |          0.557853 |
|      3 |           66 |       0.450796 |      1.07009 |   0.197598 |            0.364729 |                      4 |                  256 |       4.28664 |      1.14544 |  0.450796 |  0.578213 | 0.587458 |         0.723809 |          0.543161 |
|      4 |           31 |       0.499358 |      1.19892 |   0.337057 |            0.364729 |                      4 |                  256 |       4.33425 |      1.09895 |  0.499358 |  0.497184 | 0.667169 |         0.782622 |          0.583781 |

## Decision Gate

Direct-submit gate: MAE <= 0.48 and no Spearman collapse; otherwise only test via Caruana ADD if decorrelated.

## Final Read

Do not submit directly unless this beats the current ChemProp Strategy 3 sibling or earns non-trivial Caruana weight with low residual correlation.
