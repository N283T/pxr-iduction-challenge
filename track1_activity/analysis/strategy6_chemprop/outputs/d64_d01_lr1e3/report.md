# Buterez Strategy 6 ChemProp Report

Run name: `d64_d01_lr1e3`
Experiment: `chemprop_strategy6_adaptive_readout_d64_d01_lr1e3_umap`
Pretrain checkpoint: `/home/nagaet/pxr-iduction-challenge/track1_activity/checkpoints/chemprop_pretrain/pretrain.pt`

## OOF Metrics

Coverage: `4140 / 4140`
MAE: `0.485845`
RAE: `0.533937`
Spearman: `0.763176`
Residual r vs ens_caruana_bag20: `0.859555`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   pretrain_val_loss |   frozen_param_tensors |   encoder_hidden_dim |   target_mean |   target_std |   val_MAE |   val_RAE |   val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|--------------------:|-----------------------:|---------------------:|--------------:|-------------:|----------:|----------:|---------:|-----------------:|------------------:|
|      0 |           38 |       0.528596 |     0.985816 |   0.294492 |            0.364729 |                      4 |                  256 |       4.30623 |      1.11576 |  0.528596 |  0.583143 | 0.587317 |         0.713967 |          0.525839 |
|      1 |           68 |       0.476236 |     1.01943  |   0.284856 |            0.364729 |                      4 |                  256 |       4.34846 |      1.10393 |  0.476236 |  0.479389 | 0.698479 |         0.797784 |          0.598208 |
|      2 |           65 |       0.476928 |     0.95578  |   0.261055 |            0.364729 |                      4 |                  256 |       4.3286  |      1.14126 |  0.476928 |  0.561546 | 0.610352 |         0.766505 |          0.576245 |
|      3 |           94 |       0.450803 |     0.944156 |   0.233024 |            0.364729 |                      4 |                  256 |       4.28664 |      1.14544 |  0.450803 |  0.578222 | 0.577171 |         0.730312 |          0.548089 |
|      4 |           51 |       0.49607  |     0.99967  |   0.300364 |            0.364729 |                      4 |                  256 |       4.33425 |      1.09895 |  0.49607  |  0.493911 | 0.671632 |         0.789214 |          0.589071 |

## Decision Gate

Direct-submit gate: MAE <= 0.48 and no Spearman collapse; otherwise only test via Caruana ADD if decorrelated.

## Final Read

Do not submit directly unless this beats the current ChemProp Strategy 3 sibling or earns non-trivial Caruana weight with low residual correlation.
