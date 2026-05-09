# Buterez Strategy 6 ChemProp Report

Run name: `d128_b1_d02_lr5e4`
Experiment: `chemprop_strategy6_adaptive_readout_d128_b1_d02_lr5e4_umap`
Pretrain checkpoint: `/home/nagaet/pxr-iduction-challenge/track1_activity/checkpoints/chemprop_pretrain/pretrain.pt`

## OOF Metrics

Coverage: `4140 / 4140`
MAE: `0.480593`
RAE: `0.528165`
Spearman: `0.762776`
Residual r vs ens_caruana_bag20: `0.866745`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   pretrain_val_loss |   frozen_param_tensors |   encoder_hidden_dim |   target_mean |   target_std |   val_MAE |   val_RAE |   val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|--------------------:|-----------------------:|---------------------:|--------------:|-------------:|----------:|----------:|---------:|-----------------:|------------------:|
|      0 |           49 |       0.522533 |     1.03238  |   0.275478 |            0.364729 |                      4 |                  256 |       4.30623 |      1.11576 |  0.522533 |  0.576455 | 0.594167 |         0.719744 |          0.532037 |
|      1 |           58 |       0.47491  |     1.02064  |   0.275095 |            0.364729 |                      4 |                  256 |       4.34846 |      1.10393 |  0.47491  |  0.478055 | 0.701091 |         0.796466 |          0.598778 |
|      2 |           48 |       0.475794 |     0.916117 |   0.265311 |            0.364729 |                      4 |                  256 |       4.3286  |      1.14126 |  0.475794 |  0.56021  | 0.592491 |         0.760133 |          0.571867 |
|      3 |           57 |       0.446947 |     1.04593  |   0.247141 |            0.364729 |                      4 |                  256 |       4.28664 |      1.14544 |  0.446947 |  0.573276 | 0.586636 |         0.729131 |          0.548263 |
|      4 |           71 |       0.482236 |     1.07241  |   0.233992 |            0.364729 |                      4 |                  256 |       4.33425 |      1.09895 |  0.482236 |  0.480137 | 0.694358 |         0.789748 |          0.590821 |

## Decision Gate

Direct-submit gate: MAE <= 0.48 and no Spearman collapse; otherwise only test via Caruana ADD if decorrelated.

## Final Read

Do not submit directly unless this beats the current ChemProp Strategy 3 sibling or earns non-trivial Caruana weight with low residual correlation.
