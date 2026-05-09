# Buterez Strategy 6 ChemProp Report

Run name: `best_d128_b1_d02_lr5e4`
Experiment: `chemprop_strategy6_adaptive_readout_best_d128_b1_d02_lr5e4_umap`
Pretrain checkpoint: `/home/nagaet/pxr-iduction-challenge/track1_activity/checkpoints/chemprop_pretrain/pretrain.pt`

## OOF Metrics

Coverage: `4140 / 4140`
MAE: `0.484064`
RAE: `0.531980`
Spearman: `0.759030`
Residual r vs ens_caruana_bag20: `0.865290`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   pretrain_val_loss |   frozen_param_tensors |   encoder_hidden_dim |   target_mean |   target_std |   val_MAE |   val_RAE |   val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|--------------------:|-----------------------:|---------------------:|--------------:|-------------:|----------:|----------:|---------:|-----------------:|------------------:|
|      0 |           61 |       0.518333 |     0.975178 |   0.236306 |            0.364729 |                      4 |                  256 |       4.30623 |      1.11576 |  0.518333 |  0.571821 | 0.597837 |         0.72316  |          0.533512 |
|      1 |           63 |       0.468434 |     1.0209   |   0.267655 |            0.364729 |                      4 |                  256 |       4.34846 |      1.10393 |  0.468434 |  0.471536 | 0.7039   |         0.800523 |          0.602342 |
|      2 |           52 |       0.487765 |     0.979969 |   0.256597 |            0.364729 |                      4 |                  256 |       4.3286  |      1.14126 |  0.487765 |  0.574305 | 0.593854 |         0.75395  |          0.565686 |
|      3 |           30 |       0.457844 |     1.00156  |   0.316084 |            0.364729 |                      4 |                  256 |       4.28664 |      1.14544 |  0.457844 |  0.587253 | 0.577009 |         0.707032 |          0.525263 |
|      4 |           56 |       0.48752  |     1.02128  |   0.256223 |            0.364729 |                      4 |                  256 |       4.33425 |      1.09895 |  0.48752  |  0.485397 | 0.682677 |         0.789616 |          0.590358 |

## Decision Gate

Direct-submit gate: MAE <= 0.48 and no Spearman collapse; otherwise only test via Caruana ADD if decorrelated.

## Final Read

Do not submit directly unless this beats the current ChemProp Strategy 3 sibling or earns non-trivial Caruana weight with low residual correlation.
