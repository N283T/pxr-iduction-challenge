# Buterez Strategy 6 ChemProp Report

Run name: `tiny_lr1e3`
Experiment: `chemprop_strategy6_adaptive_readout_tiny_lr1e3_umap`
Pretrain checkpoint: `/home/nagaet/pxr-iduction-challenge/track1_activity/checkpoints/chemprop_pretrain/pretrain.pt`

## OOF Metrics

Coverage: `4140 / 4140`
MAE: `0.484447`
RAE: `0.532400`
Spearman: `0.763328`
Residual r vs ens_caruana_bag20: `0.860669`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   pretrain_val_loss |   frozen_param_tensors |   encoder_hidden_dim |   target_mean |   target_std |   val_MAE |   val_RAE |   val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|--------------------:|-----------------------:|---------------------:|--------------:|-------------:|----------:|----------:|---------:|-----------------:|------------------:|
|      0 |           68 |       0.520456 |     0.89142  |   0.27521  |            0.364729 |                      4 |                  256 |       4.30623 |      1.11576 |  0.520456 |  0.574164 | 0.599951 |         0.726322 |          0.536806 |
|      1 |           42 |       0.473901 |     1.02356  |   0.326837 |            0.364729 |                      4 |                  256 |       4.34846 |      1.10393 |  0.473901 |  0.477039 | 0.698171 |         0.80055  |          0.60311  |
|      2 |           59 |       0.479797 |     1.04406  |   0.272696 |            0.364729 |                      4 |                  256 |       4.3286  |      1.14126 |  0.479797 |  0.564923 | 0.608279 |         0.76607  |          0.574786 |
|      3 |           53 |       0.45004  |     0.888492 |   0.288481 |            0.364729 |                      4 |                  256 |       4.28664 |      1.14544 |  0.45004  |  0.577243 | 0.582229 |         0.723859 |          0.54129  |
|      4 |           51 |       0.497462 |     0.98896  |   0.313287 |            0.364729 |                      4 |                  256 |       4.33425 |      1.09895 |  0.497462 |  0.495296 | 0.676516 |         0.786302 |          0.585282 |

## Decision Gate

Direct-submit gate: MAE <= 0.48 and no Spearman collapse; otherwise only test via Caruana ADD if decorrelated.

## Final Read

Do not submit directly unless this beats the current ChemProp Strategy 3 sibling or earns non-trivial Caruana weight with low residual correlation.
