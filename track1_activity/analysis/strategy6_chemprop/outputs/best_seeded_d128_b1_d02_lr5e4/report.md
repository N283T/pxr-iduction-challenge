# Buterez Strategy 6 ChemProp Report

Run name: `best_seeded_d128_b1_d02_lr5e4`
Experiment: `chemprop_strategy6_adaptive_readout_best_seeded_d128_b1_d02_lr5e4_umap`
Pretrain checkpoint: `/home/nagaet/pxr-iduction-challenge/track1_activity/checkpoints/chemprop_pretrain/pretrain.pt`

## OOF Metrics

Coverage: `4140 / 4140`
MAE: `0.481902`
RAE: `0.529604`
Spearman: `0.761947`
Residual r vs ens_caruana_bag20: `0.868363`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   pretrain_val_loss |   frozen_param_tensors |   encoder_hidden_dim |   target_mean |   target_std |   val_MAE |   val_RAE |   val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|--------------------:|-----------------------:|---------------------:|--------------:|-------------:|----------:|----------:|---------:|-----------------:|------------------:|
|      0 |           67 |       0.521549 |     0.977412 |   0.218963 |            0.364729 |                      4 |                  256 |       4.30623 |      1.11576 |  0.521549 |  0.57537  | 0.598658 |         0.720332 |          0.529946 |
|      1 |           76 |       0.467563 |     0.893814 |   0.235784 |            0.364729 |                      4 |                  256 |       4.34846 |      1.10393 |  0.467563 |  0.470658 | 0.702649 |         0.801493 |          0.604007 |
|      2 |           42 |       0.48457  |     1.04381  |   0.270886 |            0.364729 |                      4 |                  256 |       4.3286  |      1.14126 |  0.48457  |  0.570543 | 0.592743 |         0.756143 |          0.564816 |
|      3 |           65 |       0.443119 |     1.00925  |   0.231036 |            0.364729 |                      4 |                  256 |       4.28664 |      1.14544 |  0.443119 |  0.568367 | 0.596679 |         0.73     |          0.548977 |
|      4 |           50 |       0.492079 |     1.01973  |   0.277219 |            0.364729 |                      4 |                  256 |       4.33425 |      1.09895 |  0.492079 |  0.489937 | 0.673605 |         0.783682 |          0.583839 |

## Decision Gate

Direct-submit gate: MAE <= 0.48 and no Spearman collapse; otherwise only test via Caruana ADD if decorrelated.

## Final Read

Do not submit directly unless this beats the current ChemProp Strategy 3 sibling or earns non-trivial Caruana weight with low residual correlation.

## Caruana ADD Diagnostic

One-off ADD bakeoff against the current 9-member `ENSEMBLE_MODELS` gave the Strategy 6 member only `0.0010` Caruana weight. The extended-pool OOF looked better because the bagged search reallocated weight between existing top members after the pool size changed, not because this member contributed meaningful mass. Treat this as ensemble-rejected for now.
