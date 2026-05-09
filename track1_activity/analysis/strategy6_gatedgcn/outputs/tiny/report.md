# Buterez Strategy 6 GatedGCN Report

Run name: `tiny`
Experiment: `gatedgcn_strategy6_adaptive_readout_tiny_umap`
Pretrain checkpoint: `/home/nagaet/pxr-iduction-challenge/track1_activity/checkpoints/gatedgcn_pretrain/pretrain.pt`

## OOF Metrics

MAE: `0.485509`
RAE: `0.533568`
Spearman: `0.756467`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   pretrain_best_val_loss |   val_MAE |   val_RAE |   val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|-------------------------:|----------:|----------:|---------:|-----------------:|------------------:|
|      0 |           60 |       0.515527 |      6.33548 |   0.491944 |                  0.74776 |  0.515527 |  0.568727 | 0.592935 |         0.728254 |          0.542951 |
|      1 |           43 |       0.479923 |      4.5809  |   0.498157 |                  0.74776 |  0.479923 |  0.483101 | 0.693414 |         0.799592 |          0.598988 |
|      2 |           19 |       0.495488 |      5.08037 |   0.573427 |                  0.74776 |  0.495488 |  0.583398 | 0.562763 |         0.738029 |          0.546535 |
|      3 |           48 |       0.440826 |      3.14777 |   0.473552 |                  0.74776 |  0.440826 |  0.565425 | 0.597341 |         0.723033 |          0.543533 |
|      4 |           27 |       0.495089 |      5.50026 |   0.558237 |                  0.74776 |  0.495089 |  0.492934 | 0.669267 |         0.779512 |          0.58078  |

## Decision Gate

Consider Caruana ADD only if MAE <= 0.48 or residual correlation is clearly low without major Spearman collapse.

## Final Read

This smaller readout improves over the default initial run but remains just above the 0.485 gate. The `tiny_lr1e3` variant is slightly better.
