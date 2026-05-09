# Buterez Strategy 6 GatedGCN Report

Run name: `smoke`
Experiment: `gatedgcn_strategy6_adaptive_readout_smoke_umap`
Pretrain checkpoint: `/home/nagaet/pxr-iduction-challenge/track1_activity/checkpoints/gatedgcn_pretrain/pretrain.pt`

## OOF Metrics

MAE: `0.555600`
RAE: `0.610597`
Spearman: `0.687494`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   pretrain_best_val_loss |   val_MAE |   val_RAE |   val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|-------------------------:|----------:|----------:|---------:|-----------------:|------------------:|
|      0 |            3 |       0.553472 |      2.74108 |   0.575163 |                  0.74776 |  0.553472 |  0.610587 | 0.533942 |         0.700593 |          0.51285  |
|      1 |            3 |       0.656201 |      3.39268 |   0.637757 |                  0.74776 |  0.656201 |  0.660546 | 0.405368 |         0.731154 |          0.536453 |
|      2 |            3 |       0.510668 |      2.67427 |   0.674595 |                  0.74776 |  0.510668 |  0.601272 | 0.530878 |         0.723225 |          0.535421 |
|      3 |            3 |       0.499093 |      2.95844 |   0.600245 |                  0.74776 |  0.499093 |  0.640161 | 0.492454 |         0.681704 |          0.503175 |
|      4 |            3 |       0.557694 |      2.63183 |   0.60716  |                  0.74776 |  0.557694 |  0.555266 | 0.604069 |         0.726677 |          0.532814 |

## Decision Gate

Consider Caruana ADD only if MAE <= 0.48 or residual correlation is clearly low without major Spearman collapse.
