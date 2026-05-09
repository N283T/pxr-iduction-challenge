# KAN on Frozen Embedding Report

Run name: `full256_h16_lr1e3`
Experiment: `kan_chemprop_pretrain_embed_full256_h16_lr1e3_umap`
Embedding: `/home/nagaet/pxr-iduction-challenge/data/chemprop_pretrain_embed.parquet`

## OOF Metrics

Coverage: `4140 / 4140`
MAE: `0.470122`
RAE: `0.516657`
Spearman: `0.790855`
Residual r vs ens_caruana_bag20: `0.891434`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   input_dim |   pca_variance |   target_mean |   target_std |   val_MAE |   val_RAE |   val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|------------:|---------------:|--------------:|-------------:|----------:|----------:|---------:|-----------------:|------------------:|
|      0 |           46 |       0.508676 |     0.85596  |   0.147473 |         256 |            nan |       4.30623 |      1.11576 |  0.508676 |  0.561168 | 0.629994 |         0.759421 |          0.566767 |
|      1 |           47 |       0.47069  |     0.907469 |   0.155073 |         256 |            nan |       4.34846 |      1.10393 |  0.47069  |  0.473806 | 0.724216 |         0.828649 |          0.629603 |
|      2 |           37 |       0.472558 |     0.869272 |   0.179142 |         256 |            nan |       4.3286  |      1.14126 |  0.472558 |  0.5564   | 0.617093 |         0.790046 |          0.596756 |
|      3 |           51 |       0.42264  |     0.913731 |   0.141613 |         256 |            nan |       4.28664 |      1.14544 |  0.42264  |  0.542099 | 0.650418 |         0.756356 |          0.574079 |
|      4 |           58 |       0.475302 |     0.859467 |   0.113568 |         256 |            nan |       4.33425 |      1.09895 |  0.475302 |  0.473233 | 0.692891 |         0.802905 |          0.60715  |

## Decision Gate

Use for deeper KAN/GNN-KAN only if it reaches MAE <= 0.47 or is clearly decorrelated with acceptable Spearman.

## Final Read

KAN is below the current ChemProp embedding + TabPFN sibling; do not submit directly.
