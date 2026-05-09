# KAN on Frozen Embedding Report

Run name: `full256_h64_lr1e3`
Experiment: `kan_chemprop_pretrain_embed_full256_h64_lr1e3_umap`
Embedding: `/home/nagaet/pxr-iduction-challenge/data/chemprop_pretrain_embed.parquet`

## OOF Metrics

Coverage: `4140 / 4140`
MAE: `0.471579`
RAE: `0.518258`
Spearman: `0.787525`
Residual r vs ens_caruana_bag20: `0.884999`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   input_dim |   pca_variance |   target_mean |   target_std |   val_MAE |   val_RAE |   val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|------------:|---------------:|--------------:|-------------:|----------:|----------:|---------:|-----------------:|------------------:|
|      0 |           40 |       0.50556  |     0.792483 |  0.0767115 |         256 |            nan |       4.30623 |      1.11576 |  0.50556  |  0.557731 | 0.627958 |         0.755515 |          0.56456  |
|      1 |           41 |       0.464406 |     0.783734 |  0.0707575 |         256 |            nan |       4.34846 |      1.10393 |  0.464406 |  0.46748  | 0.724989 |         0.832978 |          0.634873 |
|      2 |           36 |       0.476554 |     0.778129 |  0.102847  |         256 |            nan |       4.3286  |      1.14126 |  0.476554 |  0.561105 | 0.603202 |         0.789708 |          0.597678 |
|      3 |           49 |       0.424464 |     0.796571 |  0.0471798 |         256 |            nan |       4.28664 |      1.14544 |  0.424464 |  0.544438 | 0.630489 |         0.740858 |          0.563858 |
|      4 |           41 |       0.486158 |     0.725071 |  0.0644219 |         256 |            nan |       4.33425 |      1.09895 |  0.486158 |  0.484042 | 0.683598 |         0.800407 |          0.60372  |

## Decision Gate

Use for deeper KAN/GNN-KAN only if it reaches MAE <= 0.47 or is clearly decorrelated with acceptable Spearman.

## Final Read

KAN is below the current ChemProp embedding + TabPFN sibling; do not submit directly.
