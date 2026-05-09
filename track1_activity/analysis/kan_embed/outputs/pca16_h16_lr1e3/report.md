# KAN on Frozen Embedding Report

Run name: `pca16_h16_lr1e3`
Experiment: `kan_chemprop_pretrain_embed_pca16_h16_lr1e3_umap`
Embedding: `/home/nagaet/pxr-iduction-challenge/data/chemprop_pretrain_embed.parquet`

## OOF Metrics

Coverage: `4140 / 4140`
MAE: `0.502945`
RAE: `0.552729`
Spearman: `0.764938`
Residual r vs ens_caruana_bag20: `0.870411`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   input_dim |   pca_variance |   target_mean |   target_std |   val_MAE |   val_RAE |   val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|------------:|---------------:|--------------:|-------------:|----------:|----------:|---------:|-----------------:|------------------:|
|      0 |          110 |       0.538044 |     0.994065 |   0.265022 |          16 |       0.713727 |       4.30623 |      1.11576 |  0.538044 |  0.593567 | 0.57461  |         0.720169 |          0.531979 |
|      1 |          135 |       0.479546 |     1.04586  |   0.31015  |          16 |       0.712679 |       4.34846 |      1.10393 |  0.479546 |  0.482721 | 0.701374 |         0.807861 |          0.608182 |
|      2 |          102 |       0.50284  |     1.02105  |   0.290789 |          16 |       0.719649 |       4.3286  |      1.14126 |  0.50284  |  0.592055 | 0.560207 |         0.768775 |          0.575025 |
|      3 |          104 |       0.459447 |     0.961624 |   0.292172 |          16 |       0.712158 |       4.28664 |      1.14544 |  0.459447 |  0.589309 | 0.583819 |         0.728641 |          0.543335 |
|      4 |          180 |       0.5341   |     1.02129  |   0.282115 |          16 |       0.707792 |       4.33425 |      1.09895 |  0.5341   |  0.531775 | 0.64411  |         0.787372 |          0.581608 |

## Decision Gate

Use for deeper KAN/GNN-KAN only if it reaches MAE <= 0.47 or is clearly decorrelated with acceptable Spearman.

## Final Read

KAN is below the current ChemProp embedding + TabPFN sibling; do not submit directly.
