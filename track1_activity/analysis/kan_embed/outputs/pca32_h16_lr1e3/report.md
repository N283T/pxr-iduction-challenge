# KAN on Frozen Embedding Report

Run name: `pca32_h16_lr1e3`
Experiment: `kan_chemprop_pretrain_embed_pca32_h16_lr1e3_umap`
Embedding: `/home/nagaet/pxr-iduction-challenge/data/chemprop_pretrain_embed.parquet`

## OOF Metrics

Coverage: `4140 / 4140`
MAE: `0.503338`
RAE: `0.553162`
Spearman: `0.766286`
Residual r vs ens_caruana_bag20: `0.860443`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   input_dim |   pca_variance |   target_mean |   target_std |   val_MAE |   val_RAE |   val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|------------:|---------------:|--------------:|-------------:|----------:|----------:|---------:|-----------------:|------------------:|
|      0 |           74 |       0.538621 |     1.01044  |   0.262901 |          32 |       0.823946 |       4.30623 |      1.11576 |  0.538621 |  0.594204 | 0.583913 |         0.727314 |          0.535874 |
|      1 |          105 |       0.475873 |     0.997489 |   0.259873 |          32 |       0.822179 |       4.34846 |      1.10393 |  0.475873 |  0.479024 | 0.700615 |         0.813871 |          0.614732 |
|      2 |           96 |       0.496846 |     0.981749 |   0.239069 |          32 |       0.827566 |       4.3286  |      1.14126 |  0.496846 |  0.584998 | 0.569843 |         0.773214 |          0.581189 |
|      3 |           99 |       0.453206 |     1.00625  |   0.263939 |          32 |       0.821637 |       4.28664 |      1.14544 |  0.453206 |  0.581304 | 0.59165  |         0.722301 |          0.540715 |
|      4 |          146 |       0.551254 |     0.994393 |   0.258103 |          32 |       0.820048 |       4.33425 |      1.09895 |  0.551254 |  0.548854 | 0.622807 |         0.773651 |          0.564886 |

## Decision Gate

Use for deeper KAN/GNN-KAN only if it reaches MAE <= 0.47 or is clearly decorrelated with acceptable Spearman.

## Final Read

KAN is below the current ChemProp embedding + TabPFN sibling; do not submit directly.
