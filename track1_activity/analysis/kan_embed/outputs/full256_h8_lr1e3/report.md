# KAN on Frozen Embedding Report

Run name: `full256_h8_lr1e3`
Experiment: `kan_chemprop_pretrain_embed_full256_h8_lr1e3_umap`
Embedding: `/home/nagaet/pxr-iduction-challenge/data/chemprop_pretrain_embed.parquet`

## OOF Metrics

Coverage: `4140 / 4140`
MAE: `0.474574`
RAE: `0.521551`
Spearman: `0.786748`
Residual r vs ens_caruana_bag20: `0.887991`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   input_dim |   pca_variance |   target_mean |   target_std |   val_MAE |   val_RAE |   val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|------------:|---------------:|--------------:|-------------:|----------:|----------:|---------:|-----------------:|------------------:|
|      0 |           52 |       0.509611 |     0.927957 |   0.190155 |         256 |            nan |       4.30623 |      1.11576 |  0.509611 |  0.5622   | 0.631293 |         0.754065 |          0.561778 |
|      1 |           62 |       0.468053 |     0.979314 |   0.176865 |         256 |            nan |       4.34846 |      1.10393 |  0.468053 |  0.471152 | 0.726478 |         0.827832 |          0.62919  |
|      2 |           32 |       0.478073 |     0.917586 |   0.213604 |         256 |            nan |       4.3286  |      1.14126 |  0.478073 |  0.562894 | 0.599981 |         0.78708  |          0.595454 |
|      3 |           51 |       0.428209 |     0.916692 |   0.192228 |         256 |            nan |       4.28664 |      1.14544 |  0.428209 |  0.549242 | 0.638818 |         0.751305 |          0.570514 |
|      4 |           59 |       0.488184 |     0.878163 |   0.166592 |         256 |            nan |       4.33425 |      1.09895 |  0.488184 |  0.486059 | 0.679978 |         0.7963   |          0.598476 |

## Decision Gate

Use for deeper KAN/GNN-KAN only if it reaches MAE <= 0.47 or is clearly decorrelated with acceptable Spearman.

## Final Read

KAN is below the current ChemProp embedding + TabPFN sibling; do not submit directly.
