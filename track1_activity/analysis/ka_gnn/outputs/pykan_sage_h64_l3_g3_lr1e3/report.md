# KA-GNN Report

Run name: `pykan_sage_h64_l3_g3_lr1e3`
Experiment: `ka_gnn_pykan_sage_h64_l3_g3_lr1e3_umap`
Reference implementation: `https://github.com/LongLee220/KA-GNN`

## OOF Metrics

Coverage: `4140 / 4140`
MAE: `0.607936`
RAE: `0.668113`
Spearman: `0.631073`
Residual r vs ens_caruana_bag20: `0.743798`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   target_mean |   target_std |   node_dim |   edge_dim |   val_MAE |   val_RAE |   val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|--------------:|-------------:|-----------:|-----------:|----------:|----------:|---------:|-----------------:|------------------:|
|      0 |           76 |       0.646974 |     1.00007  |   0.478002 |       4.30623 |      1.11576 |          9 |          3 |  0.646974 |  0.713738 | 0.374275 |         0.593301 |          0.419184 |
|      1 |          100 |       0.653863 |     1.00062  |   0.51192  |       4.34846 |      1.10393 |          9 |          3 |  0.653863 |  0.658192 | 0.440743 |         0.618247 |          0.438865 |
|      2 |           72 |       0.579313 |     0.999876 |   0.476768 |       4.3286  |      1.14126 |          9 |          3 |  0.579313 |  0.682095 | 0.423686 |         0.634312 |          0.454725 |
|      3 |           59 |       0.531726 |     1.00017  |   0.456893 |       4.28664 |      1.14544 |          9 |          3 |  0.531726 |  0.682018 | 0.390193 |         0.61901  |          0.44667  |
|      4 |          100 |       0.626574 |     0.999921 |   0.475112 |       4.33425 |      1.09895 |          9 |          3 |  0.626574 |  0.623846 | 0.5088   |         0.689679 |          0.493488 |

## Decision Gate

Continue only if MAE <= 0.49 and residual correlation/candidate correlations show a new axis.

## Final Read

Do not submit directly; compare Caruana ADD and correlations first.
