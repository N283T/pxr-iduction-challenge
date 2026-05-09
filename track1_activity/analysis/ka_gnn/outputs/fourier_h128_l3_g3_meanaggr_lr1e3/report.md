# KA-GNN Report

Run name: `fourier_h128_l3_g3_meanaggr_lr1e3`
Experiment: `ka_gnn_fourier_h128_l3_g3_meanaggr_lr1e3_umap`
Reference implementation: `https://github.com/LongLee220/KA-GNN`

## OOF Metrics

Coverage: `4140 / 4140`
MAE: `0.564520`
RAE: `0.620399`
Spearman: `0.672755`
Residual r vs ens_caruana_bag20: `0.765183`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   target_mean |   target_std |   node_dim |   edge_dim |   val_MAE |   val_RAE |   val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|--------------:|-------------:|-----------:|-----------:|----------:|----------:|---------:|-----------------:|------------------:|
|      0 |           38 |       0.614659 |      1.1503  |   0.399953 |       4.30623 |      1.11576 |          9 |          3 |  0.614659 |  0.678088 | 0.46817  |         0.633245 |          0.451626 |
|      1 |           84 |       0.542666 |      1.22084 |   0.225316 |       4.34846 |      1.10393 |          9 |          3 |  0.542666 |  0.546259 | 0.603511 |         0.717654 |          0.5232   |
|      2 |           32 |       0.564345 |      1.16575 |   0.40657  |       4.3286  |      1.14126 |          9 |          3 |  0.564345 |  0.664472 | 0.444542 |         0.666825 |          0.480078 |
|      3 |           45 |       0.528727 |      1.10822 |   0.399547 |       4.28664 |      1.14544 |          9 |          3 |  0.528727 |  0.678172 | 0.439605 |         0.630243 |          0.452936 |
|      4 |           74 |       0.571607 |      1.32977 |   0.228291 |       4.33425 |      1.09895 |          9 |          3 |  0.571607 |  0.569119 | 0.561509 |         0.700177 |          0.505439 |

## Decision Gate

Continue only if MAE <= 0.49 and residual correlation/candidate correlations show a new axis.

## Final Read

Do not submit directly; compare Caruana ADD and correlations first.
