# KA-GNN Report

Run name: `fourier_h128_l3_g3_lr1e3`
Experiment: `ka_gnn_fourier_h128_l3_g3_lr1e3_umap`
Reference implementation: `https://github.com/LongLee220/KA-GNN`

## OOF Metrics

Coverage: `4140 / 4140`
MAE: `0.589573`
RAE: `0.647932`
Spearman: `0.644692`
Residual r vs ens_caruana_bag20: `0.759565`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   target_mean |   target_std |   node_dim |   edge_dim |   val_MAE |   val_RAE |   val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|--------------:|-------------:|-----------:|-----------:|----------:|----------:|---------:|-----------------:|------------------:|
|      0 |           68 |       0.596066 |      1.20876 |   0.37464  |       4.30623 |      1.11576 |          9 |          3 |  0.596066 |  0.657576 | 0.482107 |         0.637044 |          0.454879 |
|      1 |           25 |       0.674889 |      1.14237 |   0.59327  |       4.34846 |      1.10393 |          9 |          3 |  0.674889 |  0.679358 | 0.405563 |         0.623776 |          0.441968 |
|      2 |           40 |       0.56476  |      1.21093 |   0.44158  |       4.3286  |      1.14126 |          9 |          3 |  0.56476  |  0.664961 | 0.434434 |         0.655566 |          0.471448 |
|      3 |           35 |       0.53997  |      1.2159  |   0.454056 |       4.28664 |      1.14544 |          9 |          3 |  0.53997  |  0.692592 | 0.402977 |         0.59877  |          0.427879 |
|      4 |           59 |       0.571467 |      1.20984 |   0.376455 |       4.33425 |      1.09895 |          9 |          3 |  0.571467 |  0.568979 | 0.568943 |         0.697746 |          0.501148 |

## Decision Gate

Continue only if MAE <= 0.49 and residual correlation/candidate correlations show a new axis.

## Final Read

Do not submit directly; compare Caruana ADD and correlations first.
