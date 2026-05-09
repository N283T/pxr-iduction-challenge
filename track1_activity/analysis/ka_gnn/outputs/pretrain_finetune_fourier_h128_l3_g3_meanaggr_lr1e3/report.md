# KA-GNN Report

Run name: `pretrain_finetune_fourier_h128_l3_g3_meanaggr_lr1e3`
Experiment: `ka_gnn_pretrain_finetune_fourier_h128_l3_g3_meanaggr_lr1e3_umap`
Reference implementation: `https://github.com/LongLee220/KA-GNN`

## OOF Metrics

Coverage: `4140 / 4140`
MAE: `0.545506`
RAE: `0.599503`
Spearman: `0.681831`
Residual r vs ens_caruana_bag20: `0.791777`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   target_mean |   target_std |   node_dim |   edge_dim |   pretrained_tensors |   val_MAE |   val_RAE |   val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|--------------:|-------------:|-----------:|-----------:|---------------------:|----------:|----------:|---------:|-----------------:|------------------:|
|      0 |           30 |       0.602701 |     0.922539 |   0.35629  |       4.30623 |      1.11576 |          9 |          3 |                    6 |  0.602701 |  0.664896 | 0.495231 |         0.643605 |          0.460548 |
|      1 |           52 |       0.531034 |     0.831074 |   0.266807 |       4.34846 |      1.10393 |          9 |          3 |                    6 |  0.531034 |  0.534549 | 0.621778 |         0.731267 |          0.533582 |
|      2 |           31 |       0.537812 |     0.830147 |   0.329019 |       4.3286  |      1.14126 |          9 |          3 |                    6 |  0.537812 |  0.633232 | 0.483819 |         0.679896 |          0.49256  |
|      3 |           25 |       0.505629 |     0.794696 |   0.386986 |       4.28664 |      1.14544 |          9 |          3 |                    6 |  0.505629 |  0.648544 | 0.478904 |         0.638604 |          0.462365 |
|      4 |           46 |       0.549687 |     0.869537 |   0.286223 |       4.33425 |      1.09895 |          9 |          3 |                    6 |  0.549687 |  0.547294 | 0.581617 |         0.712232 |          0.517158 |

## Decision Gate

Continue only if MAE <= 0.49 and residual correlation/candidate correlations show a new axis.

## Final Read

Do not submit directly; compare Caruana ADD and correlations first.
