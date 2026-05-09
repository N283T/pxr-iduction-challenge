# Buterez Strategy 6 GatedGCN Report

Run name: `initial`
Experiment: `gatedgcn_strategy6_adaptive_readout_initial_umap`
Pretrain checkpoint: `/home/nagaet/pxr-iduction-challenge/track1_activity/checkpoints/gatedgcn_pretrain/pretrain.pt`

## OOF Metrics

MAE: `0.503973`
RAE: `0.553859`
Spearman: `0.738494`

## Fold Metrics

|   fold |   epochs_run |   best_val_mae |   loss_start |   loss_end |   pretrain_best_val_loss |   val_MAE |   val_RAE |   val_R2 |   val_Spearman_R |   val_Kendall_Tau |
|-------:|-------------:|---------------:|-------------:|-----------:|-------------------------:|----------:|----------:|---------:|-----------------:|------------------:|
|      0 |           51 |       0.533632 |      1.62221 |   0.362737 |                  0.74776 |  0.533632 |  0.588699 | 0.580399 |         0.71415  |          0.527622 |
|      1 |           67 |       0.495843 |      1.88541 |   0.327228 |                  0.74776 |  0.495843 |  0.499126 | 0.671017 |         0.788687 |          0.587232 |
|      2 |           27 |       0.517256 |      1.69299 |   0.431149 |                  0.74776 |  0.517256 |  0.609028 | 0.527203 |         0.733608 |          0.544988 |
|      3 |           56 |       0.453414 |      1.58859 |   0.355652 |                  0.74776 |  0.453414 |  0.581571 | 0.561015 |         0.727387 |          0.548245 |
|      4 |           25 |       0.518932 |      1.75507 |   0.457746 |                  0.74776 |  0.518932 |  0.516673 | 0.637785 |         0.755135 |          0.559524 |

## Decision Gate

Consider Caruana ADD only if MAE <= 0.48 or residual correlation is clearly low without major Spearman collapse.

## Final Read

This exact Set Transformer Strategy 6 run is below the ADD gate. It improves on the 3-epoch smoke but remains weaker than the existing `tabpfn_gatedgcn_pretrain_embed_umap_default` member and far below the pool-leading low-fidelity feature stacks. Do not submit directly. A smaller readout variant (`tiny_lr1e3`) improved to MAE about 0.484, making Strategy 6 weak-positive but not yet clearly pool-worthy.
