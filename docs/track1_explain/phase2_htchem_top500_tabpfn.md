# Phase 2 HTChem top500 TabPFN

Experiment: `tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_pred_htchem_top500_umap_v2_6`

`pred_htchem` was appended as one scalar to `cheme_2d_full_boltz_log2fc_pred_seed10ens`, then top500 features were selected inside each outer fold using LGBM gain before fitting TabPFN.

## Overall OOF

| metric | value |
|:--|--:|
| MAE | 0.3960 |
| RAE | 0.4352 |
| Spearman_R | 0.8480 |
| Kendall_Tau | 0.6622 |

Reference existing top500: MAE=0.3966, RAE=0.4401, Spearman=0.8458. Delta MAE=-0.0006.

## Fold Metrics

|    MAE |    RAE |     R2 |   Spearman_R |   Kendall_Tau |
|-------:|-------:|-------:|-------------:|--------------:|
| 0.4123 | 0.4549 | 0.7528 |       0.8274 |        0.6411 |
| 0.3788 | 0.3813 | 0.8168 |       0.8776 |        0.6928 |
| 0.4131 | 0.4864 | 0.6929 |       0.8427 |        0.6569 |
| 0.3568 | 0.4576 | 0.7405 |       0.8264 |        0.6474 |
| 0.4183 | 0.4164 | 0.7653 |       0.8525 |        0.6622 |

## pred_htchem Selection

|   selected_folds |   mean_rank |   min_rank |   max_rank |   mean_gain_share_pct |   mean_split |
|-----------------:|------------:|-----------:|-----------:|----------------------:|-------------:|
|           5.0000 |     11.0000 |     6.0000 |    17.0000 |                0.1765 |      54.6000 |

## Read

Use this as a SWAP diagnostic against the existing top500 member, not as an ADD by default. The gain probe said `pred_htchem` is consistently selected but low-share; the OOF result decides whether it is worth carrying into ensemble bakeoff.
