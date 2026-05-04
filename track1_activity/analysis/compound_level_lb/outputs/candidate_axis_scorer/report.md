# Candidate Axis Scorer

## Test CSV Candidates

| candidate                                | path                                                                     | exists   |   single_mae |   single_sp |   mean_delta |   mean_abs_delta |   p90_abs_delta |   max_abs_delta |   pearson |   spearman |   candidate_mean |   candidate_std | decision   | reasons     |
|:-----------------------------------------|:-------------------------------------------------------------------------|:---------|-------------:|------------:|-------------:|-----------------:|----------------:|----------------:|----------:|-----------:|-----------------:|----------------:|:-----------|:------------|
| tabpfn_drugclip_fold0_embed_umap_default | track1_activity/submissions/tabpfn_drugclip_fold0_embed_umap_default.csv | True     |       0.5416 |      0.6947 |      -0.0471 |           0.3577 |          0.7469 |          1.7151 |    0.8013 |     0.8107 |           4.7495 |          0.5686 | close      | weak_single |

## Best Summary-Only Candidates

| source                                                              | candidate                                       |   single_mae |   single_sp | decision   | reasons     |
|:--------------------------------------------------------------------|:------------------------------------------------|-------------:|------------:|:-----------|:------------|
| track1_activity/submissions/krovex_multiseed/summary.csv            | chemprop_proj32_concat_seed44 / chemprop_proj32 |       0.4889 |      0.7720 | close      | weak_single |
| track1_activity/submissions/krovex_grid/summary.csv                 | chemprop_proj32_kron                            |       0.4906 |      0.7517 | close      | weak_single |
| track1_activity/submissions/krovex_dimsweep/summary.csv             | chemprop_proj32_kron                            |       0.4906 |      0.7517 | close      | weak_single |
| track1_activity/submissions/krovex_multiseed/summary.csv            | chemprop_proj32_kron_seed42 / chemprop_proj32   |       0.4906 |      0.7517 | close      | weak_single |
| track1_activity/submissions/krovex_dimsweep/summary.csv             | chemprop_proj128_concat                         |       0.4923 |      0.7678 | close      | weak_single |
| track1_activity/submissions/krovex_multiseed/summary.csv            | chemprop_raw_concat_seed44 / chemprop_raw       |       0.4923 |      0.7725 | close      | weak_single |
| track1_activity/submissions/krovex_dimsweep/summary.csv             | chemprop_proj64_concat                          |       0.4931 |      0.7663 | close      | weak_single |
| track1_activity/submissions/krovex_multiseed/summary.csv            | chemprop_raw_concat_seed43 / chemprop_raw       |       0.4942 |      0.7744 | close      | weak_single |
| track1_activity/submissions/krovex_dimsweep/summary.csv             | chemprop_proj128_kron                           |       0.4948 |      0.7510 | close      | weak_single |
| track1_activity/submissions/krovex_grid/summary.csv                 | chemprop_proj32_concat                          |       0.4963 |      0.7728 | close      | weak_single |
| track1_activity/submissions/krovex_dimsweep/summary.csv             | chemprop_proj32_concat                          |       0.4963 |      0.7728 | close      | weak_single |
| track1_activity/submissions/krovex_multiseed/summary.csv            | chemprop_proj32_concat_seed42 / chemprop_proj32 |       0.4963 |      0.7728 | close      | weak_single |
| track1_activity/submissions/krovex_grid/summary.csv                 | chemprop256_concat                              |       0.4992 |      0.7661 | close      | weak_single |
| track1_activity/submissions/krovex_grid/summary.csv                 | chemprop256_kron                                |       0.5054 |      0.7391 | close      | weak_single |
| track1_activity/submissions/krovex_foundation/summary.csv           | molformer_kron_seed44                           |       0.5249 |      0.7034 | close      | weak_single |
| track1_activity/submissions/umsgfnet_fp_kan/phaseD2_summary.csv     | atom_gcn + bond_gcn + b1_kan                    |       0.5252 |      0.7066 | close      | weak_single |
| track1_activity/submissions/umsgfnet_fp_kan/phaseD3_summary.csv     | hier(atom+motif+super) + b1_kan                 |       0.5270 |      0.7056 | close      | weak_single |
| track1_activity/submissions/umsgfnet_fp_kan/phaseC_aggregate.csv    | kan_only / b1                                   |       0.5282 |      0.7076 | close      | weak_single |
| track1_activity/submissions/umsgfnet_fp_kan/phase_b1_a5_summary.csv | kan_only / b1                                   |       0.5290 |      0.7070 | close      | weak_single |
| track1_activity/submissions/krovex_foundation/summary.csv           | chemberta_77m_mlm_kron_seed42                   |       0.5294 |      0.7090 | close      | weak_single |
| track1_activity/submissions/umsgfnet_fp_kan/phaseD1_summary.csv     | atom_gcn + b1_kan                               |       0.5301 |      0.7047 | close      | weak_single |
| track1_activity/submissions/krovex_foundation/summary.csv           | molformer_kron_seed43                           |       0.5305 |      0.7090 | close      | weak_single |
| track1_activity/submissions/krovex_grid/summary.csv                 | cheme300_kron                                   |       0.5321 |      0.7043 | close      | weak_single |
| track1_activity/submissions/krovex_foundation/summary.csv           | molformer_kron_seed42                           |       0.5343 |      0.7101 | close      | weak_single |
| track1_activity/submissions/umsgfnet_fp_kan/phase_b2_a5_summary.csv | fused_alpha / b2                                |       0.5348 |      0.7062 | close      | weak_single |

## OOF Candidate Residual Check

| candidate                     | path                                                                          |   single_mae |   single_sp |   residual_r_vs_current_ensemble | decision   | reasons     |
|:------------------------------|:------------------------------------------------------------------------------|-------------:|------------:|---------------------------------:|:-----------|:------------|
| oof_chemprop_proj32_kron      | track1_activity/submissions/krovex_grid/oof_chemprop_proj32_kron.npy          |       0.4906 |      0.7517 |                           0.7918 | close      | weak_single |
| oof_chemprop_proj32_kron      | track1_activity/submissions/krovex_dimsweep/oof_chemprop_proj32_kron.npy      |       0.4906 |      0.7517 |                           0.7918 | close      | weak_single |
| oof_chemprop_proj128_concat   | track1_activity/submissions/krovex_dimsweep/oof_chemprop_proj128_concat.npy   |       0.4923 |      0.7678 |                           0.8256 | close      | weak_single |
| oof_chemprop_proj64_concat    | track1_activity/submissions/krovex_dimsweep/oof_chemprop_proj64_concat.npy    |       0.4931 |      0.7663 |                           0.8349 | close      | weak_single |
| oof_chemprop_proj128_kron     | track1_activity/submissions/krovex_dimsweep/oof_chemprop_proj128_kron.npy     |       0.4948 |      0.7510 |                           0.7965 | close      | weak_single |
| oof_chemprop_proj32_concat    | track1_activity/submissions/krovex_grid/oof_chemprop_proj32_concat.npy        |       0.4963 |      0.7728 |                           0.8450 | close      | weak_single |
| oof_chemprop_proj32_concat    | track1_activity/submissions/krovex_dimsweep/oof_chemprop_proj32_concat.npy    |       0.4963 |      0.7728 |                           0.8450 | close      | weak_single |
| oof_chemprop_proj64_kron      | track1_activity/submissions/krovex_dimsweep/oof_chemprop_proj64_kron.npy      |       0.4964 |      0.7394 |                           0.7956 | close      | weak_single |
| oof_chemprop_proj16_concat    | track1_activity/submissions/krovex_dimsweep/oof_chemprop_proj16_concat.npy    |       0.4982 |      0.7652 |                           0.8410 | close      | weak_single |
| oof_chemprop_raw_concat       | track1_activity/submissions/krovex_dimsweep/oof_chemprop_raw_concat.npy       |       0.4992 |      0.7661 |                           0.8358 | close      | weak_single |
| oof_chemprop256_concat        | track1_activity/submissions/krovex_grid/oof_chemprop256_concat.npy            |       0.4992 |      0.7661 |                           0.8358 | close      | weak_single |
| oof_chemprop_proj16_kron      | track1_activity/submissions/krovex_dimsweep/oof_chemprop_proj16_kron.npy      |       0.5008 |      0.7412 |                           0.7858 | close      | weak_single |
| oof_chemprop_raw_kron         | track1_activity/submissions/krovex_dimsweep/oof_chemprop_raw_kron.npy         |       0.5054 |      0.7391 |                           0.8024 | close      | weak_single |
| oof_chemprop256_kron          | track1_activity/submissions/krovex_grid/oof_chemprop256_kron.npy              |       0.5054 |      0.7391 |                           0.8024 | close      | weak_single |
| phaseD2_atom_bond_kan_oof     | track1_activity/submissions/umsgfnet_fp_kan/phaseD2_atom_bond_kan_oof.npy     |       0.5252 |      0.7066 |                           0.7700 | close      | weak_single |
| phaseC_b1_kan_only_seed44_oof | track1_activity/submissions/umsgfnet_fp_kan/phaseC_b1_kan_only_seed44_oof.npy |       0.5270 |      0.7068 |                           0.7675 | close      | weak_single |
| phaseD3_hier_kan_oof          | track1_activity/submissions/umsgfnet_fp_kan/phaseD3_hier_kan_oof.npy          |       0.5270 |      0.7056 |                           0.7785 | close      | weak_single |
| phaseC_b1_kan_only_seed43_oof | track1_activity/submissions/umsgfnet_fp_kan/phaseC_b1_kan_only_seed43_oof.npy |       0.5286 |      0.7089 |                           0.7586 | close      | weak_single |
| phase_b1_a5_kan_only_oof      | track1_activity/submissions/umsgfnet_fp_kan/phase_b1_a5_kan_only_oof.npy      |       0.5290 |      0.7070 |                           0.7662 | close      | weak_single |
| phaseC_b1_kan_only_seed42_oof | track1_activity/submissions/umsgfnet_fp_kan/phaseC_b1_kan_only_seed42_oof.npy |       0.5290 |      0.7070 |                           0.7662 | close      | weak_single |

## Interpretation

- `close`: single-model MAE is above 0.485, so do not spend LB cooldown or GPU.
- `needs_test_predictions`: OOF/summary exists, but no test-side axis is available.
- `blend_only`: candidate is too correlated or too large-shift for direct ADD.
- `review`: passes the cheap axis gate and deserves a deeper residual/OOF check.

## Closed Weak Singles

| source                                                    | candidate                                       |   single_mae |   single_sp |
|:----------------------------------------------------------|:------------------------------------------------|-------------:|------------:|
| track1_activity/submissions/krovex_multiseed/summary.csv  | chemprop_proj32_concat_seed44 / chemprop_proj32 |       0.4889 |      0.7720 |
| track1_activity/submissions/krovex_grid/summary.csv       | chemprop_proj32_kron                            |       0.4906 |      0.7517 |
| track1_activity/submissions/krovex_dimsweep/summary.csv   | chemprop_proj32_kron                            |       0.4906 |      0.7517 |
| track1_activity/submissions/krovex_multiseed/summary.csv  | chemprop_proj32_kron_seed42 / chemprop_proj32   |       0.4906 |      0.7517 |
| track1_activity/submissions/krovex_dimsweep/summary.csv   | chemprop_proj128_concat                         |       0.4923 |      0.7678 |
| track1_activity/submissions/krovex_multiseed/summary.csv  | chemprop_raw_concat_seed44 / chemprop_raw       |       0.4923 |      0.7725 |
| track1_activity/submissions/krovex_dimsweep/summary.csv   | chemprop_proj64_concat                          |       0.4931 |      0.7663 |
| track1_activity/submissions/krovex_multiseed/summary.csv  | chemprop_raw_concat_seed43 / chemprop_raw       |       0.4942 |      0.7744 |
| track1_activity/submissions/krovex_dimsweep/summary.csv   | chemprop_proj128_kron                           |       0.4948 |      0.7510 |
| track1_activity/submissions/krovex_grid/summary.csv       | chemprop_proj32_concat                          |       0.4963 |      0.7728 |
| track1_activity/submissions/krovex_dimsweep/summary.csv   | chemprop_proj32_concat                          |       0.4963 |      0.7728 |
| track1_activity/submissions/krovex_multiseed/summary.csv  | chemprop_proj32_concat_seed42 / chemprop_proj32 |       0.4963 |      0.7728 |
| track1_activity/submissions/krovex_grid/summary.csv       | chemprop256_concat                              |       0.4992 |      0.7661 |
| track1_activity/submissions/krovex_grid/summary.csv       | chemprop256_kron                                |       0.5054 |      0.7391 |
| track1_activity/submissions/krovex_foundation/summary.csv | molformer_kron_seed44                           |       0.5249 |      0.7034 |
