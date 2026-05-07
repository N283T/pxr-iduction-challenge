# Cross-Model Meta Diagnostic

This report does not explain each model's raw input features. It compares
where model OOF predictions differ using shared meta-features: assay labels,
potent46 proximity, and model-disagreement summaries.

## Overall

| model                | experiment                                                    |     mae |   spearman |   pred_mean |   pred_std |   corr_vs_ensemble |   mean_abs_delta_vs_ensemble |
|:---------------------|:--------------------------------------------------------------|--------:|-----------:|------------:|-----------:|-------------------:|-----------------------------:|
| ensemble             | ens_caruana_bag20                                             | 0.39451 |    0.84781 |     4.33303 |    0.93535 |            1.00000 |                      0.00000 |
| cheme_seed10_top500  | tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap  | 0.39676 |    0.84846 |     4.32540 |    0.98213 |            0.99248 |                      0.09037 |
| chemprop_family_meta | tabpfn_chemprop_family_meta_umap                              | 0.39760 |    0.84448 |     4.33333 |    0.94396 |            0.99859 |                      0.03618 |
| cheme_seed10_default | tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_umap_default | 0.40559 |    0.84009 |     4.33528 |    0.93608 |            0.99636 |                      0.05866 |
| mordred_singleconc   | lgbm_mordred_singleconc_umap_default                          | 0.42509 |    0.83288 |     4.32479 |    0.91210 |            0.92497 |                      0.25330 |
| chemprop_embed       | tabpfn_chemprop_pretrain_embed_umap_default                   | 0.43726 |    0.81019 |     4.33888 |    0.94374 |            0.97715 |                      0.13890 |
| kermt_embed          | tabpfn_kermt_pretrain_embed_umap_default                      | 0.44853 |    0.79189 |     4.33975 |    0.92804 |            0.96325 |                      0.17675 |
| gatedgcn_embed       | tabpfn_gatedgcn_pretrain_embed_umap_default                   | 0.47400 |    0.76666 |     4.35306 |    0.90205 |            0.94647 |                      0.21306 |
| molformer_c3_embed   | tabpfn_molformer_c3_pretrain_embed_umap                       | 0.47528 |    0.76469 |     4.31629 |    0.94005 |            0.94066 |                      0.22703 |
| attentivefp_embed    | tabpfn_attentivefp_pretrain_embed_umap_default                | 0.48437 |    0.76233 |     4.33854 |    0.90950 |            0.93988 |                      0.23070 |
| boltz_allpairs       | tabpfn_pooled_boltz_allpairs_umap_default                     | 0.48601 |    0.76201 |     4.34702 |    0.89824 |            0.92758 |                      0.24729 |
| boltz_core           | tabpfn_pooled_boltz_umap_default                              | 0.48607 |    0.75795 |     4.34794 |    0.89872 |            0.92436 |                      0.25367 |

## Key Slices

### `near_potent46_t04`

| model                |   n_true |   mae_true |   delta_true_mae_vs_ensemble |   mean_residual_true |
|:---------------------|---------:|-----------:|-----------------------------:|---------------------:|
| cheme_seed10_top500  |       73 |    0.64925 |                     -0.11470 |              0.58295 |
| mordred_singleconc   |       73 |    0.75068 |                     -0.01327 |              0.69663 |
| ensemble             |       73 |    0.76395 |                      0.00000 |              0.71136 |
| chemprop_family_meta |       73 |    0.76745 |                      0.00350 |              0.71186 |
| cheme_seed10_default |       73 |    0.77679 |                      0.01284 |              0.72598 |
| chemprop_embed       |       73 |    0.89404 |                      0.13009 |              0.82286 |
| kermt_embed          |       73 |    0.89782 |                      0.13387 |              0.84768 |
| boltz_allpairs       |       73 |    0.90763 |                      0.14368 |              0.82052 |
| boltz_core           |       73 |    0.91588 |                      0.15193 |              0.84449 |
| attentivefp_embed    |       73 |    0.94602 |                      0.18207 |              0.89014 |
| gatedgcn_embed       |       73 |    1.02900 |                      0.26505 |              0.92577 |
| molformer_c3_embed   |       73 |    1.03424 |                      0.27029 |              0.97070 |

### `near_potent46_t03`

| model                |   n_true |   mae_true |   delta_true_mae_vs_ensemble |   mean_residual_true |
|:---------------------|---------:|-----------:|-----------------------------:|---------------------:|
| cheme_seed10_top500  |      223 |    0.45729 |                     -0.03601 |              0.19349 |
| ensemble             |      223 |    0.49329 |                      0.00000 |              0.24315 |
| chemprop_family_meta |      223 |    0.49733 |                      0.00403 |              0.24389 |
| cheme_seed10_default |      223 |    0.50743 |                      0.01414 |              0.25213 |
| mordred_singleconc   |      223 |    0.53318 |                      0.03989 |              0.26514 |
| kermt_embed          |      223 |    0.56167 |                      0.06837 |              0.28305 |
| chemprop_embed       |      223 |    0.56710 |                      0.07381 |              0.27842 |
| attentivefp_embed    |      223 |    0.61573 |                      0.12243 |              0.33298 |
| boltz_core           |      223 |    0.61686 |                      0.12356 |              0.27830 |
| boltz_allpairs       |      223 |    0.62249 |                      0.12919 |              0.25077 |
| molformer_c3_embed   |      223 |    0.63759 |                      0.14429 |              0.37189 |
| gatedgcn_embed       |      223 |    0.64217 |                      0.14887 |              0.28187 |

### `no_aux`

| model                |   n_true |   mae_true |   delta_true_mae_vs_ensemble |   mean_residual_true |
|:---------------------|---------:|-----------:|-----------------------------:|---------------------:|
| ensemble             |     1294 |    0.45924 |                      0.00000 |             -0.09574 |
| chemprop_family_meta |     1294 |    0.46485 |                      0.00561 |             -0.09333 |
| cheme_seed10_default |     1294 |    0.47329 |                      0.01404 |             -0.10081 |
| cheme_seed10_top500  |     1294 |    0.47610 |                      0.01686 |             -0.06397 |
| chemprop_embed       |     1294 |    0.48624 |                      0.02700 |             -0.12407 |
| kermt_embed          |     1294 |    0.49498 |                      0.03573 |             -0.13495 |
| gatedgcn_embed       |     1294 |    0.50889 |                      0.04965 |             -0.19934 |
| molformer_c3_embed   |     1294 |    0.51643 |                      0.05718 |             -0.09657 |
| boltz_core           |     1294 |    0.52302 |                      0.06378 |             -0.17867 |
| boltz_allpairs       |     1294 |    0.52317 |                      0.06393 |             -0.18517 |
| attentivefp_embed    |     1294 |    0.52934 |                      0.07010 |             -0.18901 |
| mordred_singleconc   |     1294 |    0.53068 |                      0.07144 |             -0.05907 |

## Surrogate Quality For `pred - ensemble`

| model                | target              |   target_std |   oof_mae |   oof_r2 |   oof_spearman |
|:---------------------|:--------------------|-------------:|----------:|---------:|---------------:|
| ensemble             | pred_minus_ensemble |      0.00000 |   0.00000 |  1.00000 |      nan       |
| boltz_allpairs       | pred_minus_ensemble |      0.35081 |   0.14601 |  0.66610 |        0.76083 |
| boltz_core           | pred_minus_ensemble |      0.35849 |   0.15214 |  0.65211 |        0.75041 |
| cheme_seed10_top500  | pred_minus_ensemble |      0.12652 |   0.06889 |  0.41524 |        0.60184 |
| gatedgcn_embed       | pred_minus_ensemble |      0.30237 |   0.16620 |  0.40840 |        0.59560 |
| mordred_singleconc   | pred_minus_ensemble |      0.35856 |   0.19863 |  0.40161 |        0.58486 |
| attentivefp_embed    | pred_minus_ensemble |      0.32085 |   0.17846 |  0.39498 |        0.59173 |
| chemprop_family_meta | pred_minus_ensemble |      0.05066 |   0.02855 |  0.35716 |        0.58429 |
| molformer_c3_embed   | pred_minus_ensemble |      0.32306 |   0.18548 |  0.32756 |        0.53668 |
| kermt_embed          | pred_minus_ensemble |      0.25271 |   0.14973 |  0.28371 |        0.49828 |
| chemprop_embed       | pred_minus_ensemble |      0.20104 |   0.13308 |  0.09861 |        0.24705 |
| cheme_seed10_default | pred_minus_ensemble |      0.07982 |   0.05757 |  0.03115 |        0.17226 |

## Top SHAP Features Per Model

### `ensemble`

| model    | feature             |   mean_abs_shap |   mean_shap |
|:---------|:--------------------|----------------:|------------:|
| ensemble | pec50               |         0.00000 |     0.00000 |
| ensemble | logp                |         0.00000 |     0.00000 |
| ensemble | tpsa                |         0.00000 |     0.00000 |
| ensemble | exactmw             |         0.00000 |     0.00000 |
| ensemble | fractioncsp3        |         0.00000 |     0.00000 |
| ensemble | hba                 |         0.00000 |     0.00000 |
| ensemble | hbd                 |         0.00000 |     0.00000 |
| ensemble | num_heavy_atoms     |         0.00000 |     0.00000 |
| ensemble | num_heteroatoms     |         0.00000 |     0.00000 |
| ensemble | num_rotatable_bonds |         0.00000 |     0.00000 |

### `cheme_seed10_top500`

| model               | feature              |   mean_abs_shap |   mean_shap |
|:--------------------|:---------------------|----------------:|------------:|
| cheme_seed10_top500 | family_gap           |         0.05141 |     0.00510 |
| cheme_seed10_top500 | log2fc_8_25e_6       |         0.01000 |     0.00282 |
| cheme_seed10_top500 | pec50                |         0.00916 |     0.00181 |
| cheme_seed10_top500 | abs_family_gap       |         0.00875 |    -0.00013 |
| cheme_seed10_top500 | member_range         |         0.00780 |    -0.00037 |
| cheme_seed10_top500 | log2fc_3_30e_5       |         0.00734 |    -0.00091 |
| cheme_seed10_top500 | fractioncsp3         |         0.00661 |     0.00072 |
| cheme_seed10_top500 | exactmw              |         0.00652 |     0.00044 |
| cheme_seed10_top500 | member_std           |         0.00547 |    -0.00074 |
| cheme_seed10_top500 | nn_potent46_tanimoto |         0.00503 |     0.00095 |

### `chemprop_family_meta`

| model                | feature        |   mean_abs_shap |   mean_shap |
|:---------------------|:---------------|----------------:|------------:|
| chemprop_family_meta | family_gap     |         0.02533 |     0.00285 |
| chemprop_family_meta | log2fc_8_25e_6 |         0.00594 |    -0.00034 |
| chemprop_family_meta | abs_family_gap |         0.00531 |    -0.00023 |
| chemprop_family_meta | logp           |         0.00408 |    -0.00012 |
| chemprop_family_meta | exactmw        |         0.00373 |    -0.00010 |
| chemprop_family_meta | member_std     |         0.00371 |    -0.00029 |
| chemprop_family_meta | pec50          |         0.00346 |    -0.00040 |
| chemprop_family_meta | tpsa           |         0.00328 |     0.00014 |
| chemprop_family_meta | member_range   |         0.00302 |     0.00027 |
| chemprop_family_meta | fractioncsp3   |         0.00287 |    -0.00009 |

### `cheme_seed10_default`

| model                | feature              |   mean_abs_shap |   mean_shap |
|:---------------------|:---------------------|----------------:|------------:|
| cheme_seed10_default | pec50                |         0.00834 |    -0.00129 |
| cheme_seed10_default | logp                 |         0.00797 |    -0.00003 |
| cheme_seed10_default | log2fc_8_25e_6       |         0.00694 |    -0.00150 |
| cheme_seed10_default | exactmw              |         0.00652 |    -0.00059 |
| cheme_seed10_default | tpsa                 |         0.00571 |    -0.00053 |
| cheme_seed10_default | log2fc_3_30e_5       |         0.00560 |     0.00026 |
| cheme_seed10_default | family_gap           |         0.00509 |     0.00021 |
| cheme_seed10_default | member_range         |         0.00459 |     0.00097 |
| cheme_seed10_default | fractioncsp3         |         0.00420 |    -0.00057 |
| cheme_seed10_default | nn_potent46_tanimoto |         0.00384 |     0.00018 |

### `mordred_singleconc`

| model              | feature              |   mean_abs_shap |   mean_shap |
|:-------------------|:---------------------|----------------:|------------:|
| mordred_singleconc | log2fc_8_25e_6       |         0.13959 |     0.01287 |
| mordred_singleconc | pec50                |         0.09637 |    -0.01633 |
| mordred_singleconc | log2fc_3_30e_5       |         0.07872 |     0.00057 |
| mordred_singleconc | family_gap           |         0.07432 |    -0.00764 |
| mordred_singleconc | logp                 |         0.03332 |    -0.00046 |
| mordred_singleconc | member_std           |         0.03286 |     0.00049 |
| mordred_singleconc | num_heteroatoms      |         0.02285 |     0.00064 |
| mordred_singleconc | tpsa                 |         0.01774 |    -0.00241 |
| mordred_singleconc | nn_potent46_tanimoto |         0.01756 |    -0.00532 |
| mordred_singleconc | counter_pec50        |         0.01750 |    -0.00055 |

### `chemprop_embed`

| model          | feature        |   mean_abs_shap |   mean_shap |
|:---------------|:---------------|----------------:|------------:|
| chemprop_embed | log2fc_8_25e_6 |         0.02980 |    -0.00173 |
| chemprop_embed | family_gap     |         0.02608 |     0.00156 |
| chemprop_embed | member_std     |         0.01879 |    -0.00118 |
| chemprop_embed | pec50          |         0.01757 |    -0.00273 |
| chemprop_embed | logp           |         0.01583 |     0.00052 |
| chemprop_embed | member_range   |         0.01578 |     0.00030 |
| chemprop_embed | tpsa           |         0.01417 |     0.00015 |
| chemprop_embed | fractioncsp3   |         0.01318 |    -0.00149 |
| chemprop_embed | log2fc_3_30e_5 |         0.01298 |     0.00203 |
| chemprop_embed | exactmw        |         0.01156 |    -0.00082 |

### `kermt_embed`

| model       | feature         |   mean_abs_shap |   mean_shap |
|:------------|:----------------|----------------:|------------:|
| kermt_embed | family_gap      |         0.09039 |    -0.00953 |
| kermt_embed | exactmw         |         0.02952 |     0.00463 |
| kermt_embed | member_std      |         0.02306 |     0.00216 |
| kermt_embed | abs_family_gap  |         0.01798 |    -0.00058 |
| kermt_embed | member_range    |         0.01691 |    -0.00283 |
| kermt_embed | num_heavy_atoms |         0.01619 |     0.00245 |
| kermt_embed | pec50           |         0.01368 |     0.00286 |
| kermt_embed | logp            |         0.01332 |     0.00085 |
| kermt_embed | log2fc_3_30e_5  |         0.01314 |     0.00107 |
| kermt_embed | counter_pec50   |         0.01019 |     0.00017 |

### `gatedgcn_embed`

| model          | feature         |   mean_abs_shap |   mean_shap |
|:---------------|:----------------|----------------:|------------:|
| gatedgcn_embed | family_gap      |         0.11082 |    -0.00666 |
| gatedgcn_embed | num_heavy_atoms |         0.02495 |     0.00439 |
| gatedgcn_embed | member_std      |         0.02122 |     0.00005 |
| gatedgcn_embed | exactmw         |         0.02096 |     0.00239 |
| gatedgcn_embed | abs_family_gap  |         0.02022 |    -0.00237 |
| gatedgcn_embed | member_range    |         0.01620 |     0.00370 |
| gatedgcn_embed | log2fc_8_25e_6  |         0.01558 |    -0.00088 |
| gatedgcn_embed | num_rings       |         0.01162 |    -0.00255 |
| gatedgcn_embed | counter_emax    |         0.01143 |    -0.00172 |
| gatedgcn_embed | logp            |         0.01118 |     0.00128 |

### `molformer_c3_embed`

| model              | feature              |   mean_abs_shap |   mean_shap |
|:-------------------|:---------------------|----------------:|------------:|
| molformer_c3_embed | family_gap           |         0.13424 |    -0.00983 |
| molformer_c3_embed | member_range         |         0.03950 |    -0.00838 |
| molformer_c3_embed | logp                 |         0.03109 |    -0.00213 |
| molformer_c3_embed | abs_family_gap       |         0.02621 |     0.00277 |
| molformer_c3_embed | member_std           |         0.01825 |     0.00451 |
| molformer_c3_embed | fractioncsp3         |         0.01561 |     0.00084 |
| molformer_c3_embed | pec50                |         0.01403 |    -0.00036 |
| molformer_c3_embed | nn_potent46_tanimoto |         0.01321 |     0.00181 |
| molformer_c3_embed | exactmw              |         0.01216 |     0.00118 |
| molformer_c3_embed | tpsa                 |         0.01207 |     0.00177 |

### `attentivefp_embed`

| model             | feature             |   mean_abs_shap |   mean_shap |
|:------------------|:--------------------|----------------:|------------:|
| attentivefp_embed | family_gap          |         0.12857 |    -0.01196 |
| attentivefp_embed | logp                |         0.04519 |    -0.00189 |
| attentivefp_embed | member_range        |         0.02918 |     0.00207 |
| attentivefp_embed | abs_family_gap      |         0.02496 |    -0.00224 |
| attentivefp_embed | member_std          |         0.02283 |    -0.00245 |
| attentivefp_embed | tpsa                |         0.02026 |     0.00058 |
| attentivefp_embed | fractioncsp3        |         0.01592 |     0.00042 |
| attentivefp_embed | pec50               |         0.01512 |    -0.00249 |
| attentivefp_embed | num_rotatable_bonds |         0.01450 |     0.00102 |
| attentivefp_embed | log2fc_8_25e_6      |         0.01378 |    -0.00005 |

### `boltz_allpairs`

| model          | feature              |   mean_abs_shap |   mean_shap |
|:---------------|:---------------------|----------------:|------------:|
| boltz_allpairs | family_gap           |         0.18670 |    -0.01497 |
| boltz_allpairs | abs_family_gap       |         0.03779 |     0.00148 |
| boltz_allpairs | member_std           |         0.02953 |    -0.00597 |
| boltz_allpairs | exactmw              |         0.02915 |    -0.00550 |
| boltz_allpairs | member_range         |         0.02644 |     0.00438 |
| boltz_allpairs | logp                 |         0.01864 |    -0.00019 |
| boltz_allpairs | num_heavy_atoms      |         0.01514 |    -0.00336 |
| boltz_allpairs | log2fc_8_25e_6       |         0.01388 |    -0.00152 |
| boltz_allpairs | pec50                |         0.01092 |    -0.00119 |
| boltz_allpairs | nn_potent46_tanimoto |         0.00968 |    -0.00262 |

### `boltz_core`

| model      | feature              |   mean_abs_shap |   mean_shap |
|:-----------|:---------------------|----------------:|------------:|
| boltz_core | family_gap           |         0.18134 |    -0.01167 |
| boltz_core | abs_family_gap       |         0.03884 |    -0.00070 |
| boltz_core | exactmw              |         0.03366 |    -0.00320 |
| boltz_core | member_std           |         0.02921 |    -0.00704 |
| boltz_core | member_range         |         0.02528 |     0.00453 |
| boltz_core | logp                 |         0.01902 |    -0.00017 |
| boltz_core | log2fc_8_25e_6       |         0.01818 |    -0.00017 |
| boltz_core | num_heavy_atoms      |         0.01620 |    -0.00395 |
| boltz_core | pec50                |         0.01436 |    -0.00098 |
| boltz_core | nn_potent46_tanimoto |         0.01217 |    -0.00436 |
