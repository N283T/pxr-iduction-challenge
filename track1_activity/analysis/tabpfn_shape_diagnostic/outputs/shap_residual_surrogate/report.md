# SHAP Residual Surrogate

LightGBM surrogates are trained on low-dimensional assay-shape, proximity,
chemistry, and ensemble-disagreement features, then explained with Tree SHAP.

## Surrogate OOF Metrics

| target                          |   target_std |   oof_mae |   oof_r2 |   oof_spearman |
|:--------------------------------|-------------:|----------:|---------:|---------------:|
| top500_residual                 |      0.55233 |   0.22786 |  0.65178 |        0.79207 |
| top500_abs_error_minus_ensemble |      0.12160 |   0.07677 |  0.19418 |        0.37189 |
| top500_pred_minus_ensemble      |      0.12652 |   0.06892 |  0.41591 |        0.60409 |

## Top SHAP Features: `top500_residual`

| target          | feature                  |   mean_abs_shap |   mean_shap |   feature_mean |   feature_missing_frac |
|:----------------|:-------------------------|----------------:|------------:|---------------:|-----------------------:|
| top500_residual | pec50                    |         0.49539 |     0.10125 |        4.39294 |                0.00000 |
| top500_residual | family_gap               |         0.13970 |    -0.01285 |        0.00290 |                0.00000 |
| top500_residual | log2fc_8_25e_6           |         0.11177 |    -0.01638 |        0.83436 |                0.41500 |
| top500_residual | logp                     |         0.09818 |     0.00507 |        2.83778 |                0.00000 |
| top500_residual | exactmw                  |         0.06973 |    -0.00369 |      341.41968 |                0.00000 |
| top500_residual | member_std               |         0.05042 |    -0.00065 |        0.21165 |                0.00000 |
| top500_residual | fractioncsp3             |         0.04132 |     0.00464 |        0.37169 |                0.00000 |
| top500_residual | num_heavy_atoms          |         0.03507 |    -0.00337 |       24.14600 |                0.00000 |
| top500_residual | hba                      |         0.02883 |     0.00161 |        5.53000 |                0.00000 |
| top500_residual | member_range             |         0.02865 |     0.00413 |        0.67786 |                0.00000 |
| top500_residual | nn_potent46_tanimoto     |         0.02592 |    -0.00689 |        0.23195 |                0.00000 |
| top500_residual | hbd                      |         0.01767 |    -0.00201 |        1.23600 |                0.00000 |
| top500_residual | log2fc_3_30e_5           |         0.01756 |     0.00340 |        1.30495 |                0.42300 |
| top500_residual | tpsa                     |         0.01677 |    -0.00045 |       69.02308 |                0.00000 |
| top500_residual | counter_pec50            |         0.01475 |    -0.00199 |        3.11023 |                0.36100 |
| top500_residual | num_heteroatoms          |         0.01332 |    -0.00203 |        6.45700 |                0.00000 |
| top500_residual | counter_emax_vs_pos_ctrl |         0.01302 |    -0.00346 |        0.84889 |                0.36100 |
| top500_residual | num_rotatable_bonds      |         0.01241 |    -0.00075 |        4.46100 |                0.00000 |
| top500_residual | abs_family_gap           |         0.01058 |     0.00322 |        0.17911 |                0.00000 |
| top500_residual | counter_emax             |         0.01002 |     0.00141 |        1.55272 |                0.36100 |

## Top SHAP Features: `top500_abs_error_minus_ensemble`

| target                          | feature                  |   mean_abs_shap |   mean_shap |   feature_mean |   feature_missing_frac |
|:--------------------------------|:-------------------------|----------------:|------------:|---------------:|-----------------------:|
| top500_abs_error_minus_ensemble | pec50                    |         0.03081 |     0.00437 |        4.39294 |                0.00000 |
| top500_abs_error_minus_ensemble | family_gap               |         0.01218 |    -0.00115 |        0.00290 |                0.00000 |
| top500_abs_error_minus_ensemble | member_std               |         0.00813 |    -0.00026 |        0.21165 |                0.00000 |
| top500_abs_error_minus_ensemble | log2fc_8_25e_6           |         0.00681 |     0.00109 |        0.83436 |                0.41500 |
| top500_abs_error_minus_ensemble | member_range             |         0.00670 |     0.00157 |        0.67786 |                0.00000 |
| top500_abs_error_minus_ensemble | logp                     |         0.00597 |     0.00058 |        2.83778 |                0.00000 |
| top500_abs_error_minus_ensemble | exactmw                  |         0.00591 |    -0.00062 |      341.41968 |                0.00000 |
| top500_abs_error_minus_ensemble | log2fc_3_30e_5           |         0.00568 |     0.00064 |        1.30495 |                0.42300 |
| top500_abs_error_minus_ensemble | tpsa                     |         0.00535 |    -0.00040 |       69.02308 |                0.00000 |
| top500_abs_error_minus_ensemble | abs_family_gap           |         0.00503 |    -0.00022 |        0.17911 |                0.00000 |
| top500_abs_error_minus_ensemble | counter_emax_vs_pos_ctrl |         0.00444 |    -0.00006 |        0.84889 |                0.36100 |
| top500_abs_error_minus_ensemble | nn_potent46_tanimoto     |         0.00434 |    -0.00015 |        0.23195 |                0.00000 |
| top500_abs_error_minus_ensemble | counter_emax             |         0.00407 |     0.00015 |        1.55272 |                0.36100 |
| top500_abs_error_minus_ensemble | fractioncsp3             |         0.00401 |    -0.00095 |        0.37169 |                0.00000 |
| top500_abs_error_minus_ensemble | num_heavy_atoms          |         0.00301 |     0.00017 |       24.14600 |                0.00000 |
| top500_abs_error_minus_ensemble | counter_pec50            |         0.00282 |     0.00058 |        3.11023 |                0.36100 |
| top500_abs_error_minus_ensemble | num_heteroatoms          |         0.00236 |     0.00068 |        6.45700 |                0.00000 |
| top500_abs_error_minus_ensemble | num_rotatable_bonds      |         0.00224 |    -0.00042 |        4.46100 |                0.00000 |
| top500_abs_error_minus_ensemble | num_rings                |         0.00166 |     0.00011 |        3.08700 |                0.00000 |
| top500_abs_error_minus_ensemble | hba                      |         0.00153 |    -0.00006 |        5.53000 |                0.00000 |

## Top SHAP Features: `top500_pred_minus_ensemble`

| target                     | feature                  |   mean_abs_shap |   mean_shap |   feature_mean |   feature_missing_frac |
|:---------------------------|:-------------------------|----------------:|------------:|---------------:|-----------------------:|
| top500_pred_minus_ensemble | family_gap               |         0.05132 |     0.00524 |        0.00290 |                0.00000 |
| top500_pred_minus_ensemble | pec50                    |         0.01017 |     0.00224 |        4.39294 |                0.00000 |
| top500_pred_minus_ensemble | log2fc_8_25e_6           |         0.01003 |     0.00335 |        0.83436 |                0.41500 |
| top500_pred_minus_ensemble | abs_family_gap           |         0.00903 |    -0.00014 |        0.17911 |                0.00000 |
| top500_pred_minus_ensemble | member_range             |         0.00806 |    -0.00016 |        0.67786 |                0.00000 |
| top500_pred_minus_ensemble | log2fc_3_30e_5           |         0.00794 |    -0.00114 |        1.30495 |                0.42300 |
| top500_pred_minus_ensemble | fractioncsp3             |         0.00775 |     0.00084 |        0.37169 |                0.00000 |
| top500_pred_minus_ensemble | exactmw                  |         0.00698 |     0.00049 |      341.41968 |                0.00000 |
| top500_pred_minus_ensemble | nn_potent46_tanimoto     |         0.00591 |     0.00078 |        0.23195 |                0.00000 |
| top500_pred_minus_ensemble | member_std               |         0.00588 |    -0.00039 |        0.21165 |                0.00000 |
| top500_pred_minus_ensemble | logp                     |         0.00574 |    -0.00026 |        2.83778 |                0.00000 |
| top500_pred_minus_ensemble | tpsa                     |         0.00504 |    -0.00032 |       69.02308 |                0.00000 |
| top500_pred_minus_ensemble | num_rotatable_bonds      |         0.00484 |    -0.00045 |        4.46100 |                0.00000 |
| top500_pred_minus_ensemble | counter_emax_vs_pos_ctrl |         0.00471 |    -0.00005 |        0.84889 |                0.36100 |
| top500_pred_minus_ensemble | num_heteroatoms          |         0.00408 |    -0.00038 |        6.45700 |                0.00000 |
| top500_pred_minus_ensemble | counter_pec50            |         0.00328 |     0.00039 |        3.11023 |                0.36100 |
| top500_pred_minus_ensemble | num_heavy_atoms          |         0.00273 |     0.00062 |       24.14600 |                0.00000 |
| top500_pred_minus_ensemble | counter_emax             |         0.00265 |    -0.00035 |        1.55272 |                0.36100 |
| top500_pred_minus_ensemble | hbd                      |         0.00259 |     0.00026 |        1.23600 |                0.00000 |
| top500_pred_minus_ensemble | num_rings                |         0.00172 |     0.00012 |        3.08700 |                0.00000 |
