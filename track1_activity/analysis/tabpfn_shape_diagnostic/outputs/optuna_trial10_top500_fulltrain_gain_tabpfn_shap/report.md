# Optuna trial10 full-train top500 gain + TabPFN SHAP

Feature set: `cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens`
SHAP: `SV`, baseline imputer, budget `512`, 12 test compounds.

## LGBM Family Gain

| family           |   n_selected |   total_gain |   gain_share_pct |
|:-----------------|-------------:|-------------:|-----------------:|
| log2fc_pred      |            2 |  39846.34012 |          0.79694 |
| mordred          |          246 |   4749.08685 |          0.09498 |
| chemeleon        |          196 |   4121.32273 |          0.08243 |
| boltz_tier0      |           13 |    460.42025 |          0.00921 |
| boltz_tier1_conf |           25 |    392.44720 |          0.00785 |
| rdkit_full       |           14 |    296.76035 |          0.00594 |
| pose_jazzy       |            4 |    132.60924 |          0.00265 |

## TabPFN SHAP Family Summary

| family           |   total_abs_shap |   mean_abs_shap_per_feature |   median_abs_shap_per_feature |   mean_signed_shap |   n_selected |   total_gain |   share_abs_shap |
|:-----------------|-----------------:|----------------------------:|------------------------------:|-------------------:|-------------:|-------------:|-----------------:|
| mordred          |         18.17637 |                     0.07389 |                       0.07011 |           -0.00022 |          246 |   4749.08685 |          0.49505 |
| chemeleon        |         13.44216 |                     0.06858 |                       0.06294 |            0.00303 |          196 |   4121.32273 |          0.36611 |
| boltz_tier1_conf |          2.03317 |                     0.08133 |                       0.07082 |           -0.00342 |           25 |    392.44720 |          0.05538 |
| rdkit_full       |          0.95825 |                     0.06845 |                       0.06957 |            0.00294 |           14 |    296.76035 |          0.02610 |
| boltz_tier0      |          0.93469 |                     0.07190 |                       0.05650 |           -0.03627 |           13 |    460.42025 |          0.02546 |
| log2fc_pred      |          0.90790 |                     0.45395 |                       0.45395 |           -0.02680 |            2 |  39846.34012 |          0.02473 |
| pose_jazzy       |          0.26356 |                     0.06589 |                       0.06366 |           -0.02626 |            4 |    132.60924 |          0.00718 |

## Top TabPFN SHAP Features

|   global_feature_idx | feature                                  | family           |   mean_abs_shap |   mean_shap |   max_abs_shap |   n_explanations |   lgbm_gain |
|---------------------:|:-----------------------------------------|:-----------------|----------------:|------------:|---------------:|-----------------:|------------:|
|                 2101 | log2fc_pred__log2fc_8p25_pred            | log2fc_pred      |         0.72070 |    -0.04181 |        1.83428 |               12 | 37727.55025 |
|                 2082 | boltz_tier1__pde_pocket_ligand_min       | boltz_tier1_conf |         0.21579 |     0.18818 |        2.16399 |               12 |    33.97714 |
|                  865 | mordred__GATS6d                          | mordred          |         0.19408 |    -0.17579 |        1.26854 |               12 |    34.68611 |
|                 2102 | log2fc_pred__log2fc_33_pred              | log2fc_pred      |         0.18720 |    -0.01180 |        0.66541 |               12 |  2118.78988 |
|                  535 | mordred__MPC10                           | mordred          |         0.18248 |     0.13243 |        1.45591 |               12 |    48.24869 |
|                  193 | chemeleon_193                            | chemeleon        |         0.17069 |     0.15395 |        1.16231 |               12 |    21.09019 |
|                  946 | mordred__MATS8i                          | mordred          |         0.17022 |    -0.06299 |        1.32874 |               12 |     9.79749 |
|                  854 | mordred__GATS4v                          | mordred          |         0.16535 |     0.14279 |        1.02653 |               12 |    14.94134 |
|                  372 | mordred__JGI3                            | mordred          |         0.16374 |    -0.07605 |        1.21877 |               12 |    10.61345 |
|                 2005 | rdkit__FpDensityMorgan3                  | rdkit_full       |         0.16219 |    -0.13354 |        1.53953 |               12 |    18.47862 |
|                  102 | chemeleon_102                            | chemeleon        |         0.15522 |     0.06842 |        1.30144 |               12 |     8.74137 |
|                  160 | chemeleon_160                            | chemeleon        |         0.15515 |     0.10548 |        1.22179 |               12 |    21.75472 |
|                  682 | mordred__AATS4v                          | mordred          |         0.15133 |    -0.02333 |        0.81074 |               12 |    18.30372 |
|                  832 | mordred__GATS2c                          | mordred          |         0.15131 |    -0.10693 |        1.45813 |               12 |     8.91398 |
|                   81 | chemeleon_081                            | chemeleon        |         0.15040 |     0.14266 |        1.04580 |               12 |    11.56549 |
|                  140 | chemeleon_140                            | chemeleon        |         0.14807 |     0.12495 |        0.76469 |               12 |    20.69809 |
|                  808 | mordred__ATSC8c                          | mordred          |         0.14765 |    -0.02086 |        0.75597 |               12 |     9.13971 |
|                 2095 | boltz_tier1__plddt_res288                | boltz_tier1_conf |         0.14701 |    -0.01971 |        0.98423 |               12 |    14.35319 |
|                  798 | mordred__ATSC6v                          | mordred          |         0.14231 |    -0.11936 |        1.16285 |               12 |     9.93298 |
|                  276 | chemeleon_276                            | chemeleon        |         0.14230 |     0.12155 |        1.20092 |               12 |    17.35143 |
|                  791 | mordred__ATSC6Z                          | mordred          |         0.14143 |    -0.12512 |        0.98529 |               12 |     9.24603 |
|                  797 | mordred__ATSC6s                          | mordred          |         0.14015 |     0.11632 |        1.21844 |               12 |    33.75130 |
|                 1186 | mordred__GATS3dv                         | mordred          |         0.13991 |    -0.13015 |        1.01728 |               12 |    19.49472 |
|                  126 | chemeleon_126                            | chemeleon        |         0.13727 |     0.10687 |        1.28130 |               12 |    14.74144 |
|                 1500 | mordred__VR2_Dzse                        | mordred          |         0.13660 |    -0.12655 |        1.02007 |               12 |    25.85989 |
|                   49 | chemeleon_049                            | chemeleon        |         0.13457 |    -0.08737 |        0.83587 |               12 |    14.64007 |
|                  866 | mordred__GATS6i                          | mordred          |         0.13428 |     0.09829 |        1.36258 |               12 |    13.89144 |
|                  222 | chemeleon_222                            | chemeleon        |         0.13401 |    -0.03636 |        0.92562 |               12 |    26.35125 |
|                 2038 | boltz_tier0__affinity_pred_value         | boltz_tier0      |         0.13211 |     0.04040 |        0.84936 |               12 |    41.82944 |
|                  206 | chemeleon_206                            | chemeleon        |         0.13205 |    -0.12426 |        1.15695 |               12 |     8.79036 |
|                 2041 | boltz_tier0__affinity_probability_binary | boltz_tier0      |         0.13166 |    -0.08751 |        1.09681 |               12 |    40.05876 |
|                 1268 | mordred__SM1_Dzv                         | mordred          |         0.13113 |    -0.06337 |        0.83398 |               12 |    14.55295 |
|                  165 | chemeleon_165                            | chemeleon        |         0.13078 |     0.08574 |        0.57349 |               12 |    17.47197 |
|                  129 | chemeleon_129                            | chemeleon        |         0.12999 |     0.02742 |        0.61579 |               12 |    29.35002 |
|                 1758 | mordred__ETA_dEpsilon_C                  | mordred          |         0.12984 |     0.11283 |        0.82775 |               12 |     9.70459 |
|                 1052 | mordred__AATS6dv                         | mordred          |         0.12967 |     0.12433 |        0.70962 |               12 |    20.87644 |
|                 1654 | mordred__AETA_eta_L                      | mordred          |         0.12802 |     0.11705 |        0.52331 |               12 |    41.50379 |
|                 2073 | boltz_tier1__pae_protein_ligand_std      | boltz_tier1_conf |         0.12639 |     0.06412 |        1.08113 |               12 |     9.24711 |
|                 1467 | mordred__RotRatio                        | mordred          |         0.12516 |     0.02551 |        0.53549 |               12 |    13.08936 |
|                 2079 | boltz_tier1__pae_pocket_ligand_max       | boltz_tier1_conf |         0.12349 |     0.03984 |        0.67622 |               12 |    26.00008 |
