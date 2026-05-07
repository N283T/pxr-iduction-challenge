# Top500 Raw Feature Audit

Feature set: `cheme_2d_full_boltz_log2fc_pred_seed10ens`
Raw feature count: `2103`
Selection: per-fold LGBM gain top-500, matching the top500 TabPFN pipeline.

## Family Composition

| family           |   selected_mean |   selected_min |   selected_max |   gain_mean |   gain_share_pct |
|:-----------------|----------------:|---------------:|---------------:|------------:|-----------------:|
| log2fc_pred      |           2.000 |              2 |              2 |   31463.925 |           78.354 |
| mordred          |         248.800 |            244 |            252 |    3791.123 |            9.441 |
| chemeleon        |         197.400 |            193 |            201 |    3780.474 |            9.414 |
| boltz_tier1_conf |          22.000 |             20 |             23 |     344.774 |            0.859 |
| boltz_tier0      |          13.000 |             11 |             14 |     334.729 |            0.834 |
| rdkit_full       |          13.600 |             10 |             18 |     299.375 |            0.746 |
| pose_jazzy       |           3.200 |              2 |              5 |     141.644 |            0.353 |

## Top Features By Mean Gain

| feature                                    | family           |   selected_folds |   gain_mean |   gain_share_pct |   gain_nonzero_folds |
|:-------------------------------------------|:-----------------|-----------------:|------------:|-----------------:|---------------------:|
| log2fc_pred__log2fc_8p25_pred              | log2fc_pred      |                5 |  22029.7555 |          51.5846 |                    5 |
| log2fc_pred__log2fc_33_pred                | log2fc_pred      |                5 |   9434.1691 |          22.0909 |                    5 |
| chemeleon_067                              | chemeleon        |                5 |    129.8773 |           0.3041 |                    5 |
| chemeleon_006                              | chemeleon        |                5 |     97.6304 |           0.2286 |                    5 |
| chemeleon_175                              | chemeleon        |                5 |     86.7210 |           0.2031 |                    5 |
| chemeleon_131                              | chemeleon        |                4 |     80.8804 |           0.1894 |                    5 |
| mordred__SLogP                             | mordred          |                5 |     77.8422 |           0.1823 |                    5 |
| pose_jazzy__sa                             | pose_jazzy       |                4 |     69.5378 |           0.1628 |                    5 |
| chemeleon_002                              | chemeleon        |                5 |     67.1396 |           0.1572 |                    5 |
| rdkit__qed                                 | rdkit_full       |                5 |     55.9564 |           0.1310 |                    5 |
| pose_jazzy__dgtot                          | pose_jazzy       |                5 |     50.4909 |           0.1182 |                    5 |
| chemeleon_240                              | chemeleon        |                5 |     50.0746 |           0.1173 |                    5 |
| chemeleon_055                              | chemeleon        |                5 |     49.2420 |           0.1153 |                    5 |
| chemeleon_092                              | chemeleon        |                5 |     43.4903 |           0.1018 |                    5 |
| rdkit__BCUT2D_MRLOW                        | rdkit_full       |                5 |     42.8603 |           0.1004 |                    5 |
| chemeleon_173                              | chemeleon        |                5 |     42.0422 |           0.0984 |                    5 |
| mordred__TopoPSA(NO)                       | mordred          |                5 |     39.2669 |           0.0919 |                    5 |
| mordred__MINssCH2                          | mordred          |                5 |     39.0965 |           0.0915 |                    5 |
| chemeleon_138                              | chemeleon        |                4 |     38.4587 |           0.0901 |                    5 |
| boltz_tier0__ensemble_diff_prob            | boltz_tier0      |                5 |     38.3521 |           0.0898 |                    5 |
| chemeleon_298                              | chemeleon        |                5 |     37.7530 |           0.0884 |                    5 |
| boltz_tier0__affinity_probability_binary_1 | boltz_tier0      |                5 |     37.2966 |           0.0873 |                    5 |
| boltz_tier0__ligand_to_pocket_distance_a   | boltz_tier0      |                5 |     36.9308 |           0.0865 |                    5 |
| chemeleon_231                              | chemeleon        |                5 |     36.2619 |           0.0849 |                    5 |
| mordred__TopoPSA                           | mordred          |                5 |     35.4722 |           0.0831 |                    5 |
| mordred__ATSC1p                            | mordred          |                5 |     34.6813 |           0.0812 |                    5 |
| chemeleon_046                              | chemeleon        |                5 |     33.9905 |           0.0796 |                    5 |
| mordred__MINaaCH                           | mordred          |                5 |     33.5436 |           0.0785 |                    5 |
| chemeleon_069                              | chemeleon        |                5 |     32.5197 |           0.0761 |                    5 |
| boltz_tier0__affinity_probability_binary   | boltz_tier0      |                5 |     32.1901 |           0.0754 |                    5 |
| mordred__AATS5v                            | mordred          |                4 |     31.9803 |           0.0749 |                    5 |
| chemeleon_090                              | chemeleon        |                5 |     31.8005 |           0.0745 |                    5 |
| rdkit__FpDensityMorgan1                    | rdkit_full       |                4 |     30.6722 |           0.0718 |                    5 |
| chemeleon_143                              | chemeleon        |                5 |     30.3670 |           0.0711 |                    5 |
| chemeleon_151                              | chemeleon        |                2 |     30.2892 |           0.0709 |                    5 |
| mordred__ATSC8d                            | mordred          |                5 |     30.2007 |           0.0707 |                    5 |
| chemeleon_213                              | chemeleon        |                5 |     29.2966 |           0.0686 |                    5 |
| chemeleon_083                              | chemeleon        |                5 |     29.2579 |           0.0685 |                    5 |
| mordred__ZMIC3                             | mordred          |                5 |     29.1521 |           0.0683 |                    5 |
| boltz_tier1__plddt_ligand_min              | boltz_tier1_conf |                5 |     28.9188 |           0.0677 |                    5 |

## Stable Selected Features

Selected in all 5 folds: `103`

| feature                                    | family           |   gain_mean |   gain_share_pct |   gain_nonzero_folds |
|:-------------------------------------------|:-----------------|------------:|-----------------:|---------------------:|
| log2fc_pred__log2fc_8p25_pred              | log2fc_pred      |  22029.7555 |          51.5846 |                    5 |
| log2fc_pred__log2fc_33_pred                | log2fc_pred      |   9434.1691 |          22.0909 |                    5 |
| chemeleon_067                              | chemeleon        |    129.8773 |           0.3041 |                    5 |
| chemeleon_006                              | chemeleon        |     97.6304 |           0.2286 |                    5 |
| chemeleon_175                              | chemeleon        |     86.7210 |           0.2031 |                    5 |
| mordred__SLogP                             | mordred          |     77.8422 |           0.1823 |                    5 |
| chemeleon_002                              | chemeleon        |     67.1396 |           0.1572 |                    5 |
| rdkit__qed                                 | rdkit_full       |     55.9564 |           0.1310 |                    5 |
| pose_jazzy__dgtot                          | pose_jazzy       |     50.4909 |           0.1182 |                    5 |
| chemeleon_240                              | chemeleon        |     50.0746 |           0.1173 |                    5 |
| chemeleon_055                              | chemeleon        |     49.2420 |           0.1153 |                    5 |
| chemeleon_092                              | chemeleon        |     43.4903 |           0.1018 |                    5 |
| rdkit__BCUT2D_MRLOW                        | rdkit_full       |     42.8603 |           0.1004 |                    5 |
| chemeleon_173                              | chemeleon        |     42.0422 |           0.0984 |                    5 |
| mordred__TopoPSA(NO)                       | mordred          |     39.2669 |           0.0919 |                    5 |
| mordred__MINssCH2                          | mordred          |     39.0965 |           0.0915 |                    5 |
| boltz_tier0__ensemble_diff_prob            | boltz_tier0      |     38.3521 |           0.0898 |                    5 |
| chemeleon_298                              | chemeleon        |     37.7530 |           0.0884 |                    5 |
| boltz_tier0__affinity_probability_binary_1 | boltz_tier0      |     37.2966 |           0.0873 |                    5 |
| boltz_tier0__ligand_to_pocket_distance_a   | boltz_tier0      |     36.9308 |           0.0865 |                    5 |
| chemeleon_231                              | chemeleon        |     36.2619 |           0.0849 |                    5 |
| mordred__TopoPSA                           | mordred          |     35.4722 |           0.0831 |                    5 |
| mordred__ATSC1p                            | mordred          |     34.6813 |           0.0812 |                    5 |
| chemeleon_046                              | chemeleon        |     33.9905 |           0.0796 |                    5 |
| mordred__MINaaCH                           | mordred          |     33.5436 |           0.0785 |                    5 |
| chemeleon_069                              | chemeleon        |     32.5197 |           0.0761 |                    5 |
| boltz_tier0__affinity_probability_binary   | boltz_tier0      |     32.1901 |           0.0754 |                    5 |
| chemeleon_090                              | chemeleon        |     31.8005 |           0.0745 |                    5 |
| chemeleon_143                              | chemeleon        |     30.3670 |           0.0711 |                    5 |
| mordred__ATSC8d                            | mordred          |     30.2007 |           0.0707 |                    5 |
| chemeleon_213                              | chemeleon        |     29.2966 |           0.0686 |                    5 |
| chemeleon_083                              | chemeleon        |     29.2579 |           0.0685 |                    5 |
| mordred__ZMIC3                             | mordred          |     29.1521 |           0.0683 |                    5 |
| boltz_tier1__plddt_ligand_min              | boltz_tier1_conf |     28.9188 |           0.0677 |                    5 |
| chemeleon_130                              | chemeleon        |     28.8904 |           0.0676 |                    5 |
| chemeleon_066                              | chemeleon        |     28.1250 |           0.0659 |                    5 |
| boltz_tier0__affinity_pred_value_2         | boltz_tier0      |     28.0355 |           0.0656 |                    5 |
| chemeleon_099                              | chemeleon        |     28.0249 |           0.0656 |                    5 |
| boltz_tier1__pae_protein_ligand_max        | boltz_tier1_conf |     27.8124 |           0.0651 |                    5 |
| boltz_tier0__complex_ipde                  | boltz_tier0      |     27.3873 |           0.0641 |                    5 |
| chemeleon_070                              | chemeleon        |     27.2827 |           0.0639 |                    5 |
| rdkit__FpDensityMorgan2                    | rdkit_full       |     26.7999 |           0.0628 |                    5 |
| mordred__VSA_EState8                       | mordred          |     26.6638 |           0.0624 |                    5 |
| chemeleon_003                              | chemeleon        |     25.3799 |           0.0594 |                    5 |
| chemeleon_135                              | chemeleon        |     24.6945 |           0.0578 |                    5 |
| chemeleon_186                              | chemeleon        |     24.4086 |           0.0572 |                    5 |
| mordred__BIC2                              | mordred          |     24.4080 |           0.0572 |                    5 |
| chemeleon_082                              | chemeleon        |     24.1169 |           0.0565 |                    5 |
| chemeleon_127                              | chemeleon        |     23.5443 |           0.0551 |                    5 |
| rdkit__FpDensityMorgan3                    | rdkit_full       |     23.3663 |           0.0547 |                    5 |
| boltz_tier1__pde_pocket_ligand_min         | boltz_tier1_conf |     23.2974 |           0.0546 |                    5 |
| chemeleon_032                              | chemeleon        |     22.7617 |           0.0533 |                    5 |
| chemeleon_124                              | chemeleon        |     22.3770 |           0.0524 |                    5 |
| mordred__SMR_VSA5                          | mordred          |     21.7462 |           0.0509 |                    5 |
| chemeleon_193                              | chemeleon        |     21.2927 |           0.0499 |                    5 |
| mordred__GATS5s                            | mordred          |     21.0383 |           0.0493 |                    5 |
| chemeleon_129                              | chemeleon        |     19.6367 |           0.0460 |                    5 |
| chemeleon_163                              | chemeleon        |     19.4227 |           0.0455 |                    5 |
| boltz_tier1__pae_pocket_ligand_mean        | boltz_tier1_conf |     19.1101 |           0.0447 |                    5 |
| rdkit__BCUT2D_CHGLO                        | rdkit_full       |     18.8151 |           0.0441 |                    5 |
