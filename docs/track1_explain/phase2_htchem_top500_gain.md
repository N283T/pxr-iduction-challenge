# Phase 2 HTChem top500 gain probe

Purpose: check whether `pred_htchem` should be considered as an extra scalar in the existing LGBM-gain top500 feature selection path.

## Target Feature Ranks

| feature          | family      |   selected_folds |   mean_rank |   min_rank |   max_rank |   mean_gain_share_pct |   mean_split |   full_rank | full_selected   |   full_gain_share_pct |   full_split |
|:-----------------|:------------|-----------------:|------------:|-----------:|-----------:|----------------------:|-------------:|------------:|:----------------|----------------------:|-------------:|
| log2fc_8p25_pred | log2fc_pred |                5 |      1.2000 |          1 |          2 |               51.2720 |     257.6000 |           1 | True            |               58.3648 |          260 |
| log2fc_33_pred   | log2fc_pred |                5 |      1.8000 |          1 |          2 |               22.2984 |     173.4000 |           2 | True            |               15.1563 |          195 |
| pred_htchem      | pred_htchem |                5 |     11.0000 |          6 |         17 |                0.1765 |      54.6000 |          10 | True            |                0.1792 |           57 |

## Selected Top500 Family Gain

| fold   | family      |   n_selected |   gain_sum |   gain_share_pct |   split_sum |
|:-------|:------------|-------------:|-----------:|-----------------:|------------:|
| fold0  | log2fc_pred |            2 | 31276.6109 |          74.0271 |         401 |
| fold0  | 2d_boltz    |          304 |  5025.9703 |          11.8957 |        7972 |
| fold0  | chemeleon   |          193 |  3361.0196 |           7.9550 |        8266 |
| fold0  | pred_htchem |            1 |    57.3806 |           0.1358 |          59 |
| fold1  | log2fc_pred |            2 | 29760.9153 |          71.9356 |         436 |
| fold1  | 2d_boltz    |          290 |  4944.7656 |          11.9521 |        7589 |
| fold1  | chemeleon   |          207 |  3936.2667 |           9.5144 |        8786 |
| fold1  | pred_htchem |            1 |    55.4627 |           0.1341 |          60 |
| fold2  | log2fc_pred |            2 | 33109.6755 |          74.8575 |         450 |
| fold2  | 2d_boltz    |          295 |  4540.6513 |          10.2659 |        7584 |
| fold2  | chemeleon   |          202 |  4084.9575 |           9.2357 |        8795 |
| fold2  | pred_htchem |            1 |   102.4117 |           0.2315 |          58 |
| fold3  | log2fc_pred |            2 | 32797.8964 |          73.3677 |         420 |
| fold3  | 2d_boltz    |          300 |  5168.7967 |          11.5624 |        7712 |
| fold3  | chemeleon   |          197 |  3962.1753 |           8.8632 |        8438 |
| fold3  | pred_htchem |            1 |    85.1646 |           0.1905 |          49 |
| fold4  | log2fc_pred |            2 | 30183.7506 |          73.6643 |         448 |
| fold4  | 2d_boltz    |          297 |  4489.8830 |          10.9577 |        7763 |
| fold4  | chemeleon   |          200 |  3669.6296 |           8.9558 |        8467 |
| fold4  | pred_htchem |            1 |    78.1500 |           0.1907 |          47 |
| full   | log2fc_pred |            2 | 39261.6220 |          73.5211 |         455 |
| full   | 2d_boltz    |          292 |  5585.9843 |          10.4603 |        7637 |
| full   | chemeleon   |          205 |  4893.4247 |           9.1634 |        8778 |
| full   | pred_htchem |            1 |    95.6848 |           0.1792 |          57 |

## Read

If `pred_htchem` repeatedly lands inside top500 with non-trivial gain, the next step is a proper TabPFN top500 SWAP-style run. If it is outside top500 or only barely selected, it is better kept as a diagnostic/map axis rather than promoted into the high-weight top500 member.
