# TabPFN Shape Diagnostic

## Models

- `ensemble`: `ens_caruana_bag20`
- `tabpfn_2d_seed10_top500`: `tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap`
- `tabpfn_2d_seed10_default`: `tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_umap_default`

## Overall OOF

| model                    |     mae |   spearman |   mean_residual |   pred_mean |   pred_std |   delta_mae_vs_ensemble |   delta_spearman_vs_ensemble |
|:-------------------------|--------:|-----------:|----------------:|------------:|-----------:|------------------------:|-----------------------------:|
| ensemble                 | 0.39451 |    0.84781 |        -0.01221 |     4.33303 |    0.93535 |                 0.00000 |                      0.00000 |
| tabpfn_2d_seed10_top500  | 0.39676 |    0.84846 |        -0.00458 |     4.32540 |    0.98213 |                 0.00225 |                      0.00064 |
| tabpfn_2d_seed10_default | 0.40559 |    0.84009 |        -0.01447 |     4.33528 |    0.93608 |                 0.01108 |                     -0.00772 |

## Top500 Worse Than Ensemble On True Slice

| slice                               |   n_true |   mae_true |   ensemble_mae_true |   delta_true_mae_vs_ensemble |   mean_residual_true |
|:------------------------------------|---------:|-----------:|--------------------:|-----------------------------:|---------------------:|
| no_aux                              |     1294 |    0.47610 |             0.45924 |                      0.01686 |             -0.06397 |
| high_logp_top10                     |      414 |    0.45287 |             0.44722 |                      0.00565 |             -0.04025 |
| chemprop_family_low_vs_non_bottom10 |      414 |    0.66078 |             0.65518 |                      0.00560 |              0.01251 |
| high_mw_top10                       |      414 |    0.46431 |             0.46596 |                     -0.00166 |             -0.01355 |
| high_tpsa_top10                     |      415 |    0.46867 |             0.47127 |                     -0.00260 |              0.00616 |
| single_hi_low                       |      216 |    0.60674 |             0.61028 |                     -0.00354 |             -0.09117 |
| has_single_conc_lo                  |     2321 |    0.31137 |             0.31518 |                     -0.00381 |              0.04469 |
| has_counter                         |     2648 |    0.36227 |             0.36609 |                     -0.00383 |              0.02681 |
| has_single_conc_hi                  |     2374 |    0.32385 |             0.32879 |                     -0.00494 |              0.03700 |
| single_lo_low                       |      108 |    0.61893 |             0.62663 |                     -0.00770 |             -0.07448 |
| chemprop_family_high_vs_non_top10   |      414 |    0.41857 |             0.43515 |                     -0.01658 |             -0.01528 |
| high_family_gap_top10               |      414 |    0.63784 |             0.65468 |                     -0.01685 |              0.02888 |

## Top500 Better Than Ensemble On True Slice

| slice                             |   n_true |   mae_true |   ensemble_mae_true |   delta_true_mae_vs_ensemble |   mean_residual_true |
|:----------------------------------|---------:|-----------:|--------------------:|-----------------------------:|---------------------:|
| near_potent46_t04                 |       73 |    0.64925 |             0.76395 |                     -0.11470 |              0.58295 |
| near_potent46_t03                 |      223 |    0.45729 |             0.49329 |                     -0.03601 |              0.19349 |
| counter_above_main                |      391 |    0.57596 |             0.61098 |                     -0.03502 |             -0.33974 |
| high_member_std_top10             |      414 |    0.63686 |             0.65881 |                     -0.02195 |             -0.00924 |
| high_family_gap_top10             |      414 |    0.63784 |             0.65468 |                     -0.01685 |              0.02888 |
| chemprop_family_high_vs_non_top10 |      414 |    0.41857 |             0.43515 |                     -0.01658 |             -0.01528 |
| single_lo_low                     |      108 |    0.61893 |             0.62663 |                     -0.00770 |             -0.07448 |
| has_single_conc_hi                |     2374 |    0.32385 |             0.32879 |                     -0.00494 |              0.03700 |
| has_counter                       |     2648 |    0.36227 |             0.36609 |                     -0.00383 |              0.02681 |
| has_single_conc_lo                |     2321 |    0.31137 |             0.31518 |                     -0.00381 |              0.04469 |
| single_hi_low                     |      216 |    0.60674 |             0.61028 |                     -0.00354 |             -0.09117 |
| high_tpsa_top10                   |      415 |    0.46867 |             0.47127 |                     -0.00260 |              0.00616 |

## Top500 Quantile Variables With Largest MAE Spread

| variable                 |   mae_min |   mae_max |   mae_spread |
|:-------------------------|----------:|----------:|-------------:|
| pec50                    |   0.27651 |   0.58254 |      0.30604 |
| member_std               |   0.31004 |   0.58681 |      0.27677 |
| member_range             |   0.31032 |   0.58037 |      0.27006 |
| log2fc_3_30e_5           |   0.27020 |   0.50571 |      0.23551 |
| family_gap               |   0.33387 |   0.55348 |      0.21960 |
| abs_family_gap           |   0.33373 |   0.54164 |      0.20791 |
| log2fc_8_25e_6           |   0.29001 |   0.49477 |      0.20476 |
| counter_emax             |   0.31717 |   0.45797 |      0.14080 |
| counter_pec50            |   0.31846 |   0.45797 |      0.13951 |
| tpsa                     |   0.33449 |   0.46814 |      0.13365 |
| counter_emax_vs_pos_ctrl |   0.32434 |   0.45797 |      0.13363 |
| num_heavy_atoms          |   0.35883 |   0.45320 |      0.09437 |
| nn_potent46_tanimoto     |   0.35442 |   0.44830 |      0.09388 |
| exactmw                  |   0.34812 |   0.43918 |      0.09106 |
| num_heteroatoms          |   0.37709 |   0.46415 |      0.08706 |

## Test Prediction Summary

| model                    |   n |   pred_mean |   pred_std |   pred_min |   pred_max |   near_potent46_t03_n |   near_potent46_t03_pred_mean |   near_potent46_t03_pred_std |   near_potent46_t04_n |   near_potent46_t04_pred_mean |   near_potent46_t04_pred_std |
|:-------------------------|----:|------------:|-----------:|-----------:|-----------:|----------------------:|------------------------------:|-----------------------------:|----------------------:|------------------------------:|-----------------------------:|
| ensemble                 | 513 |     4.73307 |    0.71243 |    2.63885 |    5.92337 |                   300 |                       4.77135 |                      0.70790 |                   278 |                       4.77012 |                      0.69956 |
| tabpfn_2d_seed10_top500  | 513 |     4.72504 |    0.77145 |    2.59117 |    6.07465 |                   300 |                       4.77440 |                      0.77040 |                   278 |                       4.77330 |                      0.76178 |
| tabpfn_2d_seed10_default | 513 |     4.72849 |    0.69931 |    2.68315 |    5.85358 |                   300 |                       4.77192 |                      0.68385 |                   278 |                       4.77609 |                      0.67222 |

## Read

- Positive residual means underprediction.
- Negative residual means overprediction.
- Slices where top500 is worse than the ensemble are candidates for a CSV correction against the high-weight 2D axis.
