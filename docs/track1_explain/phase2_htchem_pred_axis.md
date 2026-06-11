# Phase 2 HTChem pred-axis probe

Purpose: use HTChem as an auxiliary external activity axis (`pred_htchem`), not as equivalent Track 1 training labels.

## Inputs

- HTChem corrected pEC50 rows: 441 unique compounds.
- Frozen representation: `data/chemprop_pretrain_embed.parquet` for challenge compounds plus `data/chemprop_pretrain_htchem_embed.parquet` extracted with the same checkpoint.
- Predictor: 5-fold activity-stratified RidgeCV on HTChem corrected pEC50.
- Existing challenge context: optuna trial10 predicted log2fc and id55 AS1 replay.

## Coverage

| asset                          |   rows |   train_cover |   test_cover |   htchem_cover |
|:-------------------------------|-------:|--------------:|-------------:|---------------:|
| chemprop_pretrain_embed        |   4653 |          4140 |          513 |              1 |
| chemprop_pretrain_htchem_embed |    441 |             1 |            0 |            441 |

## HTChem Model Check

| slice                |   n |    mae |   bias_pred_minus_true |   spearman |   pearson |   pred_mean |   true_mean |   ridge_alpha_mean |   ridge_alpha_final |
|:---------------------|----:|-------:|-----------------------:|-----------:|----------:|------------:|------------:|-------------------:|--------------------:|
| htchem_all_oof       | 441 | 0.6420 |                 0.0015 |     0.6904 |    0.6447 |      4.8548 |      4.8533 |           600.6503 |           1000.0000 |
| htchem_crude_oof     | 347 | 0.6921 |                 0.0577 |     0.6467 |    0.5979 |      4.7046 |      4.6470 |           nan      |            nan      |
| htchem_semi_pure_oof |  94 | 0.4572 |                -0.2060 |     0.6094 |    0.5931 |      5.4091 |      5.6151 |           nan      |            nan      |

## Challenge/AS1 Checks

| slice                            |    n |    mae |   bias_pred_minus_true |   spearman |   pearson |   pred_mean |   true_mean |
|:---------------------------------|-----:|-------:|-----------------------:|-----------:|----------:|------------:|------------:|
| train_true_vs_pred_htchem        | 4140 | 0.5620 |                 0.1654 |     0.7271 |    0.7264 |      4.4862 |      4.3208 |
| as1_true_vs_pred_htchem          |  253 | 0.5472 |                 0.0672 |     0.7511 |    0.7218 |      4.7313 |      4.6641 |
| as1_true_bin_lt3_vs_pred_htchem  |   24 | 1.3483 |                 1.3304 |    -0.1596 |   -0.0492 |      3.6496 |      2.3192 |
| as1_true_bin_3to4_vs_pred_htchem |   31 | 0.6121 |                 0.3270 |     0.1472 |    0.1166 |      3.8915 |      3.5645 |
| as1_true_bin_4to5_vs_pred_htchem |   86 | 0.4316 |                -0.0293 |     0.2303 |    0.2453 |      4.6147 |      4.6440 |
| as1_true_bin_5to6_vs_pred_htchem |  102 | 0.4488 |                -0.1798 |     0.2253 |    0.2167 |      5.2378 |      5.4175 |
| as1_true_bin_gte6_vs_pred_htchem |   10 | 0.4214 |                -0.4214 |     0.4667 |    0.3196 |      5.7671 |      6.1885 |

## Correlations

| slice   | x                | target          |    n |   spearman |   pearson |
|:--------|:-----------------|:----------------|-----:|-----------:|----------:|
| htchem  | log2fc_8p25_pred | corrected_pec50 |  441 |     0.6533 |    0.5843 |
| htchem  | log2fc_33_pred   | corrected_pec50 |  441 |     0.6419 |    0.6045 |
| htchem  | lf_mean          | corrected_pec50 |  441 |     0.6537 |    0.6011 |
| htchem  | pred_htchem_oof  | corrected_pec50 |  441 |     0.6904 |    0.6447 |
| train   | pred_htchem      | true_pec50      | 4140 |     0.7271 |    0.7264 |
| train   | log2fc_8p25_pred | true_pec50      | 4140 |     0.8269 |    0.7599 |
| train   | log2fc_33_pred   | true_pec50      | 4140 |     0.7442 |    0.7746 |
| train   | lf_mean          | true_pec50      | 4140 |     0.7965 |    0.7858 |
| as1     | pred_htchem      | true_pec50      |  253 |     0.7511 |    0.7218 |
| as1     | log2fc_8p25_pred | true_pec50      |  253 |     0.7931 |    0.7160 |
| as1     | log2fc_33_pred   | true_pec50      |  253 |     0.7697 |    0.7591 |
| as1     | lf_mean          | true_pec50      |  253 |     0.7878 |    0.7492 |
| as1     | pred_id55        | true_pec50      |  253 |     0.8488 |    0.8278 |
| as1     | pred_htchem      | id55_error      |  253 |    -0.0761 |   -0.0965 |
| as1     | log2fc_8p25_pred | id55_error      |  253 |    -0.0523 |   -0.1032 |
| as1     | log2fc_33_pred   | id55_error      |  253 |    -0.0591 |   -0.1225 |
| as1     | lf_mean          | id55_error      |  253 |    -0.0550 |   -0.1148 |
| as1     | pred_htchem      | id55_abs_error  |  253 |    -0.2765 |   -0.3244 |
| as1     | log2fc_8p25_pred | id55_abs_error  |  253 |    -0.3382 |   -0.3149 |
| as1     | log2fc_33_pred   | id55_abs_error  |  253 |    -0.3636 |   -0.3737 |
| as1     | lf_mean          | id55_abs_error  |  253 |    -0.3528 |   -0.3502 |

## AS1 Error By pred_htchem Quantile

| pred_htchem_quantile   |   n |   true_mean |   pred_htchem_mean |   lf_mean |   id55_mae |   id55_bias |
|:-----------------------|----:|------------:|-------------------:|----------:|-----------:|------------:|
| q1_low                 |  51 |      3.4382 |             3.5922 |    0.3080 |     0.6585 |      0.1906 |
| q2                     |  50 |      4.3021 |             4.3418 |    0.5910 |     0.4493 |      0.1198 |
| q3                     |  51 |      4.7865 |             4.7695 |    0.8019 |     0.3365 |      0.0217 |
| q4                     |  50 |      5.3531 |             5.1607 |    1.0349 |     0.2897 |     -0.1311 |
| q5_high                |  51 |      5.4471 |             5.7932 |    1.4052 |     0.2974 |      0.0548 |

## Read

This is the first usable `pred_htchem` axis. It is still a ChemProp-only frozen-embedding probe, so it should be treated like the early log2fc-axis checks: useful if it explains AS1 errors or AS2 regions differently from predicted log2fc, not automatically a member to add.
