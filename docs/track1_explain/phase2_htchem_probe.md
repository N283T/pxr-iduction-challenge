# Phase 2 HTChem probe

Diagnostic analysis for the Phase 2 HTChem release. This does not change
the production ensemble; it tests whether HTChem is a useful external SAR
axis before heavier retraining.

## HTChem label profile

| slice            |   n |   n_unique_compound |   mean_pec50 |   median_pec50 |   min_pec50 |   max_pec50 |   mean_se |   median_yield_percent |
|:-----------------|----:|--------------------:|-------------:|---------------:|------------:|------------:|----------:|-----------------------:|
| all_rows         | 441 |                 441 |       4.8533 |         5.1285 |      1.7687 |      6.9661 |    0.2444 |                53.8973 |
| unique_compounds | 441 |                 441 |       4.8533 |         5.1285 |      1.7687 |      6.9661 |    0.2444 |                53.8973 |
| crude_rows       | 347 |                 347 |       4.6470 |         4.9688 |      1.7687 |      6.8607 |    0.2649 |                55.0731 |
| semi_pure_rows   |  94 |                  94 |       5.6151 |         5.6616 |      4.0207 |      6.9661 |    0.1686 |                53.1384 |

## Morgan nearest-neighbor coverage to HTChem

| slice         |    n |   mean_nn |   p90_nn |   max_nn |   n_ge_0.3 |   n_ge_0.4 |   n_ge_0.5 |   n_ge_0.6 |
|:--------------|-----:|----------:|---------:|---------:|-----------:|-----------:|-----------:|-----------:|
| labeled_all   | 4393 |    0.2618 |   0.3225 |   1.0000 |        771 |        129 |         58 |         19 |
| labeled_train | 4140 |    0.2583 |   0.3189 |   1.0000 |        684 |         84 |         21 |          6 |
| labeled_as1   |  253 |    0.3193 |   0.5303 |   1.0000 |         87 |         45 |         37 |         13 |
| test_all      |  513 |    0.3020 |   0.5139 |   1.0000 |        135 |         68 |         57 |         24 |
| test_as1      |  253 |    0.3193 |   0.5303 |   1.0000 |         87 |         45 |         37 |         13 |
| test_as2      |  260 |    0.2852 |   0.3580 |   0.7761 |         48 |         23 |         20 |         11 |

## HTChem-only model transfer

| slice        |    n |    mae |   bias_pred_minus_true |   spearman |   pred_mean |   true_mean |
|:-------------|-----:|-------:|-----------------------:|-----------:|------------:|------------:|
| htchem_oof   |  441 | 0.7227 |                 0.2645 |     0.5802 |      5.1178 |      4.8533 |
| labeled_all  | 4393 | 0.9410 |                 0.7797 |     0.4004 |      5.1203 |      4.3406 |
| source_train | 4140 | 0.9468 |                 0.7873 |     0.3985 |      5.1081 |      4.3208 |
| source_as1   |  253 | 0.8447 |                 0.6552 |     0.3222 |      5.3193 |      4.6641 |
| true_lt3     |  719 | 2.5425 |                 2.5425 |     0.0124 |      4.8538 |      2.3112 |
| true_gte6    |   77 | 0.8695 |                -0.8613 |     0.0622 |      5.4179 |      6.2792 |

## Phase2 Morgan LGBM augmentation, all rows

| setting     | slice   |    n |    mae |   bias_pred_minus_true |   spearman |   pred_mean |   true_mean |
|:------------|:--------|-----:|-------:|-----------------------:|-----------:|------------:|------------:|
| htchem_w0p3 | all     | 4393 | 0.5735 |                 0.0512 |     0.6718 |      4.3918 |      4.3406 |
| htchem_w0p5 | all     | 4393 | 0.5736 |                 0.0508 |     0.6721 |      4.3914 |      4.3406 |
| no_htchem   | all     | 4393 | 0.5752 |                 0.0741 |     0.6716 |      4.4147 |      4.3406 |
| htchem_w0p1 | all     | 4393 | 0.5765 |                 0.0450 |     0.6698 |      4.3856 |      4.3406 |
| htchem_w1p0 | all     | 4393 | 0.5774 |                 0.0538 |     0.6679 |      4.3944 |      4.3406 |

## Phase2 Morgan LGBM augmentation, AS1 slice

| setting     | slice      |   n |    mae |   bias_pred_minus_true |   spearman |   pred_mean |   true_mean |
|:------------|:-----------|----:|-------:|-----------------------:|-----------:|------------:|------------:|
| no_htchem   | source_as1 | 253 | 0.5907 |                 0.0216 |     0.6279 |      4.6857 |      4.6641 |
| htchem_w0p1 | source_as1 | 253 | 0.5949 |                 0.0308 |     0.6240 |      4.6949 |      4.6641 |
| htchem_w0p5 | source_as1 | 253 | 0.5956 |                 0.0741 |     0.6241 |      4.7382 |      4.6641 |
| htchem_w0p3 | source_as1 | 253 | 0.5963 |                 0.0680 |     0.6320 |      4.7321 |      4.6641 |
| htchem_w1p0 | source_as1 | 253 | 0.6116 |                 0.0830 |     0.5898 |      4.7471 |      4.6641 |

## AS1 replay by HTChem nearest-neighbor coverage

| candidate             | slice           |   mean_htchem_nn |   n |    mae |   bias_pred_minus_true |   spearman |   pred_mean |   true_mean |
|:----------------------|:----------------|-----------------:|----:|-------:|-----------------------:|-----------:|------------:|------------:|
| id55_anchor           | as1_all         |           0.3193 | 253 | 0.4066 |                 0.0516 |     0.8488 |      4.7157 |      4.6641 |
| seed10_top500_temp0p7 | as1_all         |           0.3193 | 253 | 0.4107 |                -0.0181 |     0.8357 |      4.6460 |      4.6641 |
| optuna_t10_top500     | as1_all         |           0.3193 | 253 | 0.4467 |                -0.0213 |     0.8104 |      4.6428 |      4.6641 |
| seed10_top500_temp0p7 | htchem_nn_ge0p5 |           0.5912 |  37 | 0.5594 |                 0.1755 |     0.8625 |      4.3428 |      4.1673 |
| optuna_t10_top500     | htchem_nn_ge0p5 |           0.5912 |  37 | 0.5618 |                 0.1457 |     0.8385 |      4.3130 |      4.1673 |
| id55_anchor           | htchem_nn_ge0p5 |           0.5912 |  37 | 0.5673 |                 0.2170 |     0.8613 |      4.3843 |      4.1673 |

## Initial read

- Treat corrected HTChem pEC50 as the primary label and keep QC columns
  available for weighting and filtering.
- The first useful question is not whether HTChem should replace the
  current best model, but whether it provides a stable external SAR axis
  around AS2-like chemistry.
- If augmentation helps only in narrow Morgan-LGBM diagnostics, the next
  step should be a low-weight feature/member, not a broad ensemble shift.
