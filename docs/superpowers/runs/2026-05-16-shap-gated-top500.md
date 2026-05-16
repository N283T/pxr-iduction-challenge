# SHAP-Gated Top500 Probe

Date: 2026-05-16

Goal: test whether non-log2fc features from existing SHAP/gain diagnostics can
gate the top500 delta more safely than broad ensemble reweighting.

## Setup

- Anchor for candidate movement: id55
  `ens_id51_top500_potent46_t40_soft_g35`.
- Delta sources:
  - `seed10_top500_minus_current_raw`
  - `optuna_top500_minus_current_raw`
- Gate sources:
  - simple RDKit descriptors loaded by `load_train_descriptors` /
    `load_test_descriptors`
  - ensemble meta features: member spread and the chemprop/top500-family gap
- Candidate formula:

```text
candidate = id55 + gamma * gate(feature, optional_delta_sign) * (top500 - current_raw)
```

Note: measured `single_concentration.log2_fc_estimate` is not available for the
blind test compounds. A database check found 513 `test_activity` rows and 0
distinct test compounds with `single_concentration` rows, while train has 2392
distinct compounds with that auxiliary assay. Predicted log2fc features still
cover test and are the proper test-time log2fc source.

Outputs:

- `track1_activity/analysis/compound_level_lb/outputs/shap_gated_top500/shap_gated_top500_summary.csv`
- `track1_activity/analysis/compound_level_lb/outputs/shap_gated_top500/shap_gated_top500_safeish.csv`
- preflight reports under
  `track1_activity/analysis/submission_preflight/outputs/id55_shap_*`

## Result

Some SHAP-inspired gates improve OOF more than the log2fc gate, but they are
less chemically grounded and move id55 a little more broadly.

Best safe-ish candidates:

| source | gate | gamma | OOF MAE | delta vs raw | id55 p90 shift | id55 max shift | id56 projection |
|---|---|---:|---:|---:|---:|---:|---:|
| optuna_top500_minus_current_raw | positive_delta_x_num_rings_high_soft_q20 | 0.50 | 0.387464 | -0.003980 | 0.034066 | 0.103700 | 0.061398 |
| optuna_top500_minus_current_raw | family_gap_high_soft_q20 | 0.25 | 0.387999 | -0.003445 | 0.032599 | 0.086024 | 0.078768 |
| optuna_top500_minus_current_raw | positive_delta_x_family_gap_high_soft_q20 | 0.25 | 0.388166 | -0.003278 | 0.032535 | 0.086024 | 0.073448 |

Generated ignored candidate CSVs for preflight:

- `track1_activity/submissions/ens_id55_shap_numrings_pos_highsoftq20_g50.csv`
- `track1_activity/submissions/ens_id55_shap_familygap_highsoftq20_g25.csv`
- `track1_activity/submissions/ens_id55_shap_pos_familygap_highsoftq20_g25.csv`

Preflight:

| candidate | verdict | mean abs shift | p90 shift | max shift | id56 projection |
|---|---|---:|---:|---:|---:|
| positive num_rings high soft q20 gamma 0.50 | PASS | 0.009419 | 0.034066 | 0.103700 | 0.061398 |
| family_gap high soft q20 gamma 0.25 | PASS | 0.011066 | 0.032599 | 0.086024 | 0.078768 |
| positive family_gap high soft q20 gamma 0.25 | PASS | 0.009661 | 0.032535 | 0.086024 | 0.073448 |

## Interpretation

The `positive_delta_x_num_rings_high_soft_q20` gate has the strongest local
OOF support in this batch, but one blind-test compound moves more than 0.10
pEC50 and the hypothesis is weaker than the predicted-log2fc gate.

If spending one cooldown, prefer the log2fc-gated optuna candidate from
`2026-05-16-log2fc-gated-top500.md`. Keep the SHAP-gated candidates as bolder
alternates or follow-up diagnostics rather than the first submission choice.
