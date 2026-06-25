# Phase 2 external rank gate probe

Date: 2026-06-25 JST

Purpose: revisit ChEMBL and HTChem as ranking/enrichment signals for the Phase 2
tail classifiers. This intentionally avoids treating external assay values as
challenge-scale pEC50 labels.

## Feature Construction

ChEMBL PXR rows were filtered to human PXR/NR1I2 activation-like assay rows with
activity assay type, confidence >= 8, pChEMBL present, and antagonist/binding
descriptions excluded. Instead of using raw pChEMBL only, each assay was converted
to assay-local percentile ranks. Per-molecule ChEMBL features were then aggregated
and transferred to challenge compounds through nearest-neighbor Morgan Tanimoto
features after exact challenge InChIKey exclusion.

HTChem rank features were derived from the existing `pred_htchem` challenge axis:
rank percentile, z-score, `pred_htchem` minus predicted-log2fc z, and log2fc rank.

External feature cache:

- `track1_activity/analysis/phase2_classifier_gate/outputs/external_rank_features/`

ChEMBL rank coverage:

| split | n | NN >= 0.25 | NN >= 0.30 | NN max | NN median | external molecules |
|---|---:|---:|---:|---:|---:|---:|
| pool | 4393 | 681 | 151 | 1.0000 | 0.2088 | 222 |
| test | 513 | 93 | 20 | 0.3594 | 0.2152 | 222 |

The coverage is still thin on the blinded test set: only 20 / 513 compounds have
ChEMBL-rank NN similarity >= 0.30.

## Classifier Results

Main comparison:

| config | all balanced acc | AS1 balanced acc | gte6 AP | gte6 AUC | selected external cols | read |
|---|---:|---:|---:|---:|---:|---|
| TabPFN top100 baseline | 0.6279 | 0.6094 | 0.2096 | 0.8976 | 0 | best AS1 classifier quality |
| TabPFN top200 baseline | 0.6258 | 0.5925 | 0.2245 | 0.9016 | 0 | best high-tail AP |
| TabPFN top200 + ChEMBL rank | 0.6269 | 0.5820 | 0.2199 | 0.9006 | 4 | selected, but slightly worse than baseline |
| TabPFN top200 + ChEMBL + HTChem rank | 0.6313 | 0.5529 | 0.1949 | 0.8974 | 5 | all-fold score improves, AS1 degrades |
| LGBM baseline, all features | 0.4960 | 0.4540 | 0.1149 | 0.8152 | 0 | weak high classifier |
| LGBM + ChEMBL + HTChem rank | 0.5038 | 0.4421 | 0.1088 | 0.8214 | 15 | small AUC gain, AP/AS1 not better |

Selected external columns in TabPFN top200 + ChEMBL rank:

- `chembl_rank_nn_high_frac`
- `chembl_rank_top5_pct`
- `chembl_rank_nn_pct`
- `chembl_rank_top5_pchembl`

Selected external columns in TabPFN top200 + ChEMBL + HTChem rank:

- `lf_rank_pct_for_htchem`
- `htchem_minus_lf_z_rank`
- `chembl_rank_top5_pct`
- `chembl_rank_nn_pct`
- `chembl_rank_nn_high_frac`

## Gate Read

ChEMBL rank is not ignored by feature selection, which means the signal is not
pure noise. However, adding it did not improve the best existing TabPFN multiclass
gate:

- `gte6` AP decreased from 0.2245 to 0.2199 with ChEMBL rank.
- AS1 balanced accuracy decreased from 0.5925 to 0.5820 with ChEMBL rank.
- Adding HTChem rank improved all-fold balanced accuracy but hurt AS1 more.

Binary LGBM with ChEMBL + HTChem rank improved high-tail AUC from 0.8123 to
0.8229, but AP decreased from 0.1004 to 0.0974 and AS1 AP decreased from 0.2702
to 0.2231. This looks like better broad ordering but worse early enrichment.

## Conclusion

External rank features are useful diagnostically but are not ready as the primary
high-tail gate. The current best direction remains the multiclass TabPFN baseline
without external rank features.

The more promising next step is not direct concatenation into the already-strong
top-k classifier. If revisiting this, use ChEMBL rank as a separate diagnostic
stratum or as a small auxiliary score for compounds with credible ChEMBL coverage,
not as a broad feature appended to every row.
