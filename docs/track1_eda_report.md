# Track 1: Activity Prediction - EDA Report

## Competition Overview

- **Task**: Predict pEC50 (-log10(M)) for 513 blinded test compounds against human PXR (Pregnane X Receptor)
- **Evaluation**: RAE (Relative Absolute Error) as primary metric; also MAE, R², Spearman ρ, Kendall's τ
- **Submission**: `.parquet` or `.csv` with columns `SMILES`, `Molecule Name`, `pEC50` (513 rows)
- **Period**: April 1 – July 1, 2026
- **Cooldown**: 4 hours between submissions

## Dataset Summary

| Dataset | Rows | Unique SMILES | Columns | Description |
|---------|------|---------------|---------|-------------|
| Train (default) | 4,140 | 4,140 | 16 | Dose-response data with pEC50, Emax |
| Test (blinded) | 513 | 513 | 16 | All measurement columns are null |
| Counter-assay | 2,860 | 2,860 | 16 | PXR-null control assay (same schema as train) |
| Single-concentration | 21,014 | 10,875 | 18 | Single-dose screening at 4 concentrations × 3 experiments |

## SMILES Overlap

| Pair | Overlap |
|------|---------|
| Train ∩ Test | **0** (no leakage) |
| Counter-assay ∩ Train | 2,860 (100% of counter-assay) |
| Counter-assay ∩ Test | 0 |
| Single-conc ∩ Train | 2,392 |
| Single-conc ∩ Test | **0** |

**Key insight**: Test compounds have zero overlap with any training source. Counter-assay compounds are a strict subset of train.

## Target Variable: pEC50 (Train)

| Statistic | Value |
|-----------|-------|
| Count | 4,140 |
| Mean | 4.321 |
| Std | 1.122 |
| Min | 1.610 |
| Q1 | 3.640 |
| Median | 4.650 |
| Q3 | 5.130 |
| Max | 7.549 |

The distribution is slightly left-skewed, with a concentration of values between 3.5 and 5.5.

## Train Data Quality

- **Missing values**: None across all 16 columns
- **Invalid SMILES**: 0 / 4,140 (all parseable by RDKit)

## Test Data

- Columns: same 16 as train, but only `Molecule Name` and `SMILES` are populated
- All measurement columns (`pEC50`, `Emax_*`, confidence intervals, standard errors, `OCNT Batch`, `Split`) are null
- All 513 SMILES are valid in RDKit

## Molecular Descriptors: Train vs Test

| Descriptor | Train Mean | Test Mean | Δ |
|------------|-----------|----------|---|
| MW | 341.73 | 343.66 | +1.93 |
| LogP | 2.80 | 2.78 | -0.02 |
| HBA | 4.04 | 3.97 | -0.07 |
| HBD | 1.16 | 1.00 | -0.15 |
| TPSA | 69.41 | 69.70 | +0.29 |
| RotBonds | 4.41 | 4.14 | -0.27 |
| NumRings | 3.09 | 3.05 | -0.04 |
| NumAromaticRings | 2.14 | 2.03 | -0.11 |
| NumHeavyAtoms | 24.09 | 23.91 | -0.18 |
| FractionCSP3 | 0.37 | 0.39 | +0.02 |

Train and test distributions are highly similar in all descriptors. Test has narrower variance (MW std: 41 vs 77 for train), indicating fewer outlier-like molecules.

## pEC50 Correlation with Descriptors (Train)

| Descriptor | Pearson r |
|------------|-----------|
| **LogP** | **+0.482** |
| **MW** | **+0.374** |
| **NumHeavyAtoms** | **+0.370** |
| **NumRings** | **+0.345** |
| NumAromaticRings | +0.226 |
| HBD | -0.150 |
| RotBonds | +0.126 |
| TPSA | -0.074 |
| FractionCSP3 | +0.049 |
| HBA | -0.009 |

Higher lipophilicity and larger molecular size are moderately correlated with higher pEC50. This aligns with PXR's known preference for large, lipophilic ligands.

## Auxiliary Data Analysis

### Counter-Assay (2,860 compounds)

- Same schema as train (pEC50, Emax, confidence intervals)
- 100% SMILES overlap with train → paired PXR vs PXR-null measurements available
- Counter-assay pEC50: mean 3.13 (vs train mean 4.32) → lower potency in PXR-null, as expected
- 212 rows have null pEC50 (compounds with no measurable activity in PXR-null)
- **Use case**: Differential activity (train pEC50 − counter pEC50) as a feature to capture PXR specificity

### Single-Concentration (21,014 measurements)

- 10,875 unique SMILES across 4 concentrations and 3 experiments
- Concentrations: 0.98 µM, 8.25 µM, 33 µM, 99 µM
- Primary readout: `log2_fc_estimate` (log2 fold change vs baseline)
  - Mean: 0.505, Std: 0.536, Median: 0.349
- Also includes: `t_statistic`, `p_value`, `fdr_bh`, `cohens_d`, `n_replicates`
- 2,392 SMILES overlap with train; 0 overlap with test
- **Use case**: Additional training signal for compounds not in the dose-response train set (8,483 SMILES unique to single-conc). Could be used for pretraining or as weak labels.

## Chemical Space Analysis (2026-04-05)

### Nearest Neighbor Similarity (Sheridan Metric)

5-NN Tanimoto similarity (Morgan r=2, 2048 bits):

| Comparison | Mean | Std | Min | Median | Max |
|-----------|------|-----|-----|--------|-----|
| Test→Train | 0.389 | 0.045 | 0.282 | 0.385 | 0.543 |
| Train→Train | 0.345 | 0.063 | 0.119 | 0.337 | 0.715 |

**Key finding**: Test compounds are *more* similar to train than train is to itself (gap = -0.043). Chemical space shift is NOT the cause of the OOF-LB performance gap.

Test compounds with low NN similarity to train:
- < 0.3: 1.2% (6/513) — very few true outliers
- < 0.4: 63.5% — but train itself has even lower internal similarity

### Murcko Scaffold Analysis

| Set | Unique Scaffolds | Compounds | Singleton % |
|-----|-----------------|-----------|-------------|
| Train | 3,671 | 4,140 | 96.0% |
| Test | 369 | 513 | 83.5% |

- Scaffold overlap (train ∩ test): 35 scaffolds
- Test with train scaffold: 97/513 (18.9%)
- Test with novel scaffold: 416/513 (81.1%)
- **96% singleton scaffolds → scaffold split ≈ random split** (confirmed by CV comparison below)

### Butina Clustering (Train + Test)

| Cutoff | Clusters | Singletons | Test-only clusters | Test in test-only |
|--------|----------|------------|-------------------|-------------------|
| 0.3 | 4,528 | 98% | 424 | 96.9% |
| 0.4 | 4,287 | 96% | 262 | 77.8% |
| 0.5 | 3,793 | 91% | 74 | 22.4% |

At Tanimoto distance cutoff 0.5, most test compounds cluster with train compounds. The dataset has overall low pairwise similarity.

### CV Split Strategy Comparison

LightGBM (Mordred features, fixed hyperparams) across split strategies:

| Strategy | OOF RAE | LB Gap | Fold RAE Std | 5-NN Sim |
|----------|---------|--------|-------------|----------|
| Random | 0.5608 | +0.059 | 0.015 | 0.335 |
| Murcko Scaffold | 0.5615 | +0.058 | 0.034 | 0.330 |
| Butina 0.4 | 0.5604 | +0.060 | 0.011 | 0.332 |
| Butina 0.5 | 0.5675 | +0.053 | 0.011 | 0.326 |
| Butina 0.6 | 0.5660 | +0.054 | 0.026 | 0.317 |
| Butina 0.7 | 0.5671 | +0.053 | 0.014 | 0.316 |
| **UMAP 50 clusters** | **0.5804** | **+0.040** | 0.040 | **0.298** |
| **UMAP 100 clusters** | **0.5752** | **+0.045** | 0.027 | **0.301** |

- UMAP split is the strictest (lowest train-val NN similarity) and gives the closest OOF RAE to LB
- UMAP 50 clusters (RAE=0.580, gap=0.040) narrows the LB gap by 1/3 vs random/scaffold (~0.059)
- However, UMAP split has high fold variance (std=0.040) — less stable estimates
- **Conclusion**: Split strategy accounts for ~0.02 of the 0.06 LB gap. The remaining ~0.04 gap is likely due to ensemble weight overfitting (single model OOF RAE 0.56 → ensemble OOF RAE 0.539) and test distribution differences

### OOF Error Analysis (single_mordred, best single model)

- **NN similarity vs |error|**: Spearman r = -0.027 (p=0.08) — no relationship
- **Error by pEC50 range**:

| pEC50 Range | n | MAE | RAE |
|------------|---|-----|-----|
| [1, 3) | 695 | 0.853 | 0.425 |
| [3, 4) | 551 | 0.522 | 0.648 |
| [4, 5) | 1,558 | 0.353 | 1.082 |
| [5, 6) | 1,269 | 0.465 | 0.459 |
| [6, 8) | 67 | 1.362 | 0.691 |

- High-activity (pEC50 > 6) and low-activity (pEC50 < 3) compounds have the worst MAE
- Regression to the mean is the dominant error pattern
- The [4, 5) bin has low MAE but RAE > 1 due to narrow spread around the global mean

### UMAP Chemical Space Projection

Morgan FP (r=2, 2048 bits) + UMAP (Jaccard metric, n_neighbors=30):

- Test compounds are distributed throughout train chemical space — no isolated test-only regions
- **However, test is spatially biased**: under-represented in the bottom-left (low-activity) region

| UMAP Quadrant | Train % | Test % | Train mean pEC50 |
|--------------|---------|--------|-----------------|
| Bottom-Left (low activity) | 29.6% | **10.3%** | 3.96 |
| Bottom-Right | 20.6% | **38.0%** | 4.61 |
| Top-Left | 23.0% | 18.7% | 4.26 |
| Top-Right | 26.8% | **32.9%** | 4.55 |

- 42% of low-activity compounds (pEC50 < 3) cluster in the bottom-left
- Test has 3× fewer compounds in this region (10% vs 30%)
- Test is enriched in right-side (medium-to-high activity) regions (71% vs 47%)
- This spatial bias may affect how train errors translate to LB performance

Figures: `docs/figures/umap_train_test.png`, `docs/figures/umap_pec50_test_overlay.png`

### Implications for Improvement

1. **Ensemble optimization**: Must be regularized — current full-OOF weight optimization overfits by ~0.02 RAE
2. **Model diversity**: More important than split strategy. Different models may excel in different pEC50 ranges
3. **Extreme value prediction**: Improving high/low pEC50 accuracy is the highest-leverage opportunity
4. **Additional data**: Counter-assay and single-conc data may help with extreme value prediction (issue #21)

## Potential Modeling Approaches

### Baseline
- RDKit molecular descriptors + gradient boosting (LightGBM / XGBoost)
- Morgan fingerprints (ECFP) + Random Forest / Ridge regression

### Feature Engineering
- Morgan fingerprints (various radii: 2, 3)
- MACCS keys
- RDKit 2D descriptors (200+ features)
- Counter-assay differential features (for the 2,860 overlapping compounds)
- Single-concentration aggregated features (max/mean log2FC across concentrations)

### Advanced
- Graph neural networks (e.g., ChemProp, AttentiveFP)
- Pretrained molecular representations (e.g., ChemBERTa, MolBERT)
- Ensemble of fingerprint-based and GNN models

### Considerations
- Test has zero overlap with all training data → generalization is critical
- Test descriptor distributions are similar to train → domain shift is minimal
- Counter-assay provides PXR-specificity signal for ~69% of train compounds
- Single-conc data covers ~58% of train SMILES but 0% of test → limited direct utility for test prediction, but useful for model regularization
