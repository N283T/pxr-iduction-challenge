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
