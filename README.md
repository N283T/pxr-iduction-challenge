# PXR Induction Challenge

Entry for the [OpenADMET PXR Blind Challenge](https://huggingface.co/spaces/openadmet/pxr-challenge) (April–July 2026).

## Leaderboard

**4th place** (as of 2026-04-04) out of 19 teams.

| Rank | Team | RAE | R² | Spearman |
|------|------|-----|-----|----------|
| 1 | MirthEngine | 0.58 | 0.61 | 0.81 |
| 2 | Radi | 0.59 | 0.54 | 0.81 |
| 3 | N1NC1O | 0.60 | 0.52 | 0.79 |
| **4** | **N283T (us)** | **0.62** | **0.52** | **0.78** |
| 5 | jaybirdy | 0.64 | 0.51 | 0.74 |

## Overview

Predicting human Pregnane X Receptor (PXR) activity from molecular structure. PXR is a nuclear receptor critical for drug metabolism (ADMET).

### Tracks

| Track | Task | Metric | Test Size |
|-------|------|--------|-----------|
| 1 - Activity | Predict pEC50 | RAE ↓ | 513 compounds |
| 2 - Structure | Predict protein-ligand 3D | LDDT-PLI ↑ | 78 compounds |

## Approach

Following the methodology from [EUOS25 winning solution](https://www.yumizsui.com/blog/euos25-challenge/):

1. **SMILES standardization**: ChEMBL structure pipeline (salt stripping, neutralization)
2. **Diverse single-feature models**: Each model uses one feature type, individually Optuna-tuned
3. **Murcko scaffold split CV**: Prevents structural leakage in cross-validation
4. **Optuna nested CV**: Hyperparameter tuning with scaffold outer / random inner splits
5. **Weighted ensemble**: Optimized blend of 12 diverse models

### Feature Sources

| Type | Features | Source |
|------|----------|--------|
| Descriptors | RDKit 2D (41d), Mordred 2D (1402d) | RDKit, mordredcommunity |
| Fingerprints | Morgan, MACCS, Avalon, AtomPair (167-2048d) | RDKit |
| Embeddings | ChemBERTa (7 variants, 384-768d) | HuggingFace |
| Embeddings | CheMeleon MPNN (300d) | Zenodo pretrained |
| Embeddings | BERT-base-SMILES (768d) | HuggingFace |
| Graph | ChemProp D-MPNN, CheMeleon fine-tune | chemprop |

### Ensemble

12 models blended with optimized weights (scipy minimize on OOF RAE):

| Model | Weight | Solo RAE |
|-------|--------|----------|
| chemprop_scaffold | 22% | 0.620 |
| single_mordred | 20% | 0.565 |
| single_chemeleon | 17% | 0.602 |
| single_chemberta_5m_mtr | 13% | 0.614 |
| single_rdkit_desc | 10% | 0.625 |
| chemeleon_finetune | 9% | 0.657 |
| + 6 others | 9% | — |

## Setup

### Prerequisites

- [pixi](https://pixi.sh/) (package manager)
- GPU with CUDA (for GNN models and embedding computation)
- WSL2: `CONDA_OVERRIDE_CUDA=13.1` set automatically via shell config

### Install & Run

```bash
pixi install

# Download dataset
pixi run python download_data.py

# Start PostgreSQL + full DB setup
pixi run db-start
pixi run psql -h /tmp -p 5433 -c "CREATE DATABASE pxr_challenge;"
pixi run psql -h /tmp -p 5433 -d pxr_challenge -c "CREATE EXTENSION rdkit;"
pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/schema.sql
pixi run python db/load_data.py
pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/add_std_columns.sql
pixi run python db/standardize_compounds.py
pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/recompute_descriptors.sql
pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/experiments_schema.sql
pixi run python db/compute_mordred.py
pixi run python db/compute_embeddings.py
pixi run python db/compute_chemeleon.py
```

### Reproducing Results

```bash
# 1. Single-feature Optuna models (CPU, ~1 hour)
pixi run python track1_activity/scripts/run_single_feature_optuna.py

# 2. ChemProp / CheMeleon fine-tuning (GPU)
pixi run python track1_activity/scripts/run_chemprop_scaffold_cv.py

# 3. Ensemble
pixi run python track1_activity/scripts/run_ensemble.py
```

### Useful Commands

```bash
pixi run db-start / db-stop / db-psql
pixi run db-psql -c "SELECT * FROM experiment_summary;"
```

## Project Structure

```
data/                        # Parquet datasets (gitignored)
db/
  schema.sql                 # Core tables: compounds, activities
  standardize_compounds.py   # ChEMBL pipeline standardization
  compute_mordred.py         # Mordred 2D descriptors → DB (JSONB)
  compute_embeddings.py      # ChemBERTa/BERT variants → DB
  compute_chemeleon.py       # CheMeleon MPNN fingerprints → DB
  experiments_schema.sql     # Experiment tracking tables
docs/
  track1_eda_report.md       # EDA findings
  leaderboard_2026-04-04.csv # Leaderboard snapshot
track1_activity/
  src/
    data.py                  # DB loading (SQLAlchemy, ORDER BY t.id)
    features.py              # FP_REGISTRY: 8 fingerprint types
    evaluate.py              # Metrics, DB recording, OOF storage
    splits.py                # Murcko scaffold split CV
  scripts/
    run_single_feature_optuna.py  # Optuna-tuned single-feature models
    run_chemprop_scaffold_cv.py   # ChemProp/CheMeleon fine-tuning
    run_ensemble.py               # Weighted ensemble optimization
    archive/                      # Early exploration scripts
  notebooks/                 # marimo notebooks
  submissions/               # Submission CSVs (gitignored)
track2_structure/            # Structure prediction (future)
```

## Data Sources

All data from [openadmet/pxr-challenge-train-test](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test) (Apache 2.0).

| Dataset | Rows | Description |
|---------|------|-------------|
| Train (default) | 4,140 | Dose-response with pEC50, Emax |
| Test (blinded) | 513 | pEC50 to predict |
| Counter-assay | 2,860 | PXR-null control |
| Single-concentration | 21,014 | Single-dose screening |
