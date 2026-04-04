# PXR Induction Challenge

Entry for the [OpenADMET PXR Blind Challenge](https://huggingface.co/spaces/openadmet/pxr-challenge) (April–July 2026).

## Overview

Predicting human Pregnane X Receptor (PXR) activity from molecular structure. PXR is a nuclear receptor critical for drug metabolism (ADMET).

### Tracks

| Track | Task | Metric | Test Size |
|-------|------|--------|-----------|
| 1 - Activity | Predict pEC50 | RAE ↓ | 513 compounds |
| 2 - Structure | Predict protein-ligand 3D | LDDT-PLI ↑ | 78 compounds |

## Setup

### Prerequisites

- [pixi](https://pixi.sh/) (package manager)
- GPU with CUDA (optional, for GNN models and embedding computation)
- WSL2: `CONDA_OVERRIDE_CUDA=13.1` is set automatically via shell config

### Install & Run

```bash
# Install dependencies
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

### Useful Commands

```bash
pixi run db-start          # Start PostgreSQL
pixi run db-stop           # Stop PostgreSQL
pixi run db-psql           # Connect to DB

# Compare experiments
pixi run db-psql -c "SELECT * FROM experiment_summary;"
```

## Current Results (Track 1)

Top experiments sorted by RAE (lower is better):

| Experiment | Model | Features | RAE | R² | Spearman |
|-----------|-------|----------|-----|-----|----------|
| scaffold_chemeleon+desc+morgan | LightGBM | CheMeleon + RDKit desc + Morgan | 0.5507 | 0.617 | 0.744 |
| mordred+morgan_r2 | LightGBM | Mordred 2D + Morgan | 0.5531 | 0.620 | 0.739 |
| chemeleon+desc+morgan | LightGBM | CheMeleon + RDKit desc + Morgan | 0.5530 | 0.613 | 0.737 |
| desc+morgan_r2 | LightGBM | RDKit desc + Morgan | 0.5572 | 0.609 | 0.736 |
| chemprop_mpnn | ChemProp | Molecular graph (D-MPNN) | 0.6136 | 0.542 | 0.700 |
| baseline_lgbm_descriptors | LightGBM | RDKit 2D descriptors | 0.6235 | 0.518 | 0.658 |

40+ experiments tracked in DB. See `experiment_summary` view for full results.

### Approach

Following the methodology from [EUOS25 winning solution](https://www.yumizsui.com/blog/euos25-challenge/):

1. **Diverse features**: RDKit descriptors, Mordred 2D, fingerprints (Morgan/MACCS/Avalon/AtomPair), CheMeleon embeddings, ChemBERTa embeddings (7 variants), BERT-base-SMILES
2. **SMILES standardization**: ChEMBL structure pipeline (salt stripping, neutralization)
3. **Murcko scaffold split CV**: Prevents structural leakage in cross-validation
4. **Optuna nested CV**: Hyperparameter tuning with scaffold outer / random inner splits
5. **Weighted ensemble**: Combining diverse models (planned)

## Project Structure

```
data/                        # Parquet datasets (gitignored)
db/
  schema.sql                 # Core tables: compounds, activities
  standardize_compounds.py   # ChEMBL pipeline standardization
  compute_mordred.py         # Mordred 2D descriptors → DB
  compute_embeddings.py      # ChemBERTa variants → DB
  compute_chemeleon.py       # CheMeleon fingerprints → DB
  experiments_schema.sql     # Experiment tracking tables
docs/
  track1_eda_report.md       # EDA findings
track1_activity/
  src/                       # Shared modules (data, features, evaluate, splits)
  scripts/                   # Experiment scripts
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
