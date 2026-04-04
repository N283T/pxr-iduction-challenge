# PXR Induction Challenge

## Competition

OpenADMET PXR Blind Challenge (April 1 – July 1, 2026)
https://huggingface.co/spaces/openadmet/pxr-challenge

Two tracks:
- **Track 1 (Activity)**: Predict pEC50 for 513 blinded compounds. Primary metric: RAE.
- **Track 2 (Structure)**: Predict protein-ligand 3D structures for 78 compounds. Primary metric: LDDT-PLI.

Current status: **4th place** (RAE=0.62 on leaderboard, 2026-04-04).

## Environment

- **Package manager**: pixi (conda-forge + pypi)
- **Python**: 3.12
- **Database**: PostgreSQL 18 + RDKit cartridge (port 5433, socket /tmp)
- **GPU**: RTX 5080 (16GB VRAM)
- **WSL2**: `CONDA_OVERRIDE_CUDA=13.1` set globally via `~/dotfiles/home/shell.nix`

### DB Commands

```bash
pixi run db-start    # Start PostgreSQL
pixi run db-stop     # Stop PostgreSQL
pixi run db-psql     # Connect to pxr_challenge DB
```

### DB Setup (after fresh clone)

```bash
pixi run db-start
pixi run python download_data.py
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

## Project Structure

```
data/                        # Parquet files (gitignored, re-downloadable)
db/
  schema.sql                 # Core tables: compounds, train/test/counter/single_conc
  add_std_columns.sql        # Add std_smiles/std_mol columns
  standardize_compounds.py   # ChEMBL pipeline standardization
  recompute_descriptors.sql  # RDKit descriptors & fingerprints from std_mol
  compute_mordred.py         # Mordred 2D descriptors -> compound_mordred (JSONB)
  compute_embeddings.py      # ChemBERTa/BERT variants -> DB tables
  compute_chemeleon.py       # CheMeleon MPNN fingerprints -> compound_chemeleon
  experiments_schema.sql     # Experiment tracking tables + OOF predictions
  load_data.py               # Data loader script
  pgdata/                    # PostgreSQL data dir (gitignored)
docs/
  track1_eda_report.md       # EDA findings
  leaderboard_2026-04-04.csv # Leaderboard snapshot
track1_activity/
  src/
    data.py                  # DB loading (SQLAlchemy, ORDER BY t.id)
    features.py              # FP_REGISTRY: 8 fingerprint types via RDKit
    evaluate.py              # Metrics + DB recording + OOF storage
    splits.py                # Murcko scaffold split CV
  scripts/
    run_single_feature_optuna.py  # Single-feature Optuna-tuned LightGBM
    run_chemprop_scaffold_cv.py   # ChemProp/CheMeleon fine-tuning
    run_ensemble.py               # Weighted ensemble optimization
    archive/                      # Early exploration scripts
  notebooks/                 # marimo notebooks
  submissions/               # CSV submission files (gitignored)
track2_structure/            # Structure prediction (future)
```

## DB Schema

### Core Tables
- `compounds` -- SMILES + std_smiles + RDKit mol (13,136 rows)
- `train_activity` -- pEC50 dose-response data (4,140 rows)
- `test_activity` -- Blinded test compounds (513 rows)
- `counter_assay` -- PXR-null control data (2,860 rows)
- `single_concentration` -- Single-dose screening (21,014 rows)

### Pre-computed Feature Tables
- `compound_descriptors` -- 38 RDKit 2D descriptors + scaffold + formula + InChIKey
- `compound_fingerprints` -- Morgan, FeatMorgan, MACCS, AtomPair, Avalon FPs
- `compound_mordred` -- Mordred 2D descriptors (~1460 per compound, JSONB)
- `compound_chemberta` -- ChemBERTa-77M-MLM (384d)
- `compound_chemberta_mtr` / `_100m` / `_10m` / `_5m` variants
- `compound_chemberta_zinc_v1` -- ChemBERTa-zinc-v1 (768d)
- `compound_bert_smiles` -- BERT-base-SMILES (768d)
- `compound_chemeleon` -- CheMeleon MPNN fingerprints (300d)

### Experiment Tracking
- `experiments` -- Model config, hyperparameters (JSONB), submission path
- `experiment_cv_results` -- Per-fold CV metrics
- `experiment_oof_predictions` -- OOF predictions for ensemble
- `experiment_summary` -- View: aggregated metrics sorted by RAE

## Conventions

- Notebooks: use **marimo** (not Jupyter)
- All code, comments, commits in **English**
- Experiment results -> `experiments` + `experiment_cv_results` tables
- OOF predictions -> `experiment_oof_predictions` (required for ensemble)
- Submission files -> `track1_activity/submissions/` (gitignored)
- Compare experiments: `pixi run db-psql -c "SELECT * FROM experiment_summary;"`
- CV strategy: Murcko scaffold split (outer), random split for inner/tuning
- All load functions use `ORDER BY t.id` for deterministic row ordering
- Use `load_train_mordred()` / `load_test_mordred()` from data.py (not recomputing)

## Known Issues

- OOF CV RAE (0.54) is optimistic vs leaderboard RAE (0.62) -- weight optimization may overfit
- MoLFormer/IBM models disabled due to `transformers.onnx` incompatibility
