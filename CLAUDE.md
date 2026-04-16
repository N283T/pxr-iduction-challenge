# PXR Induction Challenge

## Competition

OpenADMET PXR Blind Challenge (April 1 – July 1, 2026)
https://huggingface.co/spaces/openadmet/pxr-challenge

Two tracks:
- **Track 1 (Activity)**: Predict pEC50 for 513 blinded compounds. Primary metric: RAE.
- **Track 2 (Structure)**: Predict protein-ligand 3D structures for 78 compounds. Primary metric: LDDT-PLI.

Current status: **17th place** (RAE=0.6263 on leaderboard, 2026-04-09).
See latest snapshot in `docs/leaderboard_<date>.csv`.

## Environment

- **Package manager**: pixi (conda-forge + pypi)
- **Python**: 3.12
- **Database**: PostgreSQL 18 + RDKit cartridge (port 5433, socket /tmp)
- **GPU**: RTX 5080 (16GB VRAM)
- **WSL2**: `CONDA_OVERRIDE_CUDA=13.1` set globally via `~/dotfiles/home/shell.nix`
- **Boltz-2**: installed as a separate `uv tool` (not pixi) — conflicts with pixi env
  (numpy 2.x and chembl_structure_pipeline downgrades). Invoked via the CLI.

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
pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/compound_descriptors_full_schema.sql
pixi run python db/compute_rdkit_descriptors_full.py
```

### Boltz-2 Setup (structure prediction + affinity)

One-time:
```bash
uv tool install 'boltz[cuda]' --python 3.12      # Boltz-2 CLI (~3GB model on first run)
cp /mnt/c/Users/<user>/Downloads/AF-O75469-F1-msa_v6.a3m structures/boltz2/msa/pxr.a3m
pixi run python track2_structure/scripts/boltz2_build_inputs.py     # 4653 input YAMLs
```

Full inference run (~4 days on RTX 5080, resume-safe):
```bash
tmux new -s boltz2
bash track2_structure/scripts/boltz2_full_run.sh
# Ctrl+C anytime; re-running the script resumes from cached predictions.
```

Recovery + DB registration:
```bash
# If 01576-style salt/complex fails: largest-fragment recovery
pixi run python track2_structure/scripts/boltz2_recover_01576.py
bash track2_structure/scripts/boltz2_full_run.sh   # re-run, only the missing compound

# Record permanently-failed compounds (see issue #50)
pixi run python track2_structure/scripts/boltz2_record_failures.py

# Phase 2 post-processing: ligand pkl + sdf + DB upsert
pixi run python track2_structure/scripts/boltz2_postprocess.py --db

# Pose quality validation (PoseBusters)
pixi run python track2_structure/scripts/boltz2_posebusters.py --workers 8 --db
```

## Project Structure

```
data/                        # Parquet files (gitignored, re-downloadable)
db/
  schema.sql                 # Core tables: compounds, train/test/counter/single_conc
  add_std_columns.sql        # Add std_smiles/std_mol columns
  standardize_compounds.py   # ChEMBL pipeline standardization
  recompute_descriptors.sql  # RDKit descriptors & fingerprints from std_mol (41 cols)
  compound_descriptors_full_schema.sql  # Full RDKit descriptor table (JSONB)
  compute_rdkit_descriptors_full.py     # Compute all 217 RDKit 2D descriptors
  compute_mordred.py         # Mordred 2D descriptors -> compound_mordred (JSONB)
  compute_embeddings.py      # ChemBERTa/BERT/MoLFormer variants -> DB tables
  compute_chemeleon.py       # CheMeleon MPNN fingerprints -> compound_chemeleon
  experiments_schema.sql     # Experiment tracking tables + OOF predictions
  load_data.py               # Data loader script
  pgdata/                    # PostgreSQL data dir (gitignored)
docs/
  track1_eda_report.md       # EDA findings + feature importance + distribution analysis
  leaderboard_2026-04-07.csv # Leaderboard snapshot (latest)
  literature_qsar_ml.md      # PXR QSAR/ML literature review
  literature_wet_lab.md      # PXR wet-lab/biology literature review
track1_activity/
  src/
    data.py                  # DB loading (SQLAlchemy, ORDER BY t.id)
    features.py              # FP_REGISTRY: 8 fingerprint types via RDKit
    evaluate.py              # Metrics + DB recording + OOF storage
    splits.py                # Murcko scaffold split + UMAP split CV
  scripts/
    run_train.py                  # Unified LightGBM/XGBoost/CatBoost training
    run_chemprop_optuna.py        # ChemProp D-MPNN Optuna tuning
    run_attentivefp_optuna.py     # AttentiveFP (PyG) Optuna tuning
    run_molformer_finetune.py     # MoLFormer-XL fine-tuning with Optuna
    run_residual_learning.py      # Two-stage residual learning (physprop + Mordred)
    run_ensemble.py               # Weighted ensemble optimization
    run_all_models.sh             # Sequential DL model training pipeline
    api.py                        # OpenADMET API client: fetch leaderboard + submit (gitignored, contains PII)
    archive/                      # Early exploration scripts
  notebooks/                 # marimo notebooks
  submissions/               # CSV submission files (gitignored)
track2_structure/
  src/boltz2/
    constants.py               # PXR sequence, core pocket residues, paths, chain ids
    input_builder.py           # SMILES -> Boltz-2 YAML (affinity + pocket constraint)
    postprocess.py              # pose .cif + cached .pkl -> fully-bonded RDKit Mol
  scripts/
    boltz2_build_inputs.py     # DB -> 4653 YAML files + manifest.csv
    boltz2_smoke_run.sh        # 10-compound smoke test (R1 settings, LD_LIBRARY_PATH fix)
    boltz2_full_run.sh         # 4653-compound full run (same wrapper, different dirs)
    boltz2_inspect_smoke.py    # smoke output QC (pocket distance, confidence, affinity)
    boltz2_postprocess.py      # 4653 pose pkl+sdf + metadata CSV + compound_boltz2 upsert
    boltz2_posebusters.py      # PoseBusters pose quality checks (19 booleans) + DB upsert
    boltz2_recover_01576.py    # largest-fragment recovery for 2-component salt
    boltz2_record_failures.py  # insert permanently-failed compounds into compound_boltz2
structures/
  boltz2/                      # All runtime artifacts (gitignored)
    msa/pxr.a3m                # AFDB MSA (copied once)
    inputs/<id>.yaml           # 4653 Boltz-2 inputs
    outputs/boltz_results_inputs/   # Boltz-2 output tree
      predictions/<id>/             # cif, confidence, affinity, plddt, pae, pde
      processed/                    # constraints/, structures/, mols/, msa/ caches
    ligands/<id>.{pkl,sdf}          # Phase 2 pose outputs (lossless pkl + viewer sdf)
  alphafold/                   # AF-O75469-F1-model_v6.cif.gz (PR #33)
  pxr_lbd/                     # 72 PDB holo structures (PR #33)
  aligned/, aligned_with_ligands/     # AF-aligned multi-model CIFs (PR #33)
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
  (PostgreSQL RDKit cartridge `mol_*` functions only; narrow subset for SQL-level filtering)
- `compound_descriptors_full` -- Full 217 RDKit 2D descriptors (BCUT2D, fr_*, VSA family,
  EState, MQN, etc.) computed via Python RDKit `Descriptors._descList`, JSONB storage.
  2 compounds have partial coverage (metal-containing; BCUT2D fails).
- `compound_fingerprints` -- Morgan, FeatMorgan, MACCS, AtomPair, Avalon FPs
- `compound_mordred` -- Mordred 2D descriptors (~1460 per compound, JSONB)
- `compound_chemberta` -- ChemBERTa-77M-MLM (384d)
- `compound_chemberta_mtr` / `_100m` / `_10m` / `_5m` variants
- `compound_chemberta_zinc_v1` -- ChemBERTa-zinc-v1 (768d)
- `compound_bert_smiles` -- BERT-base-SMILES (768d)
- `compound_molformer` -- MoLFormer-XL (768d, requires rotary fix)
- `compound_chemeleon` -- CheMeleon MPNN fingerprints (300d)

### Experiment Tracking
- `experiments` -- Model config, hyperparameters (JSONB), submission path
- `experiment_cv_results` -- Per-fold CV metrics
- `experiment_oof_predictions` -- OOF predictions for ensemble
- `experiment_summary` -- View: aggregated metrics sorted by RAE

### Boltz-2 Prediction Outputs (Track 2 + Track 1 structure features)
- `compound_boltz2` -- One row per compound (4653 rows). File paths (pose cif, ligand
  pkl/sdf, confidence/affinity/plddt/pae/pde), status flags (preprocessing_failed,
  ligand_oversize), 6 affinity head outputs (mean + 2 ensemble members), 9 confidence
  metrics, and geometry sanity (ligand_atom_count, ligand_to_pocket_distance_a).
  Populated by `boltz2_postprocess.py` + `boltz2_record_failures.py`.
- `compound_boltz2_posebusters` -- One row per Boltz-2 prediction. 19 boolean pose
  quality checks from PoseBusters (`minimum_distance_to_protein` = no clash, bond
  lengths/angles, aromatic flatness, internal energy, etc.) plus
  num_checks / num_passed / all_passed / intramol_passed / intermol_passed summary.
  Populated by `boltz2_posebusters.py`.

## Conventions

- Notebooks: use **marimo** (not Jupyter)
- All code, comments, commits in **English**
- Experiment results -> `experiments` + `experiment_cv_results` tables
- OOF predictions -> `experiment_oof_predictions` (required for ensemble)
- Submission files -> `track1_activity/submissions/` (gitignored)
- Compare experiments: `pixi run db-psql -c "SELECT * FROM experiment_summary;"`
- CV strategy: **UMAP split** preferred (Morgan FP + UMAP + KMeans 50 clusters, closer to LB)
- Scaffold split available as alternative (`--split scaffold`)
- Gap regularization: `--gap-lambda 1.0` penalizes train-val gap for better generalization
- All load functions use `ORDER BY t.id` for deterministic row ordering
- Use `load_train_mordred()` / `load_test_mordred()` from data.py (not recomputing)

## Known Issues

- OOF CV RAE (~0.53) is optimistic vs leaderboard RAE (0.62) -- gap ~0.09
- UMAP split narrows this gap vs scaffold split (~0.06 vs ~0.08-0.12)
- MoLFormer requires rotary embedding fix for transformers v5 (see issue #30, fix in compute_embeddings.py)
- MoLFormer embedding -> LightGBM gives RAE=0.65 (weaker than ChemBERTa); fine-tuning is better
- LogP dominates feature importance (gain 17k-24k) -- risk of "shortcut learning" on unseen chemotypes

### Boltz-2 specific
- Full inference run (4653 compounds, R1 settings) takes ~4 days on RTX 5080.
  Expect occasional MSA NPZ corruption from interrupted preprocessing; diagnose with
  `np.load(path)`, delete the affected `processed/msa/<id>*.npz` +
  `processed/{structures,constraints,mols,records}/<id>*`, and re-run.
- `uv tool` venv ships torch 2.11+cu130 libs under `nvidia/cu13/lib/`; the
  `boltz2_*_run.sh` scripts inject that path into `LD_LIBRARY_PATH` so triton /
  cuequivariance JIT kernels can dlopen `libnvrtc-builtins.so.13.0`.
- Preprocessing failed compounds (see issue #50):
  - `01576` (train) -- salt/co-crystal with two macrolide fragments. Recovered via
    RDKit `LargestFragmentChooser` (62 HAs, oversize warning applies).
  - `01657` (train) -- Auranofin, Au-containing metal complex. Excluded by Boltz-2
    standardize; drop (train-only, no metal compounds in test).
  - `03840` (train) -- (1S,4S)-2-azanorbornane (PubChem CID 131950785). RDKit ETKDGv3
    cannot embed the bridged bicyclic; drop (train-only, no external 3D source).
- 9 compounds exceed the 56-heavy-atom training cap of the Boltz-2 affinity head
  (`ligand_oversize=TRUE` in compound_boltz2). Their `affinity_pred_value` should be
  treated as low-confidence.
- Pose sanity: across 4651 predictions, ligand-to-pocket-centroid distance mean is
  ~1.55 A (max 10.89 A). Use `compound_boltz2_posebusters.minimum_distance_to_protein`
  to filter physically implausible poses.
