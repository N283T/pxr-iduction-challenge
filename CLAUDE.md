# PXR Induction Challenge

## Competition

OpenADMET PXR Blind Challenge (April 1 – July 1, 2026)
https://huggingface.co/spaces/openadmet/pxr-challenge

Two tracks:
- **Track 1 (Activity)**: Predict pEC50 for 513 blinded compounds. Primary metric: **MAE** (with RAE/R²/Spearman/Kendall as secondaries).
- **Track 2 (Structure)**: Predict protein-ligand 3D structures for 78 compounds. Primary metric: LDDT-PLI.

Current status: **7th place** (MAE=0.4358, RAE=0.5474 on leaderboard, 2026-04-21).
Research log: issue #100. See latest snapshot in `docs/leaderboard_<date>.csv`.

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
pixi run python db/fix_bridged_stereo.py --apply
pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/recompute_descriptors.sql
pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/experiments_schema.sql
pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/lb_submissions_schema.sql
pixi run python db/compute_mordred.py
pixi run python db/compute_jazzy.py
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
# Targeted re-run for a handful of compound IDs (stereo repair, failure recovery)
bash track2_structure/scripts/boltz2_recover_run.sh <compound_id> [<compound_id> ...]

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
  schema.sql                         # Core tables: compounds, train/test/counter/single_conc
  add_std_columns.sql                # Add std_smiles/std_mol columns
  standardize_compounds.py           # ChEMBL pipeline standardization
  fix_bridged_stereo.py              # Hamming-1 cis-bridgehead repair (Boltz-2 input fixup)
  recompute_descriptors.sql          # RDKit descriptors & fingerprints from std_mol (41 cols)
  compound_descriptors_full_schema.sql  # Full RDKit descriptor table (JSONB)
  compute_rdkit_descriptors_full.py  # Compute all 217 RDKit 2D descriptors
  compute_mordred.py                 # Mordred 2D descriptors -> compound_mordred (JSONB)
  compute_jazzy.py                   # Jazzy H-bond descriptors -> compound_jazzy
  compute_embeddings.py              # ChemBERTa/BERT/MoLFormer variants -> DB tables
  compute_chemeleon.py               # CheMeleon MPNN fingerprints -> compound_chemeleon
  experiments_schema.sql             # Experiment tracking tables + OOF predictions
  lb_submissions_schema.sql          # Local LB submission history + results tables
  boltz2_schema.sql                  # compound_boltz2 (pose paths, affinity, confidence)
  boltz2_posebusters_schema.sql      # compound_boltz2_posebusters (19 pose quality checks)
  load_data.py                       # Data loader script
  pgdata/                            # PostgreSQL data dir (gitignored)
docs/
  track1_eda_report.md               # EDA findings + feature importance + distribution analysis
  leaderboard_<date>.csv             # Leaderboard snapshots (latest 2026-04-21)
  literature_qsar_ml.md              # PXR QSAR/ML literature review
  literature_wet_lab.md              # PXR wet-lab/biology literature review
  superpowers/specs/                 # Approved feature design docs (YYYY-MM-DD-<topic>-design.md)
  superpowers/plans/                 # Implementation plans (YYYY-MM-DD-<topic>.md)
track1_activity/
  src/
    data.py                  # DB loading (SQLAlchemy, ORDER BY t.id)
    features.py              # FP_REGISTRY + per-feature loaders
    evaluate.py              # Metrics + DB recording + OOF storage (record_experiment
                             # supports on_conflict_replace=True for idempotent re-runs)
    splits.py                # Murcko scaffold split + UMAP split CV
    losses.py                # Custom chemprop losses (relative-distance aux — FMGCL)
    peft_backbones.py        # LoRA backbone registry (molformer_xl, molformer_c3_1_1b)
    peft_methods.py          # LoRA / PEFT method registry
    peft_trainer.py          # Shared PEFT regressor trainer (rotary-fix aware)
    pyg_training.py          # PyG graph training helpers
    pseudo_labels.py         # Weak-label prep for counter-assay
  scripts/
    run_train.py                      # Unified LightGBM/XGBoost/CatBoost/TabPFN training
    run_ensemble.py                   # Ensemble strategies (caruana_bag20 preferred;
                                      # vanilla / l2_a{0.05..0.5} / fold_l2 / simple_avg
                                      # reported side-by-side for OOF A/B).
                                      # ENSEMBLE_MODELS allow-list controls the 10-model pool.
    run_ensemble_calibrate.py         # Post-hoc regression calibration: linear, linear_pos
                                      # (slope>=0 affine), spline_k5 (PCHIP monotone),
                                      # isotonic. 4-way nested CV + MAE/Spearman guardrail
                                      # writes ens_caruana_bag20_calibrated_best.csv.
    run_chemprop_optuna.py            # ChemProp D-MPNN Optuna tuning
    run_chemprop_chemeleon.py         # CheMeleon foundation finetune (chemprop head)
    run_chemprop_pretrain.py          # Pretrain chemprop encoder on single-conc log2_fc
    run_chemprop_embed_extract.py     # Extract frozen [encoded] chemprop features
    run_chemprop_predict_log2fc.py    # Use pretrained encoder to predict log2_fc for test
    run_chemprop_finetune.py          # Frozen-encoder head FT on pEC50
    run_chemprop_multitask.py         # Multitask (pec50 + log2_fc) head
    run_chemprop_multitask_desc.py    # Multitask with descriptor aux (negative result, #86)
    run_chemprop_relative_aux.py      # FMGCL relative-distance aux loss (negative result, #97)
    run_attentivefp_optuna.py         # AttentiveFP (PyG) Optuna tuning
    run_attentivefp_pretrain_finetune.py
    run_gatedgcn_optuna.py            # GatedGCN (PyG) Optuna tuning
    run_gatedgcn_pretrain_finetune.py # GatedGCN pretrain+frozen+head FT (pool member)
    run_gin_optuna.py                 # GIN Optuna tuning
    run_graphgps_optuna.py            # GraphGPS Optuna tuning
    run_molformer_c3_pretrain.py      # Pretrain MoLFormer-c3-1.1B + LoRA on log2_fc
    run_molformer_c3_embed_extract.py # Extract [CLS] 768d embeddings from pretrained
    run_peft_finetune.py              # Generic PEFT/LoRA direct FT on pEC50 (#95,
                                      # dropped from pool after LB regression, #96)
    run_residual_learning.py          # Two-stage residual learning (physprop + Mordred)
    boltz_affhead/                    # Boltz-2 trunk embedding retarget (issue #74)
                                      # 01{,b}_pool_embeddings.py: core_pocket / allpairs pools
                                      # 02_lgbm_baseline, 03_combine_and_correlate,
                                      # 04_mlp_head (weak), 05_ensemble_dryrun,
                                      # 06_pool_rework, 07_caruana_select
    run_all_models.sh                 # Sequential DL model training pipeline
    api.py                            # OpenADMET LB client (fetch / submit / status /
                                      # cooldown). Writes lb_submissions + back-fills
                                      # LB results on each `fetch`.
    archive/                          # Early exploration scripts
  notebooks/                 # marimo notebooks
  submissions/               # CSV submission files (gitignored)
track2_structure/
  src/boltz2/
    constants.py               # PXR sequence, core pocket residues, paths, chain ids
    input_builder.py           # SMILES -> Boltz-2 YAML (affinity + pocket constraint)
    postprocess.py              # pose .cif + cached .pkl -> fully-bonded RDKit Mol
  scripts/
    boltz2_build_inputs.py     # DB -> 4653 YAML files + manifest.csv
    boltz2_full_run.sh         # 4653-compound full run
    boltz2_embeddings_run.sh   # trunk-only re-run, dumps s/z embeddings to existing outputs (issue #57)
    boltz2_recover_run.sh      # re-run selected compound IDs into the main output tree (stereo fix, failure recovery)
    boltz2_postprocess.py      # 4653 pose pkl+sdf + metadata CSV + compound_boltz2 upsert
    boltz2_posebusters.py      # PoseBusters pose quality checks (19 booleans) + DB upsert
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
- `compound_jazzy` -- Jazzy H-bond donor/acceptor descriptors (JSONB)
- `compound_chemberta` -- ChemBERTa-77M-MLM (384d)
- `compound_chemberta_mtr` / `_100m` / `_10m` / `_5m` variants
- `compound_chemberta_zinc_v1` -- ChemBERTa-zinc-v1 (768d)
- `compound_bert_smiles` -- BERT-base-SMILES (768d)
- `compound_molformer` -- MoLFormer-XL (768d, requires rotary fix)
- `compound_chemeleon` -- CheMeleon MPNN fingerprints (300d)

### Pose-derived feature tables (from Boltz-2 outputs)
- `compound_boltz2_jazzy` / `_desc3d` / `_desc3d_vector` / `_mordred3d` / `_skfp3d`
  -- Per-pose 3D features computed after ligand extraction. Populated by scripts in
  `track1_activity/scripts/` (feature bakeoff + 2d_full_boltz bundle).

### Experiment Tracking
- `experiments` -- Model config, hyperparameters (JSONB), submission path
- `experiment_cv_results` -- Per-fold CV metrics
- `experiment_oof_predictions` -- OOF predictions for ensemble
- `experiment_summary` -- View: aggregated metrics sorted by RAE

### Leaderboard submission tracking
- `lb_submissions` -- Local row per `api.py submit`: submission_name, file_path,
  experiment_name, notes (LOCAL only), submitted_at. Populated by api.py.
- `lb_submission_history` -- Back-filled LB rank/MAE/RAE etc. per fetch.

### Boltz-2 Prediction Outputs (Track 2 + Track 1 structure features)
- `compound_boltz2` -- One row per compound (4653 rows). File paths (pose cif, ligand
  pkl/sdf, confidence/affinity/plddt/pae/pde, embeddings npz), status flags (preprocessing_failed,
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
- **CV strategy: UMAP split (seed=42, n_clusters=50, Morgan+Jaccard)** — canonical
  after 12+ variant bake-off in PR #70. Scaffold split still available as
  `--split scaffold` for diagnostics.
- All load functions use `ORDER BY t.id` for deterministic row ordering
- Use `load_train_mordred()` / `load_test_mordred()` from data.py (not recomputing)
- **Ensemble strategy: `caruana_bag20`** (discrete count-based, bagged 20x per
  Caruana 2004) when adding correlated-strong members. Continuous weight optimizers
  (vanilla, L2) concentrate weight on the single best member and reallocate it
  destructively when a correlated challenger is added — see issue #82 for the LB
  regression incident that motivated this. Vanilla is still reported side-by-side
  for OOF A/B diagnostics.
- **Post-hoc calibration is part of the final submission**. `run_ensemble_calibrate.py`
  runs 4-way nested CV (linear / linear_pos / spline_k5 / isotonic) on the
  ens_caruana_bag20 output and picks the best calibrator via MAE with a
  `|ΔSpearman| < 0.005` guardrail. `linear_pos` (positive-constrained affine)
  was the 2026-04-21 LB winner, moving us rank 10 → 7 (OOF ΔMAE −0.0009 → LB ΔMAE
  −0.0065, ~7× amplification). See PR #101. Other variants (K=3..30 splines,
  full isotonic, importance-weighted, per-cluster, stacked) all regressed; see
  the PR #101 comment for the full falsification log.
- **Pretrain + frozen + embed recipe (Buterez 2024 strategy-3)**: pretrain an
  encoder on single-concentration log2_fc (13,136 compounds, transductive,
  NaN-masked MSE per concentration head), freeze it, extract embeddings for all
  compounds, run TabPFN v7 on those for pEC50. Three pool members follow this
  pattern (`tabpfn_chemprop_pretrain_embed`, `tabpfn_molformer_c3_pretrain_embed`,
  `tabpfn_2d_full_boltz_log2fc_pred`) and jointly account for >65% of caruana
  weight. Direct PEFT FT on pEC50 (PR #95 MoLFormer-XL LoRA) underperforms this
  recipe and was dropped from the pool.
- **Submission workflow**:
  1. `run_ensemble.py` -> caruana_bag20 -> `ens_caruana_bag20.csv`
  2. `run_ensemble_calibrate.py` -> 4-way nested CV -> `ens_caruana_bag20_calibrated_best.csv`
  3. `api.py cooldown` to check 4h window, `api.py submit ...` with `--notes`.
  4. `api.py fetch` after ~30 min to ~2 h to back-fill LB rank/metrics.

## Known Issues

- OOF/LB MAE gap: raw ensemble OOF MAE ~0.43, LB MAE ~0.44 (post-calibration
  LB 0.4358). The gap is driven by the test pEC50 distribution being ~12%
  narrower than train (analog enrichment), which compresses the RAE denominator
  but leaves MAE roughly faithful. Prefer MAE for ensemble selection and
  calibrator tuning; RAE is useful for LB ranking but is noisier across runs.
- MoLFormer requires rotary embedding fix for transformers v5 (see issue #30);
  `peft_trainer.py` + `compute_embeddings.py` recompute `inv_freq` and rebuild
  the cos/sin cache before PEFT wrapping.
- MoLFormer-XL direct PEFT FT (LoRA) on pEC50 underperforms the frozen-encoder
  embedding recipe; dropped from pool in PR #96.
- LogP dominates feature importance in single-feature LGBMs (gain 17k-24k) --
  risk of shortcut learning on unseen chemotypes; mitigated by ensembling with
  graph/transformer members that do not rely on LogP directly.
- Deprecation warnings from `rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect`
  (will be removed in a future RDKit release). Migrate to `MorganGenerator` when
  touching the FP code path.

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
  - `03840` (train) -- 2-azabicyclo[2.2.1] derivative. Original DB SMILES specified
    geometrically impossible trans-bridgeheads, so RDKit ETKDGv3 refused to embed.
    Repaired by `db/fix_bridged_stereo.py` (Hamming-1 cis-bridgehead enantiomer);
    Boltz-2 inference pending (delete cached failure in `compound_boltz2` and
    rerun via `boltz2_recover_run.sh 03840` once no other GPU job is active,
    then `boltz2_postprocess.py --db`).
- 9 compounds exceed the 56-heavy-atom training cap of the Boltz-2 affinity head
  (`ligand_oversize=TRUE` in compound_boltz2). Their `affinity_pred_value` should be
  treated as low-confidence.
- Pose sanity: across 4651 predictions, ligand-to-pocket-centroid distance mean is
  ~1.55 A (max 10.89 A). Use `compound_boltz2_posebusters.minimum_distance_to_protein`
  to filter physically implausible poses.
