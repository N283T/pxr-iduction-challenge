# PXR Induction Challenge

This file is the compact operating guide for coding agents in this repository.
Keep fast-changing experiment details in GitHub issues, database rows, and
leaderboard snapshots instead of expanding this file.

## Competition

OpenADMET PXR Blind Challenge (April 1 - July 1, 2026)
https://huggingface.co/spaces/openadmet/pxr-challenge

- Track 1 Activity: predict pEC50 for 513 blinded compounds. Primary metric: MAE.
- Track 2 Structure: predict protein-ligand 3D structures for 78 compounds. Primary metric: LDDT-PLI.
- Track 1 research log: GitHub issue #100.
- Leaderboard snapshots: `docs/leaderboards/activity/` and `docs/leaderboards/structure/`.
- Before quoting rank, gap, or "best", check the latest snapshot and `lb_submissions` / `lb_submission_history`. Rank changes quickly.

## Workflow Rules

- Never commit directly to `main` or `master`; use a `codex/` branch for repository changes.
- Code, comments, commit messages, PR descriptions, and technical docs must be in English unless the user-facing artifact requires another language.
- Do not delete git-tracked files unless the user explicitly asks for that deletion.
- Treat `track1_activity/scripts/api.py` as local/ignored because it contains personal account details. Do not commit it.
- Generated artifacts are not source: keep checkpoints, embedding parquets, Boltz outputs, DB data, and submissions out of git.
- This repo has no GitHub Actions CI. Use focused local checks and report `CI: N/A (no workflow)` when relevant.
- Prefer small, reversible changes. Update docs when behavior or workflow changes.

## Environment

- Package manager: pixi
- Python: 3.12
- Database: PostgreSQL 18 + RDKit cartridge on port 5433, socket `/tmp`
- GPU: RTX 5080, 16 GB VRAM
- WSL2 CUDA override: `CONDA_OVERRIDE_CUDA=13.1`
- Boltz-2 is installed separately as a `uv tool`; do not import/run it inside the pixi env unless that path has been verified.

Useful commands:

```bash
pixi run db-start
pixi run db-stop
pixi run db-psql
pixi run ruff format <file>
pixi run ruff check <file>
```

## Data And Database

Fresh clone setup is intentionally script-driven. Start the DB, download data,
create `pxr_challenge`, enable RDKit, then apply schemas and compute features from
`db/` scripts. Prefer existing loaders and feature tables over recomputing ad hoc.

Core tables:

- `compounds`: all compounds with SMILES, standardized SMILES, and RDKit mols.
- `train_activity`, `test_activity`: Track 1 train/test rows.
- `counter_assay`, `single_concentration`: auxiliary activity data.
- `experiments`, `experiment_cv_results`, `experiment_oof_predictions`: local experiment tracking.
- `lb_submissions`, `lb_submission_history`: local leaderboard submission history.

Feature tables include RDKit descriptors/fingerprints, Mordred, Jazzy, ChemBERTa
variants, BERT-SMILES, MoLFormer, CheMeleon, ChemFM, Boltz-2 pose features, and
Boltz-2 trunk-fast embeddings. Check `track1_activity/src/features.py` for the
current supported feature names.

## Project Map

```text
data/                         ignored parquet/runtime data
db/                           schemas, loaders, descriptor/embedding builders
docs/                         documentation, deepresearch, leaderboard snapshots, track2 notes
track1_activity/src/          shared data loading, features, CV splits, metrics, trainers
track1_activity/scripts/      training, ensembling, calibration, submission, experiments
track1_activity/scripts/archive/  older exploratory scripts
track1_activity/boltz2/       Boltz-2 input, inference, postprocess pipeline
track1_activity/submissions/  ignored Track 1 CSV submissions
structures/                   ignored Boltz-2 and structure runtime artifacts
track2_structure/             Track 2 work area when needed
```

## Track 1 Modeling Conventions

- Canonical CV: UMAP split, seed 42, 50 clusters, Morgan+Jaccard. Scaffold split is diagnostic only unless explicitly requested.
- All load functions should preserve deterministic ordering with `ORDER BY t.id`.
- Record experiments and OOF predictions in the DB when adding a model intended for ensembling.
- Current ensemble default: `caruana_bag20` in `run_ensemble.py`. Continuous optimizers are useful diagnostics but have caused destructive reallocation with correlated members.
- Re-run both calibrators after material pool changes:
  - `run_ensemble_calibrate.py`
  - `run_ensemble_calibrate_importance.py`
- Submission flow: run ensemble, calibrate, check cooldown, submit with explicit notes, fetch later to back-fill LB results.
- Treat tiny OOF gains as weak evidence. The public LB has repeatedly amplified small OOF moves in either direction.

Important Track 1 memory:

- The pretrain-freeze-extract recipe on single-concentration `log2_fc` has been the strongest repeatable axis. Multi-seed upgrades should usually be SWAPs, not ADDs, when predictions are highly correlated.
- `ens_meta_axis_reverse_id50_g10` was a diagnostic LB-direction probe, not a scalable model family.
- Re-pooled Boltz trunk features improved standalone trunk OOF but simple swap/drop submissions were LB-negative. Do not keep submitting small variants of that direction without new evidence.
- Direct MoLFormer-XL PEFT finetuning on pEC50 underperformed the frozen-encoder embedding recipe.
- Weak single models with non-top-tier OOF have often hurt LB even when they pass local gates.

## Boltz-2 Notes

Boltz-2 runtime artifacts live under `structures/boltz2/` and are ignored. Full
inference over 4653 compounds takes multiple days and is resume-oriented through
repository scripts.

Useful entry points:

```bash
pixi run python track1_activity/boltz2/scripts/boltz2_build_inputs.py
bash track1_activity/boltz2/scripts/boltz2_full_run.sh
bash track1_activity/boltz2/scripts/boltz2_recover_run.sh <compound_id> [<compound_id> ...]
pixi run python track1_activity/boltz2/scripts/boltz2_postprocess.py --db
pixi run python track1_activity/boltz2/scripts/boltz2_posebusters.py --workers 8 --db
```

Known boundaries:

- `compound_boltz2` full pose/confidence coverage is about 4652 compounds.
- `compound_boltz2_trunk_fast` has about 13k trunk-only embedding rows from a mix of full and cheap runs. Do not mix trunk-only coverage with full-pose features without marking that boundary.
- Known Boltz preprocessing edge cases are documented in issue #50 and relevant scripts.

## Cleanup And Safety

- If `.git` grows unexpectedly, check `git count-objects -vH` before deleting anything. A previous incident was unreachable loose blobs from accidentally staged generated checkpoints; `git prune` was used only after connectivity checks.
- Do not broad-clean ignored directories without explicit user approval.
- Keep leaderboard CSVs under `docs/leaderboards/<track>/` with timestamped names.
- Prefer GitHub issues for detailed experiment notebooks/logs; keep this file for durable operating rules only.
