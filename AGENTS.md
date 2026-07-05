# PXR Induction Challenge

This file is the compact operating guide for coding agents in this repository.
Keep fast-changing experiment details in GitHub issues, database rows, and
leaderboard snapshots instead of expanding this file.

## Competition

OpenADMET PXR Blind Challenge (April 1 - July 1, 2026)
https://huggingface.co/spaces/openadmet/pxr-challenge

- Track 1 Activity: predict pEC50 for 513 blinded compounds. Primary metric: MAE.
- Track 2 Structure: predict protein-ligand 3D structures for 78 compounds. Primary metric: LDDT-PLI.
- Track 1 Phase 1 research log: GitHub issue #100.
- Track 1 Phase 2 research log: GitHub issue #208.
- Track 1 public model report repo: https://github.com/N283T/openadmet-pxr-model-report
- Leaderboard snapshots: `docs/leaderboards/activity/`.
- Before quoting rank, gap, or "best", check the latest snapshot and `lb_submissions` / `lb_submission_history`. Rank changes quickly.

## Workflow Rules

- Never commit directly to `main` or `master`; use a `codex/` branch for repository changes.
- Code, comments, commit messages, PR descriptions, and technical docs must be in English unless the user-facing artifact requires another language.
- Do not delete git-tracked files unless the user explicitly asks for that deletion.
- Treat local submission clients such as `track1_activity/scripts/api.py` as ignored because they can contain account-specific details. Do not commit them.
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
- `test_activity_phase1_labels`: released Phase 2 labels for Analog Set 1.
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
docs/                         documentation, literature notes, leaderboard snapshots
track1_activity/src/          shared data loading, features, CV splits, metrics, trainers
track1_activity/scripts/      training, ensembling, calibration, submission, experiments
track1_activity/scripts/archive/  older exploratory scripts
track1_activity/boltz2/       Boltz-2 input, inference, postprocess pipeline
track1_activity/submissions/  ignored Track 1 CSV submissions
structures/                   ignored Boltz-2 and structure runtime artifacts
```

## Track 1 Modeling Conventions

- Canonical CV: UMAP split, seed 42, 50 clusters, Morgan+Jaccard. Scaffold split is diagnostic only unless explicitly requested.
- All load functions should preserve deterministic ordering with `ORDER BY t.id`.
- Phase 2 has released Analog Set 1 labels for 253 of the 513 Track 1 test
  compounds; 260 compounds remain blinded as Analog Set 2. Use AS1 for
  answer-checks and validation design, not as a reason to train directly on AS2
  labels that do not exist.
- Use issue #208 for Phase 2 Track 1 logs. Keep issue #100 as the Phase 1/live-leaderboard chronology.
- Record experiments and OOF predictions in the DB when adding a model intended for ensembling.
- Current ensemble default: `caruana_bag20` in `run_ensemble.py`. Continuous optimizers are useful diagnostics but have caused destructive reallocation with correlated members.
- Re-run both calibrators after material pool changes:
  - `run_ensemble_calibrate.py`
  - `run_ensemble_calibrate_importance.py`
- Before spending a cooldown on a materially changed CSV, run
  `submission_preflight.py` against a trusted anchor. Treat the PASS/CAUTION/HOLD
  label as a warning light, not an automatic submit decision; inspect shift
  counts, largest compound moves, prediction scale, and known-bad-axis alignment.
- Submission flow: run ensemble, calibrate, check cooldown, submit with explicit notes, fetch later to back-fill LB results.
- Treat tiny OOF gains as weak evidence. The public LB has repeatedly amplified small OOF moves in either direction.

Important Track 1 memory:

- Phase 2 AS1 replay confirmed the public LB target was effectively the released
  253-compound subset. Recent unique submission CSVs replay within about
  0.0003-0.0006 MAE of recorded public LB rows; older stable paths may have been
  overwritten.
- The id55/id60 anchor `ens_id51_top500_potent46_t40_soft_g35` remains the best
  recent Phase 1 anchor on AS1 (MAE about 0.4066). Its main error shape is
  extreme compression: very weak compounds are overpredicted and very strong
  compounds are underpredicted.
- Predicted `log2_fc` remains a real AS1 activity axis, but low-tail activity
  cliffs near potent train analogs are not solved by simple NN or high-LF gates.
- OOF is directionally meaningful on AS1, but small late-stage OOF gains and
  local gates remain weak evidence. The id56 top500/log2fc-heavy direction, id58
  combo gate, and id59 high-activity lift were all AS1-negative relative to id55.
- Current AS1 member replay supports re-checking low-weight diversity reserves
  before carrying them forward by inertia; weak single models can help in exact
  ensemble contexts, but that must be verified rather than assumed.
- Phase 2 final candidate submitted on 2026-06-25: `phase2_as1_aug_top500_id55blend_a0p4_pairrankchembl_q95_g0p15_labels_as1` (`lb_submissions` id 62). It fills AS1 rows with released labels, applies a 0.4 blend from the id55 anchor toward the AS1-augmented top500 TabPFN v3 model for AS2, then adds a small +0.15 high-activity lift to AS2 compounds flagged by the ChEMBL/public-PXR pairwise assay-rank gate.
- HTChem Phase 2 data is loaded for future work, but it was intentionally
  deferred from the first AS1 answer-check audit.
- The pretrain-freeze-extract recipe on single-concentration `log2_fc` has been the strongest repeatable axis. Multi-seed upgrades should usually be SWAPs, not ADDs, when predictions are highly correlated.
- `ens_meta_axis_reverse_id50_g10` was a diagnostic LB-direction probe, not a scalable model family.
- The 2026-05-07 optuna trial10 seed5ens top500 SWAP had excellent OOF but
  regressed on LB (id56). Simple new-top500 SWAP/ADD is risky unless preflight
  shows small anchored movement or a new gating hypothesis.
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
