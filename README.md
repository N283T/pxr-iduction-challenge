# PXR Induction Challenge

Repository for the OpenADMET PXR Blind Challenge (April 1 - July 1, 2026).

- Challenge page: <https://huggingface.co/spaces/openadmet/pxr-challenge>
- Track 1 Phase 1 research log: [issue #100](https://github.com/N283T/pxr-iduction-challenge/issues/100)
- Track 1 Phase 2 research log: [issue #208](https://github.com/N283T/pxr-iduction-challenge/issues/208)
- Track 2 research log: [issue #129](https://github.com/N283T/pxr-iduction-challenge/issues/129)

The GitHub issues above are the source of truth for current experiments,
leaderboard status, null results, and next-step notes. This README is only a
lightweight orientation guide and may intentionally lag behind active work.

## Tracks

| Track | Task | Primary metric | Current notes |
|---|---|---|---|
| Track 1 Activity | Predict pEC50 for blinded compounds | MAE lower is better | Phase 2 work is in issue #208; Phase 1 chronology is issue #100 |
| Track 2 Structure | Predict protein-ligand 3D structures | LDDT-PLI higher is better | See issue #129 and `docs/leaderboards/structure/` |

Before quoting a rank, gap, or "best" submission, check the latest leaderboard
snapshot and local submission history. The public leaderboard changes quickly.

## Track 1 Phase 2 status

Analog Set 1 labels have been released for 253 of the 513 Track 1 test
compounds; 260 compounds remain blinded. The first Phase 2 pass was an
answer-check audit only: existing submission CSVs, OOF summaries, production
members, and `docs/track1_explain/` claims were replayed against the released
labels without retraining or generating new predictions.

Short read from that audit:

- The Phase 1 public leaderboard effectively tracked the released Analog Set 1
  subset.
- The id55/id60 anchor `ens_id51_top500_potent46_t40_soft_g35` remains the best
  recent Phase 1 anchor on released labels.
- The main error shape is compressed extremes: very weak compounds were
  overpredicted and very strong compounds were underpredicted.
- Predicted `log2_fc` is still a strong activity axis, but small OOF gains,
  local gates, and diversity-reserve assumptions need Phase 2 revalidation.

Use issue #208 for current Track 1 Phase 2 decisions. Keep issue #100 as the
Phase 1 chronology.

## Repository map

```text
data/                         ignored runtime parquet/data artifacts
db/                           database schemas, loaders, feature builders
docs/                         notes, literature reports, leaderboard snapshots
track1_activity/src/          shared Track 1 loading, features, splits, metrics
track1_activity/scripts/      Track 1 training, ensembling, calibration, submit tools
track1_activity/boltz2/       Boltz-2 feature-generation pipeline for Track 1
track1_activity/submissions/  ignored Track 1 CSV submissions
track2_structure/             Track 2 structure-prediction pipeline and submissions
structures/                   ignored Boltz-2 / structure runtime artifacts
```

## Environment

This project uses `pixi` and a local PostgreSQL + RDKit database.

Useful commands:

```bash
pixi install
pixi run db-start
pixi run db-stop
pixi run db-psql

pixi run ruff format <file>
pixi run ruff check <file>
```

Database details used in local scripts:

- PostgreSQL 18 + RDKit cartridge
- port: `5433`
- socket: `/tmp`

GPU-heavy work was developed on an RTX 5080 with WSL2 CUDA override
`CONDA_OVERRIDE_CUDA=13.1`.

## Submission helpers

Track 1 / Track 2 submissions are handled through `track1_activity/scripts/api.py`.
The script records local submission metadata in the database when available.

```bash
pixi run python track1_activity/scripts/api.py cooldown
pixi run python track1_activity/scripts/api.py status --track activity
pixi run python track1_activity/scripts/api.py fetch --track activity
pixi run python track1_activity/scripts/api.py fetch --track structure
```

For Track 1, run a preflight report before spending a cooldown on a material CSV
change:

```bash
pixi run python track1_activity/scripts/submission_preflight.py \
  --candidate track1_activity/submissions/<candidate>.csv \
  --anchor track1_activity/submissions/<trusted-anchor>.csv \
  --name <report-name>
```

Treat `PASS` / `CAUTION` / `HOLD` as warning lights, not automatic decisions.
Inspect anchor shifts, largest compound moves, prediction scale, and known bad
axis alignment.

## Data

Challenge data comes from
[openadmet/pxr-challenge-train-test](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test).
Generated data, checkpoints, embeddings, Boltz outputs, and submission files are
kept out of git.

## Where to look first

- Track 1 Phase 2 status / decisions: issue #208
- Track 1 Phase 1 chronology: issue #100
- Track 2 status / decisions: issue #129
- Durable agent operating rules: `AGENTS.md`
- Leaderboard snapshots: `docs/leaderboards/`
- Detailed docs and archived research notes: `docs/`
