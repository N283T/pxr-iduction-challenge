# PXR Induction Challenge

Repository for my OpenADMET PXR Blind Challenge work (April 1 - July 1, 2026).

- Challenge page: <https://huggingface.co/spaces/openadmet/pxr-challenge>
- Track 1 public model report: <https://github.com/N283T/openadmet-pxr-model-report>
- Track 1 Phase 1 research log: [issue #100](https://github.com/N283T/pxr-iduction-challenge/issues/100)
- Track 1 Phase 2 research log: [issue #208](https://github.com/N283T/pxr-iduction-challenge/issues/208)
- Track 2 research log: [issue #129](https://github.com/N283T/pxr-iduction-challenge/issues/129)

The issues are intentionally part of the public record: they contain the
day-by-day experiment notes, wrong turns, leaderboard reactions, null results,
and decision logs that do not belong in a polished README. If you want the
battle record rather than the cleaned-up code map, start with #100 and #208.

## Final Result

| Track | Task | Result |
|---|---|---|
| Track 1 Activity | Predict pEC50 for 513 blinded compounds | Rank 4, MAE 0.411256332902247 in the final local snapshot (`docs/leaderboards/activity/leaderboard_2026-07-03_1426JST.csv`) |
| Track 2 Structure | Predict protein-ligand 3D structures | Not pursued to final submission; local Track 2 assets were removed after the competition |

Track 1 moved through several leaderboard phases. The best late Phase 1 public
snapshot in this repo was rank 4 / MAE 0.4059243855909984 on the 2026-06-28
snapshot; the post-deadline/final-report snapshot recorded rank 4 / MAE
0.411256332902247 with the public model report linked above.

## What Is Here

This repository keeps the reproducible Track 1 code, schemas, lightweight
documentation, and public leaderboard snapshots. Generated data, checkpoints,
embeddings, Boltz outputs, model weights, private reports, and submission CSVs
are intentionally left out of git.

```text
data/                         ignored runtime parquet/data artifacts
db/                           database schemas, loaders, feature builders
docs/                         documentation, literature notes, leaderboard snapshots
docs/leaderboards/activity/   timestamped Track 1 leaderboard snapshots
track1_activity/src/          shared Track 1 loading, features, splits, metrics
track1_activity/scripts/      training, ensembling, calibration, diagnostics
track1_activity/boltz2/       Boltz-2 feature-generation pipeline for Track 1
track1_activity/submissions/  ignored Track 1 CSV submissions
structures/                   ignored Boltz-2 / structure runtime artifacts
```

## Track 1 Notes

The final Track 1 system was an ensemble workflow built around deterministic
UMAP-split validation, feature families from 2D descriptors and foundation-model
embeddings, Boltz-2-derived features, Caruana-style ensemble selection, and
submission preflight checks against trusted anchors.

Useful context:

- Issue #100 is the Phase 1/live-leaderboard chronology.
- Issue #208 is the Phase 2 answer-check and finalization log.
- `docs/track1_explain/` keeps compact public explanation and audit notes.
- `docs/leaderboards/activity/` keeps the public leaderboard snapshots used for
  retrospective checks.
- `AGENTS.md` is the durable operating guide for coding agents in this repo.

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

## Data

Challenge data comes from
[openadmet/pxr-challenge-train-test](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test).
Fresh clone setup is intentionally script-driven through the database and data
loading scripts under `db/`.

Local submission clients can contain personal account state, so they are ignored
by git. Recreate them locally rather than committing credentials or API state.
