# Track 1 Explanation Notes

This directory is a human-facing summary layer for explaining the Track 1
activity modeling work. It should stay higher level than the experiment logs:
use it to explain what the models are, why each method was tried, what survived
into the production ensemble, and what lessons are worth carrying forward.

Source-of-truth details remain elsewhere:

- Research chronology: GitHub issue #100.
- Current ensemble allow-list: `track1_activity/scripts/run_ensemble.py`.
- Model and feature scripts: `track1_activity/scripts/` and
  `track1_activity/src/`.
- Leaderboard snapshots: `docs/leaderboards/activity/`.
- Detailed run notes: `docs/superpowers/runs/`.
- Design notes: `docs/superpowers/specs/` and `docs/superpowers/plans/`.

## Files

| File | Purpose |
|---|---|
| `current_state.md` | Current Track 1 status from issue #100 and the latest local snapshots. |
| `overall_strategy_report.md` | Overall modeling strategy: Buterez-style low-fidelity transfer, log2fc usage, TabPFN readout, and rejected alternatives. |
| `dataset_split_report.md` | Dataset construction, auxiliary assay coverage, and the rationale for the canonical Morgan UMAP split. |
| `model_inventory.md` | Practical taxonomy of the model families and methods. |
| `explanation_outline.md` | Suggested order for explaining the work to another person. |
| `models/` | Per-model notes with reproducibility checks and adoption/drop rationale. |
| `features/` | Feature-block notes used by the main tabular models. |

## Maintenance Rule

Keep this directory readable. Prefer summaries, tables, and links back to the
primary artifacts over copying long experiment logs.
