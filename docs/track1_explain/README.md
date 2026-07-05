# Track 1 Explanation Notes

This directory is a lightweight public summary layer for Track 1 activity
modeling notes. Keep it higher level than raw experiment outputs and link back
to scripts, issues, or compact audit artifacts when details matter.

Source-of-truth details remain elsewhere:

- Phase 2 research log and answer-checks: GitHub issue #208.
- Phase 1 research chronology: GitHub issue #100.
- Current ensemble allow-list: `track1_activity/scripts/run_ensemble.py`.
- Model and feature scripts: `track1_activity/scripts/` and
  `track1_activity/src/`.
- Leaderboard snapshots: `docs/leaderboards/activity/`.
- Public-facing model and feature summaries: this directory.

## Files

| File | Purpose |
|---|---|
| `current_state.md` | Current Track 1 Phase 2 status, submitted candidate, and immediate next watch items. |
| `phase2_final_submission_decision.md` | Final Phase 2 submission review: id63 hold decision, top500 v3/v2.6 checks, and hedge candidates. |
| `chembl_pairwise_deep.md` | ActFound/Boltz-style same-assay ChEMBL pairwise ChemProp work and the id63 sparse composite-gate decision. |
| `twinbooster_zero_shot.md` | Negative assay-text zero-shot probe using TwinBooster as a PXR ranking/gating prior. |
| `model_inventory.md` | Practical taxonomy of the model families and methods. |
| `explanation_outline.md` | Suggested order for explaining the work to another person. |
| `phase2_*.md` | Compact Phase 2 audit notes, candidate reviews, and post-deadline answer checks. |

## Maintenance Rule

Keep this directory readable. Prefer summaries, tables, and links back to the
primary artifacts over copying long experiment logs.
