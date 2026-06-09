# Track 1 Explanation Notes

This directory is a human-facing summary layer for explaining the Track 1
activity modeling work. It should stay higher level than the experiment logs:
use it to explain what the models are, why each method was tried, what survived
into the production ensemble, and what lessons are worth carrying forward.

Source-of-truth details remain elsewhere:

- Phase 2 research log and answer-checks: GitHub issue #208.
- Phase 1 research chronology: GitHub issue #100.
- Current ensemble allow-list: `track1_activity/scripts/run_ensemble.py`.
- Model and feature scripts: `track1_activity/scripts/` and
  `track1_activity/src/`.
- Leaderboard snapshots: `docs/leaderboards/activity/`.
- Detailed run notes: `docs/superpowers/runs/`.
- Design notes: `docs/superpowers/specs/` and `docs/superpowers/plans/`.

## Files

| File | Purpose |
|---|---|
| `current_state.md` | Phase 1 Track 1 status from issue #100 and local snapshots. Phase 2 updates live in issue #208 until refreshed. |
| `phase2_answer_check_report.md` | Japanese Phase 2 answer-check report using released Analog Set 1 labels, with figures for anchors, error bins, OOF-vs-AS1, and proxy signals. |
| `phase2_compound_case_study_report.md` | Japanese compound-level Phase 2 case study with molecule structure figures for low-tail, 3-4 bidirectional, high-tail, and well-predicted id55 cases. |
| `overall_strategy_report.md` | Overall modeling strategy: Buterez-style low-fidelity transfer, log2fc usage, TabPFN readout, and rejected alternatives. |
| `dataset_split_report.md` | Dataset construction, auxiliary assay coverage, and the rationale for the canonical Morgan UMAP split. |
| `ensemble_calibration_report.md` | Ensemble weighting, Caruana selection, post-hoc calibration, and why late small calibration moves were paused. |
| `submission_preflight_report.md` | Submission preflight checks: anchor shifts, CSV sanity, known-bad axes, and PASS/CAUTION/HOLD logic. |
| `negative_results_report.md` | High-level summary of model families and probes that were dropped because they were weak, redundant, or LB-negative. |
| `foundation_model_lessons_report.md` | Practical interpretation of why generic foundation models often underperformed on this fixed-target PXR task. |
| `external_data_report.md` | How external ChEMBL/related-target data were tested, why direct use was deferred, and what remains useful for Phase 2. |
| `model_inventory.md` | Practical taxonomy of the model families and methods. |
| `explanation_outline.md` | Suggested order for explaining the work to another person. |
| `models/` | Per-model notes with reproducibility checks and adoption/drop rationale. |
| `features/` | Feature-block notes used by the main tabular models. |

## Maintenance Rule

Keep this directory readable. Prefer summaries, tables, and links back to the
primary artifacts over copying long experiment logs.
