# GSL-MPP-Lite Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a low-risk GSL-MPP-inspired transductive residual-smoothing probe for Track 1 and produce OOF/test diagnostics before any leaderboard submission.

**Architecture:** Port the useful GSL-MPP idea—learning/using a molecule-to-molecule graph—without importing the upstream training framework into the pixi env. A focused `gsl_mpp.py` module builds Morgan/Tanimoto molecular similarity graphs and performs label-propagation-style residual smoothing around an existing anchor ensemble; a script cross-fits the correction on canonical UMAP folds, writes reports, CSVs, and optionally records a DB experiment.

**Tech Stack:** Python 3.12, pixi, NumPy, pandas, RDKit, SciPy/sklearn utilities already in the project, existing `track1_activity/src` loaders/evaluation/splits.

---

## File Structure

- Create `track1_activity/src/gsl_mpp.py`: pure functions for fingerprint matrices, Tanimoto similarity, top-k row-normalized adjacency, residual propagation, and grid evaluation.
- Create `track1_activity/scripts/run_gsl_mpp_lite.py`: CLI that loads train/test data and anchor predictions, performs canonical cross-fit smoothing, writes candidate CSVs/reports, and records the best diagnostic if requested.
- Create `track1_activity/tests/test_gsl_mpp.py`: synthetic unit tests for adjacency normalization, self-exclusion, and residual propagation behavior.
- Modify `track1_activity/scripts/README.md`: add a short entry for the GSL-MPP-lite probe.
- Write outputs under `track1_activity/analysis/gsl_mpp_lite/outputs/<run_name>/` and submissions under `track1_activity/submissions/ens_gsl_mpp_lite_<run_name>_<candidate>.csv`.

## Task 1: Core graph utilities

**Files:**
- Create: `track1_activity/src/gsl_mpp.py`
- Test: `track1_activity/tests/test_gsl_mpp.py`

- [ ] **Step 1: Write tests for top-k adjacency and propagation**

Add tests that assert: row sums are 1 for non-empty rows, diagonal self edges are removed when requested, top-k is respected, and a positive residual on a nearby labeled node increases an unlabeled anchor prediction.

- [ ] **Step 2: Run failing tests**

Run: `pixi run python -m unittest track1_activity.tests.test_gsl_mpp -v`
Expected: import failure because `gsl_mpp.py` does not exist yet.

- [ ] **Step 3: Implement `gsl_mpp.py`**

Implement these functions with deterministic NumPy behavior:
- `morgan_bit_matrix(smiles_list, radius=2, n_bits=2048) -> np.ndarray`
- `tanimoto_similarity(query_bits, anchor_bits) -> np.ndarray`
- `topk_row_normalized_adjacency(similarity, k, include_self=False) -> np.ndarray`
- `propagate_residuals(adjacency, residual_seed, labeled_mask, alpha=0.85, n_iter=50, clamp_labeled=True) -> np.ndarray`
- `apply_residual_correction(anchor_pred, propagated_residual, gamma, clip) -> np.ndarray`

- [ ] **Step 4: Run tests and ruff**

Run:
```bash
pixi run python -m unittest track1_activity.tests.test_gsl_mpp -v
pixi run ruff format track1_activity/src/gsl_mpp.py track1_activity/tests/test_gsl_mpp.py
pixi run ruff check track1_activity/src/gsl_mpp.py track1_activity/tests/test_gsl_mpp.py
```
Expected: tests pass and ruff passes.

## Task 2: Cross-fit GSL-MPP-lite script

**Files:**
- Create: `track1_activity/scripts/run_gsl_mpp_lite.py`
- Modify: `track1_activity/scripts/README.md`

- [ ] **Step 1: Implement CLI data flow**

The script should:
1. Load train/test SMILES and `pec50`.
2. Reconstruct `ens_caruana_bag20` OOF from stored Caruana weights, using the pattern in `run_ensemble_calibrate_importance.py`.
3. Load anchor test predictions from `--anchor-test-csv`, defaulting to `track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv` if present, otherwise `ens_caruana_bag20.csv`.
4. Build Morgan/Tanimoto graph over train+test.
5. For each canonical UMAP fold, seed residuals only on fold-train rows, propagate to fold-val rows, and fill OOF propagated residuals.
6. For test, seed residuals on all train rows and propagate to test rows.
7. Sweep a small grid: `k in {8,16,32}`, `alpha in {0.5,0.85}`, `gamma in {-0.5,-0.25,0.25,0.5}`, `clip in {0.03,0.06}`.
8. Write summary CSV, markdown report, best candidate CSVs, and run `submission_preflight.py` against the id55 anchor when a candidate CSV exists.

- [ ] **Step 2: Add README entry**

Add one row to `track1_activity/scripts/README.md` describing `run_gsl_mpp_lite.py` as an experimental transductive molecule-graph residual smoother.

- [ ] **Step 3: Run ruff**

Run:
```bash
pixi run ruff format track1_activity/scripts/run_gsl_mpp_lite.py
pixi run ruff check track1_activity/scripts/run_gsl_mpp_lite.py
```
Expected: ruff passes. Markdown formatting is not enforced.

## Task 3: Execute diagnostic probe

**Files:**
- Generated: `track1_activity/analysis/gsl_mpp_lite/outputs/<run_name>/report.md`
- Generated: `track1_activity/analysis/gsl_mpp_lite/outputs/<run_name>/summary.csv`
- Generated: optional `track1_activity/submissions/ens_gsl_mpp_lite_<run_name>_<candidate>.csv`

- [ ] **Step 1: Ensure DB is running**

Run: `pixi run db-start`
Expected: PostgreSQL on `/tmp/.s.PGSQL.5433` is available.

- [ ] **Step 2: Run script in dry diagnostic mode**

Run:
```bash
pixi run python track1_activity/scripts/run_gsl_mpp_lite.py \
  --run-name initial \
  --max-candidates 3
```
Expected: report and summary are written. No leaderboard submission is made.

- [ ] **Step 3: Interpret gates**

Gate for any future submission:
- OOF MAE improves by at least `0.0015` versus anchor OOF, or Spearman improves by at least `0.001` with MAE not worse by more than `0.0005`.
- Test mean absolute shift versus id55 anchor is <= `0.02`.
- Preflight verdict is not `HOLD`.
- Candidate is not positively aligned with known bad id56 axis.

- [ ] **Step 4: Commit implementation and diagnostics**

Run:
```bash
git add track1_activity/src/gsl_mpp.py track1_activity/tests/test_gsl_mpp.py \
  track1_activity/scripts/run_gsl_mpp_lite.py track1_activity/scripts/README.md \
  docs/superpowers/plans/2026-05-09-gsl-mpp-lite-probe.md \
  track1_activity/analysis/gsl_mpp_lite/outputs/initial

git commit -m "experiment(track1): add GSL-MPP-lite residual smoothing probe"
```

## Self-Review

- Spec coverage: the plan covers repository clone/audit indirectly via prior `ghq get zby961104/GSL-MPP`; implementation uses the same molecule-similarity graph idea but not the upstream framework to avoid environment drift.
- Placeholder scan: no TBD/TODO placeholders remain.
- Type consistency: all function names used by the script are defined in Task 1.
