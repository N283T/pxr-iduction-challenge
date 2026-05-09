# GSL-MPP Learned Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a learned GSL-MPP-style molecule graph residual model for Track 1 and evaluate it with canonical cross-fit OOF diagnostics.

**Architecture:** Keep the upstream GSL-MPP idea but adapt it to this repo: use Morgan fingerprints to build an initial molecule-to-molecule graph over train+test nodes, learn a weighted-cosine graph over compact node features, blend learned and initial adjacency, pass node features through dense GCN layers, and predict residuals around the current ensemble anchor. The script trains full-batch masked-loss models per UMAP fold for OOF predictions and a final all-train model for test predictions.

**Tech Stack:** Python 3.12, pixi, PyTorch 2.10, RDKit, NumPy, pandas, sklearn TruncatedSVD/StandardScaler, existing Track 1 loaders/evaluation/splits.

---

## File Structure

- Create `track1_activity/src/gsl_mpp_torch.py`: PyTorch modules and training helpers for dense learned molecule-graph regression.
- Create `track1_activity/tests/test_gsl_mpp_torch.py`: unit tests for graph learner normalization, masked loss, and learning on a toy graph.
- Create `track1_activity/scripts/run_gsl_mpp_learned.py`: CLI that builds features/graphs, runs cross-fit training, writes reports/candidates, and optionally records best OOF predictions.
- Modify `track1_activity/scripts/README.md`: add the learned GSL-MPP probe entry.
- Outputs live under `track1_activity/analysis/gsl_mpp_learned/outputs/<run_name>/`; candidate CSVs live under `track1_activity/submissions/`.

## Task 1: Learned graph model module via TDD

**Files:**
- Create: `track1_activity/tests/test_gsl_mpp_torch.py`
- Create: `track1_activity/src/gsl_mpp_torch.py`

- [ ] **Step 1: Write failing tests**

Tests should cover:
1. `build_topk_adjacency_torch` returns row-normalized adjacency with zero diagonal when `include_self=False`.
2. `masked_mae_loss` uses only mask-true entries.
3. A tiny `DenseGslMppRegressor` can overfit a 4-node residual target with two labeled nodes and produce output shape `(n_nodes,)`.

- [ ] **Step 2: Run failing tests**

Run: `pixi run python -m unittest track1_activity.tests.test_gsl_mpp_torch -v`
Expected: import failure because `gsl_mpp_torch.py` does not exist.

- [ ] **Step 3: Implement minimal model module**

Implement:
- `build_topk_adjacency_torch(similarity, k, include_self=False)`
- `masked_mae_loss(pred, target, mask)`
- `DenseGraphConvolution`
- `WeightedCosineGraphLearner`
- `DenseGslMppRegressor`
- `fit_dense_gsl_mpp(...)` returning prediction array and loss history.

- [ ] **Step 4: Verify tests and lint**

Run:
```bash
pixi run python -m unittest track1_activity.tests.test_gsl_mpp_torch -v
pixi run ruff format track1_activity/src/gsl_mpp_torch.py track1_activity/tests/test_gsl_mpp_torch.py
pixi run ruff check track1_activity/src/gsl_mpp_torch.py track1_activity/tests/test_gsl_mpp_torch.py
```
Expected: tests and ruff pass.

## Task 2: Learned GSL-MPP experiment script

**Files:**
- Create: `track1_activity/scripts/run_gsl_mpp_learned.py`
- Modify: `track1_activity/scripts/README.md`

- [ ] **Step 1: Implement CLI**

The script should:
1. Load train/test SMILES and pEC50.
2. Reconstruct `ens_caruana_bag20` OOF as the train anchor.
3. Load id55 test CSV by default as the test anchor.
4. Build Morgan fingerprints for train+test; build initial Tanimoto top-k adjacency.
5. Build compact node features: TruncatedSVD Morgan components plus standardized anchor prediction scalar.
6. For each canonical UMAP fold, train the learned graph model with loss mask on fold-train rows and predict fold-val residuals.
7. Train a final all-train model and predict test residuals.
8. Sweep gamma/clip only at correction time, write summary/report/candidate CSVs, and run preflight for top candidates.

Default knobs for a first run:
- `--svd-components 256`
- `--init-k 32`
- `--learned-k 32`
- `--epochs 600`
- `--hidden-dim 128`
- `--graph-skip 0.8`
- `--lr 0.003`
- `--weight-decay 0.0001`
- gamma grid `-0.5,-0.25,0.25,0.5`
- clip grid `0.03,0.06`

- [ ] **Step 2: Add README entry**

Add one row for `run_gsl_mpp_learned.py`.

- [ ] **Step 3: Verify lint**

Run:
```bash
pixi run ruff format track1_activity/scripts/run_gsl_mpp_learned.py
pixi run ruff check track1_activity/scripts/run_gsl_mpp_learned.py
```
Expected: ruff passes.

## Task 3: Execute first learned probe

**Files:**
- Generated: `track1_activity/analysis/gsl_mpp_learned/outputs/initial/report.md`
- Generated: `track1_activity/analysis/gsl_mpp_learned/outputs/initial/summary.csv`

- [ ] **Step 1: Run short smoke experiment**

Run:
```bash
pixi run python track1_activity/scripts/run_gsl_mpp_learned.py \
  --run-name smoke \
  --epochs 5 \
  --svd-components 32 \
  --hidden-dim 32 \
  --max-candidates 1 \
  --skip-preflight
```
Expected: script completes and writes smoke report.

- [ ] **Step 2: Run initial full diagnostic**

Run:
```bash
pixi run python track1_activity/scripts/run_gsl_mpp_learned.py \
  --run-name initial \
  --epochs 600 \
  --max-candidates 3
```
Expected: report/summary/preflight outputs are written. No leaderboard submission is made.

- [ ] **Step 3: Decision gate**

Do not submit unless:
- OOF MAE delta <= `-0.0015` or Spearman delta >= `+0.0010` with MAE delta <= `+0.0005`.
- mean absolute test shift <= `0.02`.
- preflight verdict is not `HOLD`.

## Task 4: Commit

- [ ] **Step 1: Run final verification**

Run:
```bash
pixi run python -m unittest track1_activity.tests.test_gsl_mpp track1_activity.tests.test_gsl_mpp_torch -v
pixi run ruff check track1_activity/src/gsl_mpp.py track1_activity/src/gsl_mpp_torch.py track1_activity/tests/test_gsl_mpp.py track1_activity/tests/test_gsl_mpp_torch.py track1_activity/scripts/run_gsl_mpp_lite.py track1_activity/scripts/run_gsl_mpp_learned.py
```

- [ ] **Step 2: Commit**

Run:
```bash
git add docs/superpowers/plans/2026-05-09-gsl-mpp-learned-probe.md \
  track1_activity/src/gsl_mpp_torch.py track1_activity/tests/test_gsl_mpp_torch.py \
  track1_activity/scripts/run_gsl_mpp_learned.py track1_activity/scripts/README.md \
  track1_activity/analysis/gsl_mpp_learned/outputs

git commit -m "experiment(track1): add learned GSL-MPP graph probe"
```

## Self-Review

- Spec coverage: this plan implements learned graph structure, masked transductive training, cross-fit OOF, final test prediction, reports, preflight, and no automatic submission.
- Placeholder scan: no TBD/TODO placeholders remain.
- Type consistency: function names in tests, module, and scripts match.
