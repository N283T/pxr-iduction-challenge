# Buterez Strategy 6 GatedGCN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test Buterez 2024 Strategy 6 on PXR by freezing the low-fidelity-pretrained GatedGCN encoder and fine-tuning only an adaptive graph readout on pEC50.

**Architecture:** Reuse the existing GatedGCN log2fc pretrain checkpoint as the low-fidelity encoder. Add a Set Transformer-style adaptive readout that consumes frozen node embeddings before global pooling, trains only the readout/head on pEC50 across canonical UMAP folds, and records OOF/test predictions as a standalone model for later Caruana bakeoff.

**Tech Stack:** Python 3.12, pixi, PyTorch, PyTorch Geometric, existing GatedGCN model/checkpoint, existing Track 1 data/evaluation/splits.

---

## Files

- Create `track1_activity/src/adaptive_readout.py`: Set Transformer readout modules and a small readout regressor.
- Create `track1_activity/tests/test_adaptive_readout.py`: tests for permutation invariance, masking/padding, and output shape.
- Create `track1_activity/scripts/run_gatedgcn_strategy6.py`: fold training script for frozen GatedGCN encoder + adaptive readout.
- Modify `track1_activity/scripts/README.md`: add a Strategy 6 script entry.
- Output reports under `track1_activity/analysis/strategy6_gatedgcn/outputs/<run_name>/`.

## Task 1: Adaptive readout module via TDD

- [ ] Write failing tests for permutation invariance and padding mask behavior.
- [ ] Run `pixi run python -m unittest track1_activity.tests.test_adaptive_readout -v` and confirm import failure.
- [ ] Implement `SetAttentionBlock`, `PoolingByMultiheadAttention`, `SetTransformerReadout`, and `AdaptiveReadoutRegressor`.
- [ ] Run tests and ruff on the module/test.

## Task 2: GatedGCN Strategy 6 script

- [ ] Implement a script that loads the pretrained GatedGCN encoder, freezes it, extracts per-node embeddings batch-wise, pads node sequences per batch, trains the adaptive readout on pEC50 fold-train rows, and predicts fold-val/test rows.
- [ ] Use defaults: hidden_dim from checkpoint metadata/model params, readout_dim 128, heads 4, inducing seeds 1, lr 1e-3, max_epochs 120, patience 15, batch_size 64.
- [ ] Save `summary.csv`, `fold_metrics.csv`, `report.md`, and submission CSV.
- [ ] Record experiment when `--record` is passed.
- [ ] Add README entry and run ruff.

## Task 3: Smoke and diagnostic run

- [ ] Run smoke: `pixi run python track1_activity/scripts/run_gatedgcn_strategy6.py --run-name smoke --max-epochs 3 --batch-size 128 --no-record`.
- [ ] Run diagnostic: `pixi run python track1_activity/scripts/run_gatedgcn_strategy6.py --run-name initial --no-record`.
- [ ] Gate: only consider Caruana ADD if single OOF MAE <= 0.48 or residual correlation is clearly low with no major Spearman collapse.

## Task 4: Verify and commit

- [ ] Run unit tests and ruff.
- [ ] Commit with `experiment(track1): add Buterez strategy 6 GatedGCN probe`.

## Self-Review

- This plan targets the exact Strategy 6 distinction: frozen LF encoder, trainable adaptive readout only.
- It intentionally starts with GatedGCN because the PyG encoder is easy to expose at node level.
- No leaderboard submission is automatic.
