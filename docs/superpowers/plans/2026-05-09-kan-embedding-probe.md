# KAN on Frozen Embeddings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether pykan/KAN is useful as a high-fidelity regressor on frozen low-fidelity ChemProp molecule embeddings.

**Architecture:** Reuse existing `chemprop_pretrain_embed.parquet` features and canonical UMAP folds. Train a compact pykan `KAN(width=[d, hidden, 1])` model per fold on standardized embeddings and standardized pEC50 targets, save OOF/test predictions and diagnostics. This is a cheap probe of KAN as a TabPFN/MLP replacement, not a production submission path.

**Tech Stack:** Python 3.12, pixi, pykan, PyTorch, pandas/numpy, existing DB loaders and CV/evaluation utilities.

---

### Task 1: Utilities and tests

**Files:**
- Create: `track1_activity/src/kan_embed.py`
- Create: `track1_activity/tests/test_kan_embed.py`

- [ ] Test standardization is train-fold-only and invertible for targets.
- [ ] Test KAN width construction and optional PCA feature compression helpers.
- [ ] Implement the helper functions used by the training script.

### Task 2: KAN experiment script

**Files:**
- Create: `track1_activity/scripts/run_kan_embed.py`
- Modify: `track1_activity/scripts/README.md`

- [ ] Load a named embedding parquet by train/test row order.
- [ ] Train pykan models per UMAP fold with `auto_save=False` and `symbolic_enabled=False`.
- [ ] Save fold metrics, OOF predictions, test predictions, summary, and report.
- [ ] Optionally record the best full-fold run to DB for Caruana diagnostics.

### Task 3: Probes and decision

**Files:**
- Output: `track1_activity/analysis/kan_embed/outputs/<run>/report.md`

- [ ] Run a smoke test.
- [ ] Run compact KAN sweeps on `chemprop_pretrain_embed`.
- [ ] Compare to `tabpfn_chemprop_pretrain_embed_umap_default` and decide whether KAN readout/GNN-KAN is worth deeper work.
