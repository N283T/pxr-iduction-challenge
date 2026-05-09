# Buterez Strategy 6 ChemProp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and evaluate Buterez 2024 Strategy 6 using the existing ChemProp log2_fc-pretrained D-MPNN checkpoint.

**Architecture:** Freeze the pretrained ChemProp message-passing encoder, expose per-atom hidden states before ChemProp's fixed mean/norm aggregation, and train only an adaptive Set Transformer readout plus regression head on pEC50 UMAP folds. Save OOF/test predictions, metrics, and a decision report without submitting directly.

**Tech Stack:** Python 3.12, pixi, ChemProp 2.x, PyTorch, existing `adaptive_readout.py`, PostgreSQL-backed loaders, UMAP split utilities.

---

### Task 1: Reusable ChemProp Strategy 6 module

**Files:**
- Create: `track1_activity/src/chemprop_strategy6.py`
- Test: `track1_activity/tests/test_chemprop_strategy6.py`

- [ ] Write tests for flat node embeddings -> padded graph tensor/mask.
- [ ] Run tests and confirm they fail because the module does not exist.
- [ ] Implement `pad_node_embeddings`, `ChempropNodeEncoder`, `ChempropStrategy6Regressor`, and `freeze_all`.
- [ ] Run tests and confirm they pass.

### Task 2: Training/evaluation script

**Files:**
- Create: `track1_activity/scripts/run_chemprop_strategy6.py`
- Modify: `track1_activity/scripts/README.md`

- [ ] Load `track1_activity/checkpoints/chemprop_pretrain/pretrain.pt` and rebuild the 2-head pretrain MPNN architecture.
- [ ] Load pretrained weights, freeze `message_passing`, train adaptive readout only on pEC50.
- [ ] Use canonical UMAP folds, write OOF metrics, fold metrics, residual correlation vs current ensemble, and submission CSV.
- [ ] Add a README entry for the script.

### Task 3: Run probes and decide

**Files:**
- Output: `track1_activity/analysis/strategy6_chemprop/outputs/<run>/report.md`

- [ ] Run a short smoke test with 2 folds / few epochs.
- [ ] Run at least one realistic sweep using small readout defaults first.
- [ ] Compare against GatedGCN Strategy 6, ChemProp pretrain-embed TabPFN, and the direct-submit gate.
- [ ] Commit source/docs and report generated analysis artifacts that are small enough for git.
