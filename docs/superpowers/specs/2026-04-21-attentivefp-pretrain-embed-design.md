# Design: AttentiveFP Pretrain-Embed + TabPFN (12th Caruana Pool Member)

- **Status**: Proposed (2026-04-21 PM)
- **Author**: Claude Code session, brainstormed with user
- **Related**:
  - Latest pool member: `tabpfn_kermt_pretrain_embed_umap_default` (PR #103). Lifted 11-pool caruana_bag20 OOF MAE to 0.4268, LB MAE 0.4318, LB rank 8.
  - Same recipe (Buterez 2024 strategy-3: continued-pretrain on log2_fc → frozen → embed → TabPFN) on yet another backbone.

## Goal

Apply the pretrain-embed recipe to **AttentiveFP** (PyG graph-attention network). The `attentivefp_pretrain.pt` checkpoint already exists from PR #79 (pretrain phase 1 complete, val loss 0.7012 on z-scored 2-task log2_fc). This PR implements **phase 2 only**: embedding extraction → parquet → TabPFN training → ensemble integration. Scope much smaller than KERMT.

## Rationale

1. **Pool is pretrain-embed-dominated**. After KERMT addition, 4 of top-4 caruana-weighted members (80% combined weight) are strategy-3 variants: 2d_full_boltz+predLF (0.263), chemprop_pretrain_embed (0.252), molformer_c3_pretrain_embed (0.161), kermt_pretrain_embed (0.128). AttentiveFP is the next natural backbone — graph-attention family (different from chemprop D-MPNN, molformer transformer, kermt graph-transformer).

2. **Direct-predictor AttentiveFP is weak**. `attentivefp_optuna_umap` (direct) OOF MAE 0.5280, caruana weight 0.0019. `attentivefp_pretrain_finetune_frozen_umap` (frozen+head-FT) 0.5222, weight ~0. In chemprop land the same recipe progression yielded: direct 0.528 → frozen+head-FT ~0.51 → **TabPFN-on-embed 0.437** (+0.09 MAE improvement). If AttentiveFP follows the same pattern, expected embed-MAE ~0.44-0.46.

3. **Pretrain checkpoint already validated**. `track1_activity/checkpoints/attentivefp_pretrain/pretrain.pt` exists (36 MB, created 2026-04-19). Best val loss 0.7012 on z-scored 2-head log2_fc. Reuse as-is; no retraining needed.

4. **Small scope vs KERMT PR #103**:
   - No external repo / environment split (PyG already in main pixi env)
   - No weight download
   - No upstream bug workaround
   - No pretrain phase — checkpoint is ready
   - Estimated 3-4 h end-to-end (vs KERMT's 5-6 h)

## Non-goals (deferred)

- **Re-pretrain on improved log2_fc task** — checkpoint is good enough (val loss 0.70 z-scored = 0.29 un-normalized MAE). New retraining is out of scope.
- **Finetune-then-embed variant** (pretrain + head-FT + extract post-FT embed). Chemprop precedent uses pure pretrain embed; stick with that.
- **GROVER_large follow-up, GatedGCN embed variant, other GNNs** — wait for AttentiveFP result first.
- **Ensemble+LB strategy changes** — continue the current caruana_bag20 + linear_pos calibration + LB submission cadence.

## Architecture

### Reuse existing infrastructure

- Pretrain checkpoint: `track1_activity/checkpoints/attentivefp_pretrain/pretrain.pt`
- Metadata: `track1_activity/checkpoints/attentivefp_pretrain/pretrain_meta.json` — hidden_channels=512, num_layers=4, num_timesteps=3, dropout=0.1, target_means/stds
- PyG's `torch_geometric.nn.models.AttentiveFP` (already imported in the project)
- `data/` parquet consumer pattern already established (3 pretrain-embed precedents)

### Embedding extraction (Phase 2)

Write `track1_activity/scripts/run_attentivefp_embed_extract.py`:

1. Load pretrain checkpoint + metadata (2-output-channel).
2. Instantiate `AttentiveFP(hidden_channels=512, out_channels=2, num_layers=4, num_timesteps=3, ...)`.
3. Load state_dict, move to GPU, `.eval()`.
4. **Replace `model.lin2` with `nn.Identity()`** — skips the final 512→2 projection. Forward returns the 512d molecule-level readout (post-GRU, pre-final projection). This is the "fingerprint".
5. For each of 13,136 compounds, convert SMILES → PyG `Data` via `torch_geometric.utils.from_smiles`, batch, forward, collect the 512d vector.
6. Write to `data/attentivefp_pretrain_embed.parquet` (index=compound_id, columns=emb_0000..emb_0511).

SMILES parsing: use the same `from_smiles` convention as `run_attentivefp_pretrain_finetune.py` (file already imports it).

Any SMILES that fail `from_smiles` (should be 0 — all compounds were already successfully processed during pretrain): log warnings, proceed with the 13,134+ that succeed. Converter must handle the row-count mismatch gracefully.

Output format matches existing precedents: parquet file, `compound_id` as index, 512 columns named `emb_0000..emb_0511`, float32 dtype, no NaN.

### DB / feature plumbing

- No DB schema change (parquet-only, mirrors chemprop/molformer_c3/kermt pattern).
- Register `attentivefp_pretrain_embed` in `run_train.py::load_features` (~line 436 area, right after `kermt_pretrain_embed` branch). Identical structure: read parquet, reindex by compound_id, NaN guard, return X_train/X_test.
- Add `"attentivefp_pretrain_embed"` to the VALID_FEATURES list (one line).

### Downstream TabPFN

- `run_train.py --model tabpfn --feature attentivefp_pretrain_embed --split umap --trials 0`
- Experiment name auto-derived: `tabpfn_attentivefp_pretrain_embed_umap_default`
- 512d < 2000d → **no `ignore_pretraining_limits` override needed** (works within TabPFN v2.6's supported regime — cleaner than KERMT's 3200d case).
- 5-fold UMAP (canonical: seed=42, n_clusters=50, Morgan+Jaccard).
- DB record with on_conflict_replace=True.

### Ensemble integration (Approach C: add-only)

- Append `"tabpfn_attentivefp_pretrain_embed_umap_default"` to `ENSEMBLE_MODELS` tuple in `run_ensemble.py` after the `"tabpfn_kermt_pretrain_embed_umap_default"` entry.
- Re-run `run_ensemble.py` → caruana_bag20 re-weights the 12-member pool.
- If caruana weight = 0 or 12-pool MAE regresses, keep on main but drop from allow-list in a follow-up 1-line PR.
- Re-run `run_ensemble_calibrate.py` → 4-way nested CV → `ens_caruana_bag20_calibrated_best.csv` rewritten.

## Data flow

```
[pretrain.pt (existing, 36 MB)]
     ↓ run_attentivefp_embed_extract.py (main pixi)
[data/attentivefp_pretrain_embed.parquet (13136 × 512)]
     ↓ register in run_train.py
[feature loader branch, VALID_FEATURES entry]
     ↓ run_train.py --model tabpfn --feature attentivefp_pretrain_embed --split umap --trials 0
[DB: tabpfn_attentivefp_pretrain_embed_umap_default, OOF rows, experiment_summary]
     ↓ append to ENSEMBLE_MODELS
[run_ensemble.py --strategy caruana_bag20]
[ens_caruana_bag20 OOF (12-pool) + test preds]
     ↓ run_ensemble_calibrate.py
[ens_caruana_bag20_calibrated_best.csv (513 rows)]
     ↓ optional api.py submit
[LB]
```

## Acceptance criteria (PR merge)

1. **Extraction completes on all 13,136 compounds** (matching KERMT/chemprop precedents). If some SMILES fail `from_smiles`, record the count; < 10 failures acceptable, > 10 is BLOCKED.
2. **Single-model OOF MAE ≤ 0.48** (loose bar; matches recipe precedents 0.4371-0.4752 range).
3. **caruana_bag20 weight > 0** on the new member.
4. **12-pool caruana_bag20 OOF MAE ≤ 0.4268** (post-KERMT baseline; improvement or tie required).
5. **Pearson r with each of the 4 existing pretrain-embed members < 0.96**:
   - `tabpfn_chemprop_pretrain_embed_umap_default`
   - `tabpfn_molformer_c3_pretrain_embed_umap`
   - `tabpfn_2d_full_boltz_log2fc_pred_umap_default`
   - `tabpfn_kermt_pretrain_embed_umap_default`
   
   AttentiveFP is graph-attention; closest relative is chemprop (D-MPNN, also GNN). Expect chemprop correlation highest (~0.95), but still < 0.96.

6. **ruff format + ruff check clean** on all modified/new files.
7. **on_conflict_replace=True** inherited from existing record_experiment paths.

### Failure handling

- (1) fails (many SMILES fail): inspect which compounds. If they were previously successfully pretrained, there's a bug. Report BLOCKED with the compound IDs.
- (2) fails (MAE > 0.48): AttentiveFP may have weak embeddings. Try alternative extraction points (pre-GRU vs post-GRU). Report to user before further scope changes.
- (3) fails (caruana weight 0): redundant with existing pool. Still merge (framework record), drop from allow-list in follow-up.
- (4) fails (12-pool regresses): STOP; do not submit LB; drop member from allow-list.
- (5) fails (Pearson r > 0.96 with chemprop): member is effectively a D-MPNN dup; still consider merging but note in PR body, drop from allow-list if LB disconnect risk too high.

## Testing

- Smoke: 2-fold TabPFN with first 100 compounds (if feasible). Otherwise just proceed with 5-fold full run.
- No unit tests for DL code (existing convention).
- ruff as the only automatic gate.

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| `from_smiles` fails on some of the 13,136 compounds | Low | Same function was used during pretrain with 100% success; unlikely to fail here. |
| AttentiveFP embedding is highly correlated with chemprop (both GNN, similar graph message-passing bias) | Medium | Expect r ~0.94-0.96. Caruana can still give weight to slightly decorrelated signal. If r > 0.96, drop from allow-list. |
| Pretrain checkpoint's 512d embedding has too-low-variance dims (overfitted to log2_fc) | Low | TabPFN handles low-variance features naturally. |
| 12-pool OOF regresses (caruana's new weight steals from another strong member destructively) | Low | Caruana_bag20 is discrete count-based, less prone to destructive weight reallocation than vanilla (cf. issue #82). |
| Wall-clock budget overrun (TabPFN full 5-fold on 512d) | Very low | 512d << 3200d KERMT; expect faster run (< 30 min). |

## ETA

- Embedding extraction (13,136 compounds, 512d forward on GPU): **~5-15 min**
- `run_train.py` feature registration + commit: **~10 min**
- TabPFN 5-fold training: **~20-30 min**
- Ensemble re-run + calibration: **~10-15 min**
- PR + LB submission: **~10 min**

**Total: ~1-1.5 h**. Same-day turnaround, no overnight scheduling.

## Out of scope (future PRs)

- **Apply same recipe to GatedGCN** (pretrain checkpoint exists at `track1_activity/checkpoints/gatedgcn_pretrain/`). Similar smaller-scope PR if AttentiveFP succeeds.
- **Re-pretrain AttentiveFP / GatedGCN with improved hyperparams or longer epochs** — current checkpoints are good enough; re-training is not justified by evidence.
- **Graph-level adapters / prompt-tuning on top of pretrained encoder** — speculative, not supported by literature for this task.
