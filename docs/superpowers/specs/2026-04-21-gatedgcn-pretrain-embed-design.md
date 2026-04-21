# Design: GatedGCN Pretrain-Embed + TabPFN (13th Caruana Pool Member)

- **Status**: Proposed (2026-04-21 PM)
- **Author**: Claude Code session, brainstormed with user
- **Related**:
  - Latest pool member: `tabpfn_attentivefp_pretrain_embed_umap_default` (PR #104). 12-pool caruana_bag20 OOF MAE 0.4242 (calibrated 0.4232). LB submission pending cooldown.
  - Final piece of the "apply Buterez strategy-3 to every available pretrain checkpoint" sweep (chemprop ✓, molformer_c3 ✓, kermt ✓, attentivefp ✓ — this is the last).

## Goal

Apply the pretrain-embed recipe to **GatedGCN** (ResGatedGraphConv stack). The `gatedgcn_pretrain.pt` checkpoint already exists from PR #79 (val loss 0.7394 on z-scored 2-task log2_fc). This PR implements **phase 2 only** (embed extract → TabPFN → ensemble).

## Rationale

1. **Complete the GNN-family sweep**. Pretrain-embed recipe on D-MPNN (chemprop), transformer (molformer_c3), graph-transformer (kermt), graph-attention (attentivefp) all delivered positive caruana weight. GatedGCN is the remaining available pretrain — gated edge-conditioned message passing, different inductive bias.

2. **Direct-predictor GatedGCN is modest**. `gatedgcn_pretrain_finetune_frozen_umap` OOF MAE 0.5076 (caruana weight 0.002 in 12-pool). Direct gatedgcn_optuna was dropped earlier. Recipe uplift expected in 0.46-0.50 range.

3. **Checkpoint already validated**. `track1_activity/checkpoints/gatedgcn_pretrain/pretrain.pt` (1.9 MB — notably small, hidden=128, 4 layers). `pretrain_meta.json` records val loss 0.7394.

4. **Smallest embed dim in the pool**. 128-dim (vs chemprop 256, molformer_c3 768, attentivefp 512, kermt 3200). Could be a representational bottleneck — recipe may underperform here.

## Non-goals (deferred)

- **Re-pretrain GatedGCN with higher hidden_dim** — if 128d is too narrow. Out of scope; current checkpoint stands.
- **GIN / GraphGPS pretrain from scratch** — no checkpoint available, phase-1 cost unclear ROI given they were previously pruned as weak.
- **Cross-member decorrelation optimization** — pool membership is decided by caruana_bag20 weight, not manual curation.

## Architecture

### Reuse existing infrastructure

- Pretrain checkpoint: `track1_activity/checkpoints/gatedgcn_pretrain/pretrain.pt`
- Metadata: `track1_activity/checkpoints/gatedgcn_pretrain/pretrain_meta.json` — hidden_dim=128, num_layers=4, dropout=0.05
- `GatedGCNModel` class defined in `track1_activity/scripts/run_gatedgcn_pretrain_finetune.py::84-131`
- Forward: `node_embed → convs (ResGatedGraphConv + BN) → global_mean_pool → ffn`

### Embedding extraction (Phase 2)

Write `track1_activity/scripts/run_gatedgcn_embed_extract.py`:

1. Instantiate `GatedGCNModel(in_dim, edge_dim, hidden_dim=128, num_layers=4, dropout=0.05, out_dim=2)` (pretrain had 2-head log2_fc output).
2. Load `pretrain.pt` state_dict.
3. **Replace `model.ffn` with `nn.Identity()`** — skips the final FFN head. Forward returns the 128-dim `global_mean_pool` output.
4. For each of 13,136 compounds, convert SMILES → PyG `Data` via `from_smiles`, batch, forward, collect 128-dim vectors.
5. Write to `data/gatedgcn_pretrain_embed.parquet` (index=compound_id, columns=emb_0000..emb_0127, float32, 0 NaN).

Mirrors `run_attentivefp_embed_extract.py` structurally.

### DB / feature plumbing

- No DB schema change (parquet-only).
- Register `gatedgcn_pretrain_embed` in `run_train.py::load_features` right after the `attentivefp_pretrain_embed` block.
- Add `"gatedgcn_pretrain_embed"` to the CLI `all_features` list.

### Downstream TabPFN

- `run_train.py --model tabpfn --feature gatedgcn_pretrain_embed --split umap --trials 0`
- Experiment name: `tabpfn_gatedgcn_pretrain_embed_umap_default`
- 128d — well within TabPFN 2000-dim regime; no override needed.

### Ensemble integration

- Append `"tabpfn_gatedgcn_pretrain_embed_umap_default"` to `ENSEMBLE_MODELS` in `run_ensemble.py` after the attentivefp entry.
- Re-run caruana_bag20 → 13-pool.
- Re-run calibration → updated submission CSV.

## Data flow

```
[pretrain.pt (existing, 1.9 MB)]
     ↓ run_gatedgcn_embed_extract.py (main pixi)
[data/gatedgcn_pretrain_embed.parquet (13136 × 128)]
     ↓ register in run_train.py
     ↓ run_train.py --model tabpfn --feature gatedgcn_pretrain_embed
[DB: tabpfn_gatedgcn_pretrain_embed_umap_default]
     ↓ append to ENSEMBLE_MODELS
[run_ensemble.py → caruana_bag20 → calibrate]
[ens_caruana_bag20_calibrated_best.csv]
```

## Acceptance criteria (PR merge)

1. **Extraction covers all 13,136 compounds** (0 SMILES failures tolerated; > 10 failures = BLOCKED).
2. **Single-model OOF MAE ≤ 0.48** (likely marginal like AttentiveFP — accept small overshoot with caruana evidence, following AttentiveFP precedent).
3. **caruana_bag20 weight > 0** on the new member.
4. **13-pool caruana_bag20 OOF MAE ≤ 0.4242** (12-pool baseline; improvement or tie required).
5. **Pearson r with existing 5 pretrain-embed members < 0.96**:
   - `tabpfn_chemprop_pretrain_embed_umap_default`
   - `tabpfn_molformer_c3_pretrain_embed_umap`
   - `tabpfn_2d_full_boltz_log2fc_pred_umap_default`
   - `tabpfn_kermt_pretrain_embed_umap_default`
   - `tabpfn_attentivefp_pretrain_embed_umap_default`
6. **ruff format + check clean** on new/modified files.

### Failure handling

- (2) fails by > fold-std: STOP, report BLOCKED.
- (2) fails by < fold-std: DONE_WITH_CONCERNS, let user decide whether to proceed to caruana check (AttentiveFP precedent).
- (4) fails (13-pool regresses): STOP, BLOCKED, do not submit LB.
- (5) fails: note in report, continue to Task A4. Controller decides.

## Testing

- Smoke: run full pipeline — 128d is small so TabPFN 5-fold should be < 20 min.
- No unit tests for DL code.

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| 128d is too narrow for TabPFN to exploit | Medium | Accept; if OOF MAE > 0.50, report and drop from allow-list. |
| GatedGCN encoder weights have tightly coupled feature channels (overfit to log2_fc) | Low | TabPFN is robust to low-variance features. |
| Caruana rejects as redundant with chemprop (both message-passing GNN) | Medium | Pearson r check surfaces this. Still merge even if weight = 0 (framework value). |
| 13-pool MAE flat or regresses | Medium | Accept — stop submission; drop from allow-list in follow-up. |
| Pool reaches diminishing returns (5 pretrain-embed members) | Medium | Already expected. Next exploration direction is elsewhere (re-pretrain smaller backbones, Strategy 6 adaptive readout, LB-tailored experiments). |

## ETA

- Embedding extraction: **~1-2 min** (128d, small model, 13k compounds on GPU)
- Feature registration + commit: **~5 min**
- TabPFN 5-fold: **~15-25 min**
- Ensemble + calibration: **~10-15 min**
- PR: **~5 min**

**Total: ~40-60 min**. Fastest of the four embed-PRs.

## Out of scope (future PRs)

- **Re-pretrain GatedGCN with hidden_dim=256 or 512** — if current 128d embed is weak. Needs phase 1 rerun (~30 min pretrain).
- **GIN / GraphGPS pretrain from scratch** — deferred; phase 1 cost + weak baseline.
- **Ensemble strategy reconsideration** — currently caruana_bag20; alternatives (fold_l2, simple_avg) available in run_ensemble.py for A/B.
