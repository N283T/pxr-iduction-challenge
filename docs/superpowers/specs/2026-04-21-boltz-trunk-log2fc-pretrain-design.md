# Design: Boltz-2 Trunk × log2_fc Pretrain (Buterez Strategy-3, Boltz backbone)

- **Status**: Proposed (2026-04-21 evening)
- **Author**: Claude Code session, brainstormed with user
- **Related**:
  - Completes the "Buterez strategy-3 sweep" by adding the Boltz-2 backbone. Prior strategy-3 members: chemprop (PR #87), molformer_c3 (PR #98), kermt (PR #103), attentivefp (PR #104), gatedgcn h512 (PR #105).
  - Memory note `project_boltz2_trunk_pool_retarget` (2026-04-19, #74): current `tabpfn_pooled_boltz_umap_default` (MAE 0.486) uses raw Boltz trunk without log2_fc pretrain. This PR adds the pretrain step.
  - User's Boltz fork (`~/ghq/github.com/N283T/boltz`) has `--embeddings_only` flag that runs trunk only (skips diffusion/confidence/affinity). Combined with `--recycling_steps 1` = "fast trunk" mode for bulk extraction.

## Goal

Apply the Buterez 2024 strategy-3 recipe to the Boltz-2 trunk. Expected uplift pattern (per prior backbones): existing `tabpfn_pooled_boltz_umap_default` OOF MAE **0.486** → pretrained-on-log2_fc variant OOF MAE in the **0.44–0.46** range (chemprop saw direct 0.528 → pretrain-embed 0.437, a -0.09 swing; kermt direct-frozen-head ~0.51 → pretrain-embed 0.448).

The blocker is that existing Boltz-2 trunk embeddings cover only 4,653 compounds (train + test = 4,140 + 513). Strategy-3 needs a **large weak-label corpus** for pretrain, so we extend trunk extraction to all 13,136 compounds. Quality tradeoff (reduced recycling steps) is acceptable per user.

## Rationale

1. **Strategy-3 sweep completion**. Boltz-2 is the last backbone in the current pool without a log2_fc-pretrained variant. Its 1024-dim trunk (s_prot_mean 384 + s_lig_mean 384 + z_interface_mean/max 128+128) is protein-ligand-interaction-aware, which is structurally different from the chemistry-only backbones.

2. **Expected OOF uplift matches historical pattern**. Direct-frozen Boltz trunk via TabPFN: 0.486. Pretrain-on-log2_fc adds ~0.04-0.09 improvement in other backbones' cases. Projected: MAE 0.39-0.44, solid pool contribution.

3. **Decorrelation hypothesis**. Boltz trunk captures protein-ligand structural coupling; log2_fc pretrain aligns this with our task. Expected Pearson r < 0.93 with existing chemistry-backbone members (based on current `tabpfn_pooled_boltz_umap` r = 0.90-0.95 with other pool members).

4. **User's fork already has `--embeddings_only` mode**. Trunk extraction is a supported CLI flag, not a bespoke hack. `--recycling_steps 1` (vs default 3) cuts compute ~3x.

## Non-goals (deferred)

- **Full Boltz-2 backbone retrain on log2_fc**. Requires Boltz training code + weeks of GPU. Out of scope.
- **Add affinity head finetune**. Orthogonal; could be follow-up.
- **Use 3d_ligand (4,651-compound) features separately as pool member**. Memory + experiments show weak standalone; user note "3D use するなら Uni-Mol 路線" — deferred.
- **Uni-Mol v2 pretrain on ETKDG + inference on Boltz pose**. Listed as future direction; this PR is the cheaper alternative.
- **Rerun existing 4,653-compound Boltz with rcycle=1 for consistency**. Existing trunk (rcycle=3) is higher-quality; retrain MLP on mixed-quality inputs (4,653 high + 8,483 fast) introduces input-distribution skew. Accept this skew (user-approved); if issue, redo train/test with fast mode.

## Architecture

### Phase 1: Extend Boltz-2 trunk to all 13,136 compounds (fast mode)

- Identify **8,483 compounds** without Boltz trunk (= 13,136 - 4,653)
- Build Boltz YAML inputs for these via existing builder (`track2_structure/scripts/boltz2_build_inputs.py`-style SQL query)
- Run Boltz fork with fast trunk mode:
  ```bash
  cd ~/ghq/github.com/N283T/boltz
  boltz predict <input_dir>/ \
      --embeddings_only \
      --recycling_steps 1 \
      --out_dir <output_dir>
  ```
- Per-compound output: `<output>/predictions/<id>/embeddings_<id>.npz` containing:
  - `s` (per-residue/atom single representation, ~434 prot + ligand_atoms tokens × 384 dim)
  - `z` (pair representation, tokens × tokens × 128 dim)

- Apply existing pooling (`track1_activity/scripts/boltz_affhead/01_pool_embeddings.py`-style) to derive the same 1024d feature schema:
  - `s_prot_mean` (384) — mean over 434 PXR residue tokens
  - `s_lig_mean` (384) — mean over ligand atom tokens
  - `z_if_mean` (128) — mean over (core_pocket × ligand_atoms)
  - `z_if_max` (128) — max over same

### Phase 2: New DB table `compound_boltz2_trunk_fast`

Separate from existing `compound_boltz2` to avoid confusing rcycle=3 high-quality data with rcycle=1 fast data. Schema:

```sql
CREATE TABLE IF NOT EXISTS compound_boltz2_trunk_fast (
    compound_id INTEGER PRIMARY KEY REFERENCES compounds(id),
    s_prot_mean FLOAT[] NOT NULL,       -- 384d
    s_lig_mean FLOAT[] NOT NULL,        -- 384d
    z_if_mean FLOAT[] NOT NULL,         -- 128d
    z_if_max FLOAT[] NOT NULL,          -- 128d
    recycling_steps INTEGER NOT NULL,   -- record rcycle used (1 for new fast rows)
    source_npz_path TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

For train+test (4,653 compounds) we **reuse existing `compound_boltz2` + pooled.parquet** at rcycle=3 quality. For pretrain MLP input we **concatenate 4,653 high-quality + 8,483 fast-quality** into a unified 13,136 × 1024 matrix. Flag the quality difference in the MLP training data for optional analysis.

### Phase 3: MLP pretrain on log2_fc (Buterez strategy-3 core)

Architecture variants to try (user: "MLP は色々試す"):

```
Variant A (baseline, simple):
  Linear(1024 → 256) → GELU → Dropout → Linear(256 → 2)

Variant B (wider):
  Linear(1024 → 512) → GELU → Dropout → Linear(512 → 256) → GELU → Dropout → Linear(256 → 2)

Variant C (attention):
  Linear(1024 → 256) → LayerNorm → Self-Attention(heads=4) → Linear(256 → 256) → GELU → Linear(256 → 2)
```

Training:
- Data: 13,136 compounds × 1024d Boltz trunk
- Targets: [log2fc_8p25, log2fc_33] (z-scored, NaN-masked MSE)
- Split: 90/10 random (seed=42)
- Optimizer: AdamW, lr 1e-3 (MLP) or 3e-4 (attention)
- Epochs: 50 max, early stop patience 10 on val_loss
- Batch size: 256 (fits easily in 16GB VRAM for MLP)

Pick the variant with best val_loss **that also has distinct enough penultimate representation** to be useful downstream. Quick sanity check: TabPFN OOF MAE on a 2-fold smoke test before committing to 5-fold.

### Phase 4: Extract frozen embedding

From the chosen MLP variant:
- Forward all 13,136 compounds through trained encoder
- Extract output of penultimate hidden layer (256d or 512d)
- Store as `data/boltz_trunk_pretrain_embed.parquet` (index=compound_id, columns=emb_0000..emb_NNNN)

### Phase 5: TabPFN 5-fold + ensemble integration

- Register `boltz_trunk_pretrain_embed` feature in `run_train.py::load_features`
- Run `run_train.py --model tabpfn --feature boltz_trunk_pretrain_embed --split umap --trials 0`
- Experiment: `tabpfn_boltz_trunk_pretrain_embed_umap_default`
- Bakeoff as **swap candidate** for `tabpfn_pooled_boltz_umap_default` (the raw trunk variant, MAE 0.486); if decorrelated enough, also test as add.

## Data flow

```
Phase 1: trunk extraction (~12-24h on RTX 5080, background)
  compounds (13,136 std_smiles)
    ↓ boltz2_build_inputs.py (8,483 missing)
  YAML × 8,483
    ↓ boltz predict --embeddings_only --recycling_steps 1
  embeddings_<id>.npz × 8,483
    ↓ pool_embeddings.py (reuse existing helper)
  trunk_fast_pooled.parquet (8,483 × 1024)

Phase 2: DB upsert (~5 min)
  + existing compound_boltz2 + pooled.parquet (4,653 × 1024, rcycle=3)
  → compound_boltz2_trunk_fast table (13,136 rows, rcycle column flagged)

Phase 3: MLP pretrain (~15-30 min)
  13,136 × 1024 Boltz trunk + log2_fc
    ↓ MLP variant A/B/C
  pretrain.pt (hidden 256 or 512)

Phase 4: Extract (~5 min)
  pretrain.pt + 13,136 trunks
    ↓ penultimate layer forward
  data/boltz_trunk_pretrain_embed.parquet (13,136 × 256 or 512)

Phase 5: TabPFN + ensemble (~30-45 min)
  tabpfn_boltz_trunk_pretrain_embed_umap_default
    ↓ (if gates pass) swap/add in ENSEMBLE_MODELS
  updated 8 or 9-pool caruana + calibrate + submission CSV
```

## Acceptance criteria

1. **Trunk extraction completes** on all 8,483 target compounds (tolerance: <1% failures, documented).
2. **MLP pretrain converges**: val_loss monotone-decreasing for 20+ epochs, best val_loss recorded.
3. **Single-model OOF MAE ≤ 0.46** (loose bar; expected 0.40-0.46 per pattern).
4. **caruana_bag20 weight > 0** on the new member.
5. **Swap beats or ties 8-pool baseline**:
   - Swap target: `tabpfn_pooled_boltz_umap_default` (existing trunk member, MAE 0.486)
   - Condition: swap 8-pool MAE ≤ current 0.4184 (or add-9 provides MAE ≤ 0.4180)
6. **Pearson r with existing members < 0.95** (slightly looser than prior 0.96 threshold because Boltz is the outlier family).
7. **ruff format + check clean**.

### Failure handling

- **Phase 1 fails on some compounds** (e.g., oversize ligand, MSA issue): document failures, proceed with available N. If N < 10,000 the pretrain corpus is too small — escalate.
- **rcycle=1 too noisy**: retry subset at rcycle=2, compare.
- **MLP doesn't converge**: try simpler architecture (Variant A), lower lr.
- **OOF MAE > 0.48**: single-model too weak; abort pool addition.
- **Caruana weight = 0**: DONE_WITH_CONCERNS, keep framework for reuse.
- **Swap regresses pool**: keep as add-9 or drop from allow-list.

## Testing

- Smoke on Phase 1: fast mode for 10 compounds, verify npz schema.
- Smoke on Phase 3: 2-epoch MLP train, verify loss drops.
- Smoke on Phase 4: head(100) extract, parquet shape check.
- Smoke on Phase 5: 2-fold TabPFN.
- No unit tests (existing DL convention).

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Phase 1 takes > 2 days | Medium | Monitor wall-clock; if slow, parallelize across multiple nohup tmux sessions |
| rcycle=1 trunk quality too different from rcycle=3 | Medium | Primary safety: existing train/test (4,653) kept at rcycle=3. New 8,483 at rcycle=1 are pretrain-only corpus. TabPFN evaluation uses the 4,653 high-quality trunks. Mixed-quality in MLP training is acceptable per user. |
| MLP overfits on 13k with 1024d input | Low | Standard dropout + early stopping. 13k is enough for 1024→256 compression. |
| Boltz input construction fails on some SMILES (non-trunk 8,483 set includes edge cases) | Low | Standardized SMILES + existing builder works; document any failures. |
| New DB table adds complexity; future maintainer confused by 2 tables | Low | Explicit `recycling_steps` column + docs in CLAUDE.md. |
| TabPFN treats 256d input fine but adding as 9th pool member stagnates MAE | Medium | Precedent: adding members has not always improved caruana (seen in GatedGCN h128, AttentiveFP). If regress, drop from allow-list. |

## ETA

- Phase 1 (Boltz fast trunk): **12-24h** background run on RTX 5080 (~5-10s per compound × 8,483)
- Phase 2 (DB table + upsert): **10 min**
- Phase 3 (MLP pretrain, ×3 variants): **30-60 min** (fast on GPU)
- Phase 4 (embedding extract): **5-10 min**
- Phase 5 (TabPFN 5-fold + bakeoff + commit + PR): **1-1.5h**

**Total**: 14-28h, but Phase 1 dominates and runs unattended. Active coding time ~2h.

## Implementation order (calendar)

- Today (2026-04-21 evening): spec + plan, no active compute
- Tomorrow (2026-04-22):
  - Morning: start Phase 1 background (fast mode for 8,483)
  - Afternoon: prepare Phase 2-4 scripts (spec unchanged)
  - Evening: Phase 1 finishes (~18-24h later)
- Day 3 (2026-04-23):
  - Morning: Phase 2-4 execution
  - Afternoon: Phase 5 + PR + LB submission

## Out of scope (future PRs)

- **Uni-Mol v2 on ETKDG+Boltz-pose hybrid** — explicit follow-up, different architecture
- **Add affinity-head finetune alongside log2_fc pretrain** — could stack
- **Cross-attention encoder (Boltz trunk × 2d_full_boltz) → hidden** — transformer-style mix, ambitious
- **Re-run existing 4,653 at rcycle=1** — only if mixed-quality issue is empirically observed
