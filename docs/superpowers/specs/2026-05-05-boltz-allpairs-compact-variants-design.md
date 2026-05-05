# Design: Boltz Trunk Re-Pooling and Residual Pretraining

- **Status**: Revised proposal
- **Date**: 2026-05-05
- **Context**: Track 1 activity modeling after several internal/gated axes, fixed Boltz contact features, and GatorAffinity checks failed to produce a strong candidate.
- **Related**:
  - Issue #100 research log
  - PR #74 Boltz trunk pooling
  - PR #109 fast Boltz embeddings-only extraction
  - PR #111 Boltz strategy-3 / trunk-fast DB / top-K and compression bakeoff
  - `db/boltz2_trunk_fast_schema.sql`
  - `track1_activity/scripts/boltz_affhead/08_pool_and_upsert_fast.py`
  - `track1_activity/scripts/boltz_affhead/09_mlp_pretrain.py`
  - `track1_activity/scripts/boltz_affhead/09b_extract_embed.py`

## Goal

Improve the Boltz/protein-ligand axis while respecting what has already been tried.

The target is a new Boltz-family candidate that is either:

1. a better use of the existing 13k trunk embeddings, or
2. a better 4652-compound full-structure diagnostic that explains current ensemble residuals well enough to justify a tightly controlled candidate.

This is not a GatorAffinity revival and not another fixed contact-count feature pass.

## Correct Data Model

There are three distinct Boltz data layers. Designs must keep them separate.

### Layer A: Full Boltz structure run

Table: `compound_boltz2`

- Rows: 4653
- Embedding NPZ paths: 4652
- Recycling: rcycle=3
- Has: trunk `s/z` embeddings, predicted structure/pose, confidence, affinity, ligand geometry, PoseBusters-linked downstream features
- Scope: train + test + counter-style covered compounds from the full run

This layer supports full structure-derived features and confidence/pose gating, but it does not cover all 13,136 compounds.

### Layer B: Fast embeddings-only trunk run

Table: `compound_boltz2_trunk_fast`

- Rows: 13,134
- rcycle=3 rows: 4652
- rcycle=1 rows: 8482
- Missing compound IDs: 1657 and 8624
- Source NPZ paths retained for both rcycle groups
- Has: pooled allpairs trunk arrays in DB (`s_prot_mean`, `s_lig_mean`, `z_if_mean`, `z_if_max`) plus raw source `embeddings_*.npz`
- Does not have: generated pose, confidence head, affinity head, full structure outputs for the rcycle=1 set

This layer is suitable for weak-label pretraining and raw trunk re-pooling. It is not suitable for pose/contact/confidence features on the rcycle=1 rows.

### Layer C: Existing 1024d pooled allpairs products

Files / handlers:

- `data/boltz_affhead/pooled_allpairs.parquet`
- `tabpfn_pooled_boltz_allpairs_umap_default`
- `tabpfn_pooled_boltz_umap_default`
- `boltz_trunk_pretrain_embed_*`
- `boltz_raw_plus_pretrain_concat`

These are already evaluated. The allpairs member is weak as a single model but important as a low-weight diversity reserve.

## Current Evidence

Existing Boltz-family single-model results:

- `tabpfn_boltz_raw_plus_pretrain_concat_umap_default`: OOF MAE about 0.482
- `tabpfn_boltz_trunk_pretrain_embed_c_concat_umap_default`: OOF MAE about 0.485
- `tabpfn_pooled_boltz_allpairs_umap_default`: OOF MAE about 0.486
- `tabpfn_pooled_boltz_umap_default`: OOF MAE about 0.486
- allpairs ablations showed `z_xp_max` alone was especially weak; `nozmax` was only a small diagnostic improvement, not a breakthrough

Important lessons:

- Allpairs beat interaction-only pooling in earlier PR #74 notes.
- Dropping low-weight diversity members, including `pooled_boltz_allpairs`, caused a large LB regression later. Do not remove it casually.
- Boltz strategy-3 pretraining on the 13k trunk was implemented and tested. It did not become a dominant new pool member.
- Fixed structure/contact features and ProLIF-style IFPs were null.
- GatorAffinity was already tried enough to treat it as a failed main path unless new evidence appears.

## Non-Goals

- Do not rerun full Boltz inference.
- Do not retrain Boltz itself.
- Do not resume GatorAffinity as the primary plan.
- Do not drop `tabpfn_pooled_boltz_allpairs_umap_default` as a cleanup.
- Do not submit an OOF-only -0.001 to -0.002 candidate.
- Do not mix 13k trunk-only data with 4652 pose/confidence-only data without explicitly marking the coverage boundary.
- Do not use external data in this phase.

## Design Options

### Option 1: Raw NPZ Re-Pooling From 13k Trunk

Use `source_npz_path` from `compound_boltz2_trunk_fast` and re-pool raw `s/z` tensors into new compact summaries.

Allowed signals:

- protein token `s` by residue range / region
- ligand token `s`
- protein-ligand `z` pairs
- rcycle flag

Disallowed for rcycle=1 rows:

- confidence/affinity head outputs
- ligand pose geometry
- PoseBusters/contact features

Candidate re-pools:

- region allpairs: core pocket, AF-2/H12 side, entrance/beta-sheet side, remainder
- component denoising: keep `s_prot`, `s_lig`, `z_mean`; downweight or drop `z_max`
- signed statistics: mean, std, low/high quantiles over `z` pairs instead of only mean/max
- ligand-size-normalized pooling: normalize pair aggregation by ligand atom count and track ligand-size interaction terms

Risk:

- Raw NPZ reading is expensive because each file is roughly 100 MB. Full 13k scans are feasible but should be streamed and cached to parquet.

### Option 2: Residual Pretraining on 13k Trunk Products

Do not invent more handcrafted pooling first. Instead, train a small model from trunk vectors toward a target that is closer to the current failure mode.

Potential targets:

- single-concentration log2fc, as already done, but with residual-aware weighting
- current anchor OOF residual on train rows only, with strict cross-fitting
- multi-task objective: log2fc heads + anchor residual head, with the residual head used only for train-fold-safe embeddings

This option directly addresses why earlier Boltz strategy-3 may have been weak: it optimized generic log2fc pretrain, not the current ensemble residual.

Risk:

- residual targets are easy to overfit and can point in a known bad LB direction. Any residual-trained output must be checked against id50 projection.

### Option 3: Full-Structure 4652 Diagnostic Only

Use only `compound_boltz2` rcycle=3 rows and structure/pose metadata to identify where Boltz might help.

Allowed signals:

- pose confidence
- affinity scalars
- ligand-to-pocket distance
- PoseBusters summaries
- fixed contact/IFP features already computed

Purpose:

- gate or explain residuals for test compounds
- decide whether structure confidence can modulate an allpairs/residual correction

This option should not become another high-dimensional contact feature run unless the diagnostic clearly shows residual structure.

Risk:

- Coverage is only 4652 and previous contact/IFP attempts were null. Treat this as diagnostic support, not the main candidate.

## Recommendation

Proceed in two phases:

### Phase A: Audit and Data Inventory

Build a report that establishes:

- `compound_boltz2_trunk_fast` coverage, rcycle split, missing IDs
- whether all 13,134 `source_npz_path` files are readable
- shapes and ligand token counts for a sample of rcycle=1 and rcycle=3 NPZs
- correlation and OOF metrics for all existing Boltz-family experiments
- Caruana behavior when preserving vs swapping allpairs

This phase prevents repeating the false assumption that only 1024d pooled data exists.

### Phase B: One Minimal New Candidate

After the audit, implement only one of:

1. raw NPZ region/quantile re-pooling if the raw scan is practical, or
2. residual-aware trunk pretraining if the audit shows existing pooling variants are too exhausted.

The first implementation should be narrow enough to evaluate in one session. Avoid adding three unrelated variants at once.

## Evaluation Gates

Any candidate must pass:

1. single-model OOF is within +0.05 MAE of the weakest retained pool member, or there is an explicit diagnostic reason to keep it despite being weaker
2. min residual correlation to existing pool members is meaningfully lower than same-family variants
3. Caruana ADD/SWAP gain is at least about -0.003, or a stronger anti-id50 diagnostic justifies a probe
4. projection onto the known bad id50 direction is not positive and large
5. existing `pooled_boltz_allpairs` remains in the main pool unless a replacement has overwhelming evidence
6. calibration scripts are rerun before any submission

## Expected Artifacts

Audit:

- `track1_activity/scripts/boltz_affhead/37_trunk_fast_inventory.py`
- `track1_activity/analysis/boltz_trunk_fast_inventory/outputs/report.md`

Possible candidate path:

- `track1_activity/scripts/boltz_affhead/38_repool_trunk_npz.py`
- `data/boltz_affhead/repooled_trunk_*.parquet`
- `track1_activity/scripts/boltz_affhead/39_repool_bakeoff.py`

Only add `run_train.py` feature registry entries after the audit confirms coverage and the parquet passes shape checks.

## Decision

First run the inventory/audit. Then choose between raw NPZ re-pooling and residual-aware trunk pretraining based on evidence, not memory. The likely best first candidate is a raw NPZ re-pool that uses the full 13,134 source NPZ paths but keeps pose/confidence gates restricted to the 4652 full-run subset.
