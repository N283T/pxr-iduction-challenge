# Design: Boltz Allpairs Compact Variants

- **Status**: Proposed
- **Date**: 2026-05-05
- **Context**: Track 1 activity modeling, after internal residual/gated axes and fixed Boltz contact features failed to produce a useful LB candidate.
- **Related**:
  - Issue #100 research log
  - PR #74 Boltz trunk pooling work
  - PR #109/#111 Boltz trunk log2fc-pretrain work
  - `tabpfn_pooled_boltz_allpairs_umap_default`
  - `track1_activity/scripts/boltz_affhead/01b_pool_allpairs.py`

## Goal

Improve the Boltz/protein-ligand axis without repeating failed GatorAffinity, fixed contact-count, or generic Boltz feature-concat attempts.

The immediate goal is not to build a large new protein-ligand model. It is to create a focused set of allpairs-derived compact feature variants that preserve the useful diversity of the existing `pooled_boltz_allpairs` member while testing whether more task-relevant pooling can add or replace a Boltz-family member in the current ensemble.

## Current Evidence

The existing Boltz allpairs member is weak as a single model but valuable as a diversity reserve:

- `tabpfn_pooled_boltz_allpairs_umap_default`: OOF MAE about 0.486
- `tabpfn_pooled_boltz_umap_default`: OOF MAE about 0.486
- `tabpfn_boltz_raw_plus_pretrain_concat_umap_default`: OOF MAE about 0.482
- `tabpfn_boltz_trunk_pretrain_embed_c_concat_umap_default`: OOF MAE about 0.485

Past notes show that interaction-only pooling was tested against allpairs and allpairs was better on PXR. Later drop tests also showed that removing low-weight diversity members, including `pooled_boltz_allpairs`, can regress LB despite looking harmless in OOF.

Recent failed directions:

- GatorAffinity: zero-shot/calibrated signal too weak and no successful DB-tracked pool member.
- Fixed Boltz contact/residue shell features: weak standalone and no improvement when appended to the strong chemistry/log2fc feature bundle.
- Dose-response latent variants: both cheap fixed-feature and ChemProp pretrain variants underperformed existing log2fc-pretrain embeddings.

## Non-Goals

- Do not resume GatorAffinity as the main path.
- Do not remove `tabpfn_pooled_boltz_allpairs_umap_default` from the production pool as a cleanup step.
- Do not run full Boltz inference or retrain Boltz itself.
- Do not submit a candidate based only on a small OOF gain near the known +/-0.002 bag-noise region.
- Do not use external data in this phase.

## Proposed Approach

Build compact variants from existing Boltz allpairs artifacts and evaluate them as ensemble axes, not just as single models.

Inputs:

- `data/boltz_affhead/pooled_allpairs.parquet`
- Existing Boltz pose metadata and confidence/tier-0 scalar features already available through the DB
- Existing OOF/test predictions for current anchor and recent failed LB directions

Candidate feature families:

1. **Region-pooled allpairs**
   - Split PXR residues into compact functional regions such as core pocket, H12/AF-2-adjacent, beta-sheet/entrance, and remainder.
   - Pool `z`-like allpairs summaries per region.
   - Keep dimensionality small enough for TabPFN or LGBM without another top-k machinery pass.

2. **Distance/confidence-weighted allpairs**
   - Reweight or gate existing allpairs components using ligand-to-pocket distance, pose confidence, clash/posebuster summaries, or affinity-head confidence scalars.
   - Goal is to reduce noisy pose contribution without converting this into another fixed contact-count feature.

3. **Residual-oriented compact projection**
   - Train a small, cross-fit projection from allpairs-derived features toward the current anchor residual.
   - Use strict shrinkage and evaluate projection against known bad LB directions, especially id50.
   - This is a diagnostic candidate first; only submit if it passes stronger gates.

## Evaluation Gates

Each candidate should be evaluated with the same evidence stack before any submission:

1. Single-model OOF MAE and Spearman.
2. Pearson correlation to the current anchor and to existing Boltz-family members.
3. Residual correlation against current anchor OOF residual.
4. Projection onto the known bad id50 direction.
5. Caruana ADD and SWAP tests while preserving `pooled_boltz_allpairs` unless there is overwhelming evidence for replacement.
6. Calibrated submission dry-run only if the candidate clears a practical threshold:
   - Caruana OOF gain at least about -0.003, or
   - a clearly favorable anti-id50 / residual diagnostic that justifies an LB probe despite small OOF movement.

## Artifacts

Expected new files:

- `track1_activity/scripts/boltz_affhead/37_allpairs_compact_variants.py`
- `track1_activity/scripts/boltz_affhead/38_allpairs_compact_bakeoff.py`
- `data/boltz_affhead/allpairs_compact_*.parquet`
- Optional report under `track1_activity/analysis/boltz_allpairs_compact/outputs/report.md`

`run_train.py` should only receive new feature registry entries after the generated parquets pass shape/coverage checks.

## Risks

- Allpairs variants may be too correlated with the existing Boltz pair and only reshuffle Caruana weights.
- Residual-oriented training can overfit OOF residuals and produce an id50-like bad LB direction.
- Region definitions may become arbitrary. Keep them few, documented, and based on PXR LBD structure conventions rather than high-dimensional fishing.

## Decision

Proceed with a compact allpairs-family audit and variant bakeoff. Treat this as a Boltz-axis improvement attempt, not as a broad new architecture project.
