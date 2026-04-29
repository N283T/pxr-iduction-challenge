# ADMET-AI Features as Pool Member

Date: 2026-04-29
Status: null result (4 variants tested, all blocked by pool redundancy or family share)
Owner: N283T

## Background

After Codex's "retrieval-as-pool-member" pivot was null on PXR (kNN
weight ~0, PR #153), user identified ADMET-AI (Stanford Swanson,
https://admet.ai.greenstonebio.com/) as an unexplored info source: 104
TDC-trained ADMET predictions per molecule (Chemprop-RDKit ensemble),
free, local pip install, biologically related to PXR via CYP3A4 and
metabolism endpoints.

## Goal

Determine whether ADMET-AI predictions can become a useful caruana_bag20
pool member, either standalone (orthogonal voter) or augmented (added
to the existing strongest feature stack).

Success target: caruana weight ≥ 1% AND new caruana OOF MAE Δ ≤ -0.003
AND chemprop family share remains in 0.65-0.80 zone.

## Variants tested

### Variant 1: standalone TabPFN on ADMET-AI 104

Implemented as `tabpfn_admet_ai_umap`. Single OOF MAE 0.5156, Sp 0.7115.
Residual r vs each pool member 0.87-0.90 (gate 2 fail). Caruana weight
0.0046 — null.

### Variant 2: augmented full 2207d

`tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_admet_ai_umap` =
TabPFN on (existing 2103d + ADMET 104) all features. Single OOF MAE
0.4353 (worse than existing top500 0.397 by 0.039) — TabPFN suffers
with too many noise dims.

### Variant 3: augmented + top-500 LGBM gain filter

`tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_admet_ai_top500_umap`
= TabPFN on top-500 by LGBM gain importance. 39/104 ADMET features
included (Lipophilicity_AstraZeneca rank 16, CYP3A4_Substrate
CarbonMangels rank 23, HIA_Hou rank 37). Single OOF MAE 0.3964 (tied
with existing top500), Sp 0.8490 (+0.003 marginal). Residual r vs OLD
top500 = 0.9974 (essentially identical).

### Variant 4: caruana SWAP / ADD bakeoff (using Variant 3)

- SWAP (replace OLD top500 with NEW): caruana OOF MAE 0.3956 (Δ
  -0.0001 = noise). Family share 0.773 (in zone). Tied → null.
- ADD (keep OLD, add NEW alongside): caruana OOF MAE 0.3919 (Δ -0.0039,
  above noise floor!), Sp 0.8506 (+0.0040). BUT family share 0.887 =
  hits LB regress zone (`project_family_share_lb_u_curve`: 0.85 → +0.003,
  0.94 → +0.006). High LB regress probability per id=38/40/41 trap.

## Decision

**Null overall.** All four variants either fail OOF gates or hit the
family-share trap. No LB submission — would be expected reverse amp.

The fundamental issue: ADMET-AI's Chemprop-RDKit predictions are
correlated with the existing chemprop-pretrain pool members (r 0.96-1.00
with augmented sibling), so they don't introduce orthogonal signal. The
biological link (CYP3A4 induction by PXR) is visible at the LGBM gain
level (rank 23 for CYP3A4_Substrate) but doesn't translate to caruana
ensemble benefit due to redundancy.

## Deliverables

Framework retained for future attempts:
- `track1_activity/scripts/run_admet_ai_predict.py` — batch predictor
  (4653 SMILES → 104 features, ~30s on RTX 5080)
- `track1_activity/scripts/run_admet_ai_tabpfn.py` — Variant 1 (standalone)
- `track1_activity/scripts/run_admet_ai_tabpfn_augmented.py` — Variant 2
  (full 2207d)
- `track1_activity/scripts/run_admet_ai_tabpfn_augmented_top500.py` — Variant 3
- `track1_activity/scripts/check_admet_ai_in_topk.py` — LGBM gain check
- `track1_activity/scripts/bake_admet_ai_addition.py` — Variant 1 bakeoff
- `track1_activity/scripts/bake_admet_ai_swap.py` — Variant 3/4 bakeoff
- `data/admet_ai_predictions.parquet` — cached 104-dim predictions for all
  4653 compounds (gitignored, regenerable in 30s)

## Possible follow-up (deferred)

- ADMET-AI + chemeleon + 2d_full_boltz **without** chemprop log2fc_pred —
  would not count as chemprop family, allowing ADD without family share trap.
  Gating: orthogonality vs family-share is the structural question; this
  variant might break the trap if biological signal still routes through
  CYP3A4_Substrate and similar features.

## Memory updates

- `feedback_admet_ai_null_family_share_trap` — adds 4-strike day evidence
  to PXR pool saturation analysis. PR will land alongside PR #153 (kNN null).
