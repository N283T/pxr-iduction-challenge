# MTR Domain Adaptation — Sultan 2025 recipe applied to PXR

**Date**: 2026-05-04
**Author**: N283T
**Status**: Design — pending implementation plan
**Source paper**: Sultan, Rausch-Dupont, Khan, Kalinina, Klakow, Volkamer. *Transformers for Molecular Property Prediction: Domain Adaptation Efficiently Improves Performance.* arXiv:2503.03360v3 (2025).

---

## 1. Motivation

Track 1 has been stagnant at LB rank 3 (N283T MAE 0.4091) for 4 days. The pool is saturated:

- All recent ADD candidates (tier-0, mordred3d, ChemBERTa raw, ADMET-AI, kNN, anchor-residual, region-routing) hit the **−0.002 OOF ceiling** (memory: `feedback_oof_minus_0002_ceiling`)
- Multi-seed ChemProp log2fc pretrain (Plan A, the rank-1 driver) is at the taper — 15-seed null
- Per-compound conditioning, calibrator innovation, and family-share manipulation all confirmed dead
- Cross-NR multitask was rejected by the user yesterday after Phase 3 Uni-Mol v2 closeout

The Sultan 2025 paper presents a different lever we have not tried: **multi-task regression of physicochemical descriptors as a standalone domain-adaptation objective**, applied between pre-training and fine-tuning, on a small task-domain corpus (~13k unlabeled compounds for us).

Critically, this is **not the same** as ChemProp descriptor-aux multitask (PR #86, closed null). The aux experiment used a *joint* loss `L = L_pec50 + α L_desc` during supervised fine-tuning. Sultan's MTR-DA is *standalone* — descriptors are the **sole** training objective during a separate stage, then the encoder is frozen and embeddings flow into a downstream model (RF / TabPFN / etc.).

The recipe also doubles as a **precursor validation for future data augmentation**: if MTR-DA works on the current 13k PXR corpus, the same pipeline extends naturally to 50k+ compound corpora (e.g., ChEMBL PXR-related) — Sultan 2025 itself shows that scaling the unlabeled corpus beyond ~400-800k yields no further gain, so the value is in the recipe, not the corpus size.

## 2. Goal

Build two SWAP candidates (not new pool members) that replace existing log2fc-pretrained pool members with MTR-DA-pretrained equivalents:

- **Variant C**: ChemProp encoder, MTR-pretrained from scratch (no log2fc) → SWAP candidate for `tabpfn_chemprop_pretrain_embed`
- **Variant M**: MoLFormer-c3-1.1B, MTR-DA pretrained on top of existing checkpoint → SWAP candidate for `tabpfn_molformer_c3_pretrain_embed`

Acceptance is gated by leak audit + 4 quantitative gates (G1–G4) before LB submission.

## 3. Sultan 2025 recipe — quick reference

- **MTR objective**: predict 210 physicochemical descriptors (RDKit `Descriptors._descList`) per molecule, MSE loss on standardized targets
- **Training stages**: MLM pretrain on 30% GuacaMol (~400K) → MTR-DA on small domain corpus (≤4K) → freeze encoder → RF on extracted CLS embeddings
- **Best variant in paper**: `MLM_MTR` (MLM pretrain + MTR-DA), beats both `MTR_No DA` and `MTR_MTR` on 5-7 of 7 ADME endpoints (p < 0.001)
- **Hyperparameters**: lr 3e-5, batch 16, 20 epochs, AdamW + linear schedule with 10% warmup, mean-pool over tokens, 2-layer MLP head with ReLU + dropout 0.1
- **Key finding**: scaling the pretraining corpus past 400-800k molecules does *not* help; chemically informed objectives at the DA stage *do* help (consistently, with p < 0.001 across all 7 ADME endpoints)

## 4. Architecture

```
[13,134 std_smiles]                                        [13,136 std_smiles]
         │                                                          │
         ▼                                                          ▼
  ┌──────────────────┐                                  ┌──────────────────┐
  │  MTR pretrain    │                                  │  Frozen extract  │
  │  (Variant C/M)   │  ── encoder weights freeze ──▶   │  (encoder eval)  │
  │  RDKit 217 MTR   │                                  │  per-compound    │
  └──────────────────┘                                  │  embedding       │
         │                                              └──────────────────┘
         ▼                                                          │
  scaler.json (mean/std per descriptor)                             ▼
                                                        ┌──────────────────┐
                                                        │ TabPFN UMAP CV   │
                                                        │ (5 folds × seed) │
                                                        └──────────────────┘
                                                                    │
                                                                    ▼
                                                        OOF predictions parquet
                                                            │
                                                            ▼
                                                    Gate 1/2/3 evaluation
```

### 4.1 Variant C: ChemProp from scratch + MTR pretrain

| Component | Spec |
|---|---|
| Encoder | ChemProp D-MPNN, hidden=300, depth=3, message_steps=3 (matches existing log2fc pretrain config) |
| Initial weights | Random init. The existing log2fc-pretrained ChemProp checkpoint (rank-1 driver) is **deliberately not used** as a starting point — per user instruction (`後段ではなく普通に最初から`), MTR pretrain starts from scratch so this variant is independent of the log2fc family |
| Training objective | MTR only — no log2fc, no pec50 |
| Target dimension | 217 RDKit descriptors |
| Target source | `compound_descriptors_full.descriptors` (JSONB), unpacked to 217 columns |
| Target preprocessing | StandardScaler fit on full 13,134-compound table (after row-drop) |
| Loss | MSE summed across 217 heads (no per-head NaN mask needed; rows with NaN dropped before training) |
| MTR head | MLP `300 → 300 → 217`, ReLU, dropout 0.1 |
| Hyperparameters | lr 1e-3 (matches existing chemprop pretrain), batch 64, 50 epochs, AdamW, plateau scheduler |
| Output | Encoder 300d embedding for all 13,136 compounds → parquet |
| Compute | ~30 min on RTX 5080 (seed=42) |

### 4.2 Variant M: MoLFormer-c3-1.1B + MTR-DA

| Component | Spec |
|---|---|
| Base checkpoint | Existing `peft_backbones.molformer_c3_1_1b` (already used in pool) |
| Rotary fix | Required — reuse pattern from `peft_trainer.py` (issue #30 / PR #95 reference) |
| PEFT method | LoRA (target modules: `q_proj`, `v_proj`), rank=8, alpha=16 (existing default) |
| Training objective | MTR only — no log2fc, no pec50 |
| Input | 13,134 std_smiles, max_len=128 |
| Pooling | Mean pool over non-padding tokens (MoLFormer has no CLS by default; mean is more stable than first-token in the c3-1.1B variant) |
| MTR head | MLP `768 → 768 → 217`, ReLU, dropout 0.1 |
| Loss | MSE summed across 217 heads (NaN-drop policy as Variant C) |
| Hyperparameters | lr 3e-5 (Sultan recipe), batch 16, 20 epochs, AdamW, linear schedule + 10% warmup |
| Output | Mean-pool 768d embedding for all 13,136 compounds → parquet |
| Compute | ~2-3 h on RTX 5080 (seed=42) |

### 4.3 Common downstream

- **Frozen extraction**: encoder eval mode, full 13,136 compounds (including the 2 NaN-dropped rows — forward pass works for any valid SMILES, only gradient was withheld)
- **TabPFN regressor**: `--split umap` (existing canonical), 5 fold UMAP CV with **UMAP-split-seed=42**, n_clusters=50, Morgan+Jaccard
- **OOF predictions**: per-compound 1 row → parquet → DB `experiment_oof_predictions` table

**Seed naming convention** (to avoid confusion):
- *pretrain-seed* — random init seed for the encoder MTR pretrain (single-seed phase uses `42`, multi-seed phase uses `42..46`)
- *UMAP-split-seed* — fixed at `42` throughout, follows existing canonical CV (memory: `project_cv_bakeoff_concluded`)
- They happen to coincide on `42` for the single-seed phase but are independent settings.

## 5. NaN handling: drop-row policy

Empirical NaN profile (queried 2026-05-04):

| Metric | Value |
|---|---|
| Total compounds | 13,136 |
| Total RDKit descriptors | 217 |
| Compounds with any NaN | **2** (0.015%) |
| Descriptor cols with any NaN | 12 (BCUT2D family + Partial charge family) |
| inf cells | 0 |

Affected compounds:
- `compound_id=1657` (Auranofin, Au metal complex) — 8 NaN cells (BCUT2D); train + counter membership
- `compound_id=8624` — 12 NaN cells (BCUT2D + Partial charge); single_concentration only

**Decision**: drop these 2 rows from MTR pretrain. Justification:
- 0.015% of training data is statistically irrelevant
- Dropping is simpler than per-cell NaN masking (no batch effective-size variance, no mask logic)
- Keeps Sultan's 217-head dimensionality intact (column-drop alternative would lose BCUT2D, which encodes electronic state — relevant to PXR ligand binding affinity)
- Forward pass at extraction time still works for compound 1657 → its embedding is generated, just from an encoder that never received gradient from this specific structure
- Auranofin is already a known outlier in our pipeline (Boltz-2 pre-processing failure, metal complex), so partial coverage in MTR is consistent

## 6. Leak audit

This is the user's primary concern ("leak しないように気を付けて"). Six identified risks:

| ID | Risk | Defense |
|---|---|---|
| **L1** | pec50 label leak via descriptor target | Hard rule: target source = `compound_descriptors_full` only. RDKit `Descriptors._descList` is deterministic from SMILES. Implementation reads exactly one SQL: `SELECT compound_id, descriptors FROM compound_descriptors_full`. No experiment tables touched. |
| **L2** | Test compound exposure during pretrain (transductive) | Same as existing rank-1 driver chemprop log2fc pretrain — not a new leak. Documented as transductive-by-design. |
| **L3** | Fold leak via descriptor scaler | StandardScaler fit on full 13,134-compound table, parameters frozen and saved to `models/.../scaler.json`. Pretrain is fold-agnostic. |
| **L4** | Double fold split (encoder + TabPFN) | Hard rule: pretrain fold-agnostic, UMAP fold split applied **only** at TabPFN training. Reuses `run_train.py --split umap`. |
| **L5** | Self-match via duplicate SMILES | Verify `train_activity.compound_id ∩ test_activity.compound_id = 0` AND `std_smiles` overlap = 0 in pre-pretrain audit. |
| **L6** | Mordred / Jazzy bleeding into MTR target | New scripts `run_chemprop_mtr_pretrain.py` and `run_molformer_mtr_pretrain.py` written from scratch. Existing `run_chemprop_pretrain.py` and `run_molformer_c3_pretrain.py` not modified. Single source SQL in each new script. |

### 6.1 Audit gate (G0)

New script: `track1_activity/scripts/audit_mtr_leak.py`

Performs the following checks; fails (exit 1) if any check fails:

1. `compound_id` overlap (train vs test) = 0
2. `std_smiles` overlap (train vs test) = 0
3. MTR target source SQL is exactly `SELECT compound_id, descriptors FROM compound_descriptors_full`
4. NaN-drop list is exactly `[1657, 8624]` (sanity check against current DB state — fails if a third compound appears, forcing manual review)
5. Descriptor count = 217 after column unpack
6. inf cell count = 0

Output: `track1_activity/reports/mtr_leak_audit_<date>.json`. Pretrain scripts call this first and refuse to run if audit fails.

## 7. Gates

| Gate | Metric | Threshold | Fail action |
|---|---|---|---|
| **G0** | leak audit | All 6 checks PASS | Pretrain refuses to start |
| **G1** | TabPFN OOF MAE (single seed=42) | ≤ 0.485 | Drop variant; record null PR |
| **G2** | min residual r vs **non-swap-target** 8 pool members | ≤ 0.85 (HARD RULE per `feedback_pretrain_gate2_first`) | Drop variant; close family in memory |
| **G3** | caruana_bag20 SWAP OOF MAE Δ vs current pool | ≤ −0.003 (per `feedback_oof_minus_0002_ceiling`) | Record-only PR, no LB submit |
| **G4** | LB MAE Δ vs id=43 (production importance affine, past LB best 0.4075) | ≤ −0.001 win / ≥ +0.003 loss | Revert ENSEMBLE_MODELS, document |

### 7.1 G2 detail (SWAP-specific)

Standard ADD requires `min r ≤ 0.85` against all pool members. SWAP is different: the residual correlation against the **member being replaced** is unconstrained (and is expected to be high — that's the point of SWAP).

For Variant C (replacing `tabpfn_chemprop_pretrain_embed`):
- vs `tabpfn_chemprop_pretrain_embed`: r unconstrained
- vs all other 8 pool members (incl. `tabpfn_molformer_c3_pretrain_embed`): `min r ≤ 0.85`

For Variant M (replacing `tabpfn_molformer_c3_pretrain_embed`): symmetric.

If both variants are evaluated, each is gated against its own non-swap-target 8-member set. Joint double-swap is evaluated only after each individual variant passes G2/G3.

## 8. Pool integration plan

### 8.1 Single-variant SWAP (most likely scenario)

Replace one ENSEMBLE_MODELS entry. Re-run:
- `run_ensemble.py` → caruana_bag20 → ens_caruana_bag20.csv
- `run_ensemble_calibrate.py` → 4-way nested CV
- `run_ensemble_calibrate_importance.py` → importance affine
- LB submit with importance variant (per `feedback_calibrator_importance_locked_10seed`)

### 8.2 Double-variant SWAP (jackpot scenario)

Replace both `tabpfn_chemprop_pretrain_embed` and `tabpfn_molformer_c3_pretrain_embed`. Precedent: 2026-04-25 seed5ens double-swap drove OOF −0.0116 / LB −0.0065 → rank 1. Same pattern applies if both variants pass G2/G3 individually.

### 8.3 Failure mode

If G2 fails for both variants: record null result in memory (`PXR で MTR-DA recipe 失敗`), close the family. This also blocks the future data-augmentation extension since the recipe itself is rejected.

## 9. Multi-seed extension (Phase 2, conditional)

Per `feedback_pretrain_gate2_first` HARD RULE: **multi-seed only after gate 2 passes on seed=42 baseline**. Single-seed wastes <30 min (Variant C) to ~3 h (Variant M); 5-seed wastes ~5x that with no recovery if recipe is broken.

If G2 passes:
- Run seeds 42-46 with same hyperparameters
- Per-row mean of 5 embedding parquets → final embedding
- Re-evaluate G3 with multi-seed embedding
- Memory `reference_multi_seed_pretrain_recipe` confirms variance reduction without bias change

## 10. New files

| Path | Purpose |
|---|---|
| `track1_activity/scripts/audit_mtr_leak.py` | G0 audit gate |
| `track1_activity/scripts/run_chemprop_mtr_pretrain.py` | Variant C training |
| `track1_activity/scripts/run_chemprop_mtr_extract.py` | Variant C frozen extract |
| `track1_activity/scripts/run_molformer_mtr_pretrain.py` | Variant M training |
| `track1_activity/scripts/run_molformer_mtr_extract.py` | Variant M frozen extract |
| `track1_activity/reports/mtr_leak_audit_<date>.json` | Audit output |
| `models/chemprop_mtr_seed42/` | Variant C checkpoint + scaler.json |
| `models/molformer_c3_mtr_seed42/` | Variant M checkpoint + scaler.json |
| `data/chemprop_mtr_embedding_seed42.parquet` | Variant C extracted embedding |
| `data/molformer_c3_mtr_embedding_seed42.parquet` | Variant M extracted embedding |

Existing files **NOT modified**:
- `run_chemprop_pretrain.py` (rank-1 driver script — protected)
- `run_molformer_c3_pretrain.py`
- `peft_trainer.py` (only imported, not edited)

## 11. Out of scope (explicit non-goals)

- Mordred descriptors as MTR target (NaN profile is heavier; v1 keeps to RDKit 217 per Sultan)
- Joint MTR + log2fc multi-task (the user explicitly said `後段ではなく普通に最初から` — MTR is the *sole* objective)
- Cross-NR multitask (rejected 2026-05-04)
- Larger ChemProp variant (hidden=1024, depth=6) — would re-enter the model-size axis, not multitask scaling
- Data augmentation (corpus expansion to 50k+) — recipe validation only; corpus expansion is future work conditional on G3 pass

## 12. Success criteria

**Minimum success**: at least one variant passes G1–G3 cleanly, providing evidence that MTR-DA is a viable lever on PXR. This unblocks the future data-augmentation pathway.

**Target success**: at least one variant passes G4 (LB win Δ ≤ −0.001), recovering some of the rank 1-3 gap.

**Stretch success**: both variants pass G4 individually, double-swap drives a rank-1-driver-class jump (precedent: seed5ens 2026-04-25, OOF −0.0116 / LB −0.0065).

## 13. Risk register

| Risk | Probability | Mitigation |
|---|---|---|
| MTR-DA recipe doesn't transfer to PXR (G2 fail) | Medium — past family-style nulls (KANO contrastive, Uni-Mol v2) suggest PXR is hard for raw new families | Cheap fail-fast: Variant C runs in 30 min; one shot to learn |
| Variant M rotary fix breaks (PEFT history) | Low — pattern is documented | Reuse exact `peft_trainer.py` code path |
| G3 passes, G4 reverses (OOF/LB amp, well-documented memory) | Medium — 5+ recent precedents | LB A/B with hard revert; hold importance affine |
| Hidden leak we missed in audit | Low — 6-risk audit covers known classes | Audit script is run-blocking; manual spec review before merge |
| Compute conflict with Boltz-2 jobs | Low — currently no Boltz job running | Check GPU before launch; user permission before any long run |
