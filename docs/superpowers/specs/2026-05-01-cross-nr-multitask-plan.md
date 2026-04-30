# Cross-NR Multi-Task Pretrain — Feasibility Plan (2026-05-01)

Per Codex 2026-04-30 (thread `019dde78`) top EV recommendation:
- LB upside: +0.0015 〜 +0.0045
- Material hit probability: 25-35%
- Approach: Buterez strategy-2 style (multi-task pretrain on NR family, transfer to PXR pEC50)

## Target shortlist (by ChEMBL ID)

| Target | NR family ID | ChEMBL | rationale |
|---|---|---|---|
| **PXR** (current) | NR1I2 | CHEMBL3401 | reference, do NOT use as auxiliary |
| **CAR** | NR1I3 | CHEMBL3199 | closest PXR cousin (xenobiotic sensors), highest expected transfer |
| **VDR** | NR1I1 | CHEMBL1944 | NR1I subfamily, structure-similar LBD |
| **FXR** | NR1H4 | CHEMBL2047 | NR1H, bile acid sensing (related xenobiotic biology) |
| LXR-α | NR1H3 | CHEMBL2746 | NR1H, sterol sensing |
| LXR-β | NR1H2 | CHEMBL2820 | NR1H, sterol sensing |
| PPAR-α | NR1C1 | CHEMBL239 | NR1C, lipid sensing, xenobiotic interaction |
| PPAR-γ | NR1C3 | CHEMBL235 | NR1C, drug-like inhibitor coverage |
| PPAR-δ | NR1C2 | CHEMBL3979 | NR1C |

**Priority bands:**
- **Tier 1 (must-include)**: CAR, VDR, FXR — closest to PXR, best transfer expected
- **Tier 2 (nice-to-have)**: LXR-α, LXR-β, PPAR-γ — same superfamily, drug-like overlap
- **Tier 3 (optional)**: PPAR-α/δ, GR, ER-α — broader, but transfer weakens

## Data acquisition

### Source 1: ChEMBL (REST API or togomcp `search_chembl_target`)
- pull `target_relations` filtered to wild-type human protein
- `standard_type IN ('EC50', 'IC50', 'Ki', 'AC50')`
- `standard_units = 'nM'`
- `data_validity_comment IS NULL`
- `assay_organism = 'Homo sapiens'`
- `assay_type IN ('B', 'F')` (binding or functional)
- exclude PXR compounds NOT to leak (filter by InChIKey not in PXR train+test)

### Source 2: ToxCast (via PubChem AID)
- PXR_NCGC_HG19 (PubChem AID 1224896 family)
- CAR_NCGC, VDR_NCGC etc — for cross-validation
- May overlap with Tox21 dataset

### Source 3: Tox21 (NIH)
- NR family panel
- Already has multi-task labels for PXR/CAR/AR/ER/AhR

## Expected data scale

Based on ChEMBL public counts (rough estimate, will verify):
- CAR: ~1500-3000 compounds with measurements
- VDR: ~3000-6000
- FXR: ~5000-10000
- PPAR-γ: ~8000-15000 (most studied)
- LXR-α: ~2000-4000

Total (Tier 1+2): ~20-40k compounds. Buterez strategy-2 succeeded at 21k single-conc; this scale should support same approach.

## Approach (Buterez strategy-2)

1. Pretrain ChemProp encoder on NR multi-task (per-target regression head, NaN-masked MSE)
2. Freeze encoder
3. Extract embeddings for PXR train + test
4. Train TabPFN on these embeddings → new pool member
5. Compare residual r vs current pool members (gate 2: ≤ 0.85)
6. If pass: bake-off as ADD/SWAP, strict gate evaluation

## Risk factors

- **Assay bias**: different ChEMBL assays use different protocols. Need to filter to canonical assay types per target
- **Overlap with PXR test**: if test compounds appear in CAR/VDR/FXR data with measurements, leakage. Mitigation: hold-out all InChIKeys appearing in test
- **Transfer noise**: PPAR-γ data dominates by sheer volume but may be far from PXR biology. Tier-1 only initial run
- **Compute**: chemprop multitask training at 20-40k scale is ~3-6 hours on RTX 5080 (similar to log2_fc pretrain run)

## Pilot plan (Day 1, 2026-05-01)

1. **Morning**: Pull ChEMBL data for CAR + VDR + FXR (Tier 1 only)
2. **Verify scale + quality**: row counts per target, measurement type distribution
3. **Filter + standardize**: ChEMBL pipeline (same as PXR), exclude PXR test InChIKeys
4. **Quick feasibility check**: train ChemProp single-task on each NR alone, measure CV MAE
5. **If quality OK** (single-task MAE < 0.6 reasonable): proceed to multi-task
6. **Pretrain**: chemprop encoder, multi-task heads, 100 epochs
7. **Extract + TabPFN**: embeddings for PXR pool member
8. **Gate eval**: single OOF MAE, residual r, family share, M2/Sp delta

## Decision gates

- **STOP** if Tier-1 data quality fails (>30% non-canonical assays, low coverage)
- **STOP** if pretrain converges to high CV MAE (> 0.7, suggests no transferable signal)
- **PROCEED** if PXR-side OOF gate 1 passes (single MAE ≤ 0.48)
- **SUBMIT** if strict gate (M2 ≤ -0.003, Sp ≥ -0.002, family share 0.65-0.80) passes

## Out of scope (for tomorrow)

- AF-2 helix features (separate side bet, lower priority)
- 5-pool reconstruction (depends on more LB data)
- Foundation model retry (low EV per Codex)
