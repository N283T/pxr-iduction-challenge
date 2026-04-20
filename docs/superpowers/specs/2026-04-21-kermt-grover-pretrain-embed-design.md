# Design: KERMT/GROVER Continued-Pretrain on log2_fc + TabPFN-on-Embedding

- **Status**: Proposed (2026-04-21)
- **Author**: Claude Code session, brainstormed with user
- **Related**:
  - Follows the Buterez 2024 strategy-3 recipe that produced the pool's top three caruana-weighted members: `tabpfn_chemprop_pretrain_embed` (PR #87), `tabpfn_molformer_c3_pretrain_embed` (PR #98), `tabpfn_2d_full_boltz_log2fc_pred` (PR #93).
  - Adds a **graph-transformer** backbone (KERMT / GROVER) — currently the pool has only one graph-backbone member (chemprop D-MPNN). Introduces an architecturally decorrelated representation.
  - KERMT: NVIDIA reimplementation of GROVER (https://github.com/NVIDIA-Digital-Bio/KERMT, arXiv:2510.12719). Checkpoint-compatible with original Tencent GROVER weights.

## Goal

Apply the "continued-pretrain on log2_fc → frozen → embed → TabPFN" recipe (already pool-dominant on chemprop and MoLFormer-c3 backbones) to the KERMT graph transformer. Deliver one new ensemble member (`tabpfn_kermt_pretrain_embed`) and re-run the ensemble + calibration end-to-end. If it moves the 10→11-pool caruana_bag20 OOF MAE below 0.4314 (pre-KERMT baseline), submit to LB.

## Rationale

1. **Pool strongest-three are all the same recipe**: chemprop + MoLFormer-c3 + 2d_full_boltz_log2fc_pred. Transformer-family and GNN-family both validate the pattern. A **graph-transformer** family member is the natural next slot.
2. **Backbone decorrelation matters for caruana**: Current pool has one GNN (chemprop) and two transformers (molformer-c3, RF/TabPFN on static features). A graph transformer (message passing + self-attention on molecular graphs) sits between these and should decorrelate.
3. **Public weights available**: GROVER_base (~48M params, hidden=800, depth=6, 4 heads) and GROVER_large (~108M) are mirrored on Google Drive (headless via `gdown`, no OneDrive browser auth needed). KERMT accepts these checkpoints directly.
4. **Fits 16GB VRAM**: KERMT paper's 32GB recommendation targets from-scratch ZINC pretraining with large batch. Continued-pretrain on 13k compounds with batch 32–64 should fit comfortably on RTX 5080.

## Non-goals (deferred)

- **From-scratch pretrain**: we use the published `grover_base.pt` checkpoint as starting point. Full ZINC pretrain is out of scope (~days of compute).
- **GROVER_large**: start with `grover_base`. If base succeeds and adds pool weight, evaluate `large` as a follow-up PR.
- **Direct pEC50 finetune (no log2_fc step)**: empirically weak (matches "just use frozen pretrain embed" which is known渋い per session discussion). Only the log2_fc continued-pretrain version is pursued.
- **cuik-molmaker acceleration**: KERMT's optional NVIDIA featurizer. Skip; use RDKit fallback. `--use_cuikmolmaker_featurization` off.
- **LB submission as part of the PR**: runs offline; LB submission is a separate user-confirmed step after caruana weight verification.
- **Multi-task aux heads** (3-head or 6-concentration log2_fc): matches the chemprop-pretrain / molformer-c3-pretrain 2-head recipe (`log2fc_8p25`, `log2fc_33`). Additional heads are deferred.

## Architecture

### Environment isolation

KERMT's dependency stack (DGL, cuik-molmaker build deps, older PyTorch Lightning) conflicts with our pixi env (numpy 2.x, current torch). Strategy:

- Clone to `~/ghq/github.com/NVIDIA-Digital-Bio/KERMT`
- Create separate conda env `kermt` from `environment.yml`
- KERMT runs outside pixi; only its **outputs** (embedding npz) cross the boundary
- pixi side imports the npz, writes to `db/compound_kermt_pretrain_embed`, then TabPFN/ensemble proceeds normally

### Weight acquisition

- Google Drive mirrors (Tencent original, KERMT-compatible):
  - GROVER_base: `1hiGwOzoRfbJQPWj0V_mtOffsqIIAMgjl`
  - GROVER_large: `1bMg_ntUKEoOmHM0KoUi1XYJvzPBnHeWw`
- Tool: `gdown` (installed in the `kermt` conda env)
- Checksum verification: record SHA256 on first download, re-check on subsequent machines

### New files

| File | Purpose |
|---|---|
| `track1_activity/scripts/run_kermt_pretrain.sh` | Shell wrapper: activates `kermt` conda env, runs `main.py finetune` on log2_fc CSV with `grover_base.pt` as `--checkpoint_path`. Writes finetuned checkpoint to `models/kermt/pretrain/`. |
| `track1_activity/scripts/run_kermt_embed_extract.sh` | Shell wrapper: activates `kermt` env, runs `main.py fingerprint` (or the KERMT-equivalent embedding extraction) on all 13,136 compounds. Writes npz with `{"compound_id": [...], "embedding": [...]}`. |
| `track1_activity/scripts/prepare_kermt_pretrain_csv.py` | pixi-side: export `compounds.std_smiles` + `log2fc_8p25` + `log2fc_33` (2-head, same SQL as chemprop_pretrain) to `data/kermt/pretrain.csv`. Train/val 90/10 split written as separate CSVs. |
| `db/compute_kermt_embeddings.py` | pixi-side: load npz from `run_kermt_embed_extract.sh` output, upsert into `compound_kermt_pretrain_embed` table. |
| `db/compound_kermt_pretrain_embed_schema.sql` | Schema mirroring `compound_molformer_c3_pretrain_embed`. |
| `track1_activity/scripts/run_tabpfn_kermt_embed.py` | Downstream: TabPFN v7 on the extracted embedding, 5-fold UMAP, record as `tabpfn_kermt_pretrain_embed_umap_default`, save OOF. |

### Pretrain data (reuse existing SQL)

Mirror `track1_activity/scripts/run_chemprop_pretrain.py::load_pretrain_data`:

```sql
SELECT c.id AS compound_id,
       c.std_smiles AS smiles,
       agg.log2fc_8p25,
       agg.log2fc_33
FROM compounds c
LEFT JOIN (
    -- same aggregation used by chemprop_pretrain/molformer_c3_pretrain
    ...
) agg ON agg.compound_id = c.id
-- 13,136 rows; NaN where a concentration was not screened
```

- Total: 13,136 compounds
- Labeled: 10,752 at 8.25 µM, 9,527 at 33 µM
- Transductive: test-set SMILES seen by the encoder but contribute no loss
- Correlation with pEC50: r=0.72 (8.25 µM), r=0.50 (33 µM)

Export to three CSVs (KERMT `main.py finetune` requires explicit file paths):

- `data/kermt/pretrain_train.csv` — 90% random (seed=42), columns: `smiles, log2fc_8p25, log2fc_33`
- `data/kermt/pretrain_val.csv` — 10%
- `data/kermt/pretrain_all.csv` — all 13,136 for embedding extraction input (smiles only)

### Pretrain configuration

Invocation (shell wrapper):

```bash
conda activate kermt
cd ~/ghq/github.com/NVIDIA-Digital-Bio/KERMT/code
export PYTHONPATH=$PWD
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python main.py finetune \
    --data_path ~/pxr-iduction-challenge/data/kermt/pretrain_train.csv \
    --separate_val_path ~/pxr-iduction-challenge/data/kermt/pretrain_val.csv \
    --save_dir ~/pxr-iduction-challenge/models/kermt/pretrain \
    --checkpoint_path ~/pxr-iduction-challenge/models/kermt/grover_base.pt \
    --dataset_type regression \
    --split_type scaffold_balanced \
    --ensemble_size 1 \
    --num_folds 1 \
    --no_features_scaling \
    --ffn_hidden_size 256 \
    --ffn_num_layers 3 \
    --bond_drop_rate 0.1 \
    --epochs 30 \
    --metric mae \
    --self_attention \
    --dist_coff 0.15 \
    --max_lr 1e-4 \
    --final_lr 2e-5 \
    --dropout 0.1 \
    --batch_size 32
```

Key parameter choices (differ from KERMT example defaults):

- `--ffn_hidden_size 256` (small, matches molformer-c3 2-head FFN) — weak label, avoid over-capacity head
- `--epochs 30` with manual tuning (no HPO; weak-label regression tolerates default LoRA-scale settings)
- `--batch_size 32` (conservative for 16GB VRAM with `grover_base`; bump to 64 if VRAM headroom)
- `--metric mae` (matches our evaluation protocol)
- Do **not** pass `--features_generator rdkit_2d_normalized*`; we want the model to learn from graph alone (embedding should be from graph encoder, not graph+RDKit concat)

### Embedding extraction

KERMT follows GROVER convention: after finetuning, the trained checkpoint retains both the graph encoder and FFN head. We extract **the pre-FFN readout vector** (atom-mean + bond-mean concatenation, 1600d for `grover_base` with `--embedding_output_type both`) for each compound.

Extraction script (KERMT/GROVER supports `main.py fingerprint` for this; if absent we patch a short hook):

```bash
python main.py fingerprint \
    --data_path ~/pxr-iduction-challenge/data/kermt/pretrain_all.csv \
    --checkpoint_dir ~/pxr-iduction-challenge/models/kermt/pretrain \
    --no_features_scaling \
    --output ~/pxr-iduction-challenge/data/kermt/embeddings.npz
```

Output npz schema:

```python
{
    "compound_id": np.ndarray,  # shape (13136,), int
    "embedding": np.ndarray,    # shape (13136, 1600), float32
}
```

### DB schema

```sql
CREATE TABLE IF NOT EXISTS compound_kermt_pretrain_embed (
    compound_id INTEGER PRIMARY KEY REFERENCES compounds(id),
    embedding FLOAT[] NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
```

Mirrors existing `compound_molformer_c3_pretrain_embed`. Idempotent upsert via `ON CONFLICT (compound_id) DO UPDATE`.

### Downstream (TabPFN on pEC50)

- CV: 5-fold UMAP split (seed=42, n_clusters=50, Morgan+Jaccard) — canonical
- Model: TabPFN v7, default hyperparams (matches `tabpfn_chemprop_pretrain_embed` and `tabpfn_molformer_c3_pretrain_embed`)
- Input: 1600-dim embedding (may PCA-reduce to 768 if TabPFN OOM; decide at runtime)
- DB record: `tabpfn_kermt_pretrain_embed_umap_default`, model_type=`tabpfn`, feature_set=`kermt_pretrain_embed`
- Save OOF to `experiment_oof_predictions`

### Ensemble integration (add-only, Approach C)

Append to `ENSEMBLE_MODELS` allow-list in `run_ensemble.py`:

```python
"tabpfn_kermt_pretrain_embed_umap_default",
```

Re-run `run_ensemble.py` → caruana_bag20 weights the 11-member pool. Per the "no revert on LB regression" policy, if caruana weight < 0.03 or 10→11 pool MAE regresses, remove from allow-list in a follow-up 1-line PR (keep the framework).

Post-ensemble: `run_ensemble_calibrate.py` → `linear_pos` calibrator (already the 2026-04-20 winner) → single CSV for LB.

## Data flow

```
compounds.std_smiles (13,136)
     ↓ prepare_kermt_pretrain_csv.py (pixi)
[data/kermt/pretrain_{train,val,all}.csv]
     ↓ run_kermt_pretrain.sh (kermt conda env)
[models/kermt/pretrain/fold_0/model_0/model.pt]
     ↓ run_kermt_embed_extract.sh (kermt conda env)
[data/kermt/embeddings.npz]
     ↓ compute_kermt_embeddings.py (pixi)
[db.compound_kermt_pretrain_embed]
     ↓ run_tabpfn_kermt_embed.py (pixi)
[db.experiment_oof_predictions (tabpfn_kermt_pretrain_embed_umap_default)]
     ↓ run_ensemble.py (ENSEMBLE_MODELS += new member)
[ens_caruana_bag20 OOF + test predictions]
     ↓ run_ensemble_calibrate.py
[ens_caruana_bag20_calibrated_best.csv]
```

## Acceptance criteria (PR merge)

1. **Single-model OOF MAE ≤ 0.48** — loose bar, matches the two previous pretrain-embed recipes. Better is welcome, similar-with-decorrelation is acceptable.
2. **caruana_bag20 weight > 0** on the new member.
3. **11-pool caruana_bag20 OOF MAE ≤ 0.4314** — no regression vs current 10-pool baseline. A drop of 0.001+ is the bar for LB submission consideration.
4. **OOF Pearson correlation with existing pool members < 0.96** — sanity check that the new member isn't a near-duplicate of chemprop pretrain embed.
5. **ruff format + ruff check clean** on all pixi-side new/modified files.
6. All new DB rows use `on_conflict_replace=True` (idempotent re-run).

### Failure handling

- (1) fails (MAE > 0.48): inspect pretrain val MAE curve. If val MAE flat, increase epochs or lower lr. If val MAE drops but embedding OOF bad, try `--ffn_hidden_size 512` or `grover_large`. Report to user before scope expansion.
- (2) fails (caruana weight 0): member is redundant with chemprop pretrain embed. Still merge (framework value); drop from allow-list in follow-up.
- (3) fails (pool regresses): do not submit; keep as experiment record; drop from allow-list.
- (4) fails (r > 0.96): inspect which member — if it's the chemprop-pretrain embed, consider whether GROVER_large is sufficiently different (separate PR).

## Testing

- Smoke: `--epochs 2` pretrain + `head(100)` embed extract + `--outer-folds 2` TabPFN. Target < 20 min end-to-end.
- No unit tests (DL code, existing codebase convention).
- ruff format + ruff check as automatic gates.
- Smoke run output sanity: pretrain val loss decreases monotonically over first 2 epochs, embedding npz shape is (N, 1600).

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| `environment.yml` install fails on WSL2 (DGL/CUDA version mismatch) | Medium | Fall back to KERMT Dockerfile. Extract weights into a bind mount; run finetune in container; copy embeddings npz out. |
| GROVER `main.py fingerprint` missing in KERMT (possibly renamed) | Medium | Inspect KERMT main.py on clone; if renamed, patch a 20-line extraction script that loads checkpoint + extracts graph readout pre-FFN. |
| `gdown` virus-scan prompt for large file (>100MB) | Low | `gdown --fuzzy` or manual cookie workaround; GROVER_base should be ~250MB, triggers warning but gdown handles. |
| Continued pretrain val loss does not decrease (log2_fc already well-encoded by GROVER pretrain) | Low | This is itself informative: means chemical features are already captured, embedding extraction can proceed anyway. Document and proceed. |
| VRAM OOM during finetune | Low | Drop `--batch_size` to 16; `grover_base` at batch 32 with hidden=800 should use ~6GB. |
| TabPFN OOM on 1600d embedding | Low | PCA to 768d, or rely on TabPFN's built-in feature selection. `tabpfn_2d_full_boltz` already handles >2000d. |
| Embedding extraction fails on stereo-impossible compounds | Very Low | Use `std_smiles` (stereo-repaired via `db/fix_bridged_stereo.py`); skip any SMILES KERMT rejects (record in log). |
| cuik-molmaker build hangs or fails | Already mitigated | Skip via `--use_cuikmolmaker_featurization` absent from the command; use RDKit fallback. |

## ETA

- Environment setup (conda env, weights download, KERMT clone): **~1 h** (first time; mostly conda solve + download)
- Phase 1 continued-pretrain (13k compounds, 30 epochs, batch 32, grover_base): **~1.5–3 h** on RTX 5080
- Phase 2 embedding extraction: **~15 min**
- Phase 3 TabPFN 5-fold: **~30 min**
- Phase 4 ensemble re-run + calibration: **~20 min**

**Total: ~4–6 h**. Fits within a single working day; no overnight scheduling needed.

## Out of scope (future PRs)

- **GROVER_large**: follow-up if base succeeds.
- **KERMT from-scratch pretrain on PXR-expanded chemistry**: days of compute, uncertain ROI.
- **ChemFM / ChemFMv2**: parallel transformer foundation models; consider only if KERMT's graph-transformer signal is weak and we need a different family entirely.
- **Multi-task aux beyond log2fc 2-head**: 6-concentration heads, per-concentration loss weighting, or protein-conditioned pretraining. None have evidence of helping yet.
- **Patterns (a) / (c) on KERMT**: frozen-head pEC50 FT and predicted-log2fc-as-feature (mirroring PR #92/#93 for chemprop).
