# Design: MoLFormer-c3 Pretrain on log2_fc + TabPFN-on-Embedding

- **Status**: Approved (2026-04-20 evening)
- **Author**: Claude Code session, brainstormed with user
- **Related**: Follows two negative-result PRs (PR #95 MoLFormer direct PEFT FT, PR #97 ChemProp FMGCL aux loss). User pivot to "pretrain+frozen+embedding" pattern which already yielded the pool-strongest member (`tabpfn_chemprop_pretrain_embed`, MAE 0.437).
- **Reference recipe**: PR #87 `chemprop_pretrain_finetune_frozen` + Buterez 2024 strategy-3 (low-fidelity embedding as side feature).

## Goal

Add a transformer-encoder-family pool member produced by the same recipe that made `tabpfn_chemprop_pretrain_embed` the pool-strongest: pretrain a SMILES transformer on single-concentration `log2_fc` (weak-label side assay with r=0.72 to pEC50), extract frozen [CLS] embeddings for all 13,136 compounds, then run TabPFN on those embeddings for the pEC50 regression. Uses `DeepChem/MoLFormer-c3-1.1B` (the only HF-published variant with weights) as backbone.

## Rationale

1. **Pool-strongest precedent**: `tabpfn_chemprop_pretrain_embed` (MAE 0.437) uses this exact recipe on chemprop. Pool depends heavily on it (caruana weight 0.36). Replicating with a different backbone family (transformer, not GNN) should add decorrelated signal.
2. **PR #87 ablation insight**: "pretrain helps only when the encoder is frozen" (scratch MAE 0.530, full FT 0.507, frozen 0.471 in chemprop land). PR #95 direct PEFT FT of MoLFormer validated the same for transformers (MAE 0.529 — matched the chemprop scratch/FT levels, worse than frozen+embedding would have given).
3. **PEFT framework reuse**: PR #95 shipped `peft_backbones.py` + `peft_methods.py` + `peft_trainer.py`. This PR adds a second use case (pretrain on log2_fc instead of pEC50), validating the registry pattern.
4. **HF availability constraint**: Only `DeepChem/MoLFormer-c3-1.1B` has actual weights on HF. `100M` and `550M` variant pages have only README/.gitattributes (empty). The "1.1B" refers to pretraining token count, not parameters — actual model is ~80M params (hidden=768, 12 layers, verified from config.json).

## Non-goals (deferred)

- Applying same recipe to additional backbones (ChemBERTa-77M, BERT-SMILES) — wait for this one to validate
- Three reuse patterns (a)/(b)/(c) in parallel — pattern (b) (embedding→TabPFN) is the only one with pool-strongest evidence; defer (a) frozen-head-FT and (c) predicted-log2fc-concat
- LB submission — this PR merges if OOF acceptance passes; LB is a separate user-confirmed step
- Full `peft` library fine-tune comparison vs frozen-head — PR #87 ablation already settled this for GNN; we trust the same insight transfers

## Architecture

### Backbone change (registry-only)

Add one entry to `track1_activity/src/peft_backbones.py` `BACKBONES` dict:

```python
"molformer_c3_1_1b": {
    "hf_id": "DeepChem/MoLFormer-c3-1.1B",
    "hidden_dim": 768,
    "max_length": 202,
    "trust_remote_code": True,
    "lora_target_modules_qv": ["query", "value"],
    "lora_target_modules_qkvo": ["query", "key", "value", "dense"],
    "fix_rotary": True,  # same bug as ibm/MoLFormer-XL (inherited architecture)
}
```

Architecture is identical to `ibm/MoLFormer-XL-both-10pct` (verified config.json). `fix_rotary=True` is the same issue #30 fix we already integrated. `auto_map` in the DeepChem config still references the ibm modeling code, so `trust_remote_code=True` pulls from the ibm repo.

### New files

| File | Purpose |
|---|---|
| `track1_activity/scripts/run_molformer_c3_pretrain.py` | Phase 1: pretrain MoLFormer-c3-1.1B + LoRA on 13,136 compounds with 2-head NaN-masked log2_fc MSE. Save LoRA adapter weights + projection head state_dict. |
| `track1_activity/scripts/run_molformer_c3_embed_extract.py` | Phase 2: load pretrained LoRA, extract [CLS] 768d embedding for all 13,136 compounds, write to DB table `compound_molformer_c3_pretrain_embed`. |
| `track1_activity/scripts/run_tabpfn_molformer_c3_embed.py` | Phase 3: 5-fold UMAP TabPFN on the embedding, record experiment to DB, save submission CSV. |
| `db/compound_molformer_c3_pretrain_embed_schema.sql` | Schema for embedding table (mirrors existing `compound_chemberta`). |

### Pretrain data (match PR #87 recipe)

Data loader mirrors `track1_activity/scripts/run_chemprop_pretrain.py::load_pretrain_data`:

```sql
SELECT c.id AS compound_id,
       c.std_smiles AS smiles,
       agg.log2fc_8p25,
       agg.log2fc_33
FROM compounds c
LEFT JOIN (
    -- aggregate single_concentration to one row per compound,
    -- 2-column wide form: (log2fc_8p25, log2fc_33)
    ...
) agg ON agg.compound_id = c.id
-- All 13,136 compounds, NaN where no measurement at that concentration
```

- 13,136 compounds total (includes train 4,140 + test 513 + counter 2,860 + never-tested ~5,623)
- Transductive: test SMILES' structures are seen by the encoder but contribute no loss (both target cols NaN)
- Labels: 10,752 at 8.25 µM, 9,527 at 33 µM (per PR #87 description)
- Correlation with pEC50: r=0.72 (8.25 µM), r=0.50 (33 µM) — per PR #87

### Pretrain architecture

- Backbone: `DeepChem/MoLFormer-c3-1.1B` with rotary-embedding fix (`fix_rotary=True`)
- PEFT: LoRA rank=16, alpha=32, dropout=0.1, target="qkvo", task_type=None — fixed defaults (middle of the PR #95 Optuna search range, skip Optuna for pretrain phase since the weak-label task tolerates looser hyperparameters)
- Head: 2-output MLP (hidden_dim=256, GELU, dropout=0.1) — small because the signal is weak
- Loss: NaN-masked MSE on each head separately, average the two
- Optimizer: AdamW, backbone_lr=2e-4, head_lr=1e-3, weight_decay=1e-3
- Scheduler: cosine annealing from T_max=max_epochs
- Early stopping: patience=10 on val_mse (90/10 random val split, seed=42)
- Max epochs: 50
- Batch size: 64

### Embedding extraction

- Load pretrained LoRA adapter + base model in eval mode
- For each of 13,136 compounds: forward pass, extract `last_hidden_state[:, 0, :]` ([CLS] 768d)
- Write to DB: `INSERT INTO compound_molformer_c3_pretrain_embed (compound_id, embedding) VALUES (...)` with `ON CONFLICT (compound_id) DO UPDATE` for re-runs
- Estimated time: 10 min on RTX 5080

### DB schema

```sql
CREATE TABLE compound_molformer_c3_pretrain_embed (
    compound_id INTEGER PRIMARY KEY REFERENCES compounds(id),
    embedding FLOAT[] NOT NULL
);
```

Mirrors existing embedding tables (`compound_chemberta`, `compound_molformer`, etc.).

### Downstream (TabPFN on pEC50)

- CV: 5-fold UMAP split (seed=42, n_clusters=50, Morgan+Jaccard) — canonical
- Model: TabPFN v7, default hyperparams (no Optuna — matches `tabpfn_chemprop_pretrain_embed` recipe)
- Input: 768-dim [CLS] embeddings for train+test
- DB record: name `tabpfn_molformer_c3_pretrain_embed_umap_default`, model_type=`tabpfn`, feature_set=`molformer_c3_pretrain_embed`
- OOF save for ensemble integration

## Ensemble integration (Approach C: add-only)

- Append `tabpfn_molformer_c3_pretrain_embed_umap_default` to `ENSEMBLE_MODELS` allow-list in `run_ensemble.py`
- Keep existing `tabpfn_chemprop_pretrain_embed` (the pool-strongest) — the hope is decorrelation from different backbone family
- Re-run `run_ensemble.py`; caruana decides whether to keep both

## Acceptance criteria (PR merge)

1. **Single-model OOF MAE <= 0.48** — loose bar; we want to be within 0.04 of chemprop pretrain embed (0.437) or better. A stronger lift is nice; a similar value with decorrelation is also acceptable.
2. **Caruana_bag20 weight > 0** on the new member.
3. **10-pool caruana_bag20 OOF MAE <= 0.4327** — no regression vs pre-PEFT-MoLFormer baseline. Improvement would be great.
4. **ruff format + ruff check clean**.

Failure handling:
- (1) fails (MAE > 0.48): investigate pretrain convergence (was val_mse actually decreasing?), possibly extend pretrain epochs or tune LoRA rank. Report to user before scope expansion.
- (2) fails but (1) passes: member is redundant with chemprop pretrain embed (high correlation). Still merge (experiment record and framework reuse are valuable); drop from allow-list in a 1-line follow-up if decided unnecessary.

## Testing

- Smoke test: `--max-epochs 2` on pretrain + `--head 100` rows for embed + `--outer-folds 2` for TabPFN. Should complete in 15-20 min.
- No unit tests for DL code (existing codebase convention).
- ruff format + ruff check as the only automatic gates.

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| `DeepChem/MoLFormer-c3-1.1B` has broken weights or loading issues | Medium | Fallback to `ibm/MoLFormer-XL-both-10pct` (PR #95 verified working). Document in registry comment. |
| 2-head FFN architecture needs custom subclass since chemprop RegressionFFN takes single head | Low | Chemprop `RegressionFFN(n_tasks=2)` supports multi-task; for MoLFormer we build the head ourselves (plain PyTorch `nn.Sequential` ending in `nn.Linear(hidden, 2)`) |
| Pretrain doesn't converge (log2_fc is noisy proxy) | Medium | Follow PR #87 evidence that it does converge on 13k compounds with 90/10 split. If val_mse stalls, increase warmup and lower backbone_lr. |
| TabPFN OOM on 13k-compound embedding matrix | Low | TabPFN v7 handles up to ~10k rows; we use 4,140 train rows for the pEC50 task, well within limits |
| DeepChem model needs `trust_remote_code=True` with remote code pulling from `ibm/MoLFormer-XL-both-10pct` (config `auto_map`) | Already known | Registry entry sets `trust_remote_code=True`; first load downloads ibm modeling code (~50MB), caches thereafter |

## ETA

- Phase 1 pretrain (13,136 compounds, batch 64, max 50 epochs, early stop ~15-20): **~2-3 h**
- Phase 2 embed extract: ~10 min
- Phase 3 TabPFN 5-fold: ~30 min
- Plus ensemble re-run: ~10 min

**Total: ~3-4 h on RTX 5080**. Launch this evening, results overnight.

## Out of scope (future PRs)

- Pattern (a): frozen-head direct FT on pEC50 (separate from embedding→TabPFN downstream)
- Pattern (c): predict log2_fc on test → concat as feature to `tabpfn_2d_full_boltz`
- Apply recipe to ChemBERTa variants (`seyonec/ChemBERTa-zinc-base-v1` etc.) if 1.1B MoLFormer-c3 proves the recipe transfers
- Further backbone exploration (ChemFMv2-20M, adapter-based PEFT)
