# Design: PEFT MoLFormer-XL (LoRA) for PXR pEC50 Regression

- **Status**: Approved (2026-04-20)
- **Author**: Claude Code session, brainstormed with user
- **Related**: Issue tracker -- next session candidate #4 (LoRA on MoLFormer)
- **Strategy**: Approach (A) -- sequential exploration, this PR ships only MoLFormer-XL x LoRA

## Goal

Add a transformer-encoder-based ensemble member to push pool orthogonality and improve LB rank beyond the current 9/89 (MAE 0.4414). The current ensemble pool consists of chemprop / TabPFN / Boltz-derived families only -- no SMILES transformer encoder. PEFT (LoRA) is chosen over full fine-tuning per deep research findings: with only 4,140 labels, parameter-efficient fine-tuning generalizes better and is easier to mix with descriptor-based heterogeneous ensembles (EffiChem 2025, ChemFM Nature MI 2025).

## Non-goals (deferred to later PRs)

- Adapter / last-k-layer-FT comparison on the same MoLFormer-XL backbone
- Other backbones: ChemFMv2-20M, ChemFM-1B, ChemFM-3B (QLoRA), ChemBERTa-3
- Ensemble re-weighting -- the new member is added to the existing pool and `caruana_bag20` is re-fit; no other ensemble strategy changes
- LB submission -- this PR records the experiment and submission file but does not submit (LB submission is a separate user-confirmed step)

## Architecture

### Module split

```
track1_activity/scripts/
  run_peft_finetune.py          # NEW: entry point, CLI, Optuna driver
  archive/run_molformer_finetune.py  # MOVED: prior full-FT prototype, kept for reference

track1_activity/src/
  peft_backbones.py             # NEW: backbone registry (HF model id, hidden dim, AutoModel/AutoTokenizer wiring)
  peft_methods.py               # NEW: PEFT method registry (peft library wrapper)
  peft_trainer.py               # NEW: train_one_fold(), predict(), OOF aggregation, GPU mem cleanup
```

The split exists so future PRs (adapter / last-k / new backbones) add a single registry entry rather than touching the trainer or CLI.

### Backbone registry (`peft_backbones.py`)

```python
BACKBONES = {
    "molformer_xl": {
        "hf_id": "ibm/MoLFormer-XL-both-10pct",
        "hidden_dim": 768,
        "trust_remote_code": True,
        "max_length": 202,
        "lora_target_modules_qv": ["query", "value"],
        "lora_target_modules_qkvo": ["query", "key", "value", "dense"],
    },
    # Future entries: chemfmv2_20m, chemfm_1b, chemberta3_*
}
```

Backbones are loaded with `AutoModel.from_pretrained` -- regression head is added by `peft_trainer.py`, not by registry.

### PEFT method registry (`peft_methods.py`)

```python
def build_lora_config(backbone_meta, params):
    target = backbone_meta[f"lora_target_modules_{params['lora_target']}"]
    return LoraConfig(
        r=params["lora_rank"],
        lora_alpha=params["lora_alpha"],
        lora_dropout=params["lora_dropout"],
        target_modules=target,
        bias="none",
        task_type=None,  # custom regression head, no peft task adaptation
    )

PEFT_METHODS = {"lora": build_lora_config}
# Future: "adapter", "last_k"
```

### Trainer (`peft_trainer.py`)

- `MolFormerRegressor(nn.Module)`: PEFT-wrapped backbone + 2-layer MLP regression head (Linear -> GELU -> Dropout -> Linear -> 1).
- Two parameter groups: PEFT-trainable backbone params (`backbone_lr`), head params (`head_lr`). Frozen base weights are excluded from optimizer (peft handles `requires_grad=False`).
- AdamW + CosineAnnealingLR, MSE loss, `clip_grad_norm_=1.0`.
- Early stopping on validation MAE, patience configurable (8 during Optuna, 12 final).
- After each fold: `del model; torch.cuda.empty_cache()` to avoid 16GB VRAM fragmentation across 5 folds.

## Data flow

```
load_train_smiles_target()  ->  train_smiles, y_train  (4140 rows)
load_test_smiles()          ->  test_smiles            (513 rows)
                                       |
       umap_split_indices(seed=42, n_clusters=50, k=5)  -> outer_splits
       umap_split_indices(seed=123, k=3)                -> inner_splits
                                       |
                Optuna study (n_trials=20, MedianPruner)
                  inner CV: average RAE across 3 folds
                                       |
                                   best_params
                                       |
              Final 5-fold outer CV with best_params
                  per fold: train -> val -> test
                  collect oof_preds[val_idx], test_preds_all[fold]
                                       |
        record_experiment + save_oof_predictions in DB
        submission CSV: test_preds_all.mean(axis=0)
```

## Hyperparameter search space (Optuna)

| Parameter | Distribution | Notes |
|---|---|---|
| `lora_rank` | categorical {4, 8, 16, 32} | rank=8 is common default |
| `lora_alpha` | per-trial = `lora_rank * suggest_categorical([1, 2])` | alpha = 1x or 2x rank |
| `lora_dropout` | uniform [0.0, 0.2] | |
| `lora_target` | categorical {"qv", "qkvo"} | qv = lighter, qkvo = full attention |
| `backbone_lr` | log_uniform [1e-5, 5e-4] | PEFT tolerates higher LR than full FT |
| `head_lr` | log_uniform [1e-4, 5e-3] | |
| `weight_decay` | log_uniform [1e-4, 1e-1] | |
| `batch_size` | categorical {16, 32, 64} | |
| `head_hidden_dim` | categorical {128, 256, 512} | |
| `head_dropout` | uniform [0.1, 0.4] | |
| `max_epochs` | fixed 50 (Optuna) / 80 (final) | |
| `patience` | fixed 8 (Optuna) / 12 (final) | |

Trials: 20. Direction: minimize average RAE across inner-CV folds.

## CV protocol

- Outer split: `umap_split_indices(train_smiles, n_splits=5, n_clusters=50, seed=42)` -- canonical from PR #70 CV bake-off.
- Inner split: `umap_split_indices(train_smiles, n_splits=3, n_clusters=50, seed=123)` -- same split function, different seed for fair Optuna estimation.
- OOF coverage: every train compound appears in exactly one outer-fold validation set; aggregate OOF predictions for ensemble.
- Test predictions: 5-fold mean.

## DB integration

- `experiment_name`: `peft_{backbone}_{method}_r{rank}a{alpha}_{split}_default`
  - Example: `peft_molformer_xl_lora_r8a16_umap_default`
- `model_type`: `peft_finetune` (new value; ensemble queries can filter on this)
- `feature_set`: `smiles_transformer_peft`
- `hyperparameters` JSONB: full Optuna best_params + `final_max_epochs`, `final_patience`
- `notes`: `OOF RAE=X.XXXX, MAE=X.XXXX, umap_split, optuna_n=20, peft=lora`
- `record_experiment(...)` -> `save_oof_predictions(experiment_id, oof_preds)` (existing helpers in `evaluate.py`)

## Ensemble integration

- After this PR's experiment is recorded, separately invoke:
  ```
  pixi run python track1_activity/scripts/run_ensemble.py --strategy caruana_bag20
  ```
- This refits the existing 9-model pool + the new MoLFormer-LoRA member.
- **Acceptance signal**: caruana_bag20 assigns weight > 0 to the new member.
- If `wt = 0`: the member is OOF-redundant with the existing pool. Record this finding in DB notes; do not roll back the trained model (it costs nothing to keep) but treat MoLFormer-XL + LoRA as "explored, no ensemble lift" and proceed to next backbone (ChemFMv2-20M) in PR 2.

## Acceptance criteria (PR merge)

1. **OOF MAE <= 0.50** -- on par with the weakest current pool member (chemprop frozen variants are around 0.45-0.50).
2. **Caruana_bag20 assigns weight > 0** -- demonstrates orthogonality, not just absolute strength.
3. **CI green**: `ruff format`, `ruff check --fix`, `ty check` all clean.

If (1) passes but (2) fails: still merge (the experiment record is valuable), document outcome in PR body, and mark this backbone+PEFT combo as explored.

If (1) fails: investigate (likely cause: LoRA rank too low or LR mistuned); do not merge a non-functional script.

## Testing strategy

- **No unit tests** for the trainer -- DL training scripts have brittle seed-dependent outputs and the OOF metric in DB is the authoritative correctness signal.
- **Smoke test** (manual, before pushing): `--n-trials 1 --outer-folds 2 --max-epochs 2 --patience 2` should complete in under 5 minutes on RTX 5080 without NaN losses.
- **CI**: only `ruff format && ruff check && ty check` on the new files.

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| 5-fold x 80-epoch training exceeds 8h wall clock | Medium | Cap `max_epochs` at `1.5x` Optuna best-trial epoch count; sequential folds (no parallelism on single GPU) |
| MoLFormer rotary embedding fix (issue #30) regresses under PEFT | Low | Reuse the patch from `compute_embeddings.py`; smoke test confirms forward pass before launching full Optuna |
| `peft 0.12` API drift vs `transformers 5.5.0` | Medium | Pin `peft>=0.12,<0.14` in `pyproject.toml`; if API breaks at install time, downgrade to last known-good `0.12.0` |
| GPU OOM at `batch_size=64` x `max_length=202` | Medium | Catch `torch.cuda.OutOfMemoryError` in the trial loop -> mark trial as `optuna.TrialPruned` rather than crashing the study; fall back to `batch_size in {16, 32}` if all 64-batch trials fail |
| LoRA `target_modules` name mismatch (MoLFormer linear-attention modules may not be named `query`/`key`/`value`/`dense`) | Medium | Verify actual submodule names by inspecting `dict(model.named_modules())` once during smoke test; update `lora_target_modules_qv` / `lora_target_modules_qkvo` in `peft_backbones.py` to match. Document the verified names in the registry comment. |
| 16GB VRAM fragmentation across folds | Low | Explicit `del model; torch.cuda.empty_cache()` between folds (already in prototype) |

## Dependencies

Add to `pyproject.toml` `[tool.pixi.pypi-dependencies]`:
```toml
peft = ">=0.12,<0.14"
```

`transformers` (5.5.0) and `torch` (2.10.0+cu13) are already installed and compatible.

## Out of scope / future work (not this PR)

- PR 2: MoLFormer-XL x {adapter, last-k=2, last-k=4} comparison using the same trainer
- PR 3: ChemFMv2-20M x LoRA (smallest open ChemFM, smoke test for the family)
- PR 4: ChemFM-1B x LoRA (heavier; gated by PR 3 outcome)
- PR 5: ChemFM-3B x QLoRA (4-bit; only if PRs 3-4 show ChemFM family is competitive)
- PR 6: ChemBERTa-3 framework integration (DeepChem-based; deeper integration cost)
- ChemFM license (CC BY-NC 4.0) compatibility check with OpenADMET PXR challenge submission rules -- defer until PR 3 actually uses ChemFM weights
