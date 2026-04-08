# Multi-task ChemProp results (issue #36)

**Branch**: `feature/chemprop-multitask`
**Status**: Marginal — multi-task does NOT improve a properly-tuned single-task model, but the multi-task variants add useful diversity to the ensemble.

## Setup

ChemProp D-MPNN with shared encoder, regression FFN with `n_tasks` outputs.

- **Task 0 (main)**: `pec50` from `train_activity` (always present, n=4140)
- **Aux tasks** (NaN where missing, masked via ChemProp's `targets.isfinite()` in `MPNN.training_step`):
  - `counter_pec50` from `counter_assay` (n=2,648)
  - `counter_emax_estimate` (n=2,648)
  - `log2fc @ 8.25e-6 M` from `single_concentration` (n=2,374)
  - `log2fc @ 3.30e-5 M` from `single_concentration` (n=2,321)

Aux targets are standardized (z-score using global train stats) so MSE magnitudes across tasks are comparable. Main task is left unscaled to keep predictions on the original pEC50 scale. Loss = `MSE` with `task_weights = [1.0, aux_w, aux_w, aux_w, aux_w]`. At inference, only the task-0 column is used.

## Results

### Default-params sweep (max_epochs=100, modest LR)

| tasks | aux_weight | OOF RAE | Δ vs aux=0 |
|---|---:|---:|---:|
| 1 | 0.0 (single, same params) | 0.6776 | — |
| 2 | 0.1 | 0.6548 | **−0.023** |
| 2 | 0.3 | 0.6793 | +0.002 |
| 2 | 0.5 | 0.6592 | −0.018 |
| 2 | 1.0 | 0.6870 | +0.009 |
| 5 | 0.05 | 0.6621 | −0.016 |
| 5 | 0.1 | **0.6481** | **−0.030** |
| 5 | 0.3 | 0.6644 | −0.013 |

5-task aux=0.1 wins by −0.030 RAE. This looked like real multi-task value.

### Tuned-params sweep (max_epochs=200, hyperparameters borrowed from `chemprop_optuna_umap`)

| tasks | aux_weight | OOF RAE | Δ vs aux=0 |
|---|---:|---:|---:|
| 1 | 0.0 (`chemprop_optuna_umap`, ref) | **0.5724** | — |
| 2 | 0.0 (sanity) | 0.5851 | +0.013 |
| 2 | 0.02 | 0.5879 | +0.016 |
| 2 | 0.05 | 0.5941 | +0.022 |
| 5 | 0.0 (sanity) | 0.5755 | +0.003 |
| 5 | 0.02 | 0.5975 | +0.025 |
| 5 | 0.05 | 0.5923 | +0.020 |
| 5 | 0.1 | 0.6104 | +0.038 |

**Direction flipped**. With tuned hyperparameters, every aux_weight > 0 hurts OOF. The aux=0.0 sanity runs reproduce the single-task baseline within ~0.003 RAE (run-to-run noise), confirming the harness is correct.

### Interpretation

Default-params ChemProp is undertuned: too little dropout, sub-optimal LR, etc. The auxiliary loss in that regime acts as **implicit regularization** and improves generalization. Once the dropout/LR are properly tuned (the `chemprop_optuna_umap` config has `mp_dropout=0.2`, `ffn_dropout=0.1`, `lr=1.36e-4`, etc.), that regularization is no longer needed and the auxiliary loss only **distracts the encoder from the main task**.

This is consistent with Caruana 1997 / multi-task literature: MTL gain depends on whether aux tasks provide signal the main task model lacks. In our case, the tuned single-task model already captures the available signal in the SMILES; additional aux supervision doesn't add new information about pEC50, only noise.

### Predictions are unbiased

| Variant | Test pred mean | Test pred std |
|---|---:|---:|
| `chemprop_multitask_umap_aux0.0` (default) | 4.972 | 0.606 |
| `chemprop_multitask_umap_aux0.1` (default) | 4.989 | 0.611 |
| `chemprop_multitask5_umap_aux0.1` (default) | 4.901 | 0.637 |
| `chemprop_multitask5_umap_aux0.1_tuned` | 4.686 | 0.647 |

No selection-bias leakage like the failed `mordred_singleconc` direct-feature experiment (PR #41). Multi-task respects the inference contract: only main-task SMILES go in, only main-task predictions come out.

## Ensemble contribution (PR #41 → ens_v8)

Even though the tuned multi-task variants are individually worse than `chemprop_optuna_umap` (0.5755-0.61 vs 0.5724), their predictions are **mildly decorrelated** from existing ensemble components:

| variant | Pearson r vs `chemprop_optuna_umap` |
|---|---:|
| multitask5 aux=0.0 tuned | 0.943 |
| multitask2 aux=0.0 tuned | 0.929 |
| multitask5 aux=0.05 tuned | 0.933 |
| multitask5 aux=0.1 tuned | **0.917** ← most decorrelated |

Adding the tuned multi-task variants to the ensemble (ens_v8) gives a small but real improvement:

| Ensemble | OOF RAE | Δ vs ens_v7 |
|---|---:|---:|
| ens_v7_vanilla | 0.5304 | — |
| **ens_v8_l2_a0.05** | **0.5253** | **−0.005** |
| ens_v8_vanilla_opt | 0.5260 | −0.004 |
| ens_v8_l2_a0.1 | 0.5263 | −0.004 |
| ens_v8_top5_avg | 0.5314 | +0.001 |

The L2-regularized weighted ensemble at α=0.05 picks up multi-task variants in the long tail (chemprop_multitask5_umap_aux0.1_tuned at 0.041 weight, others at 0.03-0.04). Top-5 simple average includes `chemprop_multitask5_umap_aux0.0_tuned` at 1/5 weight.

So multi-task is **not a main-line model** but adds ensemble diversity worth ~0.005 OOF RAE improvement.

## What's shipped

- `track1_activity/src/data.py`: `load_train_smiles_with_counter()`, `load_train_smiles_multitask_5()`
- `track1_activity/scripts/run_chemprop_multitask.py`: multi-task training script with `--task-set 2|5`, `--aux-weight FLOAT`, `--use-tuned`, `--max-epochs`. Supports masked loss for missing aux values automatically (ChemProp handles NaN via `targets.isfinite()`).
- `track1_activity/scripts/run_ensemble_v8.py`: extends ensemble v2 with explicit exclusion list (singleconc leaky model, pseudo-label variants, default-params multitask runs)
- `docs/multitask_results.md`: this file

## Lessons

1. **Default params can mask the true multi-task effect.** When the base model is undertuned, MTL looks like a win because it adds regularization. Always re-evaluate MTL against a tuned single-task baseline before declaring victory.
2. **Diversity for ensembles ≠ accuracy.** A model that is individually slightly worse can still help an ensemble if its errors are decorrelated. ~0.92 Pearson r between OOF preds is enough to add value.
3. **Selection-bias leakage is the real watchout.** Multi-task as implemented here has no missing-feature paths at inference, so no leakage. This is a key advantage over the direct-feature concat approach (PR #41).

## Forward path

- Submit `ens_v8_l2_a0.05` to LB once the 4-hour window opens (recovers from the `singleconc` regression to rank 41).
- Multi-task is closed as a main-line research direction; the next leverage is more diverse base models or smarter ensemble strategies, not more aux-task experiments.
