# Design: ChemProp D-MPNN with Relative-Distance Auxiliary Loss

- **Status**: Approved (2026-04-20)
- **Author**: Claude Code session, brainstormed with user
- **Related**: Deep research report `docs/papers/pxr_challenge_research/deep-research-report-2026-04-19.md` section 337 (FMGCL-style relative-distance auxiliary for rank-aware regression). Follow-up to the PEFT MoLFormer pivot -- now exploring graph-family levers.
- **Strategy**: Approach (C) -- add as a second chemprop pool member; evaluate OOF + LB; decide swap/drop in a follow-up PR.

## Goal

Add a rank-aware auxiliary loss to the existing ChemProp D-MPNN pool member. The loss supplements standard MSE with a batchwise all-pairs relative-distance term so the model explicitly learns the ordering structure between compounds. Rationale: the competition metrics (Spearman, RAE) care more about relative ranking than absolute prediction accuracy; plain MSE under-optimises this. The approach is from FMGCL (Dong et al. 2025, DOI 10.1016/j.jmgm.2025.109014) which reports meaningful rank-metric improvements on molecular regression benchmarks with this exact trick.

## Non-goals (deferred)

- Differentiable Spearman / NDCG losses -- too complex for the marginal gain over relative-distance
- Hard-pair mining -- bug risk not justified for B=64 batches
- Replacing `chemprop_optuna_umap` in the pool -- kept for A/B comparison; swap decision happens in a follow-up PR after LB measurement
- Applying the same loss to attentivefp / gatedgcn in the same PR -- single change per PR
- Submission to LB in this PR (submission happens after merge, as a separate user-confirmed step)

## Background

Current chemprop_optuna_umap pool stats:
- OOF RAE: 0.5785
- OOF MAE: 0.5208
- Caruana_bag20 weight (9-pool): ~0.04 (before PEFT MoLFormer drop)

The hypothesis: adding the rank-aware aux term should push Spearman higher and potentially MAE lower (by regularising the embedding space). If it works, the new model either beats chemprop_optuna directly (swap candidate) or decorrelates from it enough to earn additional caruana weight (add candidate).

## Architecture

### Module split

```
track1_activity/src/
  losses.py                      # NEW: RelativeDistanceMSE class only
track1_activity/scripts/
  run_chemprop_relative_aux.py   # NEW: fork of run_chemprop_optuna.py + aux_weight hyperparam
```

`losses.py` is a new src module intended to host ChempropMetric subclasses we design ourselves. Keeps the class reusable from future training scripts (attentivefp, gatedgcn) without copy-paste.

### Loss class (`losses.py`)

```python
import torch
import torch.nn.functional as F
from chemprop.nn.metrics import ChempropMetric


class RelativeDistanceMSE(ChempropMetric):
    """MSE + batchwise all-pairs relative-distance auxiliary loss.

    Main:  MSE(pred, target)  per-sample (shape (B,))
    Aux:   MSE(|pred_i - pred_j|, |target_i - target_j|)
           averaged over upper-triangle pairs  -> scalar
    Total: main + aux_weight * aux  (aux broadcast to (B,))

    The aux term teaches the model to preserve relative distances between
    compounds, which correlates with rank metrics (Spearman, RAE). With
    aux_weight <= ~0.1 the main MSE dominates; larger values sacrifice
    absolute accuracy for ranking quality.

    Reference: Dong et al. 2025, DOI 10.1016/j.jmgm.2025.109014.
    """

    def __init__(self, aux_weight: float = 0.1, task_weights=1.0):
        super().__init__(task_weights)
        self.aux_weight = aux_weight

    def _calc_unreduced_loss(self, preds, targets, *args):
        main = F.mse_loss(preds, targets, reduction="none")
        p = preds.squeeze(-1)
        t = targets.squeeze(-1)
        n = p.shape[0]
        if n < 2:
            return main
        iu = torch.triu_indices(n, n, offset=1, device=p.device)
        d_pred = torch.abs(p[iu[0]] - p[iu[1]])
        d_true = torch.abs(t[iu[0]] - t[iu[1]])
        aux = F.mse_loss(d_pred, d_true, reduction="mean")
        return main + self.aux_weight * aux.expand_as(main)
```

Key implementation notes:
- `_calc_unreduced_loss` must return per-sample shape so ChempropMetric's outer reduction (weighted mean) applies correctly. Broadcasting the scalar aux via `expand_as(main)` satisfies this.
- `n < 2` guard protects against single-sample batches at epoch boundaries.
- `squeeze(-1)` converts `(B, n_tasks=1)` tensors to `(B,)` for the pair computation. Assumes single-task regression (which is the pEC50 setup).

### CLI script (`run_chemprop_relative_aux.py`)

Fork of `run_chemprop_optuna.py`. Three mechanical changes:

1. Import `from losses import RelativeDistanceMSE`
2. `build_model()` passes `criterion=RelativeDistanceMSE(aux_weight=params["aux_weight"])` to `nn.RegressionFFN(...)`
3. Optuna search space adds:
   ```python
   "aux_weight": trial.suggest_float("aux_weight", 0.01, 1.0, log=True)
   ```

All other logic (AGG_REGISTRY, train_and_predict, run_final_cv, main, DB record, submission CSV) is copied verbatim from `run_chemprop_optuna.py`. This keeps the fork comparable and reviewable.

### Optuna search space (inherited + one new)

Inherits everything from run_chemprop_optuna.py (message_hidden_dim, depth, mp_dropout, activation, aggregation, ffn_hidden_dim, ffn_num_layers, ffn_dropout, warmup_epochs, learning_rate, lr_ratio, batch_size, max_epochs, patience). Adds:

| Parameter | Distribution | Notes |
|---|---|---|
| `aux_weight` | log_uniform [0.01, 1.0] | FMGCL paper range; expect best around 0.05-0.2 |

Trials: 40 (baseline is 30; add 10 for the extra knob). TPESampler(seed=42), MedianPruner(n_startup=5, n_warmup=1).

## CV protocol (unchanged from canonical)

- Outer: UMAP 5-fold, seed=42, n_clusters=50, Morgan+Jaccard
- Inner: UMAP 3-fold, seed=123
- Test preds: 5-fold mean
- OOF coverage: 100% of train

## DB integration

- `experiment_name`: `chemprop_relative_aux_umap_default`
- `model_type`: `chemprop_aux`
- `feature_set`: `d_mpnn_relative_distance`
- `hyperparameters` JSONB: full Optuna best_params (with `aux_weight`)
- `notes`: `OOF RAE={...}, MAE={...}, umap_split, aux_weight={...}, FMGCL-inspired`
- Standard `record_experiment()` + `save_oof_predictions()` helpers (unchanged)

## Ensemble integration (Approach C -- add only)

- Append `"chemprop_relative_aux_umap_default"` to `ENSEMBLE_MODELS` in `run_ensemble.py`
- Keep existing `chemprop_optuna_umap` in the pool
- Re-run `run_ensemble.py`; caruana_bag20 picks both if they decorrelate, concentrates on one if they don't
- This PR does NOT submit to LB. LB submission is the user-confirmed next step after merge.

## Acceptance criteria (PR merge)

1. **Single-model OOF MAE <= 0.5208** -- match or beat baseline chemprop_optuna_umap. Justification: if the aux term hurts main MSE more than it helps rank, the model is weaker at the task we're optimising. Equal or better required; tiny regression (<=0.005) tolerable if Spearman improves by >=0.02.
2. **Caruana_bag20 weight > 0** on the new member.
3. **10-pool caruana_bag20 OOF MAE <= 0.4327** -- at worst no regression vs the pool state before PEFT MoLFormer dropped out.
4. ruff format + ruff check clean (ty gate is not required per project convention).

Failure handling:
- (1) fails: investigate (aux_weight might need narrower range, or the task simply doesn't benefit from this trick). Merge decision escalated to user.
- (2) fails but (1) passes: drop the new member from ENSEMBLE_MODELS immediately in the same PR; keep the trained artifact in DB for reference.
- (3) fails: document regression, discuss with user before merging.

## Testing

- No unit tests for the loss class -- the forward pass is exercised by the smoke test.
- Smoke test (manual, before pushing): `--n-trials 1 --inner-folds 2 --outer-folds 2 --max-epochs 3 --patience 2`. Completes in under 5 minutes on RTX 5080. Confirms: loss class constructs, forward pass produces finite values, DB rows land.
- CI: ruff format + ruff check. No CI workflow in this repo.

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| aux_weight too large -> main MSE diluted -> single-model MAE regresses | Medium | Optuna range [0.01, 1.0] log; best trial should settle around 0.05-0.2 |
| aux term unstable at small batch_size | Low | `n < 2` guard; batch_size Optuna lower bound stays at 32 as in baseline |
| New model correlates r > 0.95 with chemprop_optuna_umap (pool inflation) | Medium | Approach C: observe in caruana; if wt ~= 0, drop in a 1-line follow-up PR |
| Forward pass slow (B^2 pair computation in each training step) | Low | B=32-64 means 496-2016 pairs, ~100 microseconds per batch on GPU. Negligible vs D-MPNN forward cost. |
| Chemprop 2.2.3 internal reduction breaks the broadcast trick | Low | Verified via smoke test on 100-row subset; if it breaks, fall back to returning just `main + aux_weight * aux_broadcast` with `.mean()` outside the class |

## ETA

- Optuna 40 trials x 3-fold inner x ~3 min/trial = ~2 h
- Final 5-fold outer CV x ~10 min/fold = ~50 min
- Total wall-clock: **~3 h** on RTX 5080 (noticeably faster than PEFT MoLFormer's 6-8 h)

## Out of scope / future work (not this PR)

- PR 2: decide swap vs drop vs keep-both based on PR 1 OOF + LB results
- PR 3: apply `RelativeDistanceMSE` to attentivefp / gatedgcn (if PR 1 shows lift, generalises cheaply)
- PR 4+: alternative rank-aware losses (pairwise margin ranking, listNet) if FMGCL approach plateaus
- Tuning the aux_weight upper bound or adding a warmup schedule (keep aux=0 for first N epochs)
