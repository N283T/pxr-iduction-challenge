# ChemProp Relative-Distance Aux Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a rank-aware auxiliary loss (batchwise relative-distance MSE) to ChemProp D-MPNN and ship it as a new pool member, evaluating whether the FMGCL-style aux term earns additional caruana weight alongside the existing `chemprop_optuna_umap`.

**Architecture:** A new `losses.py` src module hosts `RelativeDistanceMSE(ChempropMetric)` which returns `main_mse + aux_weight * batchwise_pair_distance_mse`. A forked training script `run_chemprop_relative_aux.py` mirrors `run_chemprop_optuna.py` with three mechanical changes: new import, `criterion=` kwarg on `RegressionFFN`, and `aux_weight` Optuna hyperparameter.

**Tech Stack:** Python 3.12, chemprop 2.2.3, torch 2.10+cu13, optuna 4.8, psycopg2, lightning 2.6 (all existing deps).

**Spec:** `docs/superpowers/specs/2026-04-20-chemprop-relative-aux-design.md`

**Branch:** `feature/chemprop-relative-aux-loss` (already created)

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `track1_activity/src/losses.py` | Create | `RelativeDistanceMSE(ChempropMetric)` — one class, FMGCL-style aux loss |
| `track1_activity/scripts/run_chemprop_relative_aux.py` | Create | Fork of `run_chemprop_optuna.py` with aux_weight hyperparam + criterion injection |
| `track1_activity/scripts/run_ensemble.py` | Modify | Append `chemprop_relative_aux_umap_default` to `ENSEMBLE_MODELS` (keep existing `chemprop_optuna_umap`; add-only per Approach C) |

---

## Task 1: Loss class (`losses.py`)

**Files:**
- Create: `track1_activity/src/losses.py`

- [ ] **Step 1.1: Write the loss class**

Create `track1_activity/src/losses.py` with EXACTLY this content:

```python
"""Custom ChempropMetric subclasses for auxiliary-loss training.

Each class extends ``chemprop.nn.metrics.ChempropMetric`` and overrides
``_calc_unreduced_loss`` to return a per-sample loss tensor that the
outer reduction (weighted mean) will aggregate. Broadcast any scalar
auxiliary terms to the per-sample shape so they survive reduction.
"""

import torch
import torch.nn.functional as F
from chemprop.nn.metrics import ChempropMetric


class RelativeDistanceMSE(ChempropMetric):
    """MSE + batchwise all-pairs relative-distance auxiliary loss.

    Main:  ``MSE(pred, target)`` per-sample (shape ``(B,)``).
    Aux:   ``MSE(|pred_i - pred_j|, |target_i - target_j|)`` averaged over
           upper-triangle pairs (i < j) -- scalar.
    Total: ``main + aux_weight * aux``, with aux broadcast to ``(B,)``.

    Teaches the model to preserve relative distances between compounds,
    which correlates with rank metrics (Spearman, RAE). With
    ``aux_weight <= 0.1`` the main MSE dominates; larger values sacrifice
    absolute accuracy for ranking quality.

    Reference: Dong et al. 2025, DOI 10.1016/j.jmgm.2025.109014 (FMGCL).
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

- [ ] **Step 1.2: Smoke check the loss constructs and returns finite values**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge && pixi run python -c "
import sys
sys.path.insert(0, 'track1_activity/src')
import torch
from losses import RelativeDistanceMSE

torch.manual_seed(0)
loss_fn = RelativeDistanceMSE(aux_weight=0.1)
preds = torch.randn(8, 1, requires_grad=True)
targets = torch.randn(8, 1)
# ChempropMetric exposes .forward(preds, targets) via torchmetrics.Metric
l = loss_fn._calc_unreduced_loss(preds, targets)
print('unreduced shape:', l.shape, 'finite:', bool(torch.isfinite(l).all()))
l.mean().backward()
print('preds grad finite:', bool(torch.isfinite(preds.grad).all()))
# Single-sample guard
l2 = loss_fn._calc_unreduced_loss(preds[:1], targets[:1])
print('n=1 guard shape:', l2.shape)
"
```
Expected output: `unreduced shape: torch.Size([8, 1]) finite: True`, `preds grad finite: True`, `n=1 guard shape: torch.Size([1, 1])`. No traceback.

- [ ] **Step 1.3: Lint**

Run: `cd /home/nagaet/pxr-iduction-challenge && pixi run ruff format track1_activity/src/losses.py && pixi run ruff check track1_activity/src/losses.py`
Expected: clean (1 file already formatted or reformatted; "All checks passed!").

- [ ] **Step 1.4: Commit**

```bash
git add track1_activity/src/losses.py
git commit -m "feat(losses): RelativeDistanceMSE ChempropMetric (FMGCL aux loss)"
```

---

## Task 2: Training CLI (`run_chemprop_relative_aux.py`)

**Files:**
- Create: `track1_activity/scripts/run_chemprop_relative_aux.py`

- [ ] **Step 2.1: Write the CLI script**

Create `track1_activity/scripts/run_chemprop_relative_aux.py` with EXACTLY this content (this is a fork of `run_chemprop_optuna.py` with 3 mechanical changes: new import, `criterion=` kwarg, `aux_weight` hyperparam, plus the DB name / notes updated):

```python
"""ChemProp D-MPNN + FMGCL relative-distance aux loss, Optuna-tuned.

Forked from run_chemprop_optuna.py. Differences:
- Imports RelativeDistanceMSE from src/losses.py
- Passes criterion=RelativeDistanceMSE(aux_weight=params["aux_weight"])
  to nn.RegressionFFN
- Adds aux_weight to the Optuna search space (log-uniform 0.01..1.0)
- Records to DB with model_type='chemprop_aux', name=chemprop_relative_aux_{split}_default
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))

import argparse

import numpy as np
import optuna
import pandas as pd
import torch
from lightning import pytorch as pl

from chemprop import data as chemprop_data
from chemprop import models, nn

from data import load_train_smiles_target, load_test_smiles
from evaluate import (
    compute_metrics,
    print_metrics,
    print_fold_summary,
    record_experiment,
    save_oof_predictions,
)
from losses import RelativeDistanceMSE
from splits import umap_split_indices, scaffold_split_indices

SUBMISSION_DIR = Path(__file__).resolve().parent.parent.joinpath("submissions")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

torch.set_float32_matmul_precision("medium")


AGG_REGISTRY = {
    "mean": nn.MeanAggregation,
    "sum": nn.SumAggregation,
    "norm": nn.NormAggregation,
}


def build_model(params: dict):
    """Build ChemProp MPNN model with relative-distance aux loss."""
    mp = nn.BondMessagePassing(
        d_h=params["message_hidden_dim"],
        depth=params["depth"],
        dropout=params["mp_dropout"],
        activation=params["activation"],
    )

    agg_cls = AGG_REGISTRY[params["aggregation"]]
    agg = agg_cls()

    criterion = RelativeDistanceMSE(aux_weight=params["aux_weight"])
    ffn = nn.RegressionFFN(
        input_dim=mp.output_dim,
        hidden_dim=params["ffn_hidden_dim"],
        n_layers=params["ffn_num_layers"],
        dropout=params["ffn_dropout"],
        criterion=criterion,
    )

    model = models.MPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        batch_norm=True,
        warmup_epochs=params["warmup_epochs"],
        init_lr=params["learning_rate"],
        max_lr=params["learning_rate"] * params["lr_ratio"],
        final_lr=params["learning_rate"] * 0.1,
    )
    return model


def train_and_predict(
    params, train_smiles, train_targets, val_smiles, val_targets, test_smiles=None
):
    """Train model, return val predictions (and optionally test predictions)."""
    train_data = [
        chemprop_data.MoleculeDatapoint.from_smi(smi, [y])
        for smi, y in zip(train_smiles, train_targets)
    ]
    val_data = [
        chemprop_data.MoleculeDatapoint.from_smi(smi, [y])
        for smi, y in zip(val_smiles, val_targets)
    ]

    train_loader = chemprop_data.build_dataloader(
        chemprop_data.MoleculeDataset(train_data),
        batch_size=params["batch_size"],
        shuffle=True,
    )
    val_loader = chemprop_data.build_dataloader(
        chemprop_data.MoleculeDataset(val_data),
        batch_size=params["batch_size"],
        shuffle=False,
    )

    model = build_model(params)

    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=params["patience"], mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=params["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[early_stop],
    )

    trainer.fit(model, train_loader, val_loader)

    val_preds = trainer.predict(model, val_loader)
    val_preds = np.concatenate([p.numpy().flatten() for p in val_preds])

    test_preds = None
    if test_smiles is not None:
        test_data = [
            chemprop_data.MoleculeDatapoint.from_smi(smi) for smi in test_smiles
        ]
        test_loader = chemprop_data.build_dataloader(
            chemprop_data.MoleculeDataset(test_data),
            batch_size=params["batch_size"],
            shuffle=False,
        )
        test_preds = trainer.predict(model, test_loader)
        test_preds = np.concatenate([p.numpy().flatten() for p in test_preds])

    # Cleanup GPU memory
    del model, trainer
    torch.cuda.empty_cache()

    return val_preds, test_preds


def objective(trial, train_smiles, y_train, inner_splits):
    """Optuna objective: average RAE across inner CV folds."""
    params = {
        "message_hidden_dim": trial.suggest_categorical(
            "message_hidden_dim", [256, 512, 768, 1024]
        ),
        "depth": trial.suggest_int("depth", 3, 6),
        "mp_dropout": trial.suggest_float("mp_dropout", 0.0, 0.3, step=0.05),
        "ffn_hidden_dim": trial.suggest_categorical("ffn_hidden_dim", [256, 512, 768]),
        "ffn_num_layers": trial.suggest_int("ffn_num_layers", 1, 3),
        "ffn_dropout": trial.suggest_float("ffn_dropout", 0.0, 0.4, step=0.05),
        "activation": trial.suggest_categorical(
            "activation", ["relu", "leakyrelu", "elu"]
        ),
        "aggregation": trial.suggest_categorical(
            "aggregation", ["mean", "sum", "norm"]
        ),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True),
        "lr_ratio": trial.suggest_categorical("lr_ratio", [5, 10, 20]),
        "warmup_epochs": trial.suggest_int("warmup_epochs", 1, 5),
        "aux_weight": trial.suggest_float("aux_weight", 0.01, 1.0, log=True),
        "max_epochs": 200,
        "patience": 15,
    }

    fold_raes = []
    for fold, (train_idx, val_idx) in enumerate(inner_splits):
        tr_smi = [train_smiles[i] for i in train_idx]
        va_smi = [train_smiles[i] for i in val_idx]
        tr_y = y_train[train_idx]
        va_y = y_train[val_idx]

        val_pred, _ = train_and_predict(params, tr_smi, tr_y, va_smi, va_y)
        if np.isnan(val_pred).any():
            raise optuna.TrialPruned()
        metrics = compute_metrics(va_y, val_pred)
        fold_raes.append(metrics["RAE"])

        trial.report(float(np.mean(fold_raes)), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_raes))


def run_final_cv(
    best_params, train_smiles, y_train, test_smiles, test_df, outer_splits, split_name
):
    """Run full outer CV with best params and save results."""
    final_params = {**best_params, "max_epochs": 200, "patience": 20}

    name = f"chemprop_relative_aux_{split_name}_default"
    print(f"\n{'=' * 60}")
    print(f"  Final CV: {name}")
    print(f"  Params: {final_params}")
    print(f"{'=' * 60}")

    oof_preds = np.zeros(len(y_train))
    test_preds_all = np.zeros((len(outer_splits), len(test_smiles)))
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(outer_splits):
        print(f"\n  --- Fold {fold} ---")
        tr_smi = [train_smiles[i] for i in train_idx]
        va_smi = [train_smiles[i] for i in val_idx]
        tr_y = y_train[train_idx]
        va_y = y_train[val_idx]

        val_pred, test_pred = train_and_predict(
            final_params, tr_smi, tr_y, va_smi, va_y, test_smiles
        )

        if np.any(np.isnan(val_pred)):
            raise ValueError(f"Fold {fold}: model produced NaN predictions")

        oof_preds[val_idx] = val_pred
        test_preds_all[fold] = test_pred

        metrics = compute_metrics(va_y, val_pred)
        fold_metrics.append(metrics)
        print_metrics(metrics, label=f"Fold {fold}")

    oof_metrics = compute_metrics(y_train, oof_preds)
    print("\n  Overall OOF:")
    print_metrics(oof_metrics)
    print_fold_summary(fold_metrics)

    test_preds_avg = test_preds_all.mean(axis=0)

    submission = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": test_preds_avg,
        }
    )
    sub_path = SUBMISSION_DIR.joinpath(f"{name}.csv")
    submission.to_csv(sub_path, index=False)

    exp_id = record_experiment(
        name=name,
        description=f"ChemProp D-MPNN + FMGCL relative-distance aux loss ({split_name} split)",
        model_type="chemprop_aux",
        feature_set="d_mpnn_relative_distance",
        hyperparameters=final_params,
        fold_metrics=fold_metrics,
        submission_path=f"track1_activity/submissions/{name}.csv",
        notes=(
            f"OOF RAE={oof_metrics['RAE']:.4f}, MAE={oof_metrics['MAE']:.4f}, "
            f"{split_name}_split, aux_weight={final_params['aux_weight']:.4f}, FMGCL-inspired"
        ),
    )
    save_oof_predictions(exp_id, oof_preds)

    return oof_metrics


def main():
    parser = argparse.ArgumentParser(description="ChemProp + FMGCL aux-loss Optuna tuning")
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--split", choices=["umap", "scaffold"], default="umap")
    parser.add_argument("--n-clusters", type=int, default=50)
    args = parser.parse_args()

    pl.seed_everything(42)

    print("Loading data...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    train_smiles = train_df["smiles"].tolist()
    y_train = train_df["pec50"].values
    test_smiles = test_df["smiles"].tolist()

    print(f"Train: {len(train_smiles)}, Test: {len(test_smiles)}")
    print(
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
    print(f"Split: {args.split}, Outer folds: {args.outer_folds}")

    if args.split == "umap":
        split_fn = lambda n, s: umap_split_indices(
            train_smiles, n_splits=n, n_clusters=args.n_clusters, seed=s
        )
    else:
        split_fn = lambda n, s: scaffold_split_indices(train_smiles, n_splits=n, seed=s)

    outer_splits = split_fn(args.outer_folds, 42)
    inner_splits = split_fn(args.inner_folds, 123)

    print(f"\n{'=' * 60}")
    print(f"  Optuna tuning: {args.n_trials} trials, {args.inner_folds}-fold inner CV")
    print(f"{'=' * 60}")

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: objective(trial, train_smiles, y_train, inner_splits),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    n_complete = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    n_pruned = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    )
    n_failed = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    )
    print(f"\n  Trials: {n_complete} complete, {n_pruned} pruned, {n_failed} FAILED")
    if n_complete == 0:
        raise RuntimeError("All Optuna trials failed. Cannot proceed.")
    print(f"  Best trial: RAE={study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    oof_metrics = run_final_cv(
        study.best_params,
        train_smiles,
        y_train,
        test_smiles,
        test_df,
        outer_splits,
        args.split,
    )

    print(f"\n  Final OOF RAE: {oof_metrics['RAE']:.4f}, MAE: {oof_metrics['MAE']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2.2: Lint**

Run: `cd /home/nagaet/pxr-iduction-challenge && pixi run ruff format track1_activity/scripts/run_chemprop_relative_aux.py && pixi run ruff check track1_activity/scripts/run_chemprop_relative_aux.py`
Expected: clean.

- [ ] **Step 2.3: Commit**

```bash
git add track1_activity/scripts/run_chemprop_relative_aux.py
git commit -m "feat(chemprop): aux-loss training script (fork of run_chemprop_optuna.py)"
```

---

## Task 3: Smoke test (end-to-end with DB)

**Files:** none (writes/deletes DB rows in the smoke path)

- [ ] **Step 3.1: Ensure DB is running**

Run: `cd /home/nagaet/pxr-iduction-challenge && pixi run db-start 2>&1 | tail -3`
Expected: server starts OR prints "another postmaster running" (both fine).

- [ ] **Step 3.2: Run smoke end-to-end**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge && mkdir -p logs && LOG="logs/smoke_chemprop_aux_$(date +%Y%m%d_%H%M).log" && time pixi run python track1_activity/scripts/run_chemprop_relative_aux.py --n-trials 1 --inner-folds 2 --outer-folds 2 2>&1 | tee "$LOG" | tail -30
```
Note: the script's `max_epochs` defaults to 200 inside `objective()` / `run_final_cv` and early-stops on patience=15/20. With UMAP 2-fold + tiny batch training on a 4140-row set this typically takes 15-25 min for the full smoke. If you want the smoke <5 min, temporarily edit `max_epochs=200 -> 5` and `patience=15/20 -> 3` in the two `final_params` / `params` dicts, re-run, then revert before Step 3.4.

Expected tail: `Final OOF RAE: <X.XXXX>, MAE: <X.XXXX>` (values are trash on smoke; we only care no crash). `Recorded experiment 'chemprop_relative_aux_umap_default' (id=<N>)` in the output.

- [ ] **Step 3.3: Verify DB rows landed**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge && pixi run python <<'PY'
import psycopg2
conn = psycopg2.connect(host='/tmp', port=5433, dbname='pxr_challenge')
cur = conn.cursor()
cur.execute("SELECT id, name, model_type FROM experiments WHERE name = 'chemprop_relative_aux_umap_default' ORDER BY id DESC LIMIT 1")
row = cur.fetchone()
print('experiment:', row)
if row:
    cur.execute("SELECT COUNT(*) FROM experiment_oof_predictions WHERE experiment_id = %s", (row[0],))
    print('oof rows:', cur.fetchone()[0])
PY
```
Expected: one row with `model_type='chemprop_aux'`, oof rows = 4140.

- [ ] **Step 3.4: Delete the smoke experiment (its 2-epoch OOF is garbage)**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge && pixi run python <<'PY'
import psycopg2
conn = psycopg2.connect(host='/tmp', port=5433, dbname='pxr_challenge')
cur = conn.cursor()
cur.execute("SELECT id FROM experiments WHERE name = 'chemprop_relative_aux_umap_default' ORDER BY id DESC LIMIT 1")
row = cur.fetchone()
if row:
    eid = row[0]
    cur.execute("DELETE FROM experiment_oof_predictions WHERE experiment_id=%s", (eid,))
    cur.execute("DELETE FROM experiment_cv_results WHERE experiment_id=%s", (eid,))
    cur.execute("DELETE FROM experiments WHERE id=%s", (eid,))
    print(f"Deleted smoke experiment id={eid}")
conn.commit()
PY
```
Expected: prints `Deleted smoke experiment id=<N>`.

- [ ] **Step 3.5: If you temporarily lowered `max_epochs` / `patience` for smoke speed, revert them now**

Open `track1_activity/scripts/run_chemprop_relative_aux.py` and confirm `max_epochs=200` in both `objective()` params dict (line near the bottom of the suggest block) and `run_final_cv()` `final_params` merge. Confirm `patience=15` in objective and `patience=20` in run_final_cv. Use `git diff track1_activity/scripts/run_chemprop_relative_aux.py` — expect no uncommitted changes if you never lowered them.

---

## Task 4: Push + open draft PR

**Files:** none

- [ ] **Step 4.1: Push branch**

Run: `cd /home/nagaet/pxr-iduction-challenge && git push -u origin feature/chemprop-relative-aux-loss`
Expected: push succeeds.

- [ ] **Step 4.2: Open draft PR**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge && gh pr create --draft --title "feat(chemprop): relative-distance aux loss pool member" --body "$(cat <<'EOF'
## Summary
- New pool member: ChemProp D-MPNN + FMGCL-style batchwise relative-distance aux loss
- `track1_activity/src/losses.py`: `RelativeDistanceMSE(ChempropMetric)` — reusable for other chemprop-based scripts
- `track1_activity/scripts/run_chemprop_relative_aux.py`: fork of `run_chemprop_optuna.py` with `aux_weight` Optuna hyperparam

## Approach C (from spec)
Add as a NEW pool member alongside existing `chemprop_optuna_umap`. If caruana decorrelates them, both stay. If not, a follow-up PR drops the weaker one. This avoids the MoLFormer pitfall of replacing a working member before LB validation.

## Spec / Plan
- Spec: `docs/superpowers/specs/2026-04-20-chemprop-relative-aux-design.md`
- Plan: `docs/superpowers/plans/2026-04-20-chemprop-relative-aux.md`

## Acceptance (post-merge measurement)
- [ ] Full Optuna 40-trial + 5-fold final CV (~3h on RTX 5080) — pending USER-CONFIRMED launch
- [ ] Single-model OOF MAE <= 0.5208 (chemprop_optuna baseline) or tolerable regression <=0.005 with Spearman +>=0.02
- [ ] caruana_bag20 assigns weight > 0 to the new member
- [ ] 10-pool caruana_bag20 OOF MAE <= 0.4327 (no regression vs pre-PEFT-MoLFormer state)
- [x] Smoke test: runs end-to-end, DB rows land
- [x] ruff format + ruff check clean
EOF
)"
```
Expected: prints PR URL. Capture the number.

---

## Task 5: Full Optuna + 5-fold CV (USER-CONFIRMED)

WARNING: USER-CONFIRMED ACTION. Per CLAUDE.md "Never run benchmarks, long-running computations, or destructive operations without explicit user permission" — stop here and ask the user to confirm before launching.

**Files:** none (writes DB + submission CSV)

- [ ] **Step 5.1: Confirm with the user**

Ask: "Ready to launch full chemprop aux-loss Optuna: 40 trials x 3 inner folds then 5-fold final CV, ETA ~3h on RTX 5080. Run now?"

Wait for explicit confirmation.

- [ ] **Step 5.2: Launch in tmux (resume-safe, detached)**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge && pixi run db-start 2>&1 | tail -2
LOG="logs/chemprop_aux_$(date +%Y%m%d_%H%M).log"
echo "Log: $LOG"
tmux new -s chemprop_aux -d "pixi run python track1_activity/scripts/run_chemprop_relative_aux.py 2>&1 | tee $LOG"
sleep 3
tmux ls
```
Expected: session `chemprop_aux` listed.

- [ ] **Step 5.3: Hand off monitoring to user**

Tell the user: "Launched in tmux session `chemprop_aux`. Progress check: `tmux capture-pane -t chemprop_aux -p | tail -20` or `tail -f <log>`. Completion marker: `Final OOF RAE: ...` in log. ETA ~3h. Ping when done."

- [ ] **Step 5.4: After completion, query DB for the experiment**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge && pixi run python <<'PY'
import psycopg2
conn = psycopg2.connect(host='/tmp', port=5433, dbname='pxr_challenge')
cur = conn.cursor()
cur.execute("""
SELECT id, name, notes,
  (SELECT AVG(rae) FROM experiment_cv_results WHERE experiment_id=experiments.id) AS rae,
  (SELECT AVG(mae) FROM experiment_cv_results WHERE experiment_id=experiments.id) AS mae,
  (SELECT COUNT(*) FROM experiment_oof_predictions WHERE experiment_id=experiments.id) AS n_oof
FROM experiments
WHERE name = 'chemprop_relative_aux_umap_default' ORDER BY id DESC LIMIT 1
""")
print(cur.fetchone())
PY
```
Expected: a row with `n_oof = 4140` and MAE/RAE values.

- [ ] **Step 5.5: Acceptance check 1 — single-model MAE**

From the row above:
- `MAE <= 0.5208`: PASS (matches baseline chemprop_optuna).
- `MAE in (0.5208, 0.5258]` AND `RAE improved by >=0.02 OR Spearman +>=0.02`: PASS with rank-metric compensation.
- Otherwise: FAIL. Report to user; do not proceed to Task 6.

To check Spearman:
```bash
cd /home/nagaet/pxr-iduction-challenge && pixi run python <<'PY'
import psycopg2
conn = psycopg2.connect(host='/tmp', port=5433, dbname='pxr_challenge')
cur = conn.cursor()
cur.execute("SELECT AVG(spearman_r) FROM experiment_cv_results r JOIN experiments e ON e.id=r.experiment_id WHERE e.name IN ('chemprop_optuna_umap', 'chemprop_relative_aux_umap_default') GROUP BY e.name ORDER BY e.name")
for row in cur.fetchall(): print(row)
PY
```

---

## Task 6: Ensemble integration

**Files:**
- Modify: `track1_activity/scripts/run_ensemble.py:80-135` (append to `ENSEMBLE_MODELS`)

- [ ] **Step 6.1: Append the new model name to ENSEMBLE_MODELS**

Open `track1_activity/scripts/run_ensemble.py` and edit the `ENSEMBLE_MODELS` tuple. Find the closing `)` of the tuple after the `# 2026-04-20 PM drop: peft_molformer_xl_lora_r32a64_umap_default ...` comment block, and add a new entry before the close-paren:

```python
    # --- ChemProp + FMGCL aux loss (1) ---
    # ChemProp D-MPNN with batchwise relative-distance auxiliary loss on top
    # of MSE (Dong et al. 2025, FMGCL). Added alongside chemprop_optuna_umap
    # per Approach C (decorrelation check). PR <PR_NUMBER>.
    "chemprop_relative_aux_umap_default",
)
```

Replace `<PR_NUMBER>` with the actual PR number from Task 4.

- [ ] **Step 6.2: Re-run the ensemble**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge && LOG="logs/ens_chemprop_aux_$(date +%Y%m%d_%H%M).log" && pixi run python track1_activity/scripts/run_ensemble.py 2>&1 | tee "$LOG" | tail -60
```
Expected: prints per-model OOF RAEs, strategy comparison. Capture: the caruana_bag20 weight on `chemprop_relative_aux_umap_default` and the overall caruana_bag20 OOF MAE.

- [ ] **Step 6.3: Acceptance check 2 & 3 — caruana weight + pool MAE**

From the ensemble output, verify:
- `chemprop_relative_aux_umap_default` weight > 0: PASS.
- `ens_caruana_bag20 OOF MAE <= 0.4327`: PASS (matches pre-PEFT-MoLFormer baseline).

If weight = 0: the new member is OOF-redundant. Still merge (experiment record is valuable) but document the finding. Decide in follow-up whether to drop.

If MAE > 0.4327: regression. Report to user.

- [ ] **Step 6.4: Commit the ENSEMBLE_MODELS change**

```bash
cd /home/nagaet/pxr-iduction-challenge
git add track1_activity/scripts/run_ensemble.py
git commit -m "ens: add chemprop_relative_aux_umap_default (caruana wt=<X.XX>)"
```
Replace `<X.XX>` with the actual caruana weight.

---

## Task 7: Finalize PR + merge approval

**Files:** none

- [ ] **Step 7.1: Push final commits**

Run: `cd /home/nagaet/pxr-iduction-challenge && git push`

- [ ] **Step 7.2: Update PR body with actual results**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge && gh pr edit --body "$(cat <<'EOF'
## Summary
- New pool member: ChemProp D-MPNN + FMGCL-style batchwise relative-distance aux loss
- `track1_activity/src/losses.py`: `RelativeDistanceMSE(ChempropMetric)` — reusable for other chemprop-based scripts
- `track1_activity/scripts/run_chemprop_relative_aux.py`: fork of `run_chemprop_optuna.py` with `aux_weight` Optuna hyperparam

## Results
- Single-model OOF: RAE <X.XXXX>, MAE <X.XXXX>, Spearman <X.XXXX>
  (baseline chemprop_optuna_umap: RAE 0.5785, MAE 0.5208)
- New member caruana weight: <X.XX>
- 10-pool caruana_bag20 OOF MAE: <X.XXXX> (was 0.4327 pre-PEFT-MoLFormer-drop, now <...>)

## Spec / Plan
- Spec: `docs/superpowers/specs/2026-04-20-chemprop-relative-aux-design.md`
- Plan: `docs/superpowers/plans/2026-04-20-chemprop-relative-aux.md`

## Test plan
- [x] Smoke test passed (no NaN, DB rows landed)
- [x] Full Optuna 40-trial + 5-fold final CV
- [x] Acceptance 1: single-model MAE
- [x] Acceptance 2: caruana weight > 0
- [x] Acceptance 3: pool MAE <= 0.4327
- [x] ruff clean
EOF
)"
```
Replace placeholders with actual values.

- [ ] **Step 7.3: Mark PR ready for review**

Run: `cd /home/nagaet/pxr-iduction-challenge && gh pr ready`

- [ ] **Step 7.4: Ask user for merge approval**

Tell the user the PR is ready with the results summary; ask "Shall I merge?"

Wait for explicit "yes" / "merge" / "OK" before running `gh pr merge`.

- [ ] **Step 7.5: After approval, merge and clean up**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge
gh pr merge --squash --delete-branch
git checkout main && git pull
git remote prune origin
git branch -a
```

---

## Self-Review Checklist

1. **Spec coverage:**
   - `RelativeDistanceMSE` loss class → Task 1 ✓
   - Fork of run_chemprop_optuna.py → Task 2 ✓
   - `aux_weight` Optuna hyperparam (log [0.01, 1.0]) → Task 2 objective() ✓
   - `criterion=` injection in RegressionFFN → Task 2 build_model() ✓
   - DB name / model_type / feature_set → Task 2 run_final_cv() ✓
   - TPESampler seeded (42) → Task 2 main() ✓
   - Smoke test → Task 3 ✓
   - Acceptance 1 (single MAE <=0.5208) → Task 5.5 ✓
   - Acceptance 2 (caruana wt >0) → Task 6.3 ✓
   - Acceptance 3 (pool MAE <=0.4327) → Task 6.3 ✓
   - Approach C (add, not replace) → Task 6.1 ✓
   - LB submission deferred → noted in Task 7 (no submit step) ✓
   - NaN guard in objective() → Task 2 `if np.isnan(val_pred).any(): raise optuna.TrialPruned()` ✓

2. **Placeholder scan:**
   - `<PR_NUMBER>` in Task 6.1 and `<X.XX>` / `<X.XXXX>` in Tasks 6.4 / 7.2 are runtime values the implementer substitutes from actual results. Each is accompanied by explicit "Replace ... with the actual ..." instruction. These are NOT plan placeholders. ✓
   - No "TBD" / "TODO" / "implement later" / vague handoffs. ✓

3. **Type consistency:**
   - `RelativeDistanceMSE(aux_weight: float = 0.1, task_weights=1.0)` in Task 1 matches constructor call in Task 2 `RelativeDistanceMSE(aux_weight=params["aux_weight"])` ✓
   - `nn.RegressionFFN(..., criterion=criterion)` — verified in pixi env (`pixi run python -c "import inspect; from chemprop import nn; print(inspect.signature(nn.RegressionFFN.__init__))"`) ✓
   - `record_experiment(name=..., description=..., model_type=..., feature_set=..., hyperparameters=..., fold_metrics=..., submission_path=..., notes=...)` call in Task 2 matches existing signature in `evaluate.py:47` ✓
   - `save_oof_predictions(exp_id, oof_preds)` 2-arg form matches existing signature ✓
   - `umap_split_indices(smiles_list, n_splits, n_clusters, seed)` call matches existing signature ✓

No issues found.
