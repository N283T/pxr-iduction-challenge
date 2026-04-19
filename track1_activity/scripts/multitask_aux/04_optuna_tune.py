#!/usr/bin/env -S pixi run python
"""Optuna sweep for chemprop multitask_desc hyperparams.

Tunes:
  * base_aux_weight : log-uniform [0.005, 0.2]
  * n_aux           : categorical [10, 15, 20, 25, 30]
  * include_boltz   : bool (whether to include boltz_tier0/tier1 aux)

Proxy objective: fold-0 val MAE on the canonical UMAP split (seed 42,
k=50, Morgan+Jaccard). Fold-0 only to keep trials fast (~6-8 min each)
while still using the official split geometry.

Trial budget: 25, TPE sampler + MedianPruner (pruned at epoch 20/60/100).

Per-trial epoch budget capped (max=120, patience=15) to fit the time
envelope. The winning config is re-run full 5-fold separately via
``run_chemprop_multitask_desc.py``.

Usage:
    pixi run python track1_activity/scripts/multitask_aux/04_optuna_tune.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from lightning import pytorch as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from data import load_test_smiles, load_train_smiles_target  # noqa: E402
from splits import umap_split_indices  # noqa: E402

# Reuse the multitask_desc machinery
from run_chemprop_multitask_desc import (  # noqa: E402
    DROP_TRAIN_CIDS,
    FAMILY_WEIGHT_MULT,
    TUNED_PARAMS,
    assert_chemprop_masks_nan,
    build_model,
    load_aux_values,
    load_compound_ids,
    make_dataloader,
    predict_pec50,
)

torch.set_float32_matmul_precision("medium")


REPORT_DIR = REPO_ROOT.joinpath("track1_activity", "reports", "multitask_aux")
TOP_AUX_CSV = REPORT_DIR.joinpath("top_aux_candidates.csv")
STUDY_PATH = REPORT_DIR.joinpath("optuna_multitask_desc.db")
TRIAL_LOG = REPORT_DIR.joinpath("optuna_multitask_desc_trials.log")


def filter_aux(top: pd.DataFrame, n_aux: int, include_boltz: bool) -> pd.DataFrame:
    """Pick the top-N aux rows, optionally dropping boltz_tier0/tier1.

    When ``include_boltz`` is False we filter out boltz_* families *then*
    take the top-N by gain, so n_aux counts only the kept families. This
    keeps the aux count stable across trials.
    """
    if not include_boltz:
        top = top[~top["family"].isin({"boltz_tier0", "boltz_tier1"})]
    return top.head(n_aux).reset_index(drop=True)


def make_pruning_callback(trial: optuna.Trial) -> pl.Callback:
    """Lightning callback that reports val_loss to Optuna at each epoch
    and raises TrialPruned if the pruner rejects the current report."""

    class OptunaPruning(pl.Callback):
        def on_validation_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            val_loss = trainer.callback_metrics.get("val_loss")
            if val_loss is None:
                return
            epoch = trainer.current_epoch
            trial.report(float(val_loss), step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned at epoch {epoch}")

    return OptunaPruning()


def build_trial_objective(
    smiles_all: list[str],
    y_main_all: np.ndarray,
    train_ids_all: list[int],
    top_full: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    params: dict,
    trial_log_path: Path,
):
    """Factory: returns a 5-fold objective closure with fold-level pruning.

    Each trial walks folds sequentially. After each fold completes we
    report running mean MAE to Optuna; MedianPruner can terminate
    underperformers after fold 1 or 2. Pruned trials cost 1-2 folds,
    promising trials run all 5.
    """

    def _train_one_fold(
        tr_idx: np.ndarray,
        va_idx: np.ndarray,
        aux_features: list[str],
        aux_families: list[str],
        base_w: float,
    ) -> float:
        Xaux = load_aux_values(aux_features, train_ids_all)
        mu = np.nanmean(Xaux, axis=0)
        sd = np.nanstd(Xaux, axis=0)
        sd[sd < 1e-6] = 1.0
        Xaux_z = ((Xaux - mu) / sd).astype(np.float32)

        targets = np.concatenate([y_main_all.reshape(-1, 1), Xaux_z], axis=1)
        n_tasks = targets.shape[1]

        weights = np.ones(n_tasks, dtype=np.float32)
        for i, fam in enumerate(aux_families, start=1):
            weights[i] = base_w * FAMILY_WEIGHT_MULT.get(fam, 1.0)
        w_t = torch.tensor(weights, dtype=torch.float32)

        tr_smi = [smiles_all[i] for i in tr_idx]
        va_smi = [smiles_all[i] for i in va_idx]
        tr_y = targets[tr_idx]
        va_y = targets[va_idx]

        train_loader = make_dataloader(tr_smi, tr_y, params["batch_size"], shuffle=True)
        val_loader = make_dataloader(va_smi, va_y, params["batch_size"], shuffle=False)

        model = build_model(params, w_t)
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

        try:
            trainer.fit(model, train_loader, val_loader)
            preds = predict_pec50(trainer, model, va_smi, params["batch_size"], n_tasks)
            mae = float(np.mean(np.abs(va_y[:, 0] - preds)))
        finally:
            del model, trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return mae

    def objective(trial: optuna.Trial) -> float:
        t0 = time.time()
        base_w = trial.suggest_float("base_aux_weight", 0.005, 0.2, log=True)
        n_aux = trial.suggest_categorical("n_aux", [10, 15, 20, 25, 30])
        include_boltz = trial.suggest_categorical("include_boltz", [True, False])

        top = filter_aux(top_full, n_aux=n_aux, include_boltz=include_boltz)
        aux_features = top["feature"].tolist()
        aux_families = top["family"].tolist()

        fold_maes: list[float] = []
        for fold_idx, (tr_idx, va_idx) in enumerate(folds):
            mae_f = _train_one_fold(tr_idx, va_idx, aux_features, aux_families, base_w)
            fold_maes.append(mae_f)
            running = float(np.mean(fold_maes))
            trial.report(running, step=fold_idx)
            if trial.should_prune():
                dt = time.time() - t0
                line = (
                    f"[TRIAL {trial.number + 1:>2d}] PRUNED@fold{fold_idx} "
                    f"base_w={base_w:.4f}  n_aux={n_aux:>2d}  "
                    f"boltz={'Y' if include_boltz else 'N'}  "
                    f"running_MAE={running:.4f}  dt={dt / 60:.1f}min"
                )
                print(line, flush=True)
                with trial_log_path.open("a") as f:
                    f.write(line + "\n")
                raise optuna.TrialPruned(
                    f"pruned at fold {fold_idx} (running MAE={running:.4f})"
                )

        cv_mae = float(np.mean(fold_maes))
        cv_std = float(np.std(fold_maes))
        dt = time.time() - t0
        try:
            best_so_far = min(trial.study.best_value, cv_mae)
        except ValueError:
            best_so_far = cv_mae

        line = (
            f"[TRIAL {trial.number + 1:>2d}] "
            f"base_w={base_w:.4f}  n_aux={n_aux:>2d}  "
            f"boltz={'Y' if include_boltz else 'N'}  "
            f"CV_MAE={cv_mae:.4f}±{cv_std:.4f}  best={best_so_far:.4f}  "
            f"dt={dt / 60:.1f}min"
        )
        print(line, flush=True)
        with trial_log_path.open("a") as f:
            f.write(line + "\n")

        return cv_mae

    return objective


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--study-name", default="multitask_desc_tune_5fold")
    args = parser.parse_args()

    print(
        f"Optuna tune (5-fold CV): n_trials={args.n_trials}, "
        f"max_epochs={args.max_epochs}, patience={args.patience}"
    )

    # Load data once
    top_full = pd.read_csv(TOP_AUX_CSV)
    train_df = load_train_smiles_target()
    train_ids = load_compound_ids("train")
    assert len(train_ids) == len(train_df)

    keep = np.array([cid not in DROP_TRAIN_CIDS for cid in train_ids], dtype=bool)
    train_df = train_df.loc[keep].reset_index(drop=True)
    train_ids = [cid for cid in train_ids if cid not in DROP_TRAIN_CIDS]
    smiles_all = train_df["smiles"].tolist()
    y_main = train_df["pec50"].to_numpy(dtype=np.float32)

    folds = umap_split_indices(smiles_all, n_splits=5, n_clusters=50, seed=42)
    for i, (tr, va) in enumerate(folds):
        print(f"  Fold {i}: train={len(tr)}, val={len(va)}")

    params = TUNED_PARAMS.copy()
    params["max_epochs"] = args.max_epochs
    params["patience"] = args.patience

    # Sanity: confirm NaN-mask still works under the largest possible n_tasks
    # (n_aux=30 + main = 31). Uses a dummy 2-weight tensor; build_model reads
    # only the size, not the actual weights.
    print("  Running NaN-mask sanity check (worst-case n_tasks=31)...")
    sanity_w = torch.ones(31, dtype=torch.float32)
    assert_chemprop_masks_nan(params, sanity_w)
    print("  Sanity check passed.")

    TRIAL_LOG.unlink(missing_ok=True)
    TRIAL_LOG.parent.mkdir(parents=True, exist_ok=True)

    storage_url = f"sqlite:///{STUDY_PATH}"
    sampler = optuna.samplers.TPESampler(seed=42)
    # Fold-level pruning: prune after fold 1 when the running mean MAE
    # exceeds the median of other trials at the same step. n_warmup_steps=0
    # because we operate on fold count (max 5), not epoch count.
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=6, n_warmup_steps=0, interval_steps=1
    )
    study = optuna.create_study(
        direction="minimize",
        storage=storage_url,
        study_name=args.study_name,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    objective = build_trial_objective(
        smiles_all=smiles_all,
        y_main_all=y_main,
        train_ids_all=train_ids,
        top_full=top_full,
        folds=folds,
        params=params,
        trial_log_path=TRIAL_LOG,
    )

    study.optimize(
        objective,
        n_trials=args.n_trials,
        gc_after_trial=True,
        show_progress_bar=False,
    )

    print("\n=== Optuna tune done ===")
    print(f"Best trial: {study.best_trial.number + 1}")
    print(f"Best val_MAE (fold {args.fold}): {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Persist the best config for the final 5-fold re-run
    best_path = REPORT_DIR.joinpath("optuna_multitask_desc_best.json")
    import json

    best_path.write_text(
        json.dumps(
            {
                "best_params": study.best_params,
                "best_value": study.best_value,
                "fold": args.fold,
                "n_trials": args.n_trials,
                "study_name": args.study_name,
            },
            indent=2,
        )
    )
    print(f"\nSaved best params -> {best_path}")

    # Top-5 trials for manual review
    trials_df = study.trials_dataframe().sort_values("value").head(10)
    print("\n=== Top 10 trials by val_MAE ===")
    cols = [
        "number",
        "value",
        "params_base_aux_weight",
        "params_n_aux",
        "params_include_boltz",
        "state",
    ]
    cols = [c for c in cols if c in trials_df.columns]
    print(trials_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
