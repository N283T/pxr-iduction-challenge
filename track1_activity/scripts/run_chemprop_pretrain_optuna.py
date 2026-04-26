#!/usr/bin/env -S pixi run python
"""Optuna search for chemprop log2fc pretrain hyperparameters.

Per Codex review (2026-04-26): the existing pretrain uses single-task
pec50 Optuna's DEFAULT_PARAMS, which is the wrong target. This script
re-tunes chemprop pretrain hparams against the actual downstream metric
that drives rank-1: TabPFN OOF MAE on cheme_2d_full_boltz_log2fc_pred
(2103d feature, UMAP 5-fold CV).

Per-trial pipeline:
  1. Pretrain chemprop on 13136 compounds (2 heads: 8.25uM + 33uM)
  2. Predict log2fc on train+test compounds (4140 + 513 = 4653)
  3. Build feature: chemeleon (300d) + 2d_full_boltz (~1801d) + log2fc_pred (2d)
  4. TabPFN UMAP 5-fold CV (canonical PXR split, seed=42, k=50)
  5. Return mean OOF MAE

Search space focuses on encoder + head + multitask weighting
(Codex: w_33 is highest-leverage knob since target is multitask MSE).

Usage:
    # Smoke test: 1 trial with current production params
    pixi run python track1_activity/scripts/run_chemprop_pretrain_optuna.py \
        --trials 1 --baseline-only

    # Full search
    pixi run python track1_activity/scripts/run_chemprop_pretrain_optuna.py \
        --trials 16 --study-name log2fc_optuna_v1
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import psycopg2
import torch
from lightning import pytorch as pl
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from chemprop import data as chemprop_data  # noqa: E402

import run_chemprop_pretrain as pretrain_mod  # noqa: E402
from data import (  # noqa: E402
    DB_PARAMS,
    load_test_smiles,
    load_train_smiles_target,
)
from run_chemprop_predict_log2fc import build_pretrain_model as build_predict_model  # noqa: E402
from run_train import load_features  # noqa: E402
from splits import umap_split_indices  # noqa: E402

torch.set_float32_matmul_precision("medium")

CKPT_BASE = REPO_ROOT.joinpath(
    "track1_activity", "checkpoints", "chemprop_pretrain_optuna"
)
DATA_BASE = REPO_ROOT.joinpath("data", "optuna_chemprop_pretrain")
DATA_BASE.mkdir(parents=True, exist_ok=True)
CKPT_BASE.mkdir(parents=True, exist_ok=True)

# Standard parquet path that run_train.py expects for `2d_full_boltz_log2fc_pred`.
STANDARD_LOG2FC_PARQUET = REPO_ROOT.joinpath(
    "data", "chemprop_pretrain_log2fc_predictions.parquet"
)

CV_FOLDS = 5
UMAP_SEED = 42
UMAP_K = 50


def sample_params(trial: optuna.Trial) -> dict:
    """Optuna search space.

    Codex priorities:
      - w_33 is highest-leverage (current 0.5 hardcoded, single-task lineage)
      - encoder hparams (dim/depth/dropout/agg/lr/batch) all need re-tune
      - ffn head is secondary
    """
    return {
        # Encoder architecture
        "message_hidden_dim": trial.suggest_categorical(
            "message_hidden_dim", [128, 192, 256, 384, 512]
        ),
        "depth": trial.suggest_categorical("depth", [3, 4, 5, 6]),
        "mp_dropout": trial.suggest_float("mp_dropout", 0.0, 0.4),
        "activation": "relu",  # fixed
        "aggregation": trial.suggest_categorical(
            "aggregation", ["mean", "sum", "norm"]
        ),
        # FFN head
        "ffn_hidden_dim": trial.suggest_categorical("ffn_hidden_dim", [128, 256, 384]),
        "ffn_num_layers": trial.suggest_int("ffn_num_layers", 1, 2),
        "ffn_dropout": trial.suggest_float("ffn_dropout", 0.0, 0.3),
        # Optimizer
        "warmup_epochs": 3,  # fixed (Optuna over warmup is wasted in budget)
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "lr_ratio": trial.suggest_categorical("lr_ratio", [3, 5, 10]),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        # Multitask weighting (KEY per Codex)
        "w_33": trial.suggest_float("w_33", 0.1, 1.0),
        # Training schedule (shorter than production for speed)
        "max_epochs": 100,
        "patience": 10,
    }


def baseline_params() -> dict:
    """Production DEFAULT_PARAMS (current rank-1 driver) for baseline check."""
    return {
        **pretrain_mod.DEFAULT_PARAMS,
        "w_33": 0.5,
        "max_epochs": 100,  # match Optuna trials' budget
        "patience": 10,
    }


def pretrain_one(params: dict, seed: int, ckpt_dir: Path) -> Path:
    """Run pretrain with given params. Returns path to pretrain.pt."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    smiles, targets, _ = pretrain_mod.load_pretrain_data()
    n_total = len(smiles)

    # Z-score
    means = np.zeros(2, dtype=np.float32)
    stds = np.ones(2, dtype=np.float32)
    for i in range(2):
        valid = np.isfinite(targets[:, i])
        means[i] = float(np.mean(targets[valid, i]))
        stds[i] = float(np.std(targets[valid, i]))
        if stds[i] < 1e-6:
            stds[i] = 1.0
    targets_z = (targets - means) / stds

    rng = np.random.default_rng(seed)
    idx = np.arange(n_total)
    rng.shuffle(idx)
    n_val = int(n_total * 0.1)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    tr_smi = [smiles[i] for i in tr_idx]
    va_smi = [smiles[i] for i in val_idx]
    tr_y = targets_z[tr_idx]
    va_y = targets_z[val_idx]

    task_weights_np = np.asarray([1.0, params["w_33"]], dtype=np.float32)
    task_weights = torch.tensor(task_weights_np, dtype=torch.float32)
    model = pretrain_mod.build_pretrain_model(params, task_weights)

    train_loader = pretrain_mod.make_dataloader(
        tr_smi, tr_y, params["batch_size"], shuffle=True
    )
    val_loader = pretrain_mod.make_dataloader(
        va_smi, va_y, params["batch_size"], shuffle=False
    )

    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=params["patience"], mode="min"
    )
    best_ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="pretrain_best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    trainer = pl.Trainer(
        max_epochs=params["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[early_stop, best_ckpt],
    )
    trainer.fit(model, train_loader, val_loader)

    state_path = ckpt_dir.joinpath("pretrain.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "params": params,
            "target_means": means.tolist(),
            "target_stds": stds.tolist(),
            "task_weights": task_weights_np.tolist(),
            "n_train": len(tr_idx),
            "n_val": len(val_idx),
            "final_val_loss": float(trainer.callback_metrics.get("val_loss", -1)),
        },
        state_path,
    )
    return state_path


def predict_log2fc(ckpt_path: Path, out_parquet: Path) -> None:
    """Forward chemprop pretrain on train+test SMILES, save log2fc parquet."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    params = ckpt["params"]
    means = np.asarray(ckpt["target_means"], dtype=np.float32)
    stds = np.asarray(ckpt["target_stds"], dtype=np.float32)

    # Load train+test compound IDs and SMILES
    sql = """
    SELECT c.id AS compound_id, c.std_smiles AS smiles
    FROM compounds c
    INNER JOIN (
        SELECT compound_id FROM train_activity
        UNION
        SELECT compound_id FROM test_activity
    ) ts ON ts.compound_id = c.id
    WHERE c.std_smiles IS NOT NULL
    ORDER BY c.id
    """
    with psycopg2.connect(**DB_PARAMS) as conn:
        df = pd.read_sql(sql, conn)
    n = len(df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_predict_model(params).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    pts = [
        chemprop_data.MoleculeDatapoint.from_smi(smi, np.full(2, 0.0, dtype=np.float32))
        for smi in df["smiles"]
    ]
    dataset = chemprop_data.MoleculeDataset(pts)
    loader = chemprop_data.build_dataloader(dataset, batch_size=256, shuffle=False)

    preds_z: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            bmg = batch.bmg
            bmg.to(device)
            V_d = batch.V_d.to(device) if batch.V_d is not None else None
            X_d = batch.X_d.to(device) if batch.X_d is not None else None
            preds = model(bmg, V_d, X_d).detach().cpu().numpy()
            preds_z.append(preds)

    preds_z_arr = np.concatenate(preds_z, axis=0)
    assert preds_z_arr.shape == (n, 2)
    preds_raw = preds_z_arr * stds + means

    out = pd.DataFrame(
        {
            "compound_id": df["compound_id"].values,
            "log2fc_8p25_pred": preds_raw[:, 0],
            "log2fc_33_pred": preds_raw[:, 1],
        }
    ).set_index("compound_id")
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet)


@contextlib.contextmanager
def swap_log2fc_parquet(trial_parquet: Path):
    """Temporarily replace the standard log2fc parquet with the trial's.

    run_train.load_features("cheme_2d_full_boltz_log2fc_pred", ...) reads
    from a hardcoded path. We swap our trial's predictions in for the
    duration of evaluation, then restore.
    """
    backup = STANDARD_LOG2FC_PARQUET.with_suffix(".parquet.optuna_bak")
    had_original = STANDARD_LOG2FC_PARQUET.exists()
    if had_original:
        if backup.exists():
            backup.unlink()
        STANDARD_LOG2FC_PARQUET.rename(backup)
    shutil.copy(trial_parquet, STANDARD_LOG2FC_PARQUET)
    try:
        yield
    finally:
        STANDARD_LOG2FC_PARQUET.unlink(missing_ok=True)
        if had_original:
            backup.rename(STANDARD_LOG2FC_PARQUET)


def evaluate_oof_mae(
    trial_parquet: Path, train_df, test_df, y_train
) -> tuple[float, float]:
    """TabPFN UMAP 5-fold CV → mean OOF MAE + Spearman.

    Uses cheme_2d_full_boltz_log2fc_pred feature (chemeleon 300d +
    2d_full_boltz 1817d + log2fc_pred 2d).
    """
    from scipy.stats import spearmanr
    from tabpfn import TabPFNRegressor

    with swap_log2fc_parquet(trial_parquet):
        X_train, _ = load_features("cheme_2d_full_boltz_log2fc_pred", train_df, test_df)

    # UMAP 5-fold CV
    smiles_train = train_df["smiles"].tolist()
    folds = umap_split_indices(
        smiles_train,
        n_splits=CV_FOLDS,
        n_clusters=UMAP_K,
        seed=UMAP_SEED,
    )

    oof_pred = np.zeros(len(y_train), dtype=np.float32)
    oof_mask = np.zeros(len(y_train), dtype=bool)

    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        X_tr = X_train[tr_idx]
        X_va = X_train[va_idx]
        y_tr = y_train[tr_idx]

        model = TabPFNRegressor(device="cuda" if torch.cuda.is_available() else "cpu")
        model.fit(X_tr, y_tr)
        pred_va = model.predict(X_va)
        oof_pred[va_idx] = pred_va
        oof_mask[va_idx] = True

    if not oof_mask.all():
        # Some folds may not cover all indices — defensive
        unc = (~oof_mask).sum()
        print(f"  WARN: {unc} indices uncovered by folds (using 0.0)")

    mae = float(np.mean(np.abs(oof_pred - y_train)))
    sp = float(spearmanr(oof_pred, y_train).correlation)
    return mae, sp


def make_objective(train_df, test_df, y_train):
    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial)
        trial_id = trial.number
        ckpt_dir = CKPT_BASE.joinpath(f"trial_{trial_id:03d}")
        log2fc_parquet = DATA_BASE.joinpath(f"trial_{trial_id:03d}_log2fc.parquet")

        t0 = time.time()
        try:
            print(f"\n[Trial {trial_id}] params: {params}")
            pretrain_one(params, seed=42, ckpt_dir=ckpt_dir)
            t_pretrain = time.time() - t0
            print(f"[Trial {trial_id}] pretrain done in {t_pretrain:.0f}s")

            t1 = time.time()
            ckpt_path = ckpt_dir.joinpath("pretrain.pt")
            predict_log2fc(ckpt_path, log2fc_parquet)
            t_predict = time.time() - t1
            print(f"[Trial {trial_id}] predict_log2fc done in {t_predict:.0f}s")

            t2 = time.time()
            mae, sp = evaluate_oof_mae(log2fc_parquet, train_df, test_df, y_train)
            t_eval = time.time() - t2
            print(
                f"[Trial {trial_id}] eval done in {t_eval:.0f}s. "
                f"OOF MAE={mae:.4f}  Spearman={sp:.4f}  "
                f"(total {time.time() - t0:.0f}s)"
            )

            trial.set_user_attr("oof_spearman", sp)
            trial.set_user_attr("pretrain_seconds", t_pretrain)
            trial.set_user_attr("predict_seconds", t_predict)
            trial.set_user_attr("eval_seconds", t_eval)
            return mae
        except Exception as exc:
            print(f"[Trial {trial_id}] FAILED: {exc}")
            raise

    return objective


def run_baseline(train_df, test_df, y_train) -> tuple[float, float]:
    """Run one trial with current production DEFAULT_PARAMS (no Optuna)."""
    params = baseline_params()
    print("\n=== Baseline (production DEFAULT_PARAMS) ===")
    print(f"  params: {params}")

    ckpt_dir = CKPT_BASE.joinpath("baseline")
    log2fc_parquet = DATA_BASE.joinpath("baseline_log2fc.parquet")

    t0 = time.time()
    pretrain_one(params, seed=42, ckpt_dir=ckpt_dir)
    print(f"  pretrain done in {time.time() - t0:.0f}s")

    ckpt_path = ckpt_dir.joinpath("pretrain.pt")
    predict_log2fc(ckpt_path, log2fc_parquet)
    print("  predict_log2fc done")

    mae, sp = evaluate_oof_mae(log2fc_parquet, train_df, test_df, y_train)
    print(f"\n  Baseline OOF MAE={mae:.4f}  Spearman={sp:.4f}")
    print("  Reference: id=31 single-seed default OOF MAE 0.4068 (seed5ens 0.4056)")
    return mae, sp


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna for chemprop log2fc pretrain")
    parser.add_argument(
        "--trials", type=int, default=12, help="Number of Optuna trials (default 12)"
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="log2fc_optuna_v1",
        help="Optuna study name (used in SQLite storage)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (default sqlite path under data/)",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only baseline check (no Optuna search)",
    )
    args = parser.parse_args()

    if args.storage is None:
        storage_path = DATA_BASE.joinpath("optuna.db")
        args.storage = f"sqlite:///{storage_path}"

    print("Loading train + test")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y_train = train_df["pec50"].to_numpy(dtype=np.float32)
    print(f"  train n={len(train_df)}  test n={len(test_df)}")

    if args.baseline_only:
        run_baseline(train_df, test_df, y_train)
        return

    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=4, n_warmup_steps=0)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )
    print(f"\nOptuna study: {args.study_name} @ {args.storage}")
    print(f"  existing trials: {len(study.trials)}")

    objective = make_objective(train_df, test_df, y_train)
    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)

    print("\n=== Search done ===")
    print(f"  best OOF MAE: {study.best_value:.4f}")
    print(f"  best params:  {json.dumps(study.best_params, indent=2)}")
    print(f"  best trial number: {study.best_trial.number}")
    print(f"  best Spearman: {study.best_trial.user_attrs.get('oof_spearman', 'N/A')}")


if __name__ == "__main__":
    main()
