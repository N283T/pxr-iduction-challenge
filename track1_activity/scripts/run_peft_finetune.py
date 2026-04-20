"""PEFT fine-tuning CLI for PXR pEC50 regression.

Trains a Hugging Face backbone wrapped with peft (LoRA by default) on
the train_activity table, runs Optuna hyperparameter search on inner
CV folds, then produces 5-fold OOF predictions plus a test submission
CSV. Records everything in the experiments / experiment_oof_predictions
DB tables so the result feeds straight into ``run_ensemble.py``.

Usage:
    pixi run python track1_activity/scripts/run_peft_finetune.py \\
        --backbone molformer_xl --peft-method lora \\
        --n-trials 20 --inner-folds 3 --outer-folds 5 --split umap

Smoke test:
    ... --n-trials 1 --inner-folds 2 --outer-folds 2 \\
        --max-epochs-final 2 --patience-final 2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))

import numpy as np  # type: ignore[import-untyped]
import optuna  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]
import torch  # type: ignore[import-untyped]

from data import load_train_smiles_target, load_test_smiles  # type: ignore[unresolved-import]
from evaluate import (  # type: ignore[unresolved-import]
    compute_metrics,
    print_fold_summary,
    print_metrics,
    record_experiment,
    save_oof_predictions,
)
from peft_trainer import get_tokenizer, train_one_fold  # type: ignore[unresolved-import]
from splits import scaffold_split_indices, umap_split_indices  # type: ignore[unresolved-import]

SUBMISSION_DIR = Path(__file__).resolve().parent.parent.joinpath("submissions")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)


def suggest_params(trial: optuna.Trial, max_epochs: int, patience: int) -> dict:
    """Optuna search space for LoRA fine-tuning."""
    rank = trial.suggest_categorical("lora_rank", [4, 8, 16, 32])
    alpha_mult = trial.suggest_categorical("lora_alpha_mult", [1, 2])
    return {
        "lora_rank": rank,
        "lora_alpha": rank * alpha_mult,
        "lora_dropout": trial.suggest_float("lora_dropout", 0.0, 0.2),
        "lora_target": trial.suggest_categorical("lora_target", ["qv", "qkvo"]),
        "head_hidden_dim": trial.suggest_categorical(
            "head_hidden_dim", [128, 256, 512]
        ),
        "head_dropout": trial.suggest_float("head_dropout", 0.1, 0.4),
        "backbone_lr": trial.suggest_float("backbone_lr", 1e-5, 5e-4, log=True),
        "head_lr": trial.suggest_float("head_lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "max_epochs": max_epochs,
        "patience": patience,
    }


def objective(
    trial: optuna.Trial,
    backbone: str,
    peft_method: str,
    tokenizer,
    train_smiles: list[str],
    y_train: np.ndarray,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    max_epochs: int,
    patience: int,
) -> float:
    params = suggest_params(trial, max_epochs, patience)
    fold_raes = []
    for fold, (tr_idx, va_idx) in enumerate(inner_splits):
        tr_smi = [train_smiles[i] for i in tr_idx]
        va_smi = [train_smiles[i] for i in va_idx]
        tr_y = y_train[tr_idx]
        va_y = y_train[va_idx]
        try:
            val_pred, _ = train_one_fold(
                params,
                backbone,
                peft_method,
                tokenizer,
                tr_smi,
                tr_y,
                va_smi,
                va_y,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()
        if np.isnan(val_pred).any():
            raise optuna.TrialPruned()
        metrics = compute_metrics(va_y, val_pred)
        fold_raes.append(metrics["RAE"])
        trial.report(float(np.mean(fold_raes)), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(fold_raes))


def run_final_cv(
    best_params: dict,
    backbone: str,
    peft_method: str,
    tokenizer,
    train_smiles: list[str],
    y_train: np.ndarray,
    test_smiles: list[str],
    test_df: pd.DataFrame,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    split_name: str,
    max_epochs_final: int,
    patience_final: int,
) -> dict:
    """Final outer CV + DB record + submission CSV."""
    final_params = {
        **best_params,
        "max_epochs": max_epochs_final,
        "patience": patience_final,
    }
    rank = final_params["lora_rank"]
    alpha = final_params["lora_alpha"]
    name = f"peft_{backbone}_{peft_method}_r{rank}a{alpha}_{split_name}_default"

    print(f"\n{'=' * 60}")
    print(f"  Final CV: {name}")
    print(f"  Params: {final_params}")
    print(f"{'=' * 60}")

    oof_preds = np.zeros(len(y_train))
    test_preds_all = np.zeros((len(outer_splits), len(test_smiles)))
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(outer_splits):
        print(f"\n  --- Fold {fold} ---")
        tr_smi = [train_smiles[i] for i in tr_idx]
        va_smi = [train_smiles[i] for i in va_idx]
        tr_y = y_train[tr_idx]
        va_y = y_train[va_idx]

        val_pred, test_pred = train_one_fold(
            final_params,
            backbone,
            peft_method,
            tokenizer,
            tr_smi,
            tr_y,
            va_smi,
            va_y,
            test_smiles,
        )
        if np.isnan(val_pred).any():
            raise ValueError(f"Fold {fold} produced NaN val predictions")
        oof_preds[va_idx] = val_pred
        test_preds_all[fold] = test_pred

        m = compute_metrics(va_y, val_pred)
        fold_metrics.append(m)
        print_metrics(m, label=f"Fold {fold}")

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
    print(f"  Wrote submission: {sub_path}")

    exp_id = record_experiment(
        name=name,
        description=f"PEFT {backbone} {peft_method} ({split_name} split, optuna-tuned)",
        model_type="peft_finetune",
        feature_set="smiles_transformer_peft",
        hyperparameters=final_params,
        fold_metrics=fold_metrics,
        submission_path=f"track1_activity/submissions/{name}.csv",
        notes=(
            f"OOF RAE={oof_metrics['RAE']:.4f}, MAE={oof_metrics['MAE']:.4f}, "
            f"{split_name}_split, peft={peft_method}"
        ),
    )
    save_oof_predictions(exp_id, oof_preds)
    return oof_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="PEFT fine-tuning with Optuna")
    parser.add_argument("--backbone", default="molformer_xl")
    parser.add_argument("--peft-method", default="lora")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--split", choices=["umap", "scaffold"], default="umap")
    parser.add_argument("--n-clusters", type=int, default=50)
    parser.add_argument("--max-epochs-trial", type=int, default=50)
    parser.add_argument("--patience-trial", type=int, default=8)
    parser.add_argument("--max-epochs-final", type=int, default=80)
    parser.add_argument("--patience-final", type=int, default=12)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    print("Loading data...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    train_smiles = train_df["smiles"].tolist()
    y_train = train_df["pec50"].to_numpy()
    test_smiles = test_df["smiles"].tolist()
    print(f"Train: {len(train_smiles)}, Test: {len(test_smiles)}")
    print(
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
    print(f"Backbone={args.backbone}, PEFT={args.peft_method}, Split={args.split}")

    print(f"Loading tokenizer for {args.backbone}...")
    tokenizer = get_tokenizer(args.backbone)

    if args.split == "umap":
        outer_splits = umap_split_indices(
            train_smiles, n_splits=args.outer_folds, n_clusters=args.n_clusters, seed=42
        )
        inner_splits = umap_split_indices(
            train_smiles,
            n_splits=args.inner_folds,
            n_clusters=args.n_clusters,
            seed=123,
        )
    else:
        outer_splits = scaffold_split_indices(
            train_smiles, n_splits=args.outer_folds, seed=42
        )
        inner_splits = scaffold_split_indices(
            train_smiles, n_splits=args.inner_folds, seed=123
        )

    print(f"\n{'=' * 60}")
    print(f"  Optuna tuning: {args.n_trials} trials, {args.inner_folds}-fold inner CV")
    print(f"{'=' * 60}")

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(
        lambda t: objective(
            t,
            args.backbone,
            args.peft_method,
            tokenizer,
            train_smiles,
            y_train,
            inner_splits,
            args.max_epochs_trial,
            args.patience_trial,
        ),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    n_complete = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    n_pruned = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    )
    n_failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
    print(f"  Trials: {n_complete} complete, {n_pruned} pruned, {n_failed} FAILED")
    if n_complete == 0:
        raise RuntimeError("All Optuna trials failed. Cannot proceed.")
    print(f"  Best trial RAE: {study.best_value:.4f}")
    print(f"  Best params:    {study.best_params}")

    # Reconstruct best_params for the trainer (lora_alpha = rank * mult).
    rank = int(study.best_params["lora_rank"])
    alpha_mult = int(study.best_params["lora_alpha_mult"])
    best_params = {
        "lora_rank": rank,
        "lora_alpha": rank * alpha_mult,
        "lora_dropout": float(study.best_params["lora_dropout"]),
        "lora_target": study.best_params["lora_target"],
        "head_hidden_dim": int(study.best_params["head_hidden_dim"]),
        "head_dropout": float(study.best_params["head_dropout"]),
        "backbone_lr": float(study.best_params["backbone_lr"]),
        "head_lr": float(study.best_params["head_lr"]),
        "weight_decay": float(study.best_params["weight_decay"]),
        "batch_size": int(study.best_params["batch_size"]),
    }

    oof_metrics = run_final_cv(
        best_params,
        args.backbone,
        args.peft_method,
        tokenizer,
        train_smiles,
        y_train,
        test_smiles,
        test_df,
        outer_splits,
        args.split,
        args.max_epochs_final,
        args.patience_final,
    )
    print(f"\n  Final OOF RAE: {oof_metrics['RAE']:.4f}, MAE: {oof_metrics['MAE']:.4f}")


if __name__ == "__main__":
    main()
