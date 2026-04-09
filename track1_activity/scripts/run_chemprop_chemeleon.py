"""ChemProp finetuning on CheMeleon foundation model with Optuna + UMAP CV.

Loads the pretrained CheMeleon BondMessagePassing weights and finetunes on the
challenge pEC50 target. Uses parameter groups so the pretrained MP can train at
a lower LR than the fresh FFN head (moka-style).

Key differences from run_chemprop_optuna.py:
  - MP architecture is fixed to CheMeleon's (d_h=2048, depth=6). Only
    mp_dropout can be overridden for regularization during finetuning.
  - FFN head is tuned by Optuna (hidden_dim, num_layers, dropout).
  - Two parameter groups: MP at `mp_lr_ratio * init_lr`, rest at `init_lr`.
  - Smaller Optuna search space (25 trials) because each trial is slow.

Saves best-model OOF predictions to DB as `chemprop_chemeleon_umap`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))

import argparse
from urllib.request import urlretrieve

import numpy as np
import optuna
import pandas as pd
import torch
from lightning import pytorch as pl
from torch import optim

from chemprop import data as chemprop_data
from chemprop import models, nn
from chemprop.schedulers import build_NoamLike_LRSched

from data import load_train_smiles_target, load_test_smiles
from evaluate import (
    compute_metrics,
    print_metrics,
    print_fold_summary,
    record_experiment,
    save_oof_predictions,
)
from splits import umap_split_indices, scaffold_split_indices

SUBMISSION_DIR = Path(__file__).resolve().parent.parent.joinpath("submissions")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
CHECKPOINT_PATH = Path.home().joinpath(".chemprop", "chemeleon_mp.pt")

torch.set_float32_matmul_precision("medium")


AGG_REGISTRY = {
    "mean": nn.MeanAggregation,
    "sum": nn.SumAggregation,
    "norm": nn.NormAggregation,
}


def load_chemeleon_checkpoint() -> dict:
    """Download CheMeleon checkpoint from Zenodo if missing, then load."""
    if not CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading CheMeleon checkpoint to {CHECKPOINT_PATH}...")
        urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)
    return torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)


class CheMeleonMPNN(models.MPNN):
    """ChemProp MPNN with two-group optimizer (MP lower LR than FFN)."""

    def __init__(self, *args, mp_lr_ratio: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.mp_lr_ratio = mp_lr_ratio

    def configure_optimizers(self):
        mp_params = list(self.message_passing.parameters())
        mp_param_ids = {id(p) for p in mp_params}
        other_params = [p for p in self.parameters() if id(p) not in mp_param_ids]

        opt = optim.Adam(
            [
                {"params": mp_params, "lr": self.init_lr * self.mp_lr_ratio},
                {"params": other_params, "lr": self.init_lr},
            ],
            lr=self.init_lr,
        )

        # Replicate parent's scheduler wiring.
        if self.trainer.train_dataloader is None:
            _ = self.trainer.estimated_stepping_batches
        steps_per_epoch = self.trainer.num_training_batches
        warmup_steps = self.warmup_epochs * steps_per_epoch
        cooldown_epochs = self.trainer.max_epochs - self.warmup_epochs
        cooldown_steps = cooldown_epochs * steps_per_epoch

        lr_sched = build_NoamLike_LRSched(
            opt,
            warmup_steps,
            cooldown_steps,
            self.init_lr,
            self.max_lr,
            self.final_lr,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": lr_sched, "interval": "step"},
        }


def build_model(params: dict, chemeleon_ckpt: dict):
    """Build CheMeleon-initialized MPNN from hyperparameters.

    MP architecture is fixed by the checkpoint. Only `mp_dropout` can be
    overridden for finetune-time regularization.
    """
    mp_kwargs = {**chemeleon_ckpt["hyper_parameters"], "dropout": params["mp_dropout"]}
    mp = nn.BondMessagePassing(**mp_kwargs)
    mp.load_state_dict(chemeleon_ckpt["state_dict"])

    agg = AGG_REGISTRY[params["aggregation"]]()

    ffn = nn.RegressionFFN(
        input_dim=mp.output_dim,
        hidden_dim=params["ffn_hidden_dim"],
        n_layers=params["ffn_num_layers"],
        dropout=params["ffn_dropout"],
    )

    model = CheMeleonMPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        batch_norm=True,
        warmup_epochs=params["warmup_epochs"],
        init_lr=params["learning_rate"],
        max_lr=params["learning_rate"] * params["lr_ratio"],
        final_lr=params["learning_rate"] * 0.1,
        mp_lr_ratio=params["mp_lr_ratio"],
    )
    return model


def train_and_predict(
    params,
    chemeleon_ckpt,
    train_smiles,
    train_targets,
    val_smiles,
    val_targets,
    test_smiles=None,
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

    model = build_model(params, chemeleon_ckpt)

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

    del model, trainer
    torch.cuda.empty_cache()

    return val_preds, test_preds


def objective(trial, chemeleon_ckpt, train_smiles, y_train, inner_splits):
    """Optuna objective: average RAE across inner CV folds."""
    params = {
        # FFN head (fully tuned; CheMeleon output is 2048-d)
        "ffn_hidden_dim": trial.suggest_categorical(
            "ffn_hidden_dim", [256, 512, 1024, 2048]
        ),
        "ffn_num_layers": trial.suggest_int("ffn_num_layers", 1, 3),
        "ffn_dropout": trial.suggest_float("ffn_dropout", 0.0, 0.4, step=0.05),
        # MP regularization (override CheMeleon's 0.0)
        "mp_dropout": trial.suggest_float("mp_dropout", 0.0, 0.3, step=0.05),
        # Aggregation (CheMeleon was pretrained agnostic of agg)
        "aggregation": trial.suggest_categorical(
            "aggregation", ["mean", "sum", "norm"]
        ),
        # Optimizer
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 1e-3, log=True),
        "lr_ratio": trial.suggest_categorical("lr_ratio", [5, 10, 20]),
        "mp_lr_ratio": trial.suggest_categorical("mp_lr_ratio", [0.1, 0.3, 1.0]),
        "warmup_epochs": trial.suggest_int("warmup_epochs", 1, 5),
        "max_epochs": 150,
        "patience": 15,
    }

    fold_raes = []
    for fold, (train_idx, val_idx) in enumerate(inner_splits):
        tr_smi = [train_smiles[i] for i in train_idx]
        va_smi = [train_smiles[i] for i in val_idx]
        tr_y = y_train[train_idx]
        va_y = y_train[val_idx]

        val_pred, _ = train_and_predict(
            params, chemeleon_ckpt, tr_smi, tr_y, va_smi, va_y
        )
        metrics = compute_metrics(va_y, val_pred)
        fold_raes.append(metrics["RAE"])

        trial.report(np.mean(fold_raes), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_raes))


def run_final_cv(
    best_params,
    chemeleon_ckpt,
    train_smiles,
    y_train,
    test_smiles,
    test_df,
    outer_splits,
    split_name,
):
    """Run full outer CV with best params and save results."""
    final_params = {**best_params, "max_epochs": 200, "patience": 20}
    name = f"chemprop_chemeleon_{split_name}"

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
            final_params, chemeleon_ckpt, tr_smi, tr_y, va_smi, va_y, test_smiles
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
        description=(
            f"CheMeleon finetune (d_h=2048, depth=6) with param groups "
            f"({split_name} split, Optuna tuned)"
        ),
        model_type="chemprop_chemeleon",
        feature_set="chemeleon_foundation",
        hyperparameters=final_params,
        fold_metrics=fold_metrics,
        submission_path=f"track1_activity/submissions/{name}.csv",
        notes=(
            f"OOF RAE={oof_metrics['RAE']:.4f}, {split_name}_split, "
            f"optuna_tuned, param_groups"
        ),
    )
    save_oof_predictions(exp_id, oof_preds)

    return oof_metrics


def main():
    parser = argparse.ArgumentParser(
        description="CheMeleon foundation finetune with Optuna"
    )
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--split", choices=["umap", "scaffold"], default="umap")
    parser.add_argument("--n-clusters", type=int, default=50)
    parser.add_argument(
        "--smoke-test", action="store_true", help="Quick 1-fold 3-epoch run"
    )
    parser.add_argument(
        "--skip-tuning", action="store_true", help="Skip Optuna, use sensible defaults"
    )
    args = parser.parse_args()

    pl.seed_everything(42)

    print("Loading CheMeleon checkpoint...")
    chemeleon_ckpt = load_chemeleon_checkpoint()
    print(
        f"  MP d_h={chemeleon_ckpt['hyper_parameters']['d_h']}, depth={chemeleon_ckpt['hyper_parameters']['depth']}"
    )

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

    def split_fn(n, s):
        if args.split == "umap":
            return umap_split_indices(
                train_smiles, n_splits=n, n_clusters=args.n_clusters, seed=s
            )
        return scaffold_split_indices(train_smiles, n_splits=n, seed=s)

    if args.smoke_test:
        print("\n*** SMOKE TEST MODE (1 fold, 3 epochs) ***\n")
        splits = split_fn(2, 42)[:1]  # 1 fold
        smoke_params = {
            "ffn_hidden_dim": 256,
            "ffn_num_layers": 1,
            "ffn_dropout": 0.1,
            "mp_dropout": 0.1,
            "aggregation": "mean",
            "batch_size": 32,
            "learning_rate": 2e-4,
            "lr_ratio": 10,
            "mp_lr_ratio": 0.1,
            "warmup_epochs": 1,
            "max_epochs": 3,
            "patience": 5,
        }
        train_idx, val_idx = splits[0]
        tr_smi = [train_smiles[i] for i in train_idx]
        va_smi = [train_smiles[i] for i in val_idx]
        val_pred, test_pred = train_and_predict(
            smoke_params,
            chemeleon_ckpt,
            tr_smi,
            y_train[train_idx],
            va_smi,
            y_train[val_idx],
            test_smiles,
        )
        metrics = compute_metrics(y_train[val_idx], val_pred)
        print_metrics(metrics, label="Smoke")
        print(f"Test preds shape: {test_pred.shape}, mean={test_pred.mean():.3f}")
        print("Smoke test OK.")
        return

    outer_splits = split_fn(args.outer_folds, 42)

    if not args.skip_tuning:
        inner_splits = split_fn(args.inner_folds, 123)

        print(f"\n{'=' * 60}")
        print(
            f"  Optuna tuning: {args.n_trials} trials, {args.inner_folds}-fold inner CV"
        )
        print(f"{'=' * 60}")

        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        )
        study.optimize(
            lambda trial: objective(
                trial, chemeleon_ckpt, train_smiles, y_train, inner_splits
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
        n_failed = len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
        )
        print(
            f"\n  Trials: {n_complete} complete, {n_pruned} pruned, {n_failed} FAILED"
        )
        if n_complete == 0:
            raise RuntimeError("All Optuna trials failed. Cannot proceed.")
        print(f"  Best trial: RAE={study.best_value:.4f}")
        print(f"  Best params: {study.best_params}")
        best_params = study.best_params
    else:
        print("  Skipping tuning, using sensible defaults")
        best_params = {
            "ffn_hidden_dim": 512,
            "ffn_num_layers": 2,
            "ffn_dropout": 0.15,
            "mp_dropout": 0.1,
            "aggregation": "mean",
            "batch_size": 32,
            "learning_rate": 2e-4,
            "lr_ratio": 10,
            "mp_lr_ratio": 0.1,
            "warmup_epochs": 2,
        }

    oof_metrics = run_final_cv(
        best_params,
        chemeleon_ckpt,
        train_smiles,
        y_train,
        test_smiles,
        test_df,
        outer_splits,
        args.split,
    )

    print(f"\n  Final OOF RAE: {oof_metrics['RAE']:.4f}")


if __name__ == "__main__":
    main()
