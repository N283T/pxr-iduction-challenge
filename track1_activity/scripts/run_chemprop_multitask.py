#!/usr/bin/env -S pixi run python
"""ChemProp multi-task: pEC50 + counter_assay pEC50 with masked loss.

Uses train compounds' counter_assay pEC50 as an auxiliary regression target
to force the shared D-MPNN encoder to learn representations that explain
both PXR-specific activity and generic off-target activation. Counter loss
is masked for the 1,492 train compounds without counter data (NaN handled
natively by ChemProp's training_step which calls targets.isfinite()).

At inference, only the PXR head is used. Test compounds never need counter
data, so there is no missing-feature leakage.

Usage:
    pixi run python track1_activity/scripts/run_chemprop_multitask.py \
        --split umap --max-epochs 100 --aux-weight 0.3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))

from chemprop import data as chemprop_data
from chemprop import models, nn
from chemprop.nn.metrics import MSE

from data import (
    load_test_smiles,
    load_train_smiles_multitask_5,
    load_train_smiles_with_counter,
)
from evaluate import (
    compute_metrics,
    print_fold_summary,
    print_metrics,
    record_experiment,
    save_oof_predictions,
)
from splits import scaffold_split_indices, umap_split_indices

SUBMISSION_DIR = Path(__file__).resolve().parent.parent.joinpath("submissions")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

torch.set_float32_matmul_precision("medium")


# Sensible defaults from existing single-task chemprop runs
DEFAULT_PARAMS = {
    "message_hidden_dim": 300,
    "depth": 3,
    "mp_dropout": 0.0,
    "activation": "relu",
    "aggregation": "mean",
    "ffn_hidden_dim": 300,
    "ffn_num_layers": 2,
    "ffn_dropout": 0.0,
    "warmup_epochs": 2,
    "learning_rate": 1e-4,
    "lr_ratio": 10.0,
    "batch_size": 64,
    "max_epochs": 100,
    "patience": 15,
}

# Hyperparameters from chemprop_optuna_umap (single-task Optuna best, OOF RAE=0.5724).
# Used when --use-tuned is passed.
TUNED_PARAMS = {
    "message_hidden_dim": 256,
    "depth": 4,
    "mp_dropout": 0.2,
    "activation": "relu",
    "aggregation": "norm",
    "ffn_hidden_dim": 256,
    "ffn_num_layers": 1,
    "ffn_dropout": 0.1,
    "warmup_epochs": 3,
    "learning_rate": 0.0001364559692954765,
    "lr_ratio": 10,
    "batch_size": 64,
    "max_epochs": 200,
    "patience": 20,
}

AGG_REGISTRY = {
    "mean": nn.MeanAggregation,
    "sum": nn.SumAggregation,
    "norm": nn.NormAggregation,
}


def build_model(params: dict, n_tasks: int, aux_weight: float):
    """Build multi-task ChemProp MPNN.

    Main task (index 0) gets weight 1.0; all auxiliary tasks share
    `aux_weight`. Aux targets are assumed already standardized so the
    cross-task loss magnitudes are comparable.
    """
    mp = nn.BondMessagePassing(
        d_h=params["message_hidden_dim"],
        depth=params["depth"],
        dropout=params["mp_dropout"],
        activation=params["activation"],
    )
    agg = AGG_REGISTRY[params["aggregation"]]()

    weights = [1.0] + [aux_weight] * (n_tasks - 1)
    task_weights = torch.tensor(weights, dtype=torch.float32)
    criterion = MSE(task_weights=task_weights)

    ffn = nn.RegressionFFN(
        n_tasks=n_tasks,
        input_dim=mp.output_dim,
        hidden_dim=params["ffn_hidden_dim"],
        n_layers=params["ffn_num_layers"],
        dropout=params["ffn_dropout"],
        criterion=criterion,
    )

    return models.MPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        batch_norm=True,
        warmup_epochs=params["warmup_epochs"],
        init_lr=params["learning_rate"],
        max_lr=params["learning_rate"] * params["lr_ratio"],
        final_lr=params["learning_rate"] * 0.1,
    )


def make_dataloader(smiles, targets, batch_size, shuffle):
    """targets is shape (N, n_tasks). NaN entries become masked at the model
    level via training_step's `targets.isfinite()` call."""
    pts = [
        chemprop_data.MoleculeDatapoint.from_smi(smi, np.asarray(y, dtype=np.float32))
        for smi, y in zip(smiles, targets)
    ]
    return chemprop_data.build_dataloader(
        chemprop_data.MoleculeDataset(pts),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def predict_pec50(trainer, model, smiles, batch_size, n_tasks):
    """Run inference and return only the pEC50 column (task 0)."""
    pts = [
        chemprop_data.MoleculeDatapoint.from_smi(
            smi, np.full(n_tasks, np.nan, dtype=np.float32)
        )
        for smi in smiles
    ]
    loader = chemprop_data.build_dataloader(
        chemprop_data.MoleculeDataset(pts),
        batch_size=batch_size,
        shuffle=False,
    )
    preds = trainer.predict(model, loader)
    arr = np.concatenate([p.numpy() for p in preds], axis=0)
    if arr.ndim == 1:
        return arr
    return arr[:, 0]


def train_fold(
    params, train_smiles, train_targets, val_smiles, val_targets, aux_weight
):
    n_tasks = train_targets.shape[1]
    train_loader = make_dataloader(
        train_smiles, train_targets, params["batch_size"], shuffle=True
    )
    val_loader = make_dataloader(
        val_smiles, val_targets, params["batch_size"], shuffle=False
    )

    model = build_model(params, n_tasks=n_tasks, aux_weight=aux_weight)
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
    val_preds = predict_pec50(trainer, model, val_smiles, params["batch_size"], n_tasks)
    return val_preds, model, trainer


def run(args):
    print(
        f"ChemProp multi-task | split={args.split} | aux_weight={args.aux_weight} "
        f"| max_epochs={args.max_epochs}"
    )

    print("Loading data...")
    if args.task_set == 5:
        train_df = load_train_smiles_multitask_5()
        target_cols = [
            "pec50",
            "counter_pec50",
            "counter_emax",
            "log2fc_8_25e_6",
            "log2fc_3_30e_5",
        ]
    else:
        train_df = load_train_smiles_with_counter()
        target_cols = ["pec50", "counter_pec50"]
    test_df = load_test_smiles()

    smiles = train_df["smiles"].tolist()
    raw_targets = train_df[target_cols].to_numpy(dtype=np.float32)

    # Standardize aux columns (cols 1..) using global train stats so cross-task
    # MSE magnitudes are comparable. Main task (col 0) is left unscaled so its
    # predictions remain on the original pEC50 scale and need no inverse transform.
    targets = raw_targets.copy()
    aux_means = np.zeros(len(target_cols), dtype=np.float32)
    aux_stds = np.ones(len(target_cols), dtype=np.float32)
    for i in range(1, len(target_cols)):
        col = raw_targets[:, i]
        valid = np.isfinite(col)
        mu = float(np.mean(col[valid]))
        sd = float(np.std(col[valid]))
        if sd < 1e-6:
            sd = 1.0
        aux_means[i] = mu
        aux_stds[i] = sd
        targets[:, i] = (col - mu) / sd

    print(f"  train: {len(smiles)}, n_tasks: {len(target_cols)}")
    for i, name in enumerate(target_cols):
        n_valid = int(np.isfinite(raw_targets[:, i]).sum())
        if i == 0:
            print(f"    [main] {name:20s} n={n_valid}")
        else:
            print(
                f"    [aux ] {name:20s} n={n_valid}  "
                f"(standardized: mu={aux_means[i]:+.3f}, sd={aux_stds[i]:.3f})"
            )
    print(f"  test:  {len(test_df)}")

    # CV splits
    if args.split == "scaffold":
        outer = scaffold_split_indices(smiles, n_splits=5, seed=42)
    else:
        outer = umap_split_indices(smiles, n_splits=5, n_clusters=50, seed=42)

    params = (TUNED_PARAMS if args.use_tuned else DEFAULT_PARAMS).copy()
    if args.max_epochs is not None:
        params["max_epochs"] = args.max_epochs

    exp_name = f"chemprop_multitask{args.task_set}_{args.split}_aux{args.aux_weight}"
    if args.use_tuned:
        exp_name += "_tuned"
    print(f"  Experiment: {exp_name}")

    oof_preds = np.zeros(len(smiles), dtype=np.float32)
    fold_metrics = []
    test_pred_per_fold = []

    for fold, (tr_idx, va_idx) in enumerate(outer):
        print(f"\n[Fold {fold}] train={len(tr_idx)}, val={len(va_idx)}")
        tr_smi = [smiles[i] for i in tr_idx]
        va_smi = [smiles[i] for i in va_idx]
        tr_y = targets[tr_idx]
        va_y = targets[va_idx]

        val_preds, model, trainer = train_fold(
            params, tr_smi, tr_y, va_smi, va_y, args.aux_weight
        )
        oof_preds[va_idx] = val_preds

        metrics = compute_metrics(va_y[:, 0], val_preds)
        fold_metrics.append(metrics)
        print_metrics(metrics, label=f"Fold {fold}")

        # Predict test for this fold (averaged later)
        test_preds = predict_pec50(
            trainer,
            model,
            test_df["smiles"].tolist(),
            params["batch_size"],
            len(target_cols),
        )
        test_pred_per_fold.append(test_preds)

        del model, trainer
        torch.cuda.empty_cache()

    oof_metrics = compute_metrics(targets[:, 0], oof_preds)
    print("\n  Overall OOF:")
    print_metrics(oof_metrics)
    print_fold_summary(fold_metrics)

    # Average test predictions across folds
    test_preds_mean = np.mean(test_pred_per_fold, axis=0)
    print(
        f"\n  Test preds: mean={test_preds_mean.mean():.3f}, "
        f"std={test_preds_mean.std():.3f}"
    )

    sub = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": test_preds_mean,
        }
    )
    sub_path = SUBMISSION_DIR.joinpath(f"{exp_name}.csv")
    sub.to_csv(sub_path, index=False)
    print(f"  Saved submission: {sub_path}")

    exp_id = record_experiment(
        name=exp_name,
        description=f"ChemProp multi-task pEC50 + counter_pec50 ({args.split} split)",
        model_type="chemprop",
        feature_set="smiles",
        hyperparameters={**params, "aux_weight": args.aux_weight, "n_tasks": 2},
        fold_metrics=fold_metrics,
        submission_path=f"track1_activity/submissions/{exp_name}.csv",
        num_boost_rounds=[0] * len(fold_metrics),
        notes=(
            f"OOF RAE={oof_metrics['RAE']:.4f}, multi-task counter_assay aux, "
            f"aux_weight={args.aux_weight}, task_set={args.task_set}"
        ),
    )
    save_oof_predictions(exp_id, oof_preds)
    print(f"\n  Done: {exp_name} -> RAE={oof_metrics['RAE']:.4f}")
    return oof_metrics


def main():
    parser = argparse.ArgumentParser(description="ChemProp multi-task training")
    parser.add_argument("--split", choices=["scaffold", "umap"], default="umap")
    parser.add_argument(
        "--task-set",
        type=int,
        choices=[2, 5],
        default=2,
        help="2 = pec50+counter_pec50; 5 = +counter_emax +log2fc(2 concs)",
    )
    parser.add_argument(
        "--aux-weight",
        type=float,
        default=0.3,
        help="Loss weight on counter_pec50 auxiliary task (main pec50 = 1.0)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Max training epochs per fold (defaults to params' max_epochs)",
    )
    parser.add_argument(
        "--use-tuned",
        action="store_true",
        help="Use chemprop_optuna_umap's tuned hyperparameters as the base",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
