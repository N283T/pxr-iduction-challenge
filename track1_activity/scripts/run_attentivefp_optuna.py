"""AttentiveFP (PyG) hyperparameter tuning with Optuna.

Uses UMAP split for more realistic CV estimates.
Saves OOF predictions to DB for ensemble.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))

import argparse
import warnings

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data, Batch
from torch_geometric.nn.models import AttentiveFP
from torch_geometric.utils import from_smiles

from data import load_train_smiles_target, load_test_smiles
from evaluate import (
    compute_metrics,
    print_metrics,
    print_fold_summary,
    record_experiment,
    save_oof_predictions,
)
from splits import umap_split_indices, scaffold_split_indices

warnings.filterwarnings("ignore", category=FutureWarning)

SUBMISSION_DIR = Path(__file__).resolve().parent.parent.joinpath("submissions")


def smiles_to_pyg_list(smiles_list):
    """Convert SMILES list to PyG Data objects."""
    data_list = []
    for smi in smiles_list:
        data = from_smiles(smi)
        data_list.append(data)
    return data_list


def make_batches(data_list, targets, batch_size, shuffle=False):
    """Create batches from PyG data list."""
    indices = list(range(len(data_list)))
    if shuffle:
        np.random.shuffle(indices)
    batches = []
    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i : i + batch_size]
        batch_data = [data_list[j] for j in batch_idx]
        batch = Batch.from_data_list(batch_data)
        if targets is not None:
            batch_y = torch.tensor(
                [targets[j] for j in batch_idx], dtype=torch.float32
            )
        else:
            batch_y = None
        batches.append((batch, batch_y))
    return batches


def train_and_predict(
    params, train_graphs, train_targets, val_graphs, val_targets, test_graphs=None
):
    """Train AttentiveFP, return val and optionally test predictions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = train_graphs[0].x.shape[1]
    edge_dim = train_graphs[0].edge_attr.shape[1]

    model = AttentiveFP(
        in_channels=in_channels,
        hidden_channels=params["hidden_channels"],
        out_channels=1,
        edge_dim=edge_dim,
        num_layers=params["num_layers"],
        num_timesteps=params["num_timesteps"],
        dropout=params["dropout"],
    ).to(device)

    optimizer = Adam(model.parameters(), lr=params["learning_rate"])
    scheduler = CosineAnnealingLR(optimizer, T_max=params["max_epochs"], eta_min=1e-7)
    criterion = nn.MSELoss()

    best_val_mae = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(params["max_epochs"]):
        # Train
        model.train()
        train_batches = make_batches(
            train_graphs, train_targets, params["batch_size"], shuffle=True
        )
        for batch, batch_y in train_batches:
            batch = batch.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            out = model(
                batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch
            ).squeeze(-1)
            loss = criterion(out, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        val_preds_list = []
        val_batches = make_batches(
            val_graphs, val_targets, params["batch_size"], shuffle=False
        )
        with torch.no_grad():
            for batch, _ in val_batches:
                batch = batch.to(device)
                out = model(
                    batch.x.float(),
                    batch.edge_index,
                    batch.edge_attr.float(),
                    batch.batch,
                ).squeeze(-1)
                val_preds_list.append(out.cpu().numpy())

        val_preds = np.concatenate(val_preds_list)
        val_mae = np.mean(np.abs(val_targets - val_preds))

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= params["patience"]:
                break

    # Load best and predict
    model.load_state_dict(best_state)
    model.eval()

    val_preds_list = []
    val_batches = make_batches(
        val_graphs, val_targets, params["batch_size"], shuffle=False
    )
    with torch.no_grad():
        for batch, _ in val_batches:
            batch = batch.to(device)
            out = model(
                batch.x.float(),
                batch.edge_index,
                batch.edge_attr.float(),
                batch.batch,
            ).squeeze(-1)
            val_preds_list.append(out.cpu().numpy())
    val_preds = np.concatenate(val_preds_list)

    test_preds = None
    if test_graphs is not None:
        test_preds_list = []
        test_batches = make_batches(
            test_graphs, None, params["batch_size"], shuffle=False
        )
        with torch.no_grad():
            for batch, _ in test_batches:
                batch = batch.to(device)
                out = model(
                    batch.x.float(),
                    batch.edge_index,
                    batch.edge_attr.float(),
                    batch.batch,
                ).squeeze(-1)
                test_preds_list.append(out.cpu().numpy())
        test_preds = np.concatenate(test_preds_list)

    del model
    torch.cuda.empty_cache()
    return val_preds, test_preds


def objective(trial, train_graphs, y_train, inner_splits):
    """Optuna objective: average RAE across inner CV folds."""
    params = {
        "hidden_channels": trial.suggest_categorical(
            "hidden_channels", [128, 200, 256, 512]
        ),
        "num_layers": trial.suggest_int("num_layers", 2, 5),
        "num_timesteps": trial.suggest_int("num_timesteps", 1, 4),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4, step=0.05),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "max_epochs": 100,
        "patience": 10,
    }

    fold_raes = []
    for fold, (train_idx, val_idx) in enumerate(inner_splits):
        tr_graphs = [train_graphs[i] for i in train_idx]
        va_graphs = [train_graphs[i] for i in val_idx]
        tr_y = y_train[train_idx]
        va_y = y_train[val_idx]

        val_pred, _ = train_and_predict(params, tr_graphs, tr_y, va_graphs, va_y)
        metrics = compute_metrics(va_y, val_pred)
        fold_raes.append(metrics["RAE"])

        trial.report(np.mean(fold_raes), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(fold_raes)


def run_final_cv(
    best_params,
    train_graphs,
    y_train,
    test_graphs,
    test_df,
    outer_splits,
    split_name,
):
    """Run full outer CV with best params and save results."""
    best_params["max_epochs"] = 150
    best_params["patience"] = 15

    name = f"attentivefp_optuna_{split_name}"
    print(f"\n{'=' * 60}")
    print(f"  Final CV: {name}")
    print(f"  Params: {best_params}")
    print(f"{'=' * 60}")

    oof_preds = np.zeros(len(y_train))
    test_preds_all = np.zeros((len(outer_splits), len(test_df)))
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(outer_splits):
        print(f"\n  --- Fold {fold} ---")
        tr_graphs = [train_graphs[i] for i in train_idx]
        va_graphs = [train_graphs[i] for i in val_idx]
        tr_y = y_train[train_idx]
        va_y = y_train[val_idx]

        val_pred, test_pred = train_and_predict(
            best_params, tr_graphs, tr_y, va_graphs, va_y, test_graphs
        )

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
        description=f"AttentiveFP (PyG) Optuna-tuned ({split_name} split)",
        model_type="attentivefp",
        feature_set="molecular_graph",
        hyperparameters=best_params,
        fold_metrics=fold_metrics,
        submission_path=f"track1_activity/submissions/{name}.csv",
        notes=f"OOF RAE={oof_metrics['RAE']:.4f}, {split_name}_split, optuna_tuned, pyg",
    )
    save_oof_predictions(exp_id, oof_preds)

    return oof_metrics


def main():
    parser = argparse.ArgumentParser(description="AttentiveFP Optuna tuning (PyG)")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--split", choices=["umap", "scaffold"], default="umap")
    parser.add_argument("--n-clusters", type=int, default=50)
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

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
    print(f"Split: {args.split}")

    print("Converting SMILES to PyG graphs...")
    train_graphs = smiles_to_pyg_list(train_smiles)
    test_graphs = smiles_to_pyg_list(test_smiles)
    print(
        f"Converted: train={len(train_graphs)}, test={len(test_graphs)}"
    )

    if args.split == "umap":
        split_fn = lambda n, s: umap_split_indices(
            train_smiles, n_splits=n, n_clusters=args.n_clusters, seed=s
        )
    else:
        split_fn = lambda n, s: scaffold_split_indices(
            train_smiles, n_splits=n, seed=s
        )

    outer_splits = split_fn(args.outer_folds, 42)
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
        lambda trial: objective(trial, train_graphs, y_train, inner_splits),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    print(f"\n  Best trial: RAE={study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    # Final outer CV with best params
    oof_metrics = run_final_cv(
        study.best_params,
        train_graphs,
        y_train,
        test_graphs,
        test_df,
        outer_splits,
        args.split,
    )

    print(f"\n  Final OOF RAE: {oof_metrics['RAE']:.4f}")


if __name__ == "__main__":
    main()
