"""GIN (GINEConv) hyperparameter tuning with Optuna.

PyG-based Graph Isomorphism Network with edge features. Uses UMAP split CV
and saves OOF predictions to the experiments table for ensembling.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))

import argparse

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool

from data import load_train_smiles_target, load_test_smiles
from evaluate import (
    compute_metrics,
    print_metrics,
    print_fold_summary,
    record_experiment,
    save_oof_predictions,
)
from pyg_training import (
    make_optuna_objective,
    run_outer_cv,
    smiles_to_pyg_list,
    train_and_predict,
)
from splits import umap_split_indices, scaffold_split_indices

SUBMISSION_DIR = Path(__file__).resolve().parent.parent.joinpath("submissions")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)


class GINModel(nn.Module):
    """GIN with shared edge embedding, stack of GINEConv layers, mean pool, FFN."""

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.node_embed = nn.Linear(in_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_embed(x)
        e = self.edge_embed(edge_attr)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, e)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        x = global_mean_pool(x, batch)
        return self.ffn(x)


def build_model(params: dict, in_dim: int, edge_dim: int) -> nn.Module:
    return GINModel(
        in_dim=in_dim,
        edge_dim=edge_dim,
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    )


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 512]),
        "num_layers": trial.suggest_int("num_layers", 3, 6),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4, step=0.05),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "max_epochs": 100,
        "patience": 10,
    }


def main():
    parser = argparse.ArgumentParser(description="GIN Optuna tuning (PyG)")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--split", choices=["umap", "scaffold"], default="umap")
    parser.add_argument("--n-clusters", type=int, default=50)
    parser.add_argument("--smoke-test", action="store_true")
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

    def split_fn(n, s):
        if args.split == "umap":
            return umap_split_indices(
                train_smiles, n_splits=n, n_clusters=args.n_clusters, seed=s
            )
        return scaffold_split_indices(train_smiles, n_splits=n, seed=s)

    if args.smoke_test:
        print("\n*** SMOKE TEST MODE ***\n")
        splits = split_fn(2, 42)[:1]
        smoke_params = {
            "hidden_dim": 128,
            "num_layers": 3,
            "dropout": 0.1,
            "learning_rate": 1e-3,
            "batch_size": 64,
            "max_epochs": 3,
            "patience": 5,
        }
        tr_idx, va_idx = splits[0]
        val_pred, test_pred = train_and_predict(
            smoke_params,
            build_model,
            [train_graphs[i] for i in tr_idx],
            y_train[tr_idx],
            [train_graphs[i] for i in va_idx],
            y_train[va_idx],
            test_graphs,
        )
        metrics = compute_metrics(y_train[va_idx], val_pred)
        print_metrics(metrics, label="Smoke")
        print(f"Test preds shape: {test_pred.shape}, mean={test_pred.mean():.3f}")
        print("GIN smoke test OK.")
        return

    outer_splits = split_fn(args.outer_folds, 42)
    inner_splits = split_fn(args.inner_folds, 123)

    print(f"\n{'=' * 60}")
    print(f"  Optuna tuning: {args.n_trials} trials, {args.inner_folds}-fold inner CV")
    print(f"{'=' * 60}")

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(
        make_optuna_objective(
            build_model, suggest_params, train_graphs, y_train, inner_splits
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
        raise RuntimeError("All Optuna trials failed.")

    print(f"  Best trial: RAE={study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    final_params = {**study.best_params, "max_epochs": 150, "patience": 15}
    name = f"gin_optuna_{args.split}"
    print(f"\n{'=' * 60}")
    print(f"  Final CV: {name}")
    print(f"  Params: {final_params}")
    print(f"{'=' * 60}")

    oof_preds, test_preds_avg, fold_metrics = run_outer_cv(
        name,
        build_model,
        final_params,
        train_graphs,
        y_train,
        test_graphs,
        outer_splits,
    )

    oof_metrics = compute_metrics(y_train, oof_preds)
    print("\n  Overall OOF:")
    print_metrics(oof_metrics)
    print_fold_summary(fold_metrics)

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
        description=f"GIN (GINEConv) Optuna-tuned ({args.split} split)",
        model_type="gin",
        feature_set="molecular_graph",
        hyperparameters=final_params,
        fold_metrics=fold_metrics,
        submission_path=f"track1_activity/submissions/{name}.csv",
        notes=f"OOF RAE={oof_metrics['RAE']:.4f}, {args.split}_split, optuna_tuned, pyg",
    )
    save_oof_predictions(exp_id, oof_preds)

    print(f"\n  Final OOF RAE: {oof_metrics['RAE']:.4f}")


if __name__ == "__main__":
    main()
