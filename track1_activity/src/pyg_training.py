"""Shared PyG training utilities for graph neural network scripts.

Used by run_gin_optuna.py / run_gatedgcn_optuna.py / run_graphgps_optuna.py.
Model-specific logic lives in each script via a `build_model(params, in_dim,
edge_dim) -> nn.Module` callback. This module handles SMILES→PyG
conversion, batching, training loops with early stopping, and the Optuna
objective wrapper.

The existing run_attentivefp_optuna.py predates this module and is kept
unchanged to avoid scope creep.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Batch
from torch_geometric.utils import from_smiles

from evaluate import compute_metrics

warnings.filterwarnings("ignore", category=FutureWarning)


ModelBuilder = Callable[[dict, int, int], nn.Module]


def smiles_to_pyg_list(smiles_list: list[str]) -> list:
    """Convert SMILES list to PyG Data objects.

    Raises on empty graphs so downstream batching never silently drops rows.
    """
    data_list = []
    for i, smi in enumerate(smiles_list):
        data = from_smiles(smi)
        if data.x is None or data.x.shape[0] == 0:
            raise ValueError(f"SMILES at index {i} produced empty graph: {smi}")
        data_list.append(data)
    return data_list


def make_batches(data_list, targets, batch_size, shuffle=False):
    """Create (batch, targets) pairs from a PyG Data list."""
    indices = list(range(len(data_list)))
    if shuffle:
        np.random.shuffle(indices)
    batches = []
    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i : i + batch_size]
        batch_data = [data_list[j] for j in batch_idx]
        batch = Batch.from_data_list(batch_data)
        if targets is not None:
            batch_y = torch.tensor([targets[j] for j in batch_idx], dtype=torch.float32)
        else:
            batch_y = None
        batches.append((batch, batch_y))
    return batches


def _forward(model, batch, device):
    """Call model(x, edge_index, edge_attr, batch) and squeeze last dim."""
    batch = batch.to(device)
    out = model(
        batch.x.float(),
        batch.edge_index,
        batch.edge_attr.float(),
        batch.batch,
    ).squeeze(-1)
    return out, batch


def _predict_loop(model, graphs, targets, batch_size, device):
    """Run eval predict loop over a graph list."""
    model.eval()
    preds = []
    batches = make_batches(graphs, targets, batch_size, shuffle=False)
    with torch.no_grad():
        for batch, _ in batches:
            out, _ = _forward(model, batch, device)
            preds.append(out.cpu().numpy())
    return np.concatenate(preds)


def train_and_predict(
    params: dict,
    build_model: ModelBuilder,
    train_graphs,
    train_targets,
    val_graphs,
    val_targets,
    test_graphs=None,
):
    """Train a PyG model and return val (+ optional test) predictions.

    `build_model(params, in_dim, edge_dim)` is called to instantiate the model.
    Trains with Adam + CosineAnnealingLR, MSE loss, gradient clipping, and
    val-MAE-based early stopping. Restores the best weights before predicting.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = train_graphs[0].x.shape[1]
    edge_dim = train_graphs[0].edge_attr.shape[1]
    model = build_model(params, in_channels, edge_dim).to(device)

    optimizer = Adam(model.parameters(), lr=params["learning_rate"])
    scheduler = CosineAnnealingLR(optimizer, T_max=params["max_epochs"], eta_min=1e-7)
    criterion = nn.MSELoss()

    best_val_mae = float("inf")
    best_state = None
    patience_counter = 0

    for _ in range(params["max_epochs"]):
        model.train()
        train_batches = make_batches(
            train_graphs, train_targets, params["batch_size"], shuffle=True
        )
        for batch, batch_y in train_batches:
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            out, _ = _forward(model, batch, device)
            loss = criterion(out, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        val_preds = _predict_loop(
            model, val_graphs, val_targets, params["batch_size"], device
        )
        val_mae = float(np.mean(np.abs(val_targets - val_preds)))

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= params["patience"]:
                break

    if best_state is None:
        raise RuntimeError(
            "Training failed: model never improved. Check learning rate."
        )
    model.load_state_dict(best_state)

    val_preds = _predict_loop(
        model, val_graphs, val_targets, params["batch_size"], device
    )
    test_preds = None
    if test_graphs is not None:
        test_preds = _predict_loop(
            model, test_graphs, None, params["batch_size"], device
        )

    del model
    torch.cuda.empty_cache()
    return val_preds, test_preds


def make_optuna_objective(
    build_model: ModelBuilder,
    suggest_params: Callable[[optuna.Trial], dict],
    train_graphs,
    y_train: np.ndarray,
    inner_splits,
):
    """Build an Optuna objective that trains across inner CV folds.

    `suggest_params(trial)` should return a params dict; training hyperparams
    (max_epochs, patience) must be included.
    """

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        fold_raes = []
        for fold, (train_idx, val_idx) in enumerate(inner_splits):
            tr_graphs = [train_graphs[i] for i in train_idx]
            va_graphs = [train_graphs[i] for i in val_idx]
            tr_y = y_train[train_idx]
            va_y = y_train[val_idx]

            val_pred, _ = train_and_predict(
                params, build_model, tr_graphs, tr_y, va_graphs, va_y
            )
            metrics = compute_metrics(va_y, val_pred)
            fold_raes.append(metrics["RAE"])

            trial.report(float(np.mean(fold_raes)), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_raes))

    return objective


def run_outer_cv(
    name: str,
    build_model: ModelBuilder,
    final_params: dict,
    train_graphs,
    y_train: np.ndarray,
    test_graphs,
    outer_splits,
):
    """Run outer CV and return (oof_preds, test_preds_mean, fold_metrics)."""
    oof_preds = np.zeros(len(y_train))
    test_preds_all = np.zeros((len(outer_splits), len(test_graphs)))
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(outer_splits):
        print(f"\n  --- {name} Fold {fold} ---")
        tr_graphs = [train_graphs[i] for i in train_idx]
        va_graphs = [train_graphs[i] for i in val_idx]
        tr_y = y_train[train_idx]
        va_y = y_train[val_idx]

        val_pred, test_pred = train_and_predict(
            final_params, build_model, tr_graphs, tr_y, va_graphs, va_y, test_graphs
        )
        if np.any(np.isnan(val_pred)):
            raise ValueError(f"Fold {fold}: model produced NaN predictions")

        oof_preds[val_idx] = val_pred
        test_preds_all[fold] = test_pred

        metrics = compute_metrics(va_y, val_pred)
        fold_metrics.append(metrics)
        print(
            f"    Fold {fold}: RAE={metrics['RAE']:.4f} "
            f"MAE={metrics['MAE']:.4f} R2={metrics['R2']:.4f}"
        )

    return oof_preds, test_preds_all.mean(axis=0), fold_metrics
