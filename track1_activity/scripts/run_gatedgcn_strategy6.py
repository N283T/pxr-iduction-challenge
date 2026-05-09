#!/usr/bin/env python
"""Buterez 2024 Strategy 6 probe: frozen GatedGCN + adaptive readout.

Strategy 6 freezes the low-fidelity-pretrained message-passing encoder and
fine-tunes only an adaptive graph readout on the high-fidelity pEC50 task.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Batch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from adaptive_readout import AdaptiveReadoutRegressor  # noqa: E402
from data import load_test_smiles, load_train_smiles_target  # noqa: E402
from evaluate import (  # noqa: E402
    compute_metrics,
    print_fold_summary,
    print_metrics,
    record_experiment,
    save_oof_predictions,
)
from run_gatedgcn_pretrain_finetune import (  # noqa: E402
    PRETRAIN_PATH,
    GatedGCNModel,
    load_pretrain_encoder,
    smiles_to_pyg,
)
from splits import umap_split_indices  # noqa: E402

SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
OUTPUT_ROOT = REPO_ROOT.joinpath(
    "track1_activity", "analysis", "strategy6_gatedgcn", "outputs"
)
META_PATH = PRETRAIN_PATH.parent.joinpath("pretrain_meta.json")


def load_pretrain_params() -> dict:
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing metadata: {META_PATH}")
    meta = json.loads(META_PATH.read_text())
    return meta["params"]


def iter_index_batches(indices: np.ndarray, batch_size: int, shuffle: bool, seed: int):
    idx = np.array(indices, copy=True)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        yield idx[start : start + batch_size]


def freeze_all(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


class GatedGCNNodeEncoder(nn.Module):
    """Expose frozen node embeddings from the pretrained GatedGCN encoder."""

    def __init__(self, model: GatedGCNModel):
        super().__init__()
        self.model = model

    def forward(self, batch: Batch) -> torch.Tensor:
        x = self.model.node_embed(batch.x.float())
        e = self.model.edge_embed(batch.edge_attr.float())
        for conv, bn in zip(self.model.convs, self.model.bns):
            h = conv(x, batch.edge_index, edge_attr=e)
            h = bn(h)
            h = torch.relu(h)
            h = self.model.dropout(h)
            x = x + h
        return x


def pad_node_embeddings(
    node_embeddings: torch.Tensor, batch_index: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert concatenated node embeddings to padded B x Nmax x D tensor."""
    n_graphs = int(batch_index.max().item()) + 1
    counts = torch.bincount(batch_index, minlength=n_graphs)
    max_nodes = int(counts.max().item())
    hidden_dim = node_embeddings.shape[1]
    padded = node_embeddings.new_zeros((n_graphs, max_nodes, hidden_dim))
    mask = torch.zeros(
        (n_graphs, max_nodes), dtype=torch.bool, device=node_embeddings.device
    )
    for graph_id in range(n_graphs):
        sel = batch_index == graph_id
        n = int(sel.sum().item())
        padded[graph_id, :n] = node_embeddings[sel]
        mask[graph_id, :n] = True
    return padded, mask


def predict_indices(
    graphs: list,
    indices: np.ndarray,
    encoder: GatedGCNNodeEncoder,
    readout: AdaptiveReadoutRegressor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    preds: list[np.ndarray] = []
    encoder.eval()
    readout.eval()
    with torch.no_grad():
        for sel in iter_index_batches(indices, batch_size, shuffle=False, seed=0):
            batch = Batch.from_data_list([graphs[i] for i in sel]).to(device)
            nodes = encoder(batch)
            padded, mask = pad_node_embeddings(nodes, batch.batch)
            pred = readout(padded, mask)
            preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds).astype(np.float64)


def train_fold(
    graphs: list,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    in_dim: int,
    edge_dim: int,
    encoder_params: dict,
    args: argparse.Namespace,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    base = GatedGCNModel(
        in_dim=in_dim,
        edge_dim=edge_dim,
        hidden_dim=encoder_params["hidden_dim"],
        num_layers=encoder_params["num_layers"],
        dropout=encoder_params["dropout"],
        out_dim=2,
    ).to(device)
    summary = load_pretrain_encoder(base, PRETRAIN_PATH)
    encoder = GatedGCNNodeEncoder(base).to(device)
    freeze_all(encoder)
    encoder.eval()

    readout = AdaptiveReadoutRegressor(
        input_dim=encoder_params["hidden_dim"],
        hidden_dim=args.readout_dim,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        num_seeds=args.num_seeds,
        dropout=args.dropout,
    ).to(device)
    optimizer = AdamW(readout.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    best_state = None
    best_val_mae = float("inf")
    patience = 0
    loss_start = None
    loss_end = None

    for epoch in range(args.max_epochs):
        readout.train()
        epoch_losses = []
        for sel in iter_index_batches(
            train_idx,
            args.batch_size,
            shuffle=True,
            seed=args.seed + 1000 * fold + epoch,
        ):
            batch = Batch.from_data_list([graphs[i] for i in sel]).to(device)
            target = torch.tensor(y[sel], dtype=torch.float32, device=device)
            with torch.no_grad():
                nodes = encoder(batch)
                padded, mask = pad_node_embeddings(nodes, batch.batch)
            optimizer.zero_grad(set_to_none=True)
            pred = readout(padded, mask)
            loss = criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(readout.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        epoch_loss = float(np.mean(epoch_losses))
        if loss_start is None:
            loss_start = epoch_loss
        loss_end = epoch_loss

        val_pred = predict_indices(
            graphs, val_idx, encoder, readout, args.batch_size, device
        )
        val_mae = float(np.mean(np.abs(y[val_idx] - val_pred)))
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {
                k: v.detach().cpu().clone() for k, v in readout.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

    if best_state is None:
        raise RuntimeError(f"fold {fold}: no best state")
    readout.load_state_dict(best_state)
    val_pred = predict_indices(
        graphs, val_idx, encoder, readout, args.batch_size, device
    )
    all_idx = np.arange(len(graphs))
    all_pred = predict_indices(
        graphs, all_idx, encoder, readout, args.batch_size, device
    )
    info = {
        "fold": fold,
        "epochs_run": epoch + 1,
        "best_val_mae": best_val_mae,
        "loss_start": loss_start,
        "loss_end": loss_end,
        "pretrain_best_val_loss": summary.get("pretrain_best_val_loss", -1),
    }
    del readout, encoder, base
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return val_pred, all_pred, info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="initial")
    parser.add_argument("--readout-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-record", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not PRETRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Missing GatedGCN pretrain checkpoint: {PRETRAIN_PATH}"
        )
    run_dir = OUTPUT_ROOT.joinpath(args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    encoder_params = load_pretrain_params()
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y = train_df["pec50"].to_numpy(dtype=np.float32)
    train_graphs = smiles_to_pyg(train_df["smiles"].tolist())
    test_graphs = smiles_to_pyg(test_df["smiles"].tolist())
    all_graphs = train_graphs + test_graphs
    in_dim = train_graphs[0].x.shape[1]
    edge_dim = train_graphs[0].edge_attr.shape[1]
    folds = umap_split_indices(train_df["smiles"].tolist())

    oof = np.zeros(len(train_df), dtype=np.float64)
    test_preds_per_fold = []
    fold_metrics = []
    fold_rows = []

    for fold, (tr_idx, va_idx) in enumerate(folds):
        print(f"\n[Fold {fold}] train={len(tr_idx)} val={len(va_idx)}")
        val_pred, all_pred, info = train_fold(
            all_graphs,
            y=np.concatenate([y, np.zeros(len(test_df), dtype=np.float32)]),
            train_idx=tr_idx,
            val_idx=va_idx,
            in_dim=in_dim,
            edge_dim=edge_dim,
            encoder_params=encoder_params,
            args=args,
            fold=fold,
        )
        oof[va_idx] = val_pred
        test_preds_per_fold.append(all_pred[len(train_df) :])
        metrics = compute_metrics(y[va_idx], val_pred)
        fold_metrics.append(metrics)
        fold_rows.append({**info, **{f"val_{k}": v for k, v in metrics.items()}})
        print_metrics(metrics, label=f"Fold {fold}")

    oof_metrics = compute_metrics(y, oof)
    test_pred = np.mean(test_preds_per_fold, axis=0)
    print("\nOverall OOF:")
    print_metrics(oof_metrics)
    print_fold_summary(fold_metrics)

    exp_name = f"gatedgcn_strategy6_adaptive_readout_{args.run_name}_umap"
    sub = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": test_pred,
        }
    )
    sub_path = SUBMISSION_DIR.joinpath(f"{exp_name}.csv")
    sub.to_csv(sub_path, index=False)

    residual_r = float("nan")
    try:
        from run_ensemble_calibrate_importance import load_caruana_oof_and_test

        anchor_oof, _anchor_test, _anchor_df = load_caruana_oof_and_test()
        residual_r = float(np.corrcoef(oof - y, anchor_oof - y)[0, 1])
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        print(f"Could not compute residual correlation vs current ensemble: {exc}")

    summary = pd.DataFrame(
        [
            {
                "name": exp_name,
                **oof_metrics,
                "residual_r_vs_ens_caruana_bag20": residual_r,
                "test_mean": float(test_pred.mean()),
                "test_std": float(test_pred.std()),
                "submission_path": str(sub_path.relative_to(REPO_ROOT)),
            }
        ]
    )
    summary.to_csv(run_dir.joinpath("summary.csv"), index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir.joinpath("fold_metrics.csv"), index=False)
    report = "\n".join(
        [
            "# Buterez Strategy 6 GatedGCN Report",
            "",
            f"Run name: `{args.run_name}`",
            f"Experiment: `{exp_name}`",
            f"Pretrain checkpoint: `{PRETRAIN_PATH}`",
            "",
            "## OOF Metrics",
            "",
            f"MAE: `{oof_metrics['MAE']:.6f}`",
            f"RAE: `{oof_metrics['RAE']:.6f}`",
            f"Spearman: `{oof_metrics['Spearman_R']:.6f}`",
            f"Residual r vs ens_caruana_bag20: `{residual_r:.6f}`",
            "",
            "## Fold Metrics",
            "",
            pd.DataFrame(fold_rows).to_markdown(index=False),
            "",
            "## Decision Gate",
            "",
            "Do not submit this model directly. Consider Caruana ADD only if MAE <= 0.48 or residual correlation is clearly low without major Spearman collapse.",
        ]
    )
    run_dir.joinpath("report.md").write_text(report + "\n")

    if not args.no_record:
        exp_id = record_experiment(
            name=exp_name,
            description="Buterez 2024 Strategy 6: frozen GatedGCN LF encoder + adaptive readout on pEC50",
            model_type="gatedgcn_strategy6",
            feature_set="molecular_graph_adaptive_readout",
            hyperparameters={
                "encoder_params": encoder_params,
                "pretrain_path": str(PRETRAIN_PATH),
                "args": vars(args),
            },
            fold_metrics=fold_metrics,
            submission_path=str(sub_path.relative_to(REPO_ROOT)),
            notes=f"OOF MAE={oof_metrics['MAE']:.4f}, Strategy 6 adaptive readout",
            on_conflict_replace=True,
        )
        save_oof_predictions(exp_id, oof)

    print(f"Saved report: {run_dir.joinpath('report.md')}")
    print(f"Saved submission: {sub_path}")


if __name__ == "__main__":
    main()
