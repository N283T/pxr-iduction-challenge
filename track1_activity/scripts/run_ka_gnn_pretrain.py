#!/usr/bin/env python
"""Pretrain KA-GNN on single-concentration log2_fc auxiliary targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Batch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402
from ka_gnn import FourierKAGNNModel, PykanSAGEModel  # noqa: E402
from pyg_training import smiles_to_pyg_list  # noqa: E402

CKPT_BASE = REPO_ROOT.joinpath("track1_activity", "checkpoints")
DEFAULT_CKPT_DIR = CKPT_BASE.joinpath("ka_gnn_pretrain")


def load_pretrain_data() -> tuple[list[str], np.ndarray, list[int]]:
    sql = """
    SELECT c.id AS compound_id,
           c.std_smiles AS smiles,
           agg.log2fc_8p25,
           agg.log2fc_33
    FROM compounds c
    LEFT JOIN (
      SELECT compound_id,
        AVG(CASE WHEN concentration_m BETWEEN 8.2e-6 AND 8.3e-6
                 THEN log2_fc_estimate END) AS log2fc_8p25,
        AVG(CASE WHEN concentration_m BETWEEN 3.28e-5 AND 3.32e-5
                 THEN log2_fc_estimate END) AS log2fc_33
      FROM single_concentration
      GROUP BY compound_id
    ) agg ON agg.compound_id = c.id
    WHERE c.std_smiles IS NOT NULL
    ORDER BY c.id
    """
    with psycopg2.connect(**DB_PARAMS) as conn:
        df = pd.read_sql(sql, conn)
    return (
        df["smiles"].tolist(),
        df[["log2fc_8p25", "log2fc_33"]].to_numpy(dtype=np.float32),
        df["compound_id"].astype(int).tolist(),
    )


def compute_feature_stats(graphs: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.cat([g.x.float() for g in graphs], dim=0)
    edge = torch.cat([g.edge_attr.float() for g in graphs], dim=0)
    return x.mean(dim=0), torch.clamp(x.std(dim=0), min=1.0), edge.mean(dim=0), torch.clamp(edge.std(dim=0), min=1.0)


def build_model(params: dict, in_dim: int, edge_dim: int, device: torch.device):
    if params["model_type"] == "pykan_sage":
        model = PykanSAGEModel(
            in_dim=in_dim,
            edge_dim=edge_dim,
            hidden_dim=params["hidden_dim"],
            out_dim=params["out_dim"],
            grid_size=params["grid_size"],
            num_layers=params["num_layers"],
            pooling=params["pooling"],
            dropout=params["dropout"],
            kan_bottleneck=params["kan_bottleneck"],
            spline_order=params["spline_order"],
            seed=params["seed"],
        )
    else:
        model = FourierKAGNNModel(
            in_dim=in_dim,
            edge_dim=edge_dim,
            hidden_dim=params["hidden_dim"],
            out_dim=params["out_dim"],
            grid_size=params["grid_size"],
            num_layers=params["num_layers"],
            pooling=params["pooling"],
            dropout=params["dropout"],
            use_bias=True,
            aggr=params["aggr"],
        )
    return model.to(device)


class PretrainHead(torch.nn.Module):
    def __init__(self, encoder, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, out_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(out_dim, 2),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        emb = self.encoder.encode_graph(
            batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch
        )
        return self.head(emb)


def iter_batches(indices: np.ndarray, batch_size: int, shuffle: bool, seed: int):
    idx = np.array(indices, copy=True)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        yield idx[start : start + batch_size]


def masked_weighted_mse(pred, target, task_weights):
    mask = torch.isfinite(target)
    if not mask.any():
        return pred.sum() * 0.0
    diff2 = (pred - torch.nan_to_num(target, nan=0.0)) ** 2
    weighted = diff2 * task_weights.view(1, -1)
    return weighted[mask].mean()


def predict_loss(graphs, targets, indices, model, batch_size, task_weights, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for sel in iter_batches(indices, batch_size, shuffle=False, seed=0):
            batch = Batch.from_data_list([graphs[int(i)] for i in sel]).to(device)
            y = torch.tensor(targets[sel], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = masked_weighted_mse(pred, y, task_weights)
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", choices=["fourier", "pykan_sage"], default="fourier")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--out-dim", type=int, default=64)
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--spline-order", type=int, default=3)
    parser.add_argument("--kan-bottleneck", type=int, default=5)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--pooling", choices=["mean", "sum", "max"], default="mean")
    parser.add_argument("--aggr", choices=["sum", "mean"], default="mean")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--w-8p25", type=float, default=1.0)
    parser.add_argument("--w-33", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="debug limit on compound count")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    smiles, targets, compound_ids = load_pretrain_data()
    if args.limit is not None:
        smiles = smiles[: args.limit]
        targets = targets[: args.limit]
        compound_ids = compound_ids[: args.limit]
    means = np.zeros(2, dtype=np.float32)
    stds = np.ones(2, dtype=np.float32)
    for i in range(2):
        valid = np.isfinite(targets[:, i])
        means[i] = float(np.mean(targets[valid, i]))
        stds[i] = float(np.std(targets[valid, i]))
        if stds[i] < 1e-6:
            stds[i] = 1.0
    targets_z = (targets - means) / stds

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(smiles))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * args.val_frac))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    print("Converting pretrain SMILES to PyG graphs...")
    graphs = smiles_to_pyg_list(smiles)
    in_dim = int(graphs[0].x.shape[1])
    edge_dim = int(graphs[0].edge_attr.shape[1])
    params = vars(args).copy()
    params["ckpt_dir"] = str(args.ckpt_dir)
    params["in_dim"] = in_dim
    params["edge_dim"] = edge_dim
    print(f"KA-GNN pretrain: n={len(graphs)} train={len(train_idx)} val={len(val_idx)} params={params}")

    encoder = build_model(params, in_dim, edge_dim, device)
    stats = compute_feature_stats([graphs[int(i)] for i in train_idx])
    encoder.set_feature_standardization(*(s.to(device) for s in stats))
    model = PretrainHead(encoder, hidden_dim=args.hidden_dim, out_dim=args.out_dim, dropout=args.head_dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.min_lr)
    task_weights = torch.tensor([args.w_8p25, args.w_33], dtype=torch.float32, device=device)

    best_state = None
    best_val_loss = float("inf")
    patience = 0
    rows = []
    for epoch in range(args.max_epochs):
        model.train()
        losses = []
        for sel in iter_batches(train_idx, args.batch_size, shuffle=True, seed=args.seed + epoch):
            batch = Batch.from_data_list([graphs[int(i)] for i in sel]).to(device)
            y = torch.tensor(targets_z[sel], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            loss = masked_weighted_mse(pred, y, task_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        train_loss = float(np.mean(losses))
        val_loss = predict_loss(graphs, targets_z, val_idx, model, args.batch_size, task_weights, device)
        rows.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch + 1:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("No best pretrain state")
    model.load_state_dict(best_state)
    state_path = args.ckpt_dir.joinpath("pretrain.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "params": params,
            "target_means": means.tolist(),
            "target_stds": stds.tolist(),
            "task_weights": [args.w_8p25, args.w_33],
            "compound_ids": compound_ids,
            "best_val_loss": best_val_loss,
            "epochs_run": len(rows),
        },
        state_path,
    )
    pd.DataFrame(rows).to_csv(args.ckpt_dir.joinpath("history.csv"), index=False)
    args.ckpt_dir.joinpath("pretrain_meta.json").write_text(
        json.dumps(
            {
                "params": params,
                "target_means": means.tolist(),
                "target_stds": stds.tolist(),
                "best_val_loss": best_val_loss,
                "epochs_run": len(rows),
                "n_compounds": len(smiles),
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "state_path": str(state_path),
            },
            indent=2,
        )
    )
    print(f"saved: {state_path}")
    print(f"best_val_loss={best_val_loss:.6f}, epochs_run={len(rows)}")


if __name__ == "__main__":
    main()
