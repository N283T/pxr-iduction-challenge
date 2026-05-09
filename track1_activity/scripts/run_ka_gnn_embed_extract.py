#!/usr/bin/env python
"""Extract frozen graph embeddings from a pretrained KA-GNN checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import torch
from torch_geometric.data import Batch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402
from ka_gnn import FourierKAGNNModel, PykanSAGEModel  # noqa: E402
from pyg_training import smiles_to_pyg_list  # noqa: E402

DEFAULT_CKPT = REPO_ROOT.joinpath("track1_activity", "checkpoints", "ka_gnn_pretrain", "pretrain.pt")
DEFAULT_OUT = REPO_ROOT.joinpath("data", "ka_gnn_pretrain_embed.parquet")


def load_target_compounds() -> pd.DataFrame:
    sql = """
    SELECT DISTINCT c.id AS compound_id, c.std_smiles AS smiles
    FROM compounds c
    WHERE c.id IN (
      SELECT compound_id FROM train_activity
      UNION
      SELECT compound_id FROM test_activity
    )
      AND c.std_smiles IS NOT NULL
    ORDER BY c.id
    """
    with psycopg2.connect(**DB_PARAMS) as conn:
        return pd.read_sql(sql, conn)


def build_encoder(params: dict, device: torch.device):
    in_dim = int(params["in_dim"])
    edge_dim = int(params["edge_dim"])
    if params["model_type"] == "pykan_sage":
        encoder = PykanSAGEModel(
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
        encoder = FourierKAGNNModel(
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
    return encoder.to(device)


def iter_batches(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        yield np.arange(start, min(n, start + batch_size))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not args.ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint: {args.ckpt}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    params = dict(ckpt["params"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    encoder = build_encoder(params, device)
    encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
    encoder.eval()

    df = load_target_compounds()
    graphs = smiles_to_pyg_list(df["smiles"].tolist())
    embeds = []
    with torch.no_grad():
        for idx in iter_batches(len(graphs), args.batch_size):
            batch = Batch.from_data_list([graphs[int(i)] for i in idx]).to(device)
            emb = encoder.encode_graph(
                batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch
            )
            embeds.append(emb.detach().cpu().numpy())
    emb_arr = np.concatenate(embeds, axis=0).astype(np.float32)
    cols = [f"emb_{i:03d}" for i in range(emb_arr.shape[1])]
    out_df = pd.DataFrame(emb_arr, columns=cols)
    out_df.insert(0, "compound_id", df["compound_id"].astype(int).values)
    out_df = out_df.set_index("compound_id")
    out_df.to_parquet(args.out)
    print(f"saved {out_df.shape} embeddings to {args.out}")
    print(f"mean_abs={float(np.mean(np.abs(emb_arr))):.6f}, std={float(np.std(emb_arr)):.6f}")


if __name__ == "__main__":
    main()
