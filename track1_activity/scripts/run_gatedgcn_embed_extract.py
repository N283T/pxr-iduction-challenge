"""Extract 512d per-compound graph-pooled embeddings from the GatedGCN
pretrain checkpoint.

Phase 2 of Buterez 2024 strategy-3 for GatedGCN. Loads the pretrain
checkpoint produced by run_gatedgcn_pretrain_finetune.py --phase
pretrain, replaces the FFN head with nn.Identity() so forward returns
the 512d global_mean_pool output (post-conv stack, pre-FFN), then
batch-extracts embeddings for all 13,136 compounds.

Output: data/gatedgcn_pretrain_embed.parquet (index=compound_id,
columns=emb_0000..emb_0511, float32).

Usage:
    pixi run python track1_activity/scripts/run_gatedgcn_embed_extract.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.utils import from_smiles

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from data import DB_PARAMS  # noqa: E402
from run_gatedgcn_pretrain_finetune import GatedGCNModel  # noqa: E402

CKPT_DIR = REPO_ROOT.joinpath("track1_activity", "checkpoints", "gatedgcn_pretrain")
PRETRAIN_PATH = CKPT_DIR.joinpath("pretrain.pt")
META_PATH = CKPT_DIR.joinpath("pretrain_meta.json")
OUT_PATH = REPO_ROOT.joinpath("data", "gatedgcn_pretrain_embed.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_compounds() -> tuple[list[int], list[str]]:
    sql = """
    SELECT id AS compound_id, std_smiles AS smiles
    FROM compounds
    WHERE std_smiles IS NOT NULL
    ORDER BY id
    """
    with psycopg2.connect(**DB_PARAMS) as conn:
        df = pd.read_sql(sql, conn)
    return df["compound_id"].astype(int).tolist(), df["smiles"].tolist()


def smiles_to_pyg(smiles_list: list[str]):
    graphs = []
    for i, smi in enumerate(smiles_list):
        g = from_smiles(smi)
        if g.x is None or g.x.shape[0] == 0:
            raise ValueError(f"SMILES[{i}] produced empty graph: {smi}")
        graphs.append(g)
    return graphs


def load_model() -> GatedGCNModel:
    with META_PATH.open() as f:
        meta = json.load(f)
    params = meta["params"]

    probe = from_smiles("CCO")
    in_dim = probe.x.shape[1]
    edge_dim = probe.edge_attr.shape[1]

    model = GatedGCNModel(
        in_dim=in_dim,
        edge_dim=edge_dim,
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        out_dim=2,  # pretrain had 2-head log2_fc
    )
    ckpt = torch.load(PRETRAIN_PATH, map_location="cpu", weights_only=False)
    state = (
        ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    )
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        print(
            f"load_state_dict: missing={result.missing_keys} "
            f"unexpected={result.unexpected_keys}"
        )

    # Replace the FFN head with Identity so forward returns the 512d
    # global_mean_pool output (post-conv stack, pre-FFN projection).
    model.ffn = nn.Identity()

    return model.to(DEVICE).eval()


@torch.no_grad()
def extract_embeddings(model: GatedGCNModel, graphs: list) -> np.ndarray:
    n = len(graphs)
    outs: list[np.ndarray] = []
    for i in range(0, n, BATCH_SIZE):
        batch = Batch.from_data_list(graphs[i : i + BATCH_SIZE]).to(DEVICE)
        emb = model(
            batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch
        )
        outs.append(emb.cpu().numpy())
        if (i // BATCH_SIZE) % 10 == 0:
            print(
                f"  batch {i // BATCH_SIZE + 1} / {(n + BATCH_SIZE - 1) // BATCH_SIZE}"
            )
    return np.concatenate(outs, axis=0).astype(np.float32)


def main() -> None:
    cids, smiles_list = load_compounds()
    print(f"Loaded {len(cids)} compounds")

    print("Converting SMILES to PyG graphs...")
    graphs = smiles_to_pyg(smiles_list)

    print(f"Loading GatedGCN pretrain checkpoint from {PRETRAIN_PATH}")
    model = load_model()

    print("Extracting embeddings...")
    emb = extract_embeddings(model, graphs)
    print(f"  shape: {emb.shape}  dtype: {emb.dtype}")
    print(f"  NaN count: {np.isnan(emb).sum()}")

    cols = [f"emb_{i:04d}" for i in range(emb.shape[1])]
    df = pd.DataFrame(emb, index=pd.Index(cids, name="compound_id"), columns=cols)
    df.to_parquet(OUT_PATH)
    print(f"Wrote {OUT_PATH}  shape {df.shape}")


if __name__ == "__main__":
    main()
