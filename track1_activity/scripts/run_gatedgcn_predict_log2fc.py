"""Predict log2_fc @ 8.25uM / 33uM for 4653 train+test compounds using
the GatedGCN pretrain checkpoint.

Phase 1 of issue #115 (log2fc_pred ensembling). Mirrors
run_chemprop_predict_log2fc.py but for the GatedGCN (PyG) pretrain.
Unlike run_gatedgcn_embed_extract.py, does NOT replace the FFN head;
keeps the full forward path to get the 2-head z-scored log2_fc
prediction, then un-z-scores using the pretrain target_means/stds.

Output: data/gatedgcn_pretrain_log2fc_predictions.parquet
        index=compound_id, columns=[log2fc_8p25_pred, log2fc_33_pred]

Usage:
    pixi run python track1_activity/scripts/run_gatedgcn_predict_log2fc.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import torch
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
OUT_PATH = REPO_ROOT.joinpath("data", "gatedgcn_pretrain_log2fc_predictions.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_target_compounds() -> pd.DataFrame:
    """Union of train_activity + test_activity compounds (4,653 unique)."""
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


def smiles_to_pyg(smiles_list: list[str]) -> list:
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
        out_dim=2,
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

    return model.to(DEVICE).eval()


@torch.no_grad()
def predict_all(model: GatedGCNModel, graphs: list) -> np.ndarray:
    n = len(graphs)
    outs: list[np.ndarray] = []
    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, n, BATCH_SIZE):
        batch = Batch.from_data_list(graphs[i : i + BATCH_SIZE]).to(DEVICE)
        preds = model(
            batch.x.float(),
            batch.edge_index,
            batch.edge_attr.float(),
            batch.batch,
        )
        outs.append(preds.cpu().numpy())
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  batch {i // BATCH_SIZE + 1} / {n_batches}")
    return np.concatenate(outs, axis=0).astype(np.float32)


def main() -> None:
    df = load_target_compounds()
    n = len(df)
    print(f"Loaded {n} compounds (train + test union)")

    ckpt = torch.load(PRETRAIN_PATH, map_location="cpu", weights_only=False)
    means = np.asarray(ckpt["target_means"], dtype=np.float32)
    stds = np.asarray(ckpt["target_stds"], dtype=np.float32)
    print(f"  means={means.tolist()}, stds={stds.tolist()}")

    print("Converting SMILES to PyG graphs...")
    graphs = smiles_to_pyg(df["smiles"].tolist())

    print(f"Loading GatedGCN pretrain checkpoint from {PRETRAIN_PATH}")
    model = load_model()

    print("Predicting log2_fc (z-scored)...")
    preds_z = predict_all(model, graphs)
    assert preds_z.shape == (n, 2)

    preds_raw = preds_z * stds + means
    out = pd.DataFrame(
        {
            "compound_id": df["compound_id"].values,
            "log2fc_8p25_pred": preds_raw[:, 0],
            "log2fc_33_pred": preds_raw[:, 1],
        }
    ).set_index("compound_id")
    out.to_parquet(OUT_PATH)

    print(f"Saved {out.shape} to {OUT_PATH}")
    print(
        f"  log2fc_8p25_pred: mean={out.log2fc_8p25_pred.mean():.3f} "
        f"std={out.log2fc_8p25_pred.std():.3f}"
    )
    print(
        f"  log2fc_33_pred:   mean={out.log2fc_33_pred.mean():.3f} "
        f"std={out.log2fc_33_pred.std():.3f}"
    )


if __name__ == "__main__":
    main()
