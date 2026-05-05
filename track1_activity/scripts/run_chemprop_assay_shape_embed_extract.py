#!/usr/bin/env -S pixi run python
"""Extract frozen embeddings from assay-shape ChemProp pretraining."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from chemprop import data as chemprop_data  # noqa: E402

from data import DB_PARAMS  # noqa: E402
from run_chemprop_assay_shape_pretrain import build_pretrain_model  # noqa: E402

DEFAULT_CKPT_PATH = REPO_ROOT.joinpath(
    "track1_activity",
    "checkpoints",
    "chemprop_assay_shape_pretrain_seed42",
    "pretrain.pt",
)
DEFAULT_OUT_PATH = REPO_ROOT.joinpath("data", "chemprop_assay_shape_embed.parquet")


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract assay-shape embeddings")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if not args.ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    params = ckpt["params"]
    task_weights = torch.tensor(ckpt["task_weights"], dtype=torch.float32)
    model = build_pretrain_model(params, task_weights)
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    df = load_target_compounds()
    n = len(df)
    n_tasks = len(ckpt["tasks"])
    pts = [
        chemprop_data.MoleculeDatapoint.from_smi(
            smi, np.full(n_tasks, 0.0, dtype=np.float32)
        )
        for smi in df["smiles"]
    ]
    loader = chemprop_data.build_dataloader(
        chemprop_data.MoleculeDataset(pts),
        batch_size=args.batch_size,
        shuffle=False,
    )

    all_embeds = []
    with torch.no_grad():
        for batch in loader:
            bmg = batch.bmg
            bmg.to(device)
            v_d = batch.V_d.to(device) if batch.V_d is not None else None
            x_d = batch.X_d.to(device) if batch.X_d is not None else None
            emb = model.fingerprint(bmg, v_d, x_d)
            all_embeds.append(emb.detach().cpu().numpy())
    arr = np.concatenate(all_embeds, axis=0)
    hidden_dim = params["message_hidden_dim"]
    if arr.shape != (n, hidden_dim):
        raise RuntimeError(
            f"unexpected embedding shape {arr.shape}, expected {(n, hidden_dim)}"
        )

    out = pd.DataFrame(arr, columns=[f"emb_{i:03d}" for i in range(hidden_dim)])
    out.insert(0, "compound_id", df["compound_id"].to_numpy())
    out = out.set_index("compound_id")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"Saved {out.shape} to {args.out}")
    print(f"tasks: {ckpt['tasks']}")


if __name__ == "__main__":
    main()
