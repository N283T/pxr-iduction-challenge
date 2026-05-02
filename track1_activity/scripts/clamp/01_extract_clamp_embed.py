#!/usr/bin/env python
"""Extract CLAMP encode_smiles embeddings for all PXR challenge compounds.

Output: data/clamp_embed.parquet with columns [compound_id, embedding (np.array)].

Note: must run inside the isolated CLAMP env at
    ~/ghq/github.com/ml-jku/clamp/.venv

Activate with:
    cd ~/ghq/github.com/ml-jku/clamp && source .venv/bin/activate
    python <repo>/track1_activity/scripts/clamp/01_extract_clamp_embed.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import torch
from tqdm import tqdm


def fetch_compounds(host: str, port: int, dbname: str) -> pd.DataFrame:
    conn = psycopg2.connect(dbname=dbname, host=host, port=port)
    df = pd.read_sql(
        "SELECT id AS compound_id, std_smiles FROM compounds "
        "WHERE std_smiles IS NOT NULL ORDER BY id",
        conn,
    )
    conn.close()
    return df


def encode_in_batches(model, smiles: list[str], batch_size: int = 256) -> np.ndarray:
    out_chunks: list[np.ndarray] = []
    for i in tqdm(range(0, len(smiles), batch_size), desc="encode"):
        batch = smiles[i : i + batch_size]
        with torch.no_grad():
            emb = model.encode_smiles(batch).cpu().numpy().astype(np.float32)
        out_chunks.append(emb)
    return np.vstack(out_chunks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/clamp_embed.parquet")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--db-host", default="/tmp")
    parser.add_argument("--db-port", type=int, default=5433)
    parser.add_argument("--db-name", default="pxr_challenge")
    parser.add_argument(
        "--repo-root",
        default=os.environ.get("PXR_REPO", "/home/nagaet/pxr-iduction-challenge"),
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    out_path = repo_root.joinpath(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import clamp

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"loading PretrainedCLAMP on {device}...", file=sys.stderr)
    model = clamp.CLAMP(device=device)
    model.to(device)
    model.eval()

    df = fetch_compounds(args.db_host, args.db_port, args.db_name)
    print(f"fetched {len(df)} compounds", file=sys.stderr)

    embed = encode_in_batches(model, df["std_smiles"].tolist(), args.batch_size)
    print(f"embed shape: {embed.shape}", file=sys.stderr)

    out_df = pd.DataFrame(
        {
            "compound_id": df["compound_id"].to_numpy(),
            "embedding": list(embed),
        }
    )
    out_df.to_parquet(out_path, index=False)
    print(f"wrote {out_path} ({embed.shape[0]} rows × {embed.shape[1]} dim)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
