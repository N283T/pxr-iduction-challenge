#!/usr/bin/env -S pixi run python
"""Extract frozen ChemProp MTR encoder embeddings for all 13,136 compounds.

The 2 NaN-dropped compounds (1657, 8624) are included here — their SMILES
are valid, so the forward pass succeeds even though they were excluded from
the pretrain target matrix.

Embeddings are extracted via model.fingerprint(bmg), which runs:
    message_passing -> aggregation -> batch_norm
and stops before the FFN head. Output dim = message_hidden_dim = 256.

Usage:
    pixi run python track1_activity/scripts/run_chemprop_mtr_extract.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
# Also expose the scripts directory so we can import from run_chemprop_mtr_pretrain
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from data import DB_PARAMS  # noqa: E402
from run_chemprop_mtr_pretrain import build_model, make_dataloader  # noqa: E402


def load_all_compounds() -> pd.DataFrame:
    """Load all 13,136 compounds ordered by compound_id."""
    with psycopg2.connect(**DB_PARAMS) as conn:
        df = pd.read_sql(
            "SELECT id AS compound_id, std_smiles FROM compounds ORDER BY id",
            conn,
        )
    return df


def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract frozen ChemProp MTR embeddings for all compounds."
    )
    p.add_argument("--seed", type=int, default=42, help="Seed used during pretrain.")
    args = p.parse_args()

    ckpt_dir = REPO_ROOT.joinpath(f"models/chemprop_mtr_seed{args.seed}")
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {ckpt_dir}. "
            "Run run_chemprop_mtr_pretrain.py first."
        )

    meta = json.loads(ckpt_dir.joinpath("meta.json").read_text())
    n_tasks = len(meta["descriptor_names"])
    embed_dim = meta["params"]["message_hidden_dim"]
    print(f"Checkpoint: seed={args.seed}, n_tasks={n_tasks}, embed_dim={embed_dim}")

    # Reconstruct model and load weights
    model = build_model(meta["params"], n_tasks=n_tasks)
    state_dict = torch.load(
        ckpt_dir.joinpath("pretrain.pt"), map_location="cpu", weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval().cuda()

    compounds = load_all_compounds()
    print(f"Extracting embeddings for {len(compounds)} compounds")

    # Dummy targets required by make_dataloader (values ignored at inference)
    dummy_targets = np.zeros((len(compounds), n_tasks), dtype=np.float32)
    loader = make_dataloader(
        compounds["std_smiles"].tolist(),
        dummy_targets,
        batch_size=256,
        shuffle=False,
    )

    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            bmg = batch.bmg.to("cuda")
            # fingerprint() = message_passing -> aggregation -> batch_norm
            # Returns graph embedding of shape (batch_size, message_hidden_dim)
            # before the FFN prediction head.
            v_d = batch.V_d.to("cuda") if batch.V_d is not None else None
            graph_emb = model.fingerprint(bmg, v_d)
            chunks.append(graph_emb.cpu().numpy())

    embeddings = np.concatenate(chunks, axis=0)
    print(f"Embedding shape: {embeddings.shape}")

    assert not np.isnan(embeddings).any(), "NaN detected in embeddings"
    assert not np.isinf(embeddings).any(), "Inf detected in embeddings"

    # Build output DataFrame with compound_id as index
    col_names = [f"chemprop_mtr_{i}" for i in range(embeddings.shape[1])]
    out_df = pd.DataFrame(
        embeddings,
        index=compounds["compound_id"].values,
        columns=col_names,
    )
    out_df.index.name = "compound_id"

    out_path = REPO_ROOT.joinpath(
        f"data/chemprop_mtr_embedding_seed{args.seed}.parquet"
    )
    out_df.to_parquet(out_path)
    print(f"Saved {out_path}  shape={embeddings.shape}")


if __name__ == "__main__":
    main()
