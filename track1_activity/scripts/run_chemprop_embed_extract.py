#!/usr/bin/env -S pixi run python
"""Extract 256d per-compound embeddings from the pretrain.pt MPNN.

Phase 0 of Buterez 2024 strategy-3 (LF embedding as input feature for
a HF tabular regressor). Loads the chemprop pretrain checkpoint built
by run_chemprop_pretrain.py and produces a parquet of molecule-level
fingerprints (the output of ``MPNN.fingerprint``, i.e. post-aggregation
post-batchnorm, pre-predictor).

Output: ``data/chemprop_pretrain_embed.parquet`` indexed by compound_id
with columns ``emb_000 .. emb_255``. Covers the union of train + test
(4653 compounds). Downstream TabPFN / LGBM consumers reindex by
compound_id.

Usage:
    pixi run python track1_activity/scripts/run_chemprop_embed_extract.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from chemprop import data as chemprop_data  # noqa: E402
from chemprop import models, nn  # noqa: E402
from chemprop.nn.metrics import MSE  # noqa: E402

from data import DB_PARAMS  # noqa: E402


CKPT_PATH = REPO_ROOT.joinpath(
    "track1_activity", "checkpoints", "chemprop_pretrain", "pretrain.pt"
)
OUT_PATH = REPO_ROOT.joinpath("data", "chemprop_pretrain_embed.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

AGG_REGISTRY = {
    "mean": nn.MeanAggregation,
    "sum": nn.SumAggregation,
    "norm": nn.NormAggregation,
}


def build_pretrain_model(params: dict):
    mp = nn.BondMessagePassing(
        d_h=params["message_hidden_dim"],
        depth=params["depth"],
        dropout=params["mp_dropout"],
        activation=params["activation"],
    )
    agg = AGG_REGISTRY[params["aggregation"]]()
    criterion = MSE(task_weights=torch.tensor([1.0, 0.5], dtype=torch.float32))
    ffn = nn.RegressionFFN(
        n_tasks=2,
        input_dim=mp.output_dim,
        hidden_dim=params["ffn_hidden_dim"],
        n_layers=params["ffn_num_layers"],
        dropout=params["ffn_dropout"],
        criterion=criterion,
    )
    return models.MPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        batch_norm=True,
        warmup_epochs=params["warmup_epochs"],
        init_lr=params["learning_rate"],
        max_lr=params["learning_rate"] * params["lr_ratio"],
        final_lr=params["learning_rate"] * 0.1,
    )


def load_target_compounds() -> pd.DataFrame:
    """Return DataFrame with compound_id + std_smiles for train + test.

    Only these compounds will appear in tabular training pipelines; the
    remaining 8483 single-only compounds are irrelevant here.
    """
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
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"missing pretrain ckpt: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    params = ckpt["params"]
    hidden_dim = params["message_hidden_dim"]
    print(f"Loaded pretrain ckpt, params: {params}")

    df = load_target_compounds()
    n = len(df)
    print(f"Target compounds: {n}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_pretrain_model(params).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Model on {device}, loaded state_dict")

    # Build datapoints with dummy targets (fingerprint() does not use them).
    pts = [
        chemprop_data.MoleculeDatapoint.from_smi(smi, np.full(2, 0.0, dtype=np.float32))
        for smi in df["smiles"]
    ]
    dataset = chemprop_data.MoleculeDataset(pts)
    loader = chemprop_data.build_dataloader(dataset, batch_size=256, shuffle=False)

    all_embeds: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            # chemprop 2.x TrainingBatch is a NamedTuple (bmg, V_d, X_d, Y, w, lt_mask, gt_mask).
            # BatchMolGraph.to(device) moves in place and RETURNS None -- do not reassign.
            bmg = batch.bmg
            bmg.to(device)
            V_d = batch.V_d.to(device) if batch.V_d is not None else None
            X_d = batch.X_d.to(device) if batch.X_d is not None else None
            fp = model.fingerprint(bmg, V_d, X_d)  # (B, 256)
            all_embeds.append(fp.detach().cpu().numpy())

    emb = np.concatenate(all_embeds, axis=0)
    assert emb.shape == (n, hidden_dim), f"emb shape {emb.shape} != ({n}, {hidden_dim})"

    cols = [f"emb_{i:03d}" for i in range(hidden_dim)]
    out_df = pd.DataFrame(emb, columns=cols)
    out_df.insert(0, "compound_id", df["compound_id"].values)
    out_df = out_df.set_index("compound_id")
    out_df.to_parquet(OUT_PATH)

    print(f"Saved {out_df.shape} embeddings to {OUT_PATH}")
    print("  mean abs =", float(np.mean(np.abs(emb))))
    print("  std per col =", float(np.mean(np.std(emb, axis=0))))


if __name__ == "__main__":
    main()
