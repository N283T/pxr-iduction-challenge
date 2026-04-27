#!/usr/bin/env python3
"""KANO contrastive pretrain encoder embedding extraction for PXR train+test.

Loads the public KANO pretrained CMPN encoder (Fang et al., Nat Mach Intell 2023)
and extracts 300-d molecular embeddings for the 4653 PXR train+test compounds.
Saved to data/kano_embed.parquet (compound_id index).

Different inductive bias from existing pool members:
- chemprop_pretrain_embed / cheme_log2fc_pred / kermt_*: all supervised on
  PXR single_concentration log2_fc
- KANO: knowledge-graph-augmented contrastive pretrain on ZINC15 250K
  (no PXR supervision, no log2_fc objective)

Hypothesis: orthogonal inductive bias may give caruana ADD value
(`feedback_new_family_threshold_min_r_085` r ≤ 0.85 gate).

Setup (run before this script):
    cd /tmp && git clone --depth 1 https://github.com/HICAI-ZJU/KANO
    uv venv --python 3.10 /home/nagaet/.cache/kano_env_parent/kano_env
    uv pip install --python /home/nagaet/.cache/kano_env_parent/kano_env/bin/python torch==1.13.1 --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python /home/nagaet/.cache/kano_env_parent/kano_env/bin/python "setuptools<70" wheel
    uv pip install --python /home/nagaet/.cache/kano_env_parent/kano_env/bin/python --no-build-isolation torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cpu.html
    uv pip install --python /home/nagaet/.cache/kano_env_parent/kano_env/bin/python rdkit-pypi "numpy<2" pandas tqdm psycopg2-binary

Usage:
    cd /home/nagaet/ghq/github.com/HICAI-ZJU/KANO  # MUST run from KANO root (featurization.py loads initial/*.pkl
                  # via relative paths at import time)
    /home/nagaet/.cache/kano_env_parent/kano_env/bin/python /home/nagaet/pxr-iduction-challenge/track1_activity/scripts/run_kano_embed_extract.py \\
        --out /home/nagaet/pxr-iduction-challenge/data/kano_embed.parquet
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

# KANO repo on sys.path
sys.path.insert(0, "/home/nagaet/ghq/github.com/HICAI-ZJU/KANO")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psycopg2  # noqa: E402
import torch  # noqa: E402

from chemprop.models import build_pretrain_model  # noqa: E402

DEFAULT_CKPT = "/home/nagaet/ghq/github.com/HICAI-ZJU/KANO/dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl"
DEFAULT_OUT = "/home/nagaet/pxr-iduction-challenge/data/kano_embed.parquet"

DB_PARAMS = dict(host="/tmp", port=5433, dbname="pxr_challenge")


def load_pxr_compounds() -> pd.DataFrame:
    """Load 4653 PXR train+test SMILES + compound_id (matches existing
    chemprop_pretrain_embed.parquet row order).
    """
    sql = """
    SELECT c.id AS compound_id, c.std_smiles AS smiles
    FROM compounds c
    INNER JOIN (
        SELECT compound_id FROM train_activity
        UNION
        SELECT compound_id FROM test_activity
    ) ts ON ts.compound_id = c.id
    WHERE c.std_smiles IS NOT NULL
    ORDER BY c.id
    """
    with psycopg2.connect(**DB_PARAMS) as conn:
        return pd.read_sql(sql, conn)


def build_kano_args() -> Namespace:
    """Minimal Namespace matching KANO's parse_train_args defaults for
    the pretrained CMPN checkpoint (atom_fdim=133, bond_fdim=147,
    hidden_size=300, depth=3 from state_dict shapes).
    """
    args = Namespace()
    # Encoder architecture (matches state_dict)
    args.hidden_size = 300
    args.depth = 3
    args.dropout = 0.0
    args.activation = "ReLU"
    args.bias = False
    args.atom_messages = False
    args.undirected = False
    args.features_only = False
    args.use_input_features = False
    # Mode: pretrain (the simplest path, no functional_prompt / KG)
    args.step = "pretrain"
    # Build/predict-time
    args.encoder_head = "CMPN"
    args.dataset_type = "regression"
    args.num_tasks = 1
    args.cuda = False
    args.no_cuda = True
    args.gpu = None
    # FFN args (used by build_pretrain_model.create_ffn even if we only use encoder)
    args.ffn_num_layers = 2
    args.ffn_hidden_size = args.hidden_size // 2
    args.features_size = 0
    args.use_input_features = False
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="KANO embed extract for PXR")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="KANO pretrained .pkl")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output parquet path")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-compounds",
        type=int,
        default=None,
        help="Cap number of compounds for smoke test",
    )
    args_cli = parser.parse_args()

    # Ensure cwd is /home/nagaet/ghq/github.com/HICAI-ZJU/KANO so featurization.py can load initial/*.pkl
    if Path.cwd() != Path("/home/nagaet/ghq/github.com/HICAI-ZJU/KANO"):
        print(
            f"WARNING: cwd is {Path.cwd()}, expected /home/nagaet/ghq/github.com/HICAI-ZJU/KANO. Continuing..."
        )
        os.chdir("/home/nagaet/ghq/github.com/HICAI-ZJU/KANO")

    print("Loading PXR train+test compounds")
    df = load_pxr_compounds()
    if args_cli.max_compounds is not None:
        df = df.head(args_cli.max_compounds)
    print(f"  n={len(df)}")
    smiles_list = df["smiles"].tolist()
    cids = df["compound_id"].astype(int).tolist()

    print(f"Building KANO model (CMPN pretrain, hidden=300)")
    args = build_kano_args()
    model = build_pretrain_model(args, encoder_name="CMPNN")

    print(f"Loading state_dict: {args_cli.ckpt}")
    sd = torch.load(args_cli.ckpt, map_location="cpu", weights_only=False)
    # The state_dict has 'encoder.*' keys; we need to load only into encoder
    encoder_sd = {
        k.replace("encoder.", "", 1): v
        for k, v in sd.items()
        if k.startswith("encoder.")
    }
    missing, unexpected = model.encoder.encoder.load_state_dict(
        encoder_sd, strict=False
    )
    if missing:
        print(
            f"  MISSING ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    if unexpected:
        print(f"  UNEXPECTED ({len(unexpected)}): {unexpected[:5]}")

    model.eval()

    print(f"Extracting embeddings (batch_size={args_cli.batch_size})")
    # KANO encoder accepts raw SMILES list (mol2graph called inside it)
    embs = []
    n = len(smiles_list)
    bs = args_cli.batch_size
    for start in range(0, n, bs):
        end = min(start + bs, n)
        batch_smi = smiles_list[start:end]
        with torch.no_grad():
            batch_embs = model.encoder(args.step, False, batch_smi, None)
        batch_embs_np = batch_embs.cpu().numpy()
        embs.append(batch_embs_np)
        if (start // bs) % 10 == 0:
            print(
                f"  batch {start // bs + 1}/{(n + bs - 1) // bs}: shape={batch_embs_np.shape}"
            )

    embs = np.vstack(embs)
    assert embs.shape == (n, 300), f"unexpected emb shape {embs.shape}"

    out_df = pd.DataFrame(embs, index=pd.Index(cids, name="compound_id"))
    out_df.columns = [f"kano_{i}" for i in range(embs.shape[1])]

    out_path = Path(args_cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path)
    print(f"\nSaved: {out_path}  shape={out_df.shape}")
    print(
        f"  mean={embs.mean():.4f}  std={embs.std():.4f}  "
        f"min={embs.min():.4f}  max={embs.max():.4f}"
    )


if __name__ == "__main__":
    main()
