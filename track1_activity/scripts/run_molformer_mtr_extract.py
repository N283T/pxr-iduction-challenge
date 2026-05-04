#!/usr/bin/env -S pixi run python
"""Extract frozen MoLFormer-c3 MTR encoder embeddings for all 13,136 compounds (Variant M).

The 2 NaN-dropped compounds (1657, 8624) are included here — their SMILES
are valid, so the forward pass succeeds even though they were excluded from
the pretrain target matrix.

Embeddings are extracted via mean-pooled last_hidden_state BEFORE the MLP
head (same 'pooled' tensor as in MolformerMTRModel.forward). Output dim = 768
(MoLFormer-c3-1.1B hidden size).

Usage:
    pixi run python track1_activity/scripts/run_molformer_mtr_extract.py --seed 42
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
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(SCRIPTS_DIR))

from data import DB_PARAMS  # noqa: E402
from run_molformer_mtr_pretrain import MolformerMTRModel, SmilesMTRDataset  # noqa: E402


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
        description="Extract frozen MoLFormer-c3 MTR embeddings for all compounds (Variant M)."
    )
    p.add_argument("--seed", type=int, default=42, help="Seed used during pretrain.")
    args = p.parse_args()

    ckpt_dir = REPO_ROOT.joinpath(f"models/molformer_c3_mtr_seed{args.seed}")

    meta_path = ckpt_dir.joinpath("meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"meta.json not found in {ckpt_dir}. "
            "Was run_molformer_mtr_pretrain.py completed successfully? "
            f"Expected path: {meta_path}"
        )
    meta = json.loads(meta_path.read_text())

    ckpt_path = ckpt_dir.joinpath("pretrain.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"pretrain.pt not found in {ckpt_dir}. "
            "Was run_molformer_mtr_pretrain.py completed successfully? "
            f"Expected path: {ckpt_path}"
        )

    n_tasks = len(meta["descriptor_names"])
    backbone_name = meta["backbone_name"]
    peft_method = meta["peft_method"]
    peft_params = meta["peft_params"]
    head_dim = meta["head_dim"]

    print(
        f"Checkpoint: seed={args.seed}, backbone={backbone_name}, "
        f"n_tasks={n_tasks}, head_dim={head_dim}"
    )

    # Load state_dict — trust our own checkpoint, weights_only=False required
    # because the PEFT-wrapped module may contain HF objects outside the
    # strict weights_only=True allowlist.
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Guard: reject Lightning checkpoints (they wrap the state_dict in a dict)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        raise ValueError(
            "pretrain.pt looks like a Lightning checkpoint (contains 'state_dict' key). "
            "Expected a raw MTR state_dict produced by run_molformer_mtr_pretrain.py."
        )

    # Reconstruct model with same architecture used at pretrain time
    model = MolformerMTRModel(
        backbone_name=backbone_name,
        peft_method=peft_method,
        peft_params=peft_params,
        n_tasks=n_tasks,
        head_dim=head_dim,
        head_dropout=0.0,  # eval mode, dropout disabled anyway; 0.0 for clarity
    )
    model.load_state_dict(state_dict)
    model.eval().cuda()

    # Load tokenizer from backbone HF id
    from peft_backbones import get_backbone  # noqa: E402 (lazy import for --help speed)

    backbone_meta = get_backbone(backbone_name)
    tokenizer = AutoTokenizer.from_pretrained(
        backbone_meta["hf_id"], trust_remote_code=backbone_meta["trust_remote_code"]
    )
    max_length = backbone_meta["max_length"]
    embed_dim = backbone_meta["hidden_dim"]

    # Load all 13,136 compounds (including NaN-dropped 1657 & 8624 from pretrain)
    compounds = load_all_compounds()
    print(f"Extracting embeddings for {len(compounds)} compounds")

    # Dummy targets — values ignored; only input_ids/attention_mask are used below
    dummy_targets = np.zeros((len(compounds), n_tasks), dtype=np.float32)
    dataset = SmilesMTRDataset(
        compounds["std_smiles"].tolist(),
        dummy_targets,
        tokenizer,
        max_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].cuda()
            mask = batch["attention_mask"].cuda()
            out = model.backbone(input_ids=ids, attention_mask=mask)
            m = mask.unsqueeze(-1).float()
            pooled = (out.last_hidden_state * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            chunks.append(pooled.cpu().numpy())

    embeddings = np.concatenate(chunks, axis=0)
    print(f"Embedding shape: {embeddings.shape}")

    assert embeddings.shape == (len(compounds), embed_dim), (
        f"Expected ({len(compounds)}, {embed_dim}), got {embeddings.shape}"
    )
    assert not np.isnan(embeddings).any(), "NaN detected in embeddings"
    assert not np.isinf(embeddings).any(), "Inf detected in embeddings"

    # Build output DataFrame indexed by compound_id
    col_names = [f"molformer_c3_mtr_{i}" for i in range(embeddings.shape[1])]
    out_df = pd.DataFrame(
        embeddings,
        index=compounds["compound_id"].values,
        columns=col_names,
    )
    out_df.index.name = "compound_id"

    out_path = REPO_ROOT.joinpath(
        f"data/molformer_c3_mtr_embedding_seed{args.seed}.parquet"
    )
    out_df.to_parquet(out_path)
    print(f"Saved {out_path}  shape={embeddings.shape}")


if __name__ == "__main__":
    main()
