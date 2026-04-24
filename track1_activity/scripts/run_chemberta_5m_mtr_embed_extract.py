"""Extract 384d [CLS] embeddings from the ChemBERTa-5M-MTR continued-
pretrain checkpoint.

Phase B of #100 BERT-family follow-up. Mirrors
run_molformer_c3_embed_extract.py but for the RoBERTa-3L 5M ChemBERTa.
Output: data/chemberta_5m_mtr_pretrain_embed.parquet (index=compound_id,
columns=emb_000..emb_383).

Downstream consumer: run_train.py --feature chemberta_5m_mtr_pretrain_embed.

Usage:
    pixi run python track1_activity/scripts/run_chemberta_5m_mtr_embed_extract.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402
from peft_backbones import get_backbone  # noqa: E402
from peft_methods import get_peft_builder  # noqa: E402

from peft import get_peft_model  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

CKPT_PATH = REPO_ROOT.joinpath(
    "track1_activity", "checkpoints", "chemberta_5m_mtr_pretrain", "pretrain.pt"
)
OUT_PATH = REPO_ROOT.joinpath("data", "chemberta_5m_mtr_pretrain_embed.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


class SmilesOnlyDataset(Dataset):
    def __init__(self, smiles, tokenizer, max_length):
        self.smiles = smiles
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.smiles[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


class ChemBertaPretrainModel(nn.Module):
    """Must match the structure saved in pretrain.pt exactly so load_state_dict works."""

    def __init__(
        self, backbone_name, peft_params, head_hidden_dim, head_dropout, n_tasks=2
    ):
        super().__init__()
        meta = get_backbone(backbone_name)
        base = AutoModel.from_pretrained(
            meta["hf_id"], trust_remote_code=meta["trust_remote_code"]
        )
        peft_config = get_peft_builder("lora")(meta, peft_params)
        self.backbone = get_peft_model(base, peft_config)
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(meta["hidden_dim"], head_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, n_tasks),
        )


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


def main():
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"missing pretrain ckpt: {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    params = ckpt["params"]
    backbone_name = ckpt["backbone"]
    meta = get_backbone(backbone_name)
    hidden_dim = meta["hidden_dim"]
    print(f"Loaded pretrain ckpt: backbone={backbone_name}, params={params}")

    df = load_target_compounds()
    n = len(df)
    print(f"Target compounds (train + test union): {n}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        meta["hf_id"], trust_remote_code=meta["trust_remote_code"]
    )

    peft_params = {
        "lora_rank": params["lora_rank"],
        "lora_alpha": params["lora_alpha"],
        "lora_dropout": params["lora_dropout"],
        "lora_target": params["lora_target"],
    }
    model = ChemBertaPretrainModel(
        backbone_name=backbone_name,
        peft_params=peft_params,
        head_hidden_dim=params["head_hidden_dim"],
        head_dropout=params["head_dropout"],
        n_tasks=2,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dataset = SmilesOnlyDataset(df["smiles"].tolist(), tokenizer, meta["max_length"])
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    chunks = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model.backbone(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            chunks.append(cls.detach().cpu().numpy())

    emb = np.concatenate(chunks, axis=0)
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
