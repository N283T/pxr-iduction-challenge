"""Predict log2_fc @ 8.25uM / 33uM for 4653 train+test compounds using
the MoLFormer-c3 pretrain checkpoint.

Phase 1 of issue #115 (log2fc_pred ensembling). Mirrors
run_chemprop_predict_log2fc.py but for the MoLFormer-c3-1.1B LoRA
pretrain. Loads model.backbone + model.head from the checkpoint, runs
full forward on train+test SMILES, un-z-scores using the pretrain
target_means/stds.

Output: data/molformer_c3_pretrain_log2fc_predictions.parquet
        index=compound_id, columns=[log2fc_8p25_pred, log2fc_33_pred]

Usage:
    pixi run python track1_activity/scripts/run_molformer_c3_predict_log2fc.py
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
    "track1_activity", "checkpoints", "molformer_c3_pretrain", "pretrain.pt"
)
OUT_PATH = REPO_ROOT.joinpath(
    "data", "molformer_c3_pretrain_log2fc_predictions.parquet"
)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


class SmilesOnlyDataset(Dataset):
    def __init__(self, smiles: list[str], tokenizer, max_length: int) -> None:
        self.smiles = smiles
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
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


class MolformerPretrainModel(nn.Module):
    """Identical to run_molformer_c3_pretrain.py::MolformerPretrainModel."""

    def __init__(
        self,
        backbone_name: str,
        peft_params: dict,
        head_hidden_dim: int,
        head_dropout: float,
        n_tasks: int = 2,
    ) -> None:
        super().__init__()
        meta = get_backbone(backbone_name)
        base = AutoModel.from_pretrained(
            meta["hf_id"], trust_remote_code=meta["trust_remote_code"]
        )
        if meta.get("fix_rotary", False):
            for layer in base.encoder.layer:
                rotary = layer.attention.self.rotary_embeddings
                rotary.inv_freq = 1.0 / (
                    rotary.base ** (torch.arange(0, rotary.dim, 2).float() / rotary.dim)
                )
                rotary._set_cos_sin_cache(
                    seq_len=rotary.max_position_embeddings,
                    device=rotary.inv_freq.device,
                    dtype=torch.float32,
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

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.head(cls)


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


def main() -> None:
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"missing pretrain ckpt: {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    params = ckpt["params"]
    backbone_name = ckpt["backbone"]
    means = np.asarray(ckpt["target_means"], dtype=np.float32)
    stds = np.asarray(ckpt["target_stds"], dtype=np.float32)
    print(
        f"Loaded pretrain ckpt: backbone={backbone_name}, "
        f"means={means.tolist()}, stds={stds.tolist()}"
    )

    df = load_target_compounds()
    n = len(df)
    print(f"Target compounds (train + test union): {n}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = get_backbone(backbone_name)
    tokenizer = AutoTokenizer.from_pretrained(
        meta["hf_id"], trust_remote_code=meta["trust_remote_code"]
    )

    peft_params = {
        "lora_rank": params["lora_rank"],
        "lora_alpha": params["lora_alpha"],
        "lora_dropout": params["lora_dropout"],
        "lora_target": params["lora_target"],
    }
    model = MolformerPretrainModel(
        backbone_name=backbone_name,
        peft_params=peft_params,
        head_hidden_dim=params["head_hidden_dim"],
        head_dropout=params["head_dropout"],
        n_tasks=2,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Model on {device}, loaded state_dict")

    dataset = SmilesOnlyDataset(df["smiles"].tolist(), tokenizer, meta["max_length"])
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    preds_z: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            preds = model(input_ids, attention_mask).detach().cpu().numpy()
            preds_z.append(preds)

    preds_z_arr = np.concatenate(preds_z, axis=0).astype(np.float32)
    assert preds_z_arr.shape == (n, 2)
    preds_raw = preds_z_arr * stds + means

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
