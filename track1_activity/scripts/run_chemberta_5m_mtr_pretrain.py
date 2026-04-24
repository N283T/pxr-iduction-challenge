"""ChemBERTa-5M-MTR continued-pretrain on 2-head log2_fc (8.25uM + 33uM).

Phase B of #100 BERT-family audit follow-up. Mirrors
run_molformer_c3_pretrain.py exactly but swaps the backbone to
DeepChem/ChemBERTa-5M-MTR (RoBERTa 3L, 384d, ~5M params).

Motivation: Phase B1 9-variant audit showed that raw ChemBERTa
embeddings (no adapt) give single-model MAE 0.53-0.67 -- 5m_mtr
was the best at 0.5287. Caruana ADD bakeoff gave Δ -0.0020 (below
-0.003 threshold after tier-0 LB regression). Hypothesis: continued
pretrain on log2fc will push single-model MAE down (more task-aligned
embedding) so caruana ADD meets the stricter gate.

Usage:
    pixi run python track1_activity/scripts/run_chemberta_5m_mtr_pretrain.py

Output: track1_activity/checkpoints/chemberta_5m_mtr_pretrain/pretrain.pt
        + pretrain_meta.json with target z-score stats.
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
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402
from peft_backbones import get_backbone  # noqa: E402
from peft_methods import get_peft_builder  # noqa: E402

from peft import get_peft_model  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

torch.set_float32_matmul_precision("medium")

CKPT_DIR = REPO_ROOT.joinpath(
    "track1_activity", "checkpoints", "chemberta_5m_mtr_pretrain"
)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Same head / optimizer hyperparams as run_molformer_c3_pretrain for
# apples-to-apples comparison. Smaller backbone tolerates larger LR
# and bigger batch; can tune later if convergence is slow.
DEFAULT_PARAMS = {
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "lora_target": "qkvo",
    "head_hidden_dim": 256,
    "head_dropout": 0.1,
    "backbone_lr": 2e-4,
    "head_lr": 1e-3,
    "weight_decay": 1e-3,
    "batch_size": 128,
    "max_epochs": 80,
    "patience": 10,
}


class SmilesDataset(Dataset):
    def __init__(self, smiles, targets, tokenizer, max_length):
        self.smiles = smiles
        self.targets = targets
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
            "targets": torch.tensor(self.targets[idx], dtype=torch.float32),
        }


class ChemBertaPretrainModel(nn.Module):
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

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # RoBERTa returns last_hidden_state: (B, L, H). Use [CLS] token = index 0.
        cls = out.last_hidden_state[:, 0, :]
        return self.head(cls)


def masked_mse(preds, targets):
    """Per-sample MSE averaged over non-NaN targets. preds/targets: (B, n_tasks)."""
    mask = torch.isfinite(targets)
    sq = (preds - torch.where(mask, targets, torch.zeros_like(targets))) ** 2
    sq = sq * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return sq.sum() / denom


def load_pretrain_data():
    """All 13,136 compounds + 2-head log2_fc targets. NaN where no measurement.

    Identical SQL to run_molformer_c3_pretrain / run_chemprop_pretrain so
    the seed=42 + val_frac=0.1 shuffle produces the same val compound_ids.
    """
    sql = """
    SELECT c.id AS compound_id,
           c.std_smiles AS smiles,
           agg.log2fc_8p25,
           agg.log2fc_33
    FROM compounds c
    LEFT JOIN (
      SELECT compound_id,
        AVG(CASE WHEN concentration_m BETWEEN 8.2e-6 AND 8.3e-6
                 THEN log2_fc_estimate END) AS log2fc_8p25,
        AVG(CASE WHEN concentration_m BETWEEN 3.28e-5 AND 3.32e-5
                 THEN log2_fc_estimate END) AS log2fc_33
      FROM single_concentration
      GROUP BY compound_id
    ) agg ON agg.compound_id = c.id
    WHERE c.std_smiles IS NOT NULL
    ORDER BY c.id
    """
    with psycopg2.connect(**DB_PARAMS) as conn:
        df = pd.read_sql(sql, conn)
    return (
        df["smiles"].tolist(),
        df[["log2fc_8p25", "log2fc_33"]].to_numpy(dtype=np.float32),
        df["compound_id"].astype(int).tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description="ChemBERTa-5M-MTR continued-pretrain")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", default="chemberta_5m_mtr")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    params = DEFAULT_PARAMS.copy()
    if args.max_epochs is not None:
        params["max_epochs"] = args.max_epochs
    print(f"ChemBERTa continued-pretrain: {args.backbone}")
    print(f"  params: {params}")

    smiles, targets, _cids = load_pretrain_data()
    n_total = len(smiles)
    n_valid_8 = int(np.isfinite(targets[:, 0]).sum())
    n_valid_33 = int(np.isfinite(targets[:, 1]).sum())
    print(
        f"  data: {n_total} compounds, "
        f"log2fc_8p25 valid={n_valid_8}, log2fc_33 valid={n_valid_33}"
    )

    means = np.zeros(2, dtype=np.float32)
    stds = np.ones(2, dtype=np.float32)
    for i in range(2):
        valid = np.isfinite(targets[:, i])
        means[i] = float(np.mean(targets[valid, i]))
        stds[i] = float(np.std(targets[valid, i]))
        if stds[i] < 1e-6:
            stds[i] = 1.0
    targets_z = (targets - means) / stds
    print(f"  target means: {means.tolist()}, stds: {stds.tolist()}")

    rng = np.random.default_rng(args.seed)
    idx = np.arange(n_total)
    rng.shuffle(idx)
    n_val = int(n_total * args.val_frac)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    tr_smi = [smiles[i] for i in tr_idx]
    va_smi = [smiles[i] for i in val_idx]
    tr_y = targets_z[tr_idx]
    va_y = targets_z[val_idx]
    print(f"  split: train={len(tr_idx)}, val={len(val_idx)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = get_backbone(args.backbone)
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
        backbone_name=args.backbone,
        peft_params=peft_params,
        head_hidden_dim=params["head_hidden_dim"],
        head_dropout=params["head_dropout"],
        n_tasks=2,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.backbone.parameters() if p.requires_grad],
                "lr": params["backbone_lr"],
            },
            {"params": model.head.parameters(), "lr": params["head_lr"]},
        ],
        weight_decay=params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params["max_epochs"], eta_min=1e-7
    )

    train_ds = SmilesDataset(tr_smi, tr_y, tokenizer, meta["max_length"])
    val_ds = SmilesDataset(va_smi, va_y, tokenizer, meta["max_length"])
    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False)

    best_val = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(params["max_epochs"]):
        model.train()
        train_losses = []
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)
            optimizer.zero_grad()
            preds = model(input_ids, attention_mask)
            loss = masked_mse(preds, targets)
            if torch.isnan(loss):
                raise RuntimeError(f"NaN train loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["targets"].to(device)
                preds = model(input_ids, attention_mask)
                val_losses.append(float(masked_mse(preds, targets).item()))
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        print(f"  epoch {epoch:03d}: train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= params["patience"]:
                print(f"  early stop at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("no best state captured (max_epochs=0?)")

    state_path = CKPT_DIR.joinpath("pretrain.pt")
    torch.save(
        {
            "state_dict": best_state,
            "params": params,
            "target_means": means.tolist(),
            "target_stds": stds.tolist(),
            "n_train": len(tr_idx),
            "n_val": len(val_idx),
            "best_val_loss": best_val,
            "backbone": args.backbone,
        },
        state_path,
    )
    print(f"  saved: {state_path}  best_val={best_val:.4f}")

    meta_path = CKPT_DIR.joinpath("pretrain_meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "params": params,
                "backbone": args.backbone,
                "target_means": means.tolist(),
                "target_stds": stds.tolist(),
                "n_valid_target": {"8.25uM": n_valid_8, "33uM": n_valid_33},
                "n_train_compounds": len(tr_idx),
                "n_val_compounds": len(val_idx),
                "best_val_loss": best_val,
            },
            indent=2,
        )
    )
    print(f"  meta: {meta_path}")


if __name__ == "__main__":
    main()
