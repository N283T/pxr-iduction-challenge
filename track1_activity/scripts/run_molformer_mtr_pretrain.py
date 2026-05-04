#!/usr/bin/env -S pixi run python
"""MoLFormer-c3-1.1B MTR domain-adaptation pretrain (Variant M).

Descriptor multi-task regression on 13,134 PXR compounds (2 NaN rows
dropped: 1657, 8624). 217 RDKit descriptors as targets, StandardScaler-
normalized, MSE loss summed across heads. Existing MLM-pretrained backbone
+ LoRA + mean-pool head (Sultan 2025 MLM_MTR configuration).

Output: models/molformer_c3_mtr_seed{N}/{pretrain.pt, scaler.json, meta.json}

Usage:
    pixi run python track1_activity/scripts/run_molformer_mtr_pretrain.py --seed 42
    pixi run python track1_activity/scripts/run_molformer_mtr_pretrain.py --smoke --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(SCRIPTS_DIR))

from peft import get_peft_model  # noqa: E402
from peft_backbones import get_backbone  # noqa: E402
from peft_methods import get_peft_builder  # noqa: E402
from run_chemprop_mtr_pretrain import (  # noqa: E402
    apply_scaler,
    fit_scaler,
    load_descriptor_targets,
    run_audit_or_die,
)

torch.set_float32_matmul_precision("medium")

CKPT_BASE = REPO_ROOT.joinpath("models")

# Sultan 2025 MLM_MTR hyperparameters
DEFAULT_PARAMS = {
    "backbone": "molformer_c3_1_1b",
    "peft_method": "lora",
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "lora_target": "qv",
    "head_dim": 256,
    "head_dropout": 0.1,
    "lr": 3e-5,
    "batch_size": 16,
    "max_epochs": 20,
    "warmup_ratio": 0.10,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "val_frac": 0.1,
}


class SmilesMTRDataset(Dataset):
    """Same as SmilesDataset in run_molformer_c3_pretrain but multi-target (N, 217)."""

    def __init__(
        self, smiles: list[str], targets: np.ndarray, tokenizer, max_length: int
    ):
        self.smiles = smiles
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> dict:
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
            "targets": self.targets[idx],
        }


class MolformerMTRModel(nn.Module):
    """MoLFormer-c3 + LoRA + mean-pool + MLP head → 217 descriptors."""

    def __init__(
        self,
        backbone_name: str,
        peft_method: str,
        peft_params: dict,
        n_tasks: int,
        head_dim: int,
        head_dropout: float,
    ):
        super().__init__()
        meta = get_backbone(backbone_name)
        base = AutoModel.from_pretrained(
            meta["hf_id"], trust_remote_code=meta["trust_remote_code"]
        )
        # === ROTARY FIX (verbatim from peft_trainer.py:72-82, issue #30) ===
        # Fix corrupted rotary-embedding buffers (MoLFormer-XL + transformers v5
        # bug: inv_freq loaded from checkpoint contains garbage values).
        # Recompute inv_freq and rebuild the cos/sin cache before PEFT wrapping.
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
        # === END ROTARY FIX ===
        peft_config = get_peft_builder(peft_method)(meta, peft_params)
        self.backbone = get_peft_model(base, peft_config)

        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(meta["hidden_dim"], head_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_dim, n_tasks),
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Mean-pool over non-padding tokens (spec §4.2)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(
            min=1
        )
        return self.head(pooled)


def main() -> None:
    p = argparse.ArgumentParser(
        description="MoLFormer-c3 MTR domain-adaptation pretrain"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke run: 3 epochs, 200 compounds, no checkpoint write",
    )
    args = p.parse_args()

    run_audit_or_die()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    smiles, raw_targets, ids, desc_names = load_descriptor_targets()

    if args.smoke:
        smiles = smiles[:200]
        raw_targets = raw_targets[:200]
        ids = ids[:200]

    print(f"Loaded {len(smiles)} compounds × {raw_targets.shape[1]} descriptors")

    scaler = fit_scaler(raw_targets)
    targets = apply_scaler(raw_targets, scaler)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(smiles))
    n_val = int(DEFAULT_PARAMS["val_frac"] * len(perm))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    params = dict(DEFAULT_PARAMS)
    if args.smoke:
        params["max_epochs"] = 3

    backbone_name = params["backbone"]
    meta = get_backbone(backbone_name)
    tokenizer = AutoTokenizer.from_pretrained(
        meta["hf_id"], trust_remote_code=meta["trust_remote_code"]
    )

    train_smi = [smiles[i] for i in train_idx]
    val_smi = [smiles[i] for i in val_idx]
    tr_y = targets[train_idx]
    va_y = targets[val_idx]
    print(f"  split: train={len(train_idx)}, val={len(val_idx)}")

    train_ds = SmilesMTRDataset(train_smi, tr_y, tokenizer, meta["max_length"])
    val_ds = SmilesMTRDataset(val_smi, va_y, tokenizer, meta["max_length"])
    train_loader = DataLoader(
        train_ds,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    peft_params = {
        "lora_rank": params["lora_rank"],
        "lora_alpha": params["lora_alpha"],
        "lora_dropout": params["lora_dropout"],
        "lora_target": params["lora_target"],
    }
    model = MolformerMTRModel(
        backbone_name=backbone_name,
        peft_method=params["peft_method"],
        peft_params=peft_params,
        n_tasks=raw_targets.shape[1],
        head_dim=params["head_dim"],
        head_dropout=params["head_dropout"],
    ).to(device)

    n_steps_per_epoch = len(train_loader)
    total_steps = n_steps_per_epoch * params["max_epochs"]
    warmup_steps = int(total_steps * params["warmup_ratio"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val = float("inf")
    best_state: dict | None = None
    final_val_loss: float = float("inf")

    for epoch in range(params["max_epochs"]):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_targets = batch["targets"].to(device)
            optimizer.zero_grad()
            preds = model(input_ids, attention_mask)
            loss = F.mse_loss(preds, batch_targets)
            if torch.isnan(loss):
                raise RuntimeError(
                    f"NaN train loss at epoch {epoch} — check rotary fix"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), params["grad_clip"])
            optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                batch_targets = batch["targets"].to(device)
                preds = model(input_ids, attention_mask)
                val_losses.append(float(F.mse_loss(preds, batch_targets).item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        final_val_loss = val_loss
        print(f"  epoch {epoch:03d}: train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No best state captured (max_epochs=0?)")

    if args.smoke:
        print("Smoke run completed; no checkpoint written.")
        return

    out_dir = CKPT_BASE.joinpath(f"molformer_c3_mtr_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(best_state, out_dir.joinpath("pretrain.pt"))
    out_dir.joinpath("scaler.json").write_text(json.dumps(scaler, indent=2))
    out_dir.joinpath("meta.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "n_compounds": len(smiles),
                "descriptor_names": desc_names,
                "params": params,
                "head_dim": params["head_dim"],
                "backbone_name": backbone_name,
                "peft_method": params["peft_method"],
                "peft_params": peft_params,
                "best_val_loss": best_val,
                "final_val_loss": final_val_loss,
            },
            indent=2,
        )
    )
    print(f"Saved {out_dir}  best_val={best_val:.4f}")


if __name__ == "__main__":
    main()
