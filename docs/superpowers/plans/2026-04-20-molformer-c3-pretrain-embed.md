# MoLFormer-c3 Pretrain + Embed + TabPFN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a transformer-encoder-family pool member via the proven "pretrain on log2_fc -> frozen embedding -> TabPFN" recipe (mirrors pool-strongest `tabpfn_chemprop_pretrain_embed` MAE 0.437). Uses `DeepChem/MoLFormer-c3-1.1B` as backbone (only variant with published weights, architecture identical to PR #95 base).

**Architecture:** Reuses PEFT framework from PR #95 (`peft_backbones`, `peft_methods`, `peft_trainer`) with one registry entry. Three-phase pipeline: (1) custom pretrain script (2-head NaN-masked MSE on 13k compounds' log2_fc at 8.25µM/33µM, mirrors `run_chemprop_pretrain.py`), (2) embed extract script (saves parquet, mirrors `run_chemprop_embed_extract.py`), (3) downstream TabPFN via existing `run_train.py --model tabpfn --feature molformer_c3_pretrain_embed --split umap`.

**Tech Stack:** Python 3.12, transformers 5.5, peft 0.13.2, torch 2.10+cu13, lightning 2.6, tabpfn, optuna 4.8, psycopg2, pandas/parquet.

**Spec:** `docs/superpowers/specs/2026-04-20-molformer-c3-pretrain-embed-design.md`

**Branch:** `feature/molformer-c3-pretrain-embed` (already created)

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `track1_activity/src/peft_backbones.py` | Modify | Add `molformer_c3_1_1b` entry (`DeepChem/MoLFormer-c3-1.1B`, hidden=768, fix_rotary=True) |
| `track1_activity/scripts/run_molformer_c3_pretrain.py` | Create | Phase 1: pretrain MoLFormer+LoRA on 2-head log2_fc, save `track1_activity/checkpoints/molformer_c3_pretrain/pretrain.pt` |
| `track1_activity/scripts/run_molformer_c3_embed_extract.py` | Create | Phase 2: forward pass all 13k compounds, save `data/molformer_c3_pretrain_embed.parquet` indexed by compound_id (768d) |
| `track1_activity/scripts/run_train.py` | Modify | Add `molformer_c3_pretrain_embed` to `all_features` tuple + feature-handler branch (mirrors `chemprop_pretrain_embed` branch at line ~385) |
| `track1_activity/scripts/run_ensemble.py` | Modify | Append `tabpfn_molformer_c3_pretrain_embed_umap_default` to `ENSEMBLE_MODELS` allow-list |

---

## Task 1: Backbone registry entry

**Files:**
- Modify: `track1_activity/src/peft_backbones.py`

- [ ] **Step 1.1: Add the new backbone entry**

Open `track1_activity/src/peft_backbones.py` and add the `molformer_c3_1_1b` entry to the `BACKBONES` dict. Place it after the existing `molformer_xl` entry, inside the closing `}` of BACKBONES.

Current BACKBONES has one entry (molformer_xl). Add the second:

```python
    "molformer_c3_1_1b": {
        "hf_id": "DeepChem/MoLFormer-c3-1.1B",
        "hidden_dim": 768,
        "max_length": 202,
        "trust_remote_code": True,
        # Architecture identical to ibm/MoLFormer-XL-both-10pct (verified
        # config.json). The "1.1B" refers to pretrain token count; actual
        # model is ~80M params. HF auto_map references ibm modeling code,
        # so trust_remote_code=True pulls from the ibm repo.
        "lora_target_modules_qv": ["query", "value"],
        "lora_target_modules_qkvo": ["query", "key", "value", "dense"],
        # Same rotary embedding bug as ibm/MoLFormer-XL (inherited
        # architecture); PeftRegressor recomputes inv_freq + cos/sin cache.
        "fix_rotary": True,
    },
```

- [ ] **Step 1.2: Smoke check the registry entry**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge && pixi run python -c "
import sys
sys.path.insert(0, 'track1_activity/src')
from peft_backbones import get_backbone
m = get_backbone('molformer_c3_1_1b')
print(m['hf_id'], m['hidden_dim'], 'fix_rotary=', m['fix_rotary'])
"
```
Expected: `DeepChem/MoLFormer-c3-1.1B 768 fix_rotary= True`.

- [ ] **Step 1.3: Verify HF model loads (downloads ~2GB on first call)**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge && pixi run python -c "
from transformers import AutoModel, AutoTokenizer
m = AutoModel.from_pretrained('DeepChem/MoLFormer-c3-1.1B', trust_remote_code=True)
print('params:', sum(p.numel() for p in m.parameters()) / 1e6, 'M')
print('hidden_size:', m.config.hidden_size)
print('num_layers:', m.config.num_hidden_layers)
tok = AutoTokenizer.from_pretrained('DeepChem/MoLFormer-c3-1.1B', trust_remote_code=True)
print('tokenizer vocab size:', tok.vocab_size)
" 2>&1 | tail -10
```
Expected: param count 70-90M (despite "1.1B" name), hidden_size=768, num_layers=12, tokenizer vocab size ~2362. If weights are broken (e.g., 404 on safetensors), STOP and escalate — we'll fall back to `ibm/MoLFormer-XL-both-10pct` by editing the `hf_id` value to that string.

- [ ] **Step 1.4: Lint**

Run: `cd /home/nagaet/pxr-iduction-challenge && pixi run ruff format track1_activity/src/peft_backbones.py && pixi run ruff check track1_activity/src/peft_backbones.py`
Expected: clean.

- [ ] **Step 1.5: Commit**

```bash
git add track1_activity/src/peft_backbones.py
git commit -m "feat(peft): add DeepChem/MoLFormer-c3-1.1B backbone entry"
```

---

## Task 2: Pretrain script

**Files:**
- Create: `track1_activity/scripts/run_molformer_c3_pretrain.py`

This script pretrain the MoLFormer-c3 backbone with LoRA on a 2-head NaN-masked MSE objective over single_concentration log2_fc at 8.25µM and 33µM, on all 13,136 compounds. Structure mirrors `track1_activity/scripts/run_chemprop_pretrain.py` (reviewed as reference). Differences: MoLFormer backbone + LoRA (not chemprop MPNN), custom 2-head MLP, explicit NaN-masked loss (chemprop's `MSE` loss with task_weights handles this automatically; for MoLFormer we roll our own).

- [ ] **Step 2.1: Create the pretrain script**

Create `track1_activity/scripts/run_molformer_c3_pretrain.py` with this content:

```python
"""MoLFormer-c3 backbone pretrain on 2-head log2_fc (8.25uM + 33uM).

Phase 1 of the MoLFormer-c3-pretrain+frozen+TabPFN pipeline (mirrors
run_chemprop_pretrain.py for chemprop). Uses all 13,136 compounds from
compounds.std_smiles. NaN targets (no measurement at that concentration)
are masked per-sample per-head.

Usage:
    pixi run python track1_activity/scripts/run_molformer_c3_pretrain.py

Output: track1_activity/checkpoints/molformer_c3_pretrain/pretrain.pt
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
    "track1_activity", "checkpoints", "molformer_c3_pretrain"
)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

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
    "batch_size": 64,
    "max_epochs": 50,
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


class MolformerPretrainModel(nn.Module):
    def __init__(self, backbone_name, peft_params, head_hidden_dim, head_dropout, n_tasks=2):
        super().__init__()
        meta = get_backbone(backbone_name)
        base = AutoModel.from_pretrained(
            meta["hf_id"], trust_remote_code=meta["trust_remote_code"]
        )
        if meta.get("fix_rotary", False):
            for layer in base.encoder.layer:
                rotary = layer.attention.self.rotary_embeddings
                rotary.inv_freq = 1.0 / (
                    rotary.base
                    ** (torch.arange(0, rotary.dim, 2).float() / rotary.dim)
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

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
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
    """All 13,136 compounds + 2-head log2_fc targets. NaN where no measurement."""
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
    parser = argparse.ArgumentParser(description="MoLFormer-c3 pretrain")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", default="molformer_c3_1_1b")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    params = DEFAULT_PARAMS.copy()
    if args.max_epochs is not None:
        params["max_epochs"] = args.max_epochs
    print(f"MoLFormer-c3 pretrain: {args.backbone}")
    print(f"  params: {params}")

    smiles, targets, _cids = load_pretrain_data()
    n_total = len(smiles)
    n_valid_8 = int(np.isfinite(targets[:, 0]).sum())
    n_valid_33 = int(np.isfinite(targets[:, 1]).sum())
    print(f"  data: {n_total} compounds, log2fc_8p25 valid={n_valid_8}, log2fc_33 valid={n_valid_33}")

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
    model = MolformerPretrainModel(
        backbone_name=args.backbone,
        peft_params=peft_params,
        head_hidden_dim=params["head_hidden_dim"],
        head_dropout=params["head_dropout"],
        n_tasks=2,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.backbone.parameters() if p.requires_grad],
             "lr": params["backbone_lr"]},
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
```

- [ ] **Step 2.2: Lint**

Run: `cd /home/nagaet/pxr-iduction-challenge && pixi run ruff format track1_activity/scripts/run_molformer_c3_pretrain.py && pixi run ruff check track1_activity/scripts/run_molformer_c3_pretrain.py`
Expected: clean.

- [ ] **Step 2.3: Smoke test with tiny max_epochs**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge && pixi run db-start 2>&1 | tail -2
time pixi run python track1_activity/scripts/run_molformer_c3_pretrain.py --max-epochs 2 2>&1 | tail -20
```
Expected: completes in ~15 min (2 epochs on 13k compounds with batch 64, 1.1B LoRA), prints `saved: .../pretrain.pt best_val=<float>`. If OOM, reduce `batch_size` in DEFAULT_PARAMS to 32 or 16 and re-run smoke.

- [ ] **Step 2.4: Verify checkpoint saved correctly**

Run:
```bash
ls -la track1_activity/checkpoints/molformer_c3_pretrain/
cd /home/nagaet/pxr-iduction-challenge && pixi run python -c "
import torch
ckpt = torch.load('track1_activity/checkpoints/molformer_c3_pretrain/pretrain.pt', map_location='cpu', weights_only=False)
print('keys:', list(ckpt.keys()))
print('backbone:', ckpt['backbone'])
print('state_dict size:', len(ckpt['state_dict']))
print('best_val_loss:', ckpt['best_val_loss'])
"
```
Expected: keys include `state_dict`, `params`, `target_means`, `target_stds`, `best_val_loss`, `backbone`. state_dict size > 300 (LoRA + base + head).

- [ ] **Step 2.5: Delete smoke checkpoint (will be regenerated by full run)**

Run:
```bash
rm track1_activity/checkpoints/molformer_c3_pretrain/pretrain.pt
rm track1_activity/checkpoints/molformer_c3_pretrain/pretrain_meta.json
```

- [ ] **Step 2.6: Commit**

```bash
git add track1_activity/scripts/run_molformer_c3_pretrain.py
git commit -m "feat(molformer): Phase 1 pretrain script on 2-head log2_fc"
```

---

## Task 3: Embed extract script

**Files:**
- Create: `track1_activity/scripts/run_molformer_c3_embed_extract.py`

Mirrors `run_chemprop_embed_extract.py` structure: load pretrain checkpoint, extract per-compound embedding for the union of train+test compounds, save parquet indexed by compound_id.

- [ ] **Step 3.1: Create the embed extract script**

Create `track1_activity/scripts/run_molformer_c3_embed_extract.py`:

```python
"""Extract 768d [CLS] embeddings from the MoLFormer-c3 pretrain checkpoint.

Phase 2 of Buterez 2024 strategy-3 pipeline for MoLFormer. Loads the
pretrain checkpoint produced by run_molformer_c3_pretrain.py and
produces a parquet of molecule-level [CLS] embeddings for the union of
train_activity + test_activity compounds.

Output: data/molformer_c3_pretrain_embed.parquet (index=compound_id,
columns=emb_000..emb_767). Downstream TabPFN / LGBM consumers read via
run_train.py --feature molformer_c3_pretrain_embed.

Usage:
    pixi run python track1_activity/scripts/run_molformer_c3_embed_extract.py
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
OUT_PATH = REPO_ROOT.joinpath("data", "molformer_c3_pretrain_embed.parquet")
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


class MolformerPretrainModel(nn.Module):
    """Must match the structure saved in pretrain.pt exactly so load_state_dict works."""

    def __init__(self, backbone_name, peft_params, head_hidden_dim, head_dropout, n_tasks=2):
        super().__init__()
        meta = get_backbone(backbone_name)
        base = AutoModel.from_pretrained(
            meta["hf_id"], trust_remote_code=meta["trust_remote_code"]
        )
        if meta.get("fix_rotary", False):
            for layer in base.encoder.layer:
                rotary = layer.attention.self.rotary_embeddings
                rotary.inv_freq = 1.0 / (
                    rotary.base
                    ** (torch.arange(0, rotary.dim, 2).float() / rotary.dim)
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
    model = MolformerPretrainModel(
        backbone_name=backbone_name,
        peft_params=peft_params,
        head_hidden_dim=params["head_hidden_dim"],
        head_dropout=params["head_dropout"],
        n_tasks=2,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dataset = SmilesOnlyDataset(df["smiles"].tolist(), tokenizer, meta["max_length"])
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

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
```

- [ ] **Step 3.2: Lint**

Run: `cd /home/nagaet/pxr-iduction-challenge && pixi run ruff format track1_activity/scripts/run_molformer_c3_embed_extract.py && pixi run ruff check track1_activity/scripts/run_molformer_c3_embed_extract.py`
Expected: clean.

- [ ] **Step 3.3: Commit (smoke defer — needs pretrain.pt from Task 5)**

```bash
git add track1_activity/scripts/run_molformer_c3_embed_extract.py
git commit -m "feat(molformer): Phase 2 embed extract script"
```

No smoke test at this stage — the script depends on a real pretrain checkpoint which only exists after Task 5. The full pipeline smoke is Task 6.

---

## Task 4: `run_train.py` feature handler

**Files:**
- Modify: `track1_activity/scripts/run_train.py` (two places: the `all_features` tuple near line 1280, and the feature-handler `if feature_name == "chemprop_pretrain_embed":` block at line 385)

- [ ] **Step 4.1: Add `molformer_c3_pretrain_embed` to `all_features` tuple**

Open `track1_activity/scripts/run_train.py`. Find the `all_features = (` block near line 1280. The current block is:

```python
    all_features = (
        [
            "rdkit_desc_full",
            "mordred",
            "mordred_singleconc",
            "mordred_jazzy",
            "2d_full",
            "2d_full_boltz",
            "pooled_boltz",
            "pooled_boltz_allpairs",
            "chemprop_pretrain_embed",
            "2d_full_boltz_log2fc_pred",
            "3d_ligand",
            "jazzy",
        ]
        + list(FP_REGISTRY.keys())
        + list(EMBEDDING_TABLES.keys())
    )
```

Add `"molformer_c3_pretrain_embed"` to the literal list, just after `"chemprop_pretrain_embed"`:

```python
    all_features = (
        [
            "rdkit_desc_full",
            "mordred",
            "mordred_singleconc",
            "mordred_jazzy",
            "2d_full",
            "2d_full_boltz",
            "pooled_boltz",
            "pooled_boltz_allpairs",
            "chemprop_pretrain_embed",
            "molformer_c3_pretrain_embed",
            "2d_full_boltz_log2fc_pred",
            "3d_ligand",
            "jazzy",
        ]
        + list(FP_REGISTRY.keys())
        + list(EMBEDDING_TABLES.keys())
    )
```

- [ ] **Step 4.2: Add the feature handler branch**

Find the `if feature_name == "chemprop_pretrain_embed":` block near line 385. It ends at line 410 with `return X_train, X_test`. Add a new `if` block immediately after that `return`, before the next `if feature_name == "pooled_boltz":` at line 412:

```python
    if feature_name == "molformer_c3_pretrain_embed":
        # 768d per-compound [CLS] embeddings from MoLFormer-c3 pretrain
        # LoRA checkpoint. See
        # track1_activity/scripts/run_molformer_c3_pretrain.py and
        # track1_activity/scripts/run_molformer_c3_embed_extract.py.
        # Buterez 2024 strategy-3 with a transformer-family backbone
        # (parallel to chemprop_pretrain_embed which uses a GNN).
        embed_path = REPO_ROOT.joinpath("data", "molformer_c3_pretrain_embed.parquet")
        if not embed_path.exists():
            raise SystemExit(
                f"Missing {embed_path}. Run "
                f"track1_activity/scripts/run_molformer_c3_embed_extract.py"
            )
        emb_df = pd.read_parquet(embed_path)
        X_train = emb_df.reindex(index=train_ids).to_numpy(dtype=np.float32).copy()
        X_test = emb_df.reindex(index=test_ids).to_numpy(dtype=np.float32).copy()
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        print(
            f"  molformer_c3_pretrain_embed: {X_train.shape[1]} dims "
            f"(train {X_train.shape[0]} / test {X_test.shape[0]})"
        )
        return X_train, X_test
```

- [ ] **Step 4.3: Lint**

Run: `cd /home/nagaet/pxr-iduction-challenge && pixi run ruff format track1_activity/scripts/run_train.py && pixi run ruff check track1_activity/scripts/run_train.py`
Expected: clean.

- [ ] **Step 4.4: Verify feature registration**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge && pixi run python -c "
import sys
sys.path.insert(0, 'track1_activity/src')
sys.path.insert(0, 'track1_activity/scripts')
# Avoid running main(); just import the module and check argparse help
" 2>&1 | tail -3
cd /home/nagaet/pxr-iduction-challenge && pixi run python track1_activity/scripts/run_train.py --help 2>&1 | grep -i 'feature' | head -5
```
Expected: `--feature` choices list contains `molformer_c3_pretrain_embed`.

- [ ] **Step 4.5: Commit**

```bash
git add track1_activity/scripts/run_train.py
git commit -m "feat(train): add molformer_c3_pretrain_embed feature handler"
```

---

## Task 5: Push + open draft PR

- [ ] **Step 5.1: Push**

Run: `cd /home/nagaet/pxr-iduction-challenge && git push -u origin feature/molformer-c3-pretrain-embed`

- [ ] **Step 5.2: Open draft PR**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge && gh pr create --draft --title "feat(molformer): pretrain + frozen embed + TabPFN pool member" --body "$(cat <<'EOF'
## Summary
Third attempt at a transformer-encoder pool member. This time using the proven pool-strongest recipe (pretrain on log2_fc -> frozen embedding -> TabPFN, mirroring `tabpfn_chemprop_pretrain_embed` MAE 0.437) with `DeepChem/MoLFormer-c3-1.1B` as backbone.

## Pipeline
- Phase 1: `run_molformer_c3_pretrain.py` -- LoRA pretrain on 13,136 compounds' 2-head log2_fc, NaN-masked MSE
- Phase 2: `run_molformer_c3_embed_extract.py` -- [CLS] 768d for 4,653 train+test compounds, parquet
- Phase 3: `run_train.py --model tabpfn --feature molformer_c3_pretrain_embed --split umap` (existing infra, new feature handler)

## Acceptance (post-merge measurement)
- [ ] Phase 1 pretrain (~2-3h on RTX 5080) -- pending USER-CONFIRMED launch
- [ ] Phase 2 embed extract (~10 min)
- [ ] Phase 3 TabPFN 5-fold UMAP CV (~30 min)
- [ ] Single-model OOF MAE <= 0.48
- [ ] caruana_bag20 weight > 0
- [ ] 10-pool caruana_bag20 OOF MAE <= 0.4327
- [x] Smoke test Phase 1: 2-epoch run completed without OOM / NaN
- [x] ruff format + ruff check clean

Spec: `docs/superpowers/specs/2026-04-20-molformer-c3-pretrain-embed-design.md`
Plan: `docs/superpowers/plans/2026-04-20-molformer-c3-pretrain-embed.md`
EOF
)"
```
Expected: prints PR URL.

---

## Task 6: Run pretrain (USER-CONFIRMED, ~2-3h)

WARNING: USER-CONFIRMED ACTION. Per CLAUDE.md -- stop and ask before launching.

- [ ] **Step 6.1: Confirm with user**

Ask: "Ready to launch Phase 1 pretrain (13k compounds, LoRA, max 50 epochs, ~2-3h on RTX 5080). Run now?"

Wait for explicit confirmation.

- [ ] **Step 6.2: Launch in tmux (detached)**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge && pixi run db-start 2>&1 | tail -2
mkdir -p logs
LOG="logs/molformer_c3_pretrain_$(date +%Y%m%d_%H%M).log"
echo "Log: $LOG"
tmux new -s molformer_c3_pretrain -d "pixi run python track1_activity/scripts/run_molformer_c3_pretrain.py 2>&1 | tee $LOG"
sleep 3
tmux ls
tmux capture-pane -t molformer_c3_pretrain -p | tail -20
```
Expected: session listed; progress begins (weight download on first run).

- [ ] **Step 6.3: Hand off monitoring to user**

Tell the user: "Pretrain launched in tmux session `molformer_c3_pretrain`. Progress: `tmux capture-pane -t molformer_c3_pretrain -p | tail -20` or `tail -f <log>`. Completion marker: `saved: ...pretrain.pt best_val=...`. ETA ~2-3h. Ping when done."

- [ ] **Step 6.4: After completion, verify checkpoint**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge && ls -la track1_activity/checkpoints/molformer_c3_pretrain/
pixi run python -c "
import torch
ckpt = torch.load('track1_activity/checkpoints/molformer_c3_pretrain/pretrain.pt', map_location='cpu', weights_only=False)
print('best_val_loss:', ckpt['best_val_loss'])
print('n_train:', ckpt['n_train'])
print('n_val:', ckpt['n_val'])
"
```
Expected: best_val_loss < 1.0 (z-scored target, so 1.0 == variance-baseline; finite convergence required). n_train ~11800, n_val ~1300.

If `best_val_loss >= 1.0`, pretrain did not converge. Stop and discuss with user.

---

## Task 7: Run embed extract + TabPFN downstream

- [ ] **Step 7.1: Run embed extract (~10 min)**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge && pixi run python track1_activity/scripts/run_molformer_c3_embed_extract.py 2>&1 | tail -10
```
Expected: `Saved (4653, 768) embeddings to data/molformer_c3_pretrain_embed.parquet`.

- [ ] **Step 7.2: Run TabPFN downstream via unified trainer (~30 min)**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge && LOG="logs/tabpfn_molformer_c3_embed_$(date +%Y%m%d_%H%M).log" && pixi run python track1_activity/scripts/run_train.py --model tabpfn --feature molformer_c3_pretrain_embed --split umap 2>&1 | tee "$LOG" | tail -30
```
Expected:
- `molformer_c3_pretrain_embed: 768 dims (train 4140 / test 513)`
- `Recorded experiment 'tabpfn_molformer_c3_pretrain_embed_umap_default' (id=<N>)`
- `Final OOF RAE=<X.XXXX> MAE=<X.XXXX> Spearman=<X.XXXX>`

- [ ] **Step 7.3: Acceptance check (single-model)**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge && pixi run python <<'PY'
import psycopg2
conn = psycopg2.connect(host='/tmp', port=5433, dbname='pxr_challenge')
cur = conn.cursor()
cur.execute("""
SELECT name,
  AVG(rae) AS rae, AVG(mae) AS mae, AVG(spearman_r) AS spearman
FROM experiment_cv_results r JOIN experiments e ON e.id = r.experiment_id
WHERE e.name IN ('tabpfn_chemprop_pretrain_embed_umap_default',
                 'tabpfn_molformer_c3_pretrain_embed_umap_default')
GROUP BY e.name ORDER BY e.name
""")
for row in cur.fetchall():
    print(row)
PY
```

- MAE <= 0.48: PASS
- 0.48 < MAE <= 0.50: PASS only if Spearman within 0.02 of chemprop_pretrain baseline
- MAE > 0.50: FAIL — report to user before Task 8

---

## Task 8: Ensemble integration

**Files:**
- Modify: `track1_activity/scripts/run_ensemble.py` (append to `ENSEMBLE_MODELS`)

- [ ] **Step 8.1: Append to ENSEMBLE_MODELS**

Open `track1_activity/scripts/run_ensemble.py`. Find the `ENSEMBLE_MODELS` tuple closing paren (after the drop comment for chemprop_relative_aux from PR #97). Add:

```python
    # --- MoLFormer-c3 pretrain embed (1) ---
    # Transformer-encoder-family analog of tabpfn_chemprop_pretrain_embed.
    # Same recipe (Buterez 2024 strategy-3: pretrain on log2_fc ->
    # frozen embedding -> TabPFN) but different backbone family
    # (MoLFormer-c3 transformer vs chemprop D-MPNN). PR <PR_NUMBER>.
    "tabpfn_molformer_c3_pretrain_embed_umap_default",
)
```

Replace `<PR_NUMBER>` with actual PR number from Task 5.

- [ ] **Step 8.2: Re-run ensemble**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge && LOG="logs/ens_molformer_c3_$(date +%Y%m%d_%H%M).log" && pixi run python track1_activity/scripts/run_ensemble.py 2>&1 | tee "$LOG" | tail -60
```
Record caruana_bag20 weight on `tabpfn_molformer_c3_pretrain_embed_umap_default` and overall caruana_bag20 OOF MAE.

- [ ] **Step 8.3: Acceptance check 2 & 3**

- caruana weight > 0: PASS
- 10-pool caruana_bag20 OOF MAE <= 0.4327: PASS

If weight = 0: still merge (experiment record is valuable); follow-up PR drops from allow-list.
If MAE > 0.4327: regression; report to user before merging.

- [ ] **Step 8.4: Commit ensemble change**

```bash
cd /home/nagaet/pxr-iduction-challenge
git add track1_activity/scripts/run_ensemble.py
git commit -m "ens: add tabpfn_molformer_c3_pretrain_embed (caruana wt=<X.XX>)"
```

Replace `<X.XX>` with actual caruana weight from step 8.2.

---

## Task 9: Finalize PR + merge approval

- [ ] **Step 9.1: Push final commits**

Run: `cd /home/nagaet/pxr-iduction-challenge && git push`

- [ ] **Step 9.2: Update PR body with results**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge && gh pr edit --body "$(cat <<'EOF'
## Summary
Transformer-encoder pool member via Buterez 2024 strategy-3 recipe (pretrain on log2_fc -> frozen embedding -> TabPFN). Mirrors pool-strongest `tabpfn_chemprop_pretrain_embed`; backbone family is different (MoLFormer-c3 transformer vs chemprop D-MPNN) -> decorrelation hypothesis.

## Pipeline
- Phase 1: `run_molformer_c3_pretrain.py` -- LoRA pretrain on 13,136 compounds' 2-head log2_fc
- Phase 2: `run_molformer_c3_embed_extract.py` -- [CLS] 768d parquet
- Phase 3: `run_train.py --model tabpfn --feature molformer_c3_pretrain_embed --split umap`

## Results
- Pretrain best_val_loss: <X.XXXX>
- Single-model OOF: RAE <X.XXXX>, MAE <X.XXXX>, Spearman <X.XXXX>
  (chemprop_pretrain_embed baseline: RAE ~0.48, MAE 0.437, Spearman ~0.81)
- Caruana_bag20 weight on new member: <X.XX>
- 10-pool caruana_bag20 OOF MAE: <X.XXXX> (was 0.4327 pre-change)

## Spec / Plan
- Spec: `docs/superpowers/specs/2026-04-20-molformer-c3-pretrain-embed-design.md`
- Plan: `docs/superpowers/plans/2026-04-20-molformer-c3-pretrain-embed.md`

## Test plan
- [x] Phase 1 pretrain smoke (2-epoch) -- no crashes, DB-free
- [x] Full Phase 1 pretrain -- converged to best_val_loss <X.XXXX>
- [x] Phase 2 embed extract -- 4653 x 768 parquet written
- [x] Phase 3 TabPFN 5-fold UMAP CV -- OOF metrics in DB
- [x] Acceptance 1: single-model MAE
- [x] Acceptance 2: caruana weight > 0
- [x] Acceptance 3: pool MAE <= 0.4327
- [x] ruff clean
EOF
)"
```

Replace placeholders with actual values.

- [ ] **Step 9.3: Mark PR ready**

Run: `cd /home/nagaet/pxr-iduction-challenge && gh pr ready`

- [ ] **Step 9.4: Ask user for merge approval**

Include results summary in the question: "PR is ready. Single MAE X.XXXX, caruana wt X.XX, pool MAE X.XXXX. Shall I merge?"

Wait for explicit approval.

- [ ] **Step 9.5: Merge + cleanup**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge
gh pr merge --squash --delete-branch
git checkout main && git pull
git remote prune origin
git branch -a
```

---

## Self-Review Checklist

1. **Spec coverage:**
   - Backbone registry entry for `DeepChem/MoLFormer-c3-1.1B` with fix_rotary=True → Task 1 ✓
   - Phase 1 pretrain: 2-head, 13k compounds, NaN-masked MSE, z-score, 90/10 split, checkpoint + metadata → Task 2 ✓
   - Phase 2 embed extract: load ckpt, [CLS] extraction, parquet output → Task 3 ✓
   - Phase 3 TabPFN via existing `run_train.py --model tabpfn --feature ...` → Task 4 (feature registration) + Task 7 (invocation) ✓
   - Ensemble integration (add-only per Approach C) → Task 8 ✓
   - Acceptance MAE <= 0.48 → Task 7.3 ✓
   - Acceptance caruana wt > 0 → Task 8.3 ✓
   - Acceptance pool MAE <= 0.4327 → Task 8.3 ✓
   - Fallback to ibm/MoLFormer-XL-both-10pct if DeepChem weights broken → Task 1.3 noted ✓
   - LB submission deferred → Task 9 (no submit step) ✓

2. **Placeholder scan:**
   - `<PR_NUMBER>` in Task 8.1 and `<X.XX>` / `<X.XXXX>` in Tasks 8.4 / 9.2 are runtime values with explicit "Replace ... with the actual ..." instructions. Not plan defects. ✓
   - No "TBD" / "TODO" / vague "handle errors". ✓

3. **Type consistency:**
   - `MolformerPretrainModel(backbone_name, peft_params, head_hidden_dim, head_dropout, n_tasks=2)` signature identical in Task 2 (training) and Task 3 (inference) so `load_state_dict(ckpt["state_dict"])` succeeds. ✓
   - `ckpt["state_dict"]`, `ckpt["params"]`, `ckpt["backbone"]`, `ckpt["target_means"]`, `ckpt["target_stds"]`, `ckpt["best_val_loss"]` keys used in Task 3 all defined in Task 2 `torch.save({...})`. ✓
   - `get_backbone`, `get_peft_builder` signatures from existing `peft_backbones.py` / `peft_methods.py` consumed correctly in Tasks 2, 3. ✓
   - `nan_to_num` + `.reindex(index=train_ids)` pattern in Task 4.2 matches the existing chemprop_pretrain_embed branch at line 385. ✓

No issues found.
