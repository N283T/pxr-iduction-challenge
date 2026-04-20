# PEFT MoLFormer-XL (LoRA) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a generic PEFT trainer to track1_activity, train MoLFormer-XL with LoRA, save OOF predictions to DB, and verify the new member is picked up by the existing `caruana_bag20` ensemble.

**Architecture:** Three new src modules form a registry-pattern PEFT framework: `peft_backbones.py` (HF model id + LoRA target module names), `peft_methods.py` (peft library wrapper), `peft_trainer.py` (PyTorch model + train/predict loops). One CLI script `run_peft_finetune.py` drives Optuna-tuned 5-fold CV and writes results into the existing `experiments` / `experiment_oof_predictions` tables.

**Tech Stack:** Python 3.12, transformers 5.5, torch 2.10+cu13, peft >=0.12 (new dep), optuna 4.8, psycopg2 (existing); UMAP-split CV from `track1_activity/src/splits.py`; experiment helpers from `track1_activity/src/evaluate.py`.

**Spec:** `docs/superpowers/specs/2026-04-20-peft-molformer-lora-design.md`

**Branch:** `feature/peft-molformer-lora` (already created)

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | Modify (line 19-22) | Add `peft = ">=0.12,<0.14"` to `[tool.pixi.pypi-dependencies]` |
| `track1_activity/scripts/run_molformer_finetune.py` | Move | Relocate to `track1_activity/scripts/archive/run_molformer_finetune.py` (kept for historical reference) |
| `track1_activity/src/peft_backbones.py` | Create | Backbone registry (HF id, hidden dim, max_length, LoRA target module names) |
| `track1_activity/src/peft_methods.py` | Create | PEFT method registry; only `lora` for this PR |
| `track1_activity/src/peft_trainer.py` | Create | `MolFormerRegressor` nn.Module, `train_one_fold`, `predict`, GPU mem cleanup |
| `track1_activity/scripts/run_peft_finetune.py` | Create | CLI entry: argparse, Optuna driver, final CV, DB record, submission CSV |
| `track1_activity/scripts/run_ensemble.py` | Modify (line 80-132) | Append the new experiment name to `ENSEMBLE_MODELS` (only after Task 8 confirms the experiment exists in DB) |

---

## Task 1: Workspace prep — move legacy script, add peft dependency

**Files:**
- Move: `track1_activity/scripts/run_molformer_finetune.py` → `track1_activity/scripts/archive/run_molformer_finetune.py`
- Modify: `pyproject.toml:19-22`

- [ ] **Step 1.1: Move legacy MoLFormer full-FT prototype to archive**

```bash
git mv track1_activity/scripts/run_molformer_finetune.py track1_activity/scripts/archive/run_molformer_finetune.py
```

- [ ] **Step 1.2: Add peft to pyproject.toml**

Open `pyproject.toml` and edit the `[tool.pixi.pypi-dependencies]` block to add the `peft` line. The block currently looks like:

```toml
[tool.pixi.pypi-dependencies]
deepchem = ">=2.8, <3"
prolif = ">=2.1.0,<3"
pdbfixer = ">=1.11"
```

Change to:

```toml
[tool.pixi.pypi-dependencies]
deepchem = ">=2.8, <3"
prolif = ">=2.1.0,<3"
pdbfixer = ">=1.11"
peft = ">=0.12,<0.14"
```

- [ ] **Step 1.3: Install the new dep into the pixi env**

Run: `cd /home/nagaet/pxr-iduction-challenge && pixi install`
Expected: peft is downloaded and resolved cleanly. If it fails because of a transformers version conflict, capture the error verbatim and stop — do not downgrade transformers.

- [ ] **Step 1.4: Verify import works**

Run: `pixi run python -c "import peft; print('peft', peft.__version__); from peft import LoraConfig, get_peft_model; print('LoraConfig OK')"`
Expected: prints `peft 0.12.x` (or 0.13.x) and `LoraConfig OK`. No traceback.

- [ ] **Step 1.5: Commit**

```bash
git add pyproject.toml track1_activity/scripts/run_molformer_finetune.py track1_activity/scripts/archive/run_molformer_finetune.py
git commit -m "chore: archive legacy molformer full-FT script, add peft dep"
```

---

## Task 2: Backbone registry (`peft_backbones.py`)

**Files:**
- Create: `track1_activity/src/peft_backbones.py`

This module is a pure-data registry. No torch import here — keep it import-light so callers can `from peft_backbones import BACKBONES` without paying the torch import cost when only reading metadata.

The LoRA `target_modules` names are **placeholders that must be verified against the actual MoLFormer module tree** in Task 5 (smoke test). MoLFormer-XL uses linear-attention layers whose internal naming is not the standard BERT `query/key/value/dense`. We accept this as a known risk per the spec; Task 5 includes the verification + correction step.

- [ ] **Step 2.1: Write the registry file**

Create `track1_activity/src/peft_backbones.py` with this content:

```python
"""Backbone registry for PEFT fine-tuning.

Each entry describes a Hugging Face model that can be wrapped with peft
(LoRA / adapter / etc.). The registry is intentionally pure data: no
torch / transformers / peft imports here so callers can read metadata
without paying the model-loading cost.

Adding a new backbone:
1. Append a new entry below.
2. Verify the LoRA target_modules names by inspecting
   ``dict(AutoModel.from_pretrained(hf_id).named_modules()).keys()``
   once during smoke test, and update lora_target_modules_* if needed.
"""

BACKBONES: dict[str, dict] = {
    "molformer_xl": {
        "hf_id": "ibm/MoLFormer-XL-both-10pct",
        "hidden_dim": 768,
        "max_length": 202,
        "trust_remote_code": True,
        # LoRA target submodule name fragments. peft matches these as
        # substrings against module.named_modules() keys, so partial names
        # are fine. These BERT-style names are placeholders -- verify and
        # update during smoke test (see Task 5).
        "lora_target_modules_qv": ["query", "value"],
        "lora_target_modules_qkvo": ["query", "key", "value", "dense"],
    },
}


def get_backbone(name: str) -> dict:
    """Return the backbone metadata dict, or raise KeyError with a helpful list."""
    if name not in BACKBONES:
        available = ", ".join(sorted(BACKBONES))
        raise KeyError(f"Unknown backbone '{name}'. Available: {available}")
    return BACKBONES[name]
```

- [ ] **Step 2.2: Smoke check the registry**

Run: `cd /home/nagaet/pxr-iduction-challenge && pixi run python -c "import sys; sys.path.insert(0, 'track1_activity/src'); from peft_backbones import get_backbone; m = get_backbone('molformer_xl'); print(m['hf_id'], m['hidden_dim'])"`
Expected: prints `ibm/MoLFormer-XL-both-10pct 768`. No traceback.

- [ ] **Step 2.3: Run linters**

Run: `cd /home/nagaet/pxr-iduction-challenge && pixi run ruff format track1_activity/src/peft_backbones.py && pixi run ruff check track1_activity/src/peft_backbones.py && pixi run ty check track1_activity/src/peft_backbones.py`
Expected: ruff prints "1 file reformatted" or "1 file already formatted", check prints "All checks passed", ty prints no errors.

- [ ] **Step 2.4: Commit**

```bash
git add track1_activity/src/peft_backbones.py
git commit -m "feat(peft): backbone registry with MoLFormer-XL entry"
```

---

## Task 3: PEFT method registry (`peft_methods.py`)

**Files:**
- Create: `track1_activity/src/peft_methods.py`

- [ ] **Step 3.1: Write the registry**

Create `track1_activity/src/peft_methods.py` with:

```python
"""PEFT method registry.

Each entry maps a method name to a builder that turns hyperparameters
into a peft Config object. Only LoRA is implemented in this PR; adapter
and last-k-layer FT will be added in follow-up PRs.
"""

from typing import Callable

from peft import LoraConfig, PeftConfig


def build_lora_config(backbone_meta: dict, params: dict) -> LoraConfig:
    """Build a LoraConfig from hyperparameters.

    ``params`` keys:
        lora_rank: int
        lora_alpha: int
        lora_dropout: float
        lora_target: "qv" or "qkvo"
    """
    target_key = f"lora_target_modules_{params['lora_target']}"
    target_modules = backbone_meta[target_key]
    return LoraConfig(
        r=params["lora_rank"],
        lora_alpha=params["lora_alpha"],
        lora_dropout=params["lora_dropout"],
        target_modules=target_modules,
        bias="none",
        # Custom regression head -- do not let peft inject a task head.
        task_type=None,
    )


PEFT_METHODS: dict[str, Callable[[dict, dict], PeftConfig]] = {
    "lora": build_lora_config,
}


def get_peft_builder(method: str) -> Callable[[dict, dict], PeftConfig]:
    if method not in PEFT_METHODS:
        available = ", ".join(sorted(PEFT_METHODS))
        raise KeyError(f"Unknown PEFT method '{method}'. Available: {available}")
    return PEFT_METHODS[method]
```

- [ ] **Step 3.2: Smoke check**

Run: `cd /home/nagaet/pxr-iduction-challenge && pixi run python -c "
import sys
sys.path.insert(0, 'track1_activity/src')
from peft_backbones import get_backbone
from peft_methods import get_peft_builder
meta = get_backbone('molformer_xl')
cfg = get_peft_builder('lora')(meta, {'lora_rank': 8, 'lora_alpha': 16, 'lora_dropout': 0.1, 'lora_target': 'qv'})
print(type(cfg).__name__, 'rank=', cfg.r, 'alpha=', cfg.lora_alpha, 'targets=', cfg.target_modules)
"`
Expected: prints `LoraConfig rank= 8 alpha= 16 targets= ['query', 'value']`. No traceback.

- [ ] **Step 3.3: Lint**

Run: `cd /home/nagaet/pxr-iduction-challenge && pixi run ruff format track1_activity/src/peft_methods.py && pixi run ruff check track1_activity/src/peft_methods.py && pixi run ty check track1_activity/src/peft_methods.py`
Expected: clean.

- [ ] **Step 3.4: Commit**

```bash
git add track1_activity/src/peft_methods.py
git commit -m "feat(peft): method registry with LoRA builder"
```

---

## Task 4: Trainer module (`peft_trainer.py`)

**Files:**
- Create: `track1_activity/src/peft_trainer.py`

This is the largest file (~250 lines). It wraps a peft-augmented HF backbone with a regression head, trains one fold with early stopping on validation MAE, and produces predictions.

- [ ] **Step 4.1: Write the trainer**

Create `track1_activity/src/peft_trainer.py` with:

```python
"""PEFT fine-tuning trainer for SMILES regression.

Wraps a Hugging Face backbone with a peft adapter (LoRA / etc.) and a
2-layer MLP regression head. Provides ``train_one_fold`` which returns
val and (optionally) test predictions for a single CV fold, and frees
GPU memory at the end so 5-fold runs stay inside 16GB VRAM.
"""

import numpy as np
import torch
import torch.nn as nn
from peft import get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from peft_backbones import get_backbone
from peft_methods import get_peft_builder


class SmilesDataset(Dataset):
    """Tokenises SMILES on the fly, optionally returns regression targets."""

    def __init__(
        self,
        smiles_list: list[str],
        tokenizer,
        targets: np.ndarray | None = None,
        max_length: int = 202,
    ):
        self.smiles = smiles_list
        self.tokenizer = tokenizer
        self.targets = targets
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.smiles[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        if self.targets is not None:
            item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return item


class PeftRegressor(nn.Module):
    """Generic PEFT-wrapped backbone + 2-layer MLP regression head."""

    def __init__(
        self,
        backbone_name: str,
        peft_method: str,
        peft_params: dict,
        head_hidden_dim: int,
        head_dropout: float,
    ):
        super().__init__()
        meta = get_backbone(backbone_name)
        base = AutoModel.from_pretrained(
            meta["hf_id"], trust_remote_code=meta["trust_remote_code"]
        )
        peft_config = get_peft_builder(peft_method)(meta, peft_params)
        self.backbone = get_peft_model(base, peft_config)

        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(meta["hidden_dim"], head_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token embedding (index 0).
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_emb).squeeze(-1)


def _predict(model: PeftRegressor, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            preds = model(input_ids, attention_mask)
            chunks.append(preds.cpu().numpy())
    return np.concatenate(chunks)


def get_tokenizer(backbone_name: str):
    """Load and cache the tokenizer for a backbone."""
    meta = get_backbone(backbone_name)
    return AutoTokenizer.from_pretrained(
        meta["hf_id"], trust_remote_code=meta["trust_remote_code"]
    )


def train_one_fold(
    params: dict,
    backbone_name: str,
    peft_method: str,
    tokenizer,
    train_smiles: list[str],
    train_targets: np.ndarray,
    val_smiles: list[str],
    val_targets: np.ndarray,
    test_smiles: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Train one CV fold; return (val_preds, test_preds_or_None).

    ``params`` keys consumed:
        lora_rank, lora_alpha, lora_dropout, lora_target  (peft method args)
        head_hidden_dim, head_dropout                     (head args)
        backbone_lr, head_lr, weight_decay                (optimizer)
        batch_size, max_epochs, patience                  (training loop)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = get_backbone(backbone_name)

    train_ds = SmilesDataset(
        train_smiles, tokenizer, train_targets, max_length=meta["max_length"]
    )
    val_ds = SmilesDataset(
        val_smiles, tokenizer, val_targets, max_length=meta["max_length"]
    )
    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False)

    peft_params = {
        "lora_rank": params["lora_rank"],
        "lora_alpha": params["lora_alpha"],
        "lora_dropout": params["lora_dropout"],
        "lora_target": params["lora_target"],
    }
    model = PeftRegressor(
        backbone_name=backbone_name,
        peft_method=peft_method,
        peft_params=peft_params,
        head_hidden_dim=params["head_hidden_dim"],
        head_dropout=params["head_dropout"],
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
    criterion = nn.MSELoss()

    best_val_mae = float("inf")
    best_state = None
    patience_counter = 0

    for _epoch in range(params["max_epochs"]):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            preds = model(input_ids, attention_mask)
            loss = criterion(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        val_preds = _predict(model, val_loader, device)
        val_mae = float(np.mean(np.abs(val_targets - val_preds)))

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= params["patience"]:
                break

    if best_state is None:
        # Should not happen unless max_epochs=0; fail loudly so the caller
        # does not silently use a random-init model.
        del model
        torch.cuda.empty_cache()
        raise RuntimeError(
            f"Training never improved val MAE; best={best_val_mae}. "
            "Check learning rate / max_epochs / data."
        )

    model.load_state_dict(best_state)
    val_preds = _predict(model, val_loader, device)

    test_preds = None
    if test_smiles is not None:
        test_ds = SmilesDataset(test_smiles, tokenizer, max_length=meta["max_length"])
        test_loader = DataLoader(test_ds, batch_size=params["batch_size"], shuffle=False)
        test_preds = _predict(model, test_loader, device)

    del model
    torch.cuda.empty_cache()

    return val_preds, test_preds
```

- [ ] **Step 4.2: Lint**

Run: `cd /home/nagaet/pxr-iduction-challenge && pixi run ruff format track1_activity/src/peft_trainer.py && pixi run ruff check track1_activity/src/peft_trainer.py && pixi run ty check track1_activity/src/peft_trainer.py`
Expected: clean. If ty complains about peft type stubs missing, that is OK — those are runtime objects without type info; if it errors, add `# ty: ignore[unresolved-import]` on the peft import line.

- [ ] **Step 4.3: Commit**

```bash
git add track1_activity/src/peft_trainer.py
git commit -m "feat(peft): trainer module with PeftRegressor and train_one_fold"
```

---

## Task 5: Smoke test trainer + verify LoRA target_modules

This is the critical risk-mitigation step from the spec. Before writing the CLI / Optuna driver we must confirm the LoRA target module names actually match MoLFormer-XL's submodules.

**Files:** none (read-only inspection + possibly `peft_backbones.py` correction)

- [ ] **Step 5.1: Inspect MoLFormer-XL module tree**

Run: `cd /home/nagaet/pxr-iduction-challenge && pixi run python -c "
from transformers import AutoModel
m = AutoModel.from_pretrained('ibm/MoLFormer-XL-both-10pct', trust_remote_code=True)
linear_names = sorted({n.split('.')[-1] for n, mod in m.named_modules() if type(mod).__name__ == 'Linear'})
print('Linear submodule names:', linear_names)
print()
print('Sample module paths (first 30):')
for i, (n, mod) in enumerate(m.named_modules()):
    if type(mod).__name__ == 'Linear':
        print(' ', n)
    if i > 80: break
" 2>&1 | tail -40`

Read the output. The interesting names are typically things like `query`, `key`, `value`, `dense`, `q_proj`, `k_proj`, `v_proj`, `out_proj`, `output.dense`, `attention.output.dense`. Note the actual names.

- [ ] **Step 5.2: Update `peft_backbones.py` if names differ**

If the printed names do not include `query`/`key`/`value`/`dense`, edit `track1_activity/src/peft_backbones.py` and replace the placeholder lists in the `molformer_xl` entry with the actual names. Two cases to handle:

  - **Case A (BERT-style)**: names contain `query`, `key`, `value`, `dense` — leave `peft_backbones.py` unchanged.
  - **Case B (proj-style)**: names contain `q_proj`, `k_proj`, `v_proj`, `out_proj` — change the registry to:
    ```python
    "lora_target_modules_qv": ["q_proj", "v_proj"],
    "lora_target_modules_qkvo": ["q_proj", "k_proj", "v_proj", "out_proj"],
    ```
  - **Case C (other naming)**: pick the attention input/output linear layers analogously and document the naming in a comment above the dict entry.

If you make changes, re-run the linters from Task 2.3.

- [ ] **Step 5.3: Smoke train one tiny fold**

Run: `cd /home/nagaet/pxr-iduction-challenge && pixi run db-start && pixi run python -c "
import sys
sys.path.insert(0, 'track1_activity/src')
import numpy as np
from data import load_train_smiles_target
from peft_trainer import get_tokenizer, train_one_fold

df = load_train_smiles_target().head(100)
smi = df['smiles'].tolist()
y = df['pec50'].to_numpy()
tok = get_tokenizer('molformer_xl')

params = {
    'lora_rank': 8, 'lora_alpha': 16, 'lora_dropout': 0.1, 'lora_target': 'qv',
    'head_hidden_dim': 128, 'head_dropout': 0.2,
    'backbone_lr': 1e-4, 'head_lr': 1e-3, 'weight_decay': 1e-3,
    'batch_size': 16, 'max_epochs': 2, 'patience': 2,
}
val_pred, _ = train_one_fold(params, 'molformer_xl', 'lora', tok, smi[:80], y[:80], smi[80:], y[80:])
print('val_pred shape:', val_pred.shape, 'mean:', float(val_pred.mean()), 'has_nan:', bool(np.isnan(val_pred).any()))
" 2>&1 | tail -20`

Expected: prints `val_pred shape: (20,) mean: <some_float> has_nan: False`. If you get a `ValueError: Target modules ... not found` from peft, your target_modules names are wrong — go back to step 5.2.

- [ ] **Step 5.4: Commit (only if you changed `peft_backbones.py`)**

If step 5.2 modified the registry:

```bash
git add track1_activity/src/peft_backbones.py
git commit -m "fix(peft): correct MoLFormer-XL LoRA target_modules names"
```

If unchanged: skip this step.

---

## Task 6: CLI entry point (`run_peft_finetune.py`)

**Files:**
- Create: `track1_activity/scripts/run_peft_finetune.py`

- [ ] **Step 6.1: Write the CLI script**

Create `track1_activity/scripts/run_peft_finetune.py` with:

```python
"""PEFT fine-tuning CLI for PXR pEC50 regression.

Trains a Hugging Face backbone wrapped with peft (LoRA by default) on
the train_activity table, runs Optuna hyperparameter search on inner
CV folds, then produces 5-fold OOF predictions plus a test submission
CSV. Records everything in the experiments / experiment_oof_predictions
DB tables so the result feeds straight into ``run_ensemble.py``.

Usage:
    pixi run python track1_activity/scripts/run_peft_finetune.py \\
        --backbone molformer_xl --peft-method lora \\
        --n-trials 20 --inner-folds 3 --outer-folds 5 --split umap

Smoke test:
    ... --n-trials 1 --inner-folds 2 --outer-folds 2 \\
        --max-epochs-final 2 --patience-final 2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))

import numpy as np
import optuna
import pandas as pd
import torch

from data import load_train_smiles_target, load_test_smiles
from evaluate import (
    compute_metrics,
    print_fold_summary,
    print_metrics,
    record_experiment,
    save_oof_predictions,
)
from peft_trainer import get_tokenizer, train_one_fold
from splits import scaffold_split_indices, umap_split_indices

SUBMISSION_DIR = Path(__file__).resolve().parent.parent.joinpath("submissions")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)


def suggest_params(
    trial: optuna.Trial, max_epochs: int, patience: int
) -> dict:
    """Optuna search space for LoRA fine-tuning."""
    rank = trial.suggest_categorical("lora_rank", [4, 8, 16, 32])
    alpha_mult = trial.suggest_categorical("lora_alpha_mult", [1, 2])
    return {
        "lora_rank": rank,
        "lora_alpha": rank * alpha_mult,
        "lora_dropout": trial.suggest_float("lora_dropout", 0.0, 0.2),
        "lora_target": trial.suggest_categorical("lora_target", ["qv", "qkvo"]),
        "head_hidden_dim": trial.suggest_categorical(
            "head_hidden_dim", [128, 256, 512]
        ),
        "head_dropout": trial.suggest_float("head_dropout", 0.1, 0.4),
        "backbone_lr": trial.suggest_float("backbone_lr", 1e-5, 5e-4, log=True),
        "head_lr": trial.suggest_float("head_lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "max_epochs": max_epochs,
        "patience": patience,
    }


def objective(
    trial: optuna.Trial,
    backbone: str,
    peft_method: str,
    tokenizer,
    train_smiles: list[str],
    y_train: np.ndarray,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    max_epochs: int,
    patience: int,
) -> float:
    params = suggest_params(trial, max_epochs, patience)
    fold_raes = []
    for fold, (tr_idx, va_idx) in enumerate(inner_splits):
        tr_smi = [train_smiles[i] for i in tr_idx]
        va_smi = [train_smiles[i] for i in va_idx]
        tr_y = y_train[tr_idx]
        va_y = y_train[va_idx]
        try:
            val_pred, _ = train_one_fold(
                params, backbone, peft_method, tokenizer,
                tr_smi, tr_y, va_smi, va_y,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()
        if np.isnan(val_pred).any():
            raise optuna.TrialPruned()
        metrics = compute_metrics(va_y, val_pred)
        fold_raes.append(metrics["RAE"])
        trial.report(float(np.mean(fold_raes)), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(fold_raes))


def run_final_cv(
    best_params: dict,
    backbone: str,
    peft_method: str,
    tokenizer,
    train_smiles: list[str],
    y_train: np.ndarray,
    test_smiles: list[str],
    test_df: pd.DataFrame,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    split_name: str,
    max_epochs_final: int,
    patience_final: int,
) -> dict:
    """Final outer CV + DB record + submission CSV."""
    final_params = {
        **best_params,
        "max_epochs": max_epochs_final,
        "patience": patience_final,
    }
    rank = final_params["lora_rank"]
    alpha = final_params["lora_alpha"]
    name = f"peft_{backbone}_{peft_method}_r{rank}a{alpha}_{split_name}_default"

    print(f"\n{'=' * 60}")
    print(f"  Final CV: {name}")
    print(f"  Params: {final_params}")
    print(f"{'=' * 60}")

    oof_preds = np.zeros(len(y_train))
    test_preds_all = np.zeros((len(outer_splits), len(test_smiles)))
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(outer_splits):
        print(f"\n  --- Fold {fold} ---")
        tr_smi = [train_smiles[i] for i in tr_idx]
        va_smi = [train_smiles[i] for i in va_idx]
        tr_y = y_train[tr_idx]
        va_y = y_train[va_idx]

        val_pred, test_pred = train_one_fold(
            final_params, backbone, peft_method, tokenizer,
            tr_smi, tr_y, va_smi, va_y, test_smiles,
        )
        if np.isnan(val_pred).any():
            raise ValueError(f"Fold {fold} produced NaN val predictions")
        oof_preds[va_idx] = val_pred
        test_preds_all[fold] = test_pred

        m = compute_metrics(va_y, val_pred)
        fold_metrics.append(m)
        print_metrics(m, label=f"Fold {fold}")

    oof_metrics = compute_metrics(y_train, oof_preds)
    print("\n  Overall OOF:")
    print_metrics(oof_metrics)
    print_fold_summary(fold_metrics)

    test_preds_avg = test_preds_all.mean(axis=0)
    submission = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": test_preds_avg,
        }
    )
    sub_path = SUBMISSION_DIR.joinpath(f"{name}.csv")
    submission.to_csv(sub_path, index=False)
    print(f"  Wrote submission: {sub_path}")

    exp_id = record_experiment(
        name=name,
        description=f"PEFT {backbone} {peft_method} ({split_name} split, optuna-tuned)",
        model_type="peft_finetune",
        feature_set="smiles_transformer_peft",
        hyperparameters=final_params,
        fold_metrics=fold_metrics,
        submission_path=f"track1_activity/submissions/{name}.csv",
        notes=(
            f"OOF RAE={oof_metrics['RAE']:.4f}, MAE={oof_metrics['MAE']:.4f}, "
            f"{split_name}_split, peft={peft_method}"
        ),
    )
    save_oof_predictions(exp_id, oof_preds)
    return oof_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="PEFT fine-tuning with Optuna")
    parser.add_argument("--backbone", default="molformer_xl")
    parser.add_argument("--peft-method", default="lora")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--split", choices=["umap", "scaffold"], default="umap")
    parser.add_argument("--n-clusters", type=int, default=50)
    parser.add_argument("--max-epochs-trial", type=int, default=50)
    parser.add_argument("--patience-trial", type=int, default=8)
    parser.add_argument("--max-epochs-final", type=int, default=80)
    parser.add_argument("--patience-final", type=int, default=12)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    print("Loading data...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    train_smiles = train_df["smiles"].tolist()
    y_train = train_df["pec50"].to_numpy()
    test_smiles = test_df["smiles"].tolist()
    print(f"Train: {len(train_smiles)}, Test: {len(test_smiles)}")
    print(
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
    print(f"Backbone={args.backbone}, PEFT={args.peft_method}, Split={args.split}")

    print(f"Loading tokenizer for {args.backbone}...")
    tokenizer = get_tokenizer(args.backbone)

    if args.split == "umap":
        outer_splits = umap_split_indices(
            train_smiles, n_splits=args.outer_folds, n_clusters=args.n_clusters, seed=42
        )
        inner_splits = umap_split_indices(
            train_smiles, n_splits=args.inner_folds, n_clusters=args.n_clusters, seed=123
        )
    else:
        outer_splits = scaffold_split_indices(train_smiles, n_splits=args.outer_folds, seed=42)
        inner_splits = scaffold_split_indices(train_smiles, n_splits=args.inner_folds, seed=123)

    print(f"\n{'=' * 60}")
    print(f"  Optuna tuning: {args.n_trials} trials, {args.inner_folds}-fold inner CV")
    print(f"{'=' * 60}")

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(
        lambda t: objective(
            t, args.backbone, args.peft_method, tokenizer, train_smiles, y_train,
            inner_splits, args.max_epochs_trial, args.patience_trial,
        ),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    n_failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
    print(f"  Trials: {n_complete} complete, {n_pruned} pruned, {n_failed} FAILED")
    if n_complete == 0:
        raise RuntimeError("All Optuna trials failed. Cannot proceed.")
    print(f"  Best trial RAE: {study.best_value:.4f}")
    print(f"  Best params:    {study.best_params}")

    # Reconstruct best_params for the trainer (lora_alpha = rank * mult).
    rank = int(study.best_params["lora_rank"])
    alpha_mult = int(study.best_params["lora_alpha_mult"])
    best_params = {
        "lora_rank": rank,
        "lora_alpha": rank * alpha_mult,
        "lora_dropout": float(study.best_params["lora_dropout"]),
        "lora_target": study.best_params["lora_target"],
        "head_hidden_dim": int(study.best_params["head_hidden_dim"]),
        "head_dropout": float(study.best_params["head_dropout"]),
        "backbone_lr": float(study.best_params["backbone_lr"]),
        "head_lr": float(study.best_params["head_lr"]),
        "weight_decay": float(study.best_params["weight_decay"]),
        "batch_size": int(study.best_params["batch_size"]),
    }

    oof_metrics = run_final_cv(
        best_params, args.backbone, args.peft_method, tokenizer,
        train_smiles, y_train, test_smiles, test_df,
        outer_splits, args.split, args.max_epochs_final, args.patience_final,
    )
    print(f"\n  Final OOF RAE: {oof_metrics['RAE']:.4f}, MAE: {oof_metrics['MAE']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.2: Lint**

Run: `cd /home/nagaet/pxr-iduction-challenge && pixi run ruff format track1_activity/scripts/run_peft_finetune.py && pixi run ruff check track1_activity/scripts/run_peft_finetune.py && pixi run ty check track1_activity/scripts/run_peft_finetune.py`
Expected: clean.

- [ ] **Step 6.3: Smoke test the CLI end-to-end**

Run: `cd /home/nagaet/pxr-iduction-challenge && pixi run db-start && time pixi run python track1_activity/scripts/run_peft_finetune.py --n-trials 1 --inner-folds 2 --outer-folds 2 --max-epochs-trial 2 --patience-trial 2 --max-epochs-final 2 --patience-final 2 2>&1 | tail -40`

Expected: completes in under 10 minutes wall clock; final output shows `Final OOF RAE: <number>, MAE: <number>` and `Recorded experiment 'peft_molformer_xl_lora_r{R}a{A}_umap_default' (id={N})`. The recorded RAE will be poor (only 2 epochs on tiny folds) — this is fine; we only need to confirm no crashes and that DB rows land.

- [ ] **Step 6.4: Verify smoke-test DB rows**

Run: `cd /home/nagaet/pxr-iduction-challenge && pixi run python -c "
import psycopg2
conn = psycopg2.connect(host='/tmp', port=5433, dbname='pxr_challenge')
cur = conn.cursor()
cur.execute(\"SELECT id, name, model_type FROM experiments WHERE name LIKE 'peft_molformer_xl_lora_%' ORDER BY id DESC LIMIT 3\")
print('experiments:', cur.fetchall())
cur.execute(\"SELECT experiment_id, COUNT(*) FROM experiment_oof_predictions WHERE experiment_id IN (SELECT id FROM experiments WHERE name LIKE 'peft_molformer_xl_lora_%') GROUP BY experiment_id ORDER BY experiment_id DESC LIMIT 3\")
print('oof rows:', cur.fetchall())
"`
Expected: at least one experiment row with `model_type='peft_finetune'`, and an oof_rows count equal to the train set size (4140) for that experiment.

- [ ] **Step 6.5: Delete the smoke-test experiment from DB**

Smoke-test trained on only 2 epochs / 2 folds, so the OOF predictions are garbage and would corrupt the ensemble. Delete them before continuing.

Run: `cd /home/nagaet/pxr-iduction-challenge && pixi run python -c "
import psycopg2
conn = psycopg2.connect(host='/tmp', port=5433, dbname='pxr_challenge')
cur = conn.cursor()
cur.execute(\"SELECT id FROM experiments WHERE name LIKE 'peft_molformer_xl_lora_%' AND notes LIKE '%peft=lora%' ORDER BY id DESC LIMIT 1\")
ids = [r[0] for r in cur.fetchall()]
for eid in ids:
    cur.execute('DELETE FROM experiment_oof_predictions WHERE experiment_id=%s', (eid,))
    cur.execute('DELETE FROM experiment_cv_results WHERE experiment_id=%s', (eid,))
    cur.execute('DELETE FROM experiments WHERE id=%s', (eid,))
    print(f'Deleted smoke-test experiment id={eid}')
conn.commit()
"`
Expected: prints `Deleted smoke-test experiment id=<N>`.

- [ ] **Step 6.6: Commit**

```bash
git add track1_activity/scripts/run_peft_finetune.py
git commit -m "feat(peft): CLI entry with Optuna driver and 5-fold final CV"
```

---

## Task 7: Push, open draft PR, run CI

The full Optuna run takes hours. Push first so CI runs in parallel and so the user has a reviewable surface while we wait.

**Files:** none

- [ ] **Step 7.1: Push branch**

Run: `cd /home/nagaet/pxr-iduction-challenge && git push -u origin feature/peft-molformer-lora`
Expected: push succeeds. If it complains about `--no-verify`, do not bypass — investigate and fix the failing pre-push hook.

- [ ] **Step 7.2: Open draft PR**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge && gh pr create --draft --title "feat(peft): MoLFormer-XL LoRA fine-tune as ensemble member" --body "$(cat <<'EOF'
## Summary
- Generic PEFT trainer (peft_backbones / peft_methods / peft_trainer) keyed by registry pattern so future PRs add backbones / methods with one entry each
- MoLFormer-XL x LoRA via `pixi run python track1_activity/scripts/run_peft_finetune.py`
- Records into experiments table as `peft_molformer_xl_lora_r{R}a{A}_umap_default` with full OOF predictions for ensemble integration

Spec: docs/superpowers/specs/2026-04-20-peft-molformer-lora-design.md
Plan: docs/superpowers/plans/2026-04-20-peft-molformer-lora.md

## Test plan
- [x] Smoke test: `--n-trials 1 --outer-folds 2 --max-epochs-final 2` completed without NaN, DB rows land
- [ ] Full Optuna 20-trial + 5-fold final CV (running in background)
- [ ] OOF MAE <= 0.50
- [ ] caruana_bag20 assigns weight > 0 to the new member
- [ ] CI green (ruff + ty)
EOF
)"
```
Expected: prints PR URL. Capture the PR number for later steps.

- [ ] **Step 7.3: Watch CI**

Run: `cd /home/nagaet/pxr-iduction-challenge && gh pr checks --watch`
Expected: ruff/ty checks pass. If they fail, fix and re-push before continuing.

---

## Task 8: Run full Optuna + final 5-fold CV

⚠️ **USER-CONFIRMED ACTION**: This is the long-running compute step (estimated 6-12h on RTX 5080). Per CLAUDE.md "Never run benchmarks, long-running computations, or destructive operations without explicit user permission" — **stop here and ask the user to confirm before launching**.

**Files:** none (writes DB + submission CSV)

- [ ] **Step 8.1: Confirm with the user**

Ask: "Ready to launch the full Optuna run (20 trials × 3 inner folds, then 5-fold final CV with 80 epoch cap). ETA 6-12h on RTX 5080. Run now or wait?"

Wait for explicit confirmation. Do NOT proceed without it.

- [ ] **Step 8.2: Launch in tmux (resume-safe)**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge && pixi run db-start
tmux new -s peft_lora -d "pixi run python track1_activity/scripts/run_peft_finetune.py 2>&1 | tee logs/peft_molformer_lora_$(date +%Y%m%d_%H%M).log"
```
Expected: tmux session `peft_lora` is detached and running. Verify with `tmux ls`.

- [ ] **Step 8.3: Periodically check progress**

Run: `cd /home/nagaet/pxr-iduction-challenge && tmux capture-pane -t peft_lora -p | tail -30`
Expected: see Optuna trial progress bars or fold output. If you see a stack trace, capture it and stop.

- [ ] **Step 8.4: After completion, query DB for the experiment**

Run: `cd /home/nagaet/pxr-iduction-challenge && pixi run python -c "
import psycopg2
conn = psycopg2.connect(host='/tmp', port=5433, dbname='pxr_challenge')
cur = conn.cursor()
cur.execute(\"\"\"
SELECT id, name, notes,
  (SELECT AVG(rae) FROM experiment_cv_results WHERE experiment_id=experiments.id) AS rae,
  (SELECT AVG(mae) FROM experiment_cv_results WHERE experiment_id=experiments.id) AS mae,
  (SELECT COUNT(*) FROM experiment_oof_predictions WHERE experiment_id=experiments.id) AS n_oof
FROM experiments
WHERE name LIKE 'peft_molformer_xl_lora_%' ORDER BY id DESC LIMIT 1
\"\"\")
print(cur.fetchone())
"`
Expected: a row showing OOF MAE roughly between 0.45 and 0.55, n_oof = 4140. Capture the experiment name (will be needed in Task 9).

- [ ] **Step 8.5: Acceptance check 1 — OOF MAE <= 0.50**

If the printed MAE > 0.50: stop, report to the user, and do NOT continue to Task 9. The model is too weak to ensemble-evaluate cleanly. Possible fixes: increase `--max-epochs-final`, try `--n-trials 30`, or pivot to ChemFM (PR 3).

If MAE <= 0.50: proceed.

---

## Task 9: Ensemble integration

**Files:**
- Modify: `track1_activity/scripts/run_ensemble.py:80-132` (append to `ENSEMBLE_MODELS`)

- [ ] **Step 9.1: Append the new model name to ENSEMBLE_MODELS**

Open `track1_activity/scripts/run_ensemble.py` and edit the `ENSEMBLE_MODELS` tuple to append the experiment name from Task 8.4. Find the closing `)` of the tuple (around line 132 — currently after `"tabpfn_pooled_boltz_allpairs_umap_default",`) and add a new entry just before the close-paren:

```python
    # --- PEFT transformer encoder (1) ---
    # First transformer encoder in the pool (chemprop / TabPFN / Boltz only
    # before this). Aims to add encoder-family orthogonality. PR
    # feature/peft-molformer-lora.
    "peft_molformer_xl_lora_r<R>a<A>_umap_default",
)
```

Replace `<R>a<A>` with the actual rank/alpha values from the experiment name captured in Task 8.4.

- [ ] **Step 9.2: Re-run the ensemble**

Run: `cd /home/nagaet/pxr-iduction-challenge && pixi run python track1_activity/scripts/run_ensemble.py 2>&1 | tee logs/peft_lora_ens_$(date +%Y%m%d_%H%M).log | tail -80`
Expected: prints per-model OOF RAEs (the new model included), then strategy comparison including `caruana_bag20` with the new model's weight. Capture the new model's caruana weight.

- [ ] **Step 9.3: Acceptance check 2 — caruana_bag20 weight > 0**

Read the `ens_caruana_bag20` block in the printed output. The new member should appear with a numeric weight.

- If weight > 0: success. Document in PR body. Continue to Task 10.
- If weight = 0: the model is OOF-redundant with the existing pool. Per spec, this still merges but is documented as "explored, no ensemble lift". Update PR body with this finding. Continue to Task 10.

- [ ] **Step 9.4: Commit ENSEMBLE_MODELS change**

```bash
cd /home/nagaet/pxr-iduction-challenge
git add track1_activity/scripts/run_ensemble.py
git commit -m "ens: add peft_molformer_xl_lora to allow-list (caruana wt=<X.XX>)"
```

Replace `<X.XX>` with the actual caruana weight from step 9.3.

---

## Task 10: Finalize PR, ask user for merge approval

**Files:** none (PR body update + git push)

- [ ] **Step 10.1: Push final commits**

Run: `cd /home/nagaet/pxr-iduction-challenge && git push`
Expected: push succeeds.

- [ ] **Step 10.2: Update PR body with results**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge && gh pr edit --body "$(cat <<'EOF'
## Summary
- Generic PEFT trainer (peft_backbones / peft_methods / peft_trainer) keyed by registry pattern so future PRs add backbones / methods with one entry each
- MoLFormer-XL x LoRA via `pixi run python track1_activity/scripts/run_peft_finetune.py`
- New ensemble member: `peft_molformer_xl_lora_r<R>a<A>_umap_default`

## Results
- OOF MAE: <X.XXXX> (RAE: <X.XXXX>)
- Caruana_bag20 weight on new member: <X.XX>
- Pool size: 9 -> 10
- New ensemble OOF MAE (caruana_bag20): <X.XXXX> (was <X.XXXX>)

Spec: docs/superpowers/specs/2026-04-20-peft-molformer-lora-design.md
Plan: docs/superpowers/plans/2026-04-20-peft-molformer-lora.md

## Test plan
- [x] Smoke test passed (no NaN, DB rows landed)
- [x] Full Optuna 20-trial + 5-fold final CV
- [x] OOF MAE acceptance (<= 0.50)
- [x] Caruana_bag20 weight check
- [x] CI green
EOF
)"
```

Replace placeholders with actual numbers from Tasks 8.4 and 9.2.

- [ ] **Step 10.3: Mark PR ready for review**

Run: `cd /home/nagaet/pxr-iduction-challenge && gh pr ready`
Expected: PR transitions from draft to ready.

- [ ] **Step 10.4: Ask user for merge approval**

Verify CI: `cd /home/nagaet/pxr-iduction-challenge && gh pr checks`

Then ask the user explicitly:
> "PR is ready to merge. CI: <CI_STATUS>. New ensemble caruana MAE: <NEW_MAE> (was <OLD_MAE>). Shall I merge?"

Wait for explicit "yes" / "merge" / "OK" before running `gh pr merge`.

- [ ] **Step 10.5: After user approval, merge and clean up**

Run:
```bash
cd /home/nagaet/pxr-iduction-challenge
gh pr merge --squash --delete-branch
git checkout main && git pull
git remote prune origin
git branch -a
```
Expected: PR merged, feature branch deleted on remote, local main up to date.

---

## Self-Review Checklist (run after writing the plan, fix inline)

1. **Spec coverage:**
   - Module split (peft_backbones/methods/trainer + CLI) → Tasks 2, 3, 4, 6 ✓
   - Backbone registry with MoLFormer-XL → Task 2 ✓
   - LoRA method only (no adapter / last-k) → Task 3 ✓
   - LoRA target_modules verification risk → Task 5 ✓
   - Optuna search space (12 params) → Task 6 step 6.1 (`suggest_params`) ✓
   - 5-fold UMAP outer / 3-fold UMAP inner → Task 6 step 6.1 (CLI defaults) ✓
   - DB record + OOF save → Task 6 step 6.1 (`record_experiment` + `save_oof_predictions`) ✓
   - Acceptance: OOF MAE <= 0.50 → Task 8 step 8.5 ✓
   - Acceptance: caruana_bag20 weight > 0 → Task 9 step 9.3 ✓
   - CI = ruff + ty only, no unit tests → Tasks 2.3, 3.3, 4.2, 6.2 ✓
   - Smoke test before full run → Task 6 step 6.3 + cleanup 6.5 ✓
   - OOM mitigation (catch + TrialPruned) → Task 6 step 6.1 (`objective` body) ✓
   - Risk: long wall-clock → Task 8 user-confirm gate ✓
   - Submission CSV but no LB submission → Task 6 step 6.1 (writes file, no upload) ✓

2. **Placeholder scan:** Found `<R>`, `<A>`, `<X.XX>`, `<NEW_MAE>` etc. in PR body templates and ENSEMBLE_MODELS append — these are intentional runtime values that the implementing agent must substitute from actual results. They are NOT plan placeholders. Each is accompanied by an explicit instruction "Replace ... with the actual ..." so the engineer knows what to do. ✓

3. **Type consistency:**
   - `train_one_fold(...)` signature in `peft_trainer.py` (Task 4 step 4.1) matches all call sites in `run_peft_finetune.py` (Task 6 step 6.1, both `objective` and `run_final_cv`) ✓
   - `get_backbone(name)` and `get_peft_builder(method)` consumed in `peft_trainer.py` match definitions in `peft_backbones.py` / `peft_methods.py` ✓
   - `record_experiment(...)` call (Task 6 step 6.1) uses kwargs that match the existing function signature in `evaluate.py` (verified via Read) ✓
   - `save_oof_predictions(exp_id, oof_preds)` call matches the existing 2-arg form ✓
   - `umap_split_indices(smiles_list, n_splits, n_clusters, seed)` call matches the existing signature ✓

No issues found. Plan is internally consistent.
