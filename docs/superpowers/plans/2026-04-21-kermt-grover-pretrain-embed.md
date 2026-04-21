# KERMT/GROVER Pretrain-Embed Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `tabpfn_kermt_pretrain_embed_umap_default` as an 11th caruana_bag20 pool member via the Buterez strategy-3 recipe (continued-pretrain GROVER_base on `single_concentration.log2_fc` → frozen → embed → TabPFN), then re-run ensemble + linear_pos calibration.

**Architecture:** KERMT is the NVIDIA reimplementation of GROVER (graph transformer). Its dependency stack conflicts with the main pixi env, so we isolate it as a **separate pixi project** inside the `~/ghq/github.com/NVIDIA-Digital-Bio/KERMT` clone. Shell wrappers in `track1_activity/scripts/` cross the boundary; the embedding artifact is a parquet file consumed by the existing `run_train.py --feature` plumbing.

**Tech Stack:** KERMT (GROVER fork), PyTorch, DGL, pixi (separate project), gdown, TabPFN v7, scikit-learn (UMAP), LightGBM, psycopg2, RDKit (fallback featurization, no cuik-molmaker).

**Related spec:** `docs/superpowers/specs/2026-04-21-kermt-grover-pretrain-embed-design.md`

---

## File map

| Path | Create / Modify | Responsibility |
|---|---|---|
| `~/ghq/github.com/NVIDIA-Digital-Bio/KERMT/` | clone (external) | KERMT source tree |
| `~/ghq/github.com/NVIDIA-Digital-Bio/KERMT/pixi.toml` | create (external) | KERMT-local pixi environment |
| `~/ghq/github.com/NVIDIA-Digital-Bio/KERMT/pixi.lock` | create (external, generated) | Locked env |
| `models/kermt/grover_base.pt` | create (gitignored) | Downloaded GROVER_base weights |
| `models/kermt/pretrain/` | create (gitignored) | Finetuned checkpoint output |
| `data/kermt/pretrain_train.csv` | create (gitignored) | 90% train split for log2_fc pretrain |
| `data/kermt/pretrain_val.csv` | create (gitignored) | 10% val split |
| `data/kermt/pretrain_all.csv` | create (gitignored) | All 13,136 compounds for embed extraction |
| `data/kermt_pretrain_embed.parquet` | create (gitignored) | Final per-compound embedding (indexed by compound_id) |
| `track1_activity/scripts/prepare_kermt_pretrain_csv.py` | create | Main-pixi: export SQL → 3 CSVs |
| `track1_activity/scripts/run_kermt_pretrain.sh` | create | Shell wrapper: run KERMT finetune in KERMT-local pixi |
| `track1_activity/scripts/run_kermt_embed_extract.sh` | create | Shell wrapper: run fingerprint extraction in KERMT-local pixi |
| `track1_activity/scripts/kermt_fingerprint_fallback.py` | create (only if KERMT's `main.py fingerprint` is missing) | Load checkpoint, extract pre-FFN pooled embedding |
| `track1_activity/scripts/kermt_embed_npz_to_parquet.py` | create | Main-pixi: convert KERMT npz output to parquet |
| `track1_activity/scripts/run_train.py` | modify (~line 435) | Add `kermt_pretrain_embed` feature branch |
| `track1_activity/scripts/run_ensemble.py` | modify (~line 151) | Append `tabpfn_kermt_pretrain_embed_umap_default` to `ENSEMBLE_MODELS` |
| `.gitignore` | modify | Add `models/kermt/`, `data/kermt/`, `data/kermt_pretrain_embed.parquet` |

---

## Conventions recap

- **Branch**: `feature/kermt-grover-embed-pretrain` (already created; commits go here).
- **No CI in this repo.** `gh pr checks` reports "no checks" — expected.
- **No unit tests for DL code** (existing convention). Gates are `pixi run ruff format <file>` + `pixi run ruff check <file>`, plus smoke runs with reduced epochs/rows.
- **Idempotent DB writes**: use `record_experiment(..., on_conflict_replace=True)`.
- **Commit frequently**: one commit per task (or per logical sub-step within a task).

---

## Task 1: Set up KERMT clone + local pixi env

**Files:**
- Create: `~/ghq/github.com/NVIDIA-Digital-Bio/KERMT/` (external clone)
- Create: `~/ghq/github.com/NVIDIA-Digital-Bio/KERMT/pixi.toml`

- [ ] **Step 1.1: Clone KERMT into ghq**

Run:
```bash
mkdir -p ~/ghq/github.com/NVIDIA-Digital-Bio
cd ~/ghq/github.com/NVIDIA-Digital-Bio
git clone https://github.com/NVIDIA-Digital-Bio/KERMT.git
cd KERMT
git log --oneline -3
```

Expected: 3 recent commits shown, confirming clone success.

- [ ] **Step 1.2: Inspect environment.yml for porting**

Run:
```bash
cat ~/ghq/github.com/NVIDIA-Digital-Bio/KERMT/environment.yml
```

Expected: conda-style YAML listing python, pytorch, dgl, rdkit, pytorch-lightning, pandas, numpy, scikit-learn, tqdm, tensorboard, etc. Note the versions for porting.

- [ ] **Step 1.3: pixi init in KERMT clone**

Run:
```bash
cd ~/ghq/github.com/NVIDIA-Digital-Bio/KERMT
pixi init
```

Expected: `pixi.toml` created with default template.

- [ ] **Step 1.4: Port environment.yml into pixi.toml**

Edit `~/ghq/github.com/NVIDIA-Digital-Bio/KERMT/pixi.toml` to this content (adjust versions only if Step 1.2 showed different pins):

```toml
[project]
name = "kermt"
version = "0.1.0"
description = "NVIDIA KERMT (GROVER reimpl) — isolated env for PXR Track 1 pretrain-embed"
authors = []
channels = ["conda-forge", "pytorch", "nvidia"]
platforms = ["linux-64"]

[dependencies]
python = "3.10.*"
pytorch = { version = ">=2.1,<2.4", channel = "pytorch" }
pytorch-cuda = { version = "12.1.*", channel = "pytorch" }
dgl = ">=1.1,<2.1"
rdkit = ">=2023.09"
pytorch-lightning = ">=2.0,<2.5"
numpy = ">=1.24,<2.0"
pandas = ">=2.0"
scikit-learn = ">=1.3"
tqdm = "*"
tensorboard = "*"
scipy = ">=1.10"
pyyaml = "*"
six = "*"

[pypi-dependencies]
gdown = ">=5.0"
descriptastorus = "*"
```

- [ ] **Step 1.5: Install the local pixi env**

Run:
```bash
cd ~/ghq/github.com/NVIDIA-Digital-Bio/KERMT
pixi install
```

Expected: resolves and installs in a few minutes. If resolution fails, relax version pins (drop `<2.4` on pytorch, try pytorch-cuda 11.8) and retry.

- [ ] **Step 1.6: Smoke test imports**

Run:
```bash
cd ~/ghq/github.com/NVIDIA-Digital-Bio/KERMT
pixi run python -c "import torch, dgl, rdkit, pytorch_lightning; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('dgl', dgl.__version__); print('rdkit', rdkit.__version__); print('pl', pytorch_lightning.__version__)"
```

Expected: prints versions, `cuda True`.

- [ ] **Step 1.7: Smoke test KERMT CLI help**

Run:
```bash
cd ~/ghq/github.com/NVIDIA-Digital-Bio/KERMT/code
pixi run python main.py --help 2>&1 | head -40
```

Expected: usage text showing `finetune`, `predict`, and possibly `fingerprint` subcommands. **Record** whether `fingerprint` appears — if missing, flag for Task 6 fallback.

- [ ] **Step 1.8: Commit nothing (external repo)**

This task does not produce changes in the PXR repo; no commit here.

---

## Task 2: Download GROVER_base weights

**Files:**
- Create: `models/kermt/grover_base.pt` (gitignored)
- Modify: `.gitignore` to cover `models/kermt/`

- [ ] **Step 2.1: Add gitignore entry**

Edit `.gitignore` to append:

```
# KERMT pipeline artifacts (weights + CSVs + embed parquet)
models/kermt/
data/kermt/
data/kermt_pretrain_embed.parquet
```

- [ ] **Step 2.2: Verify entry applied**

Run:
```bash
git check-ignore -v models/kermt/grover_base.pt data/kermt/pretrain_train.csv data/kermt_pretrain_embed.parquet
```

Expected: each path resolves to the new `.gitignore` entry.

- [ ] **Step 2.3: Commit gitignore**

```bash
git add .gitignore
git commit -m "chore: gitignore KERMT pipeline artifacts"
```

- [ ] **Step 2.4: Create models dir**

```bash
mkdir -p models/kermt
```

- [ ] **Step 2.5: Download GROVER_base via gdown**

Run (from the KERMT-local pixi env that has gdown):
```bash
cd ~/ghq/github.com/NVIDIA-Digital-Bio/KERMT
pixi run gdown --fuzzy "https://drive.google.com/file/d/1hiGwOzoRfbJQPWj0V_mtOffsqIIAMgjl/view" -O ~/pxr-iduction-challenge/models/kermt/grover_base.pt
```

Expected: ~250 MB downloaded. If gdown reports a virus-scan prompt, retry with `--fuzzy` (already set) or add `--continue`.

- [ ] **Step 2.6: Verify weights load**

Run:
```bash
cd ~/ghq/github.com/NVIDIA-Digital-Bio/KERMT
pixi run python -c "
import torch
sd = torch.load('/home/nagaet/pxr-iduction-challenge/models/kermt/grover_base.pt', map_location='cpu', weights_only=False)
print('top keys:', list(sd.keys())[:10] if isinstance(sd, dict) else type(sd))
print('size (MB):', round(__import__('os').path.getsize('/home/nagaet/pxr-iduction-challenge/models/kermt/grover_base.pt') / 1e6, 1))
"
```

Expected: prints a dict with keys like `state_dict`, `args`, or similar; size ~250 MB.

- [ ] **Step 2.7: Record SHA256 for reproducibility**

Run:
```bash
sha256sum ~/pxr-iduction-challenge/models/kermt/grover_base.pt
```

Expected: 64-char hex. Copy the hash into the next commit message.

- [ ] **Step 2.8: Commit a README note with the checksum**

Create `models/kermt/README.md`:

```markdown
# KERMT model weights

- `grover_base.pt` — GROVER_base pretrained checkpoint (Tencent mirror, KERMT-compatible).
- Source: Google Drive ID `1hiGwOzoRfbJQPWj0V_mtOffsqIIAMgjl`.
- SHA256: `<paste from Step 2.7>`
- Size: ~250 MB.
- GROVER_large (ID `1bMg_ntUKEoOmHM0KoUi1XYJvzPBnHeWw`) is a follow-up if base succeeds.
```

```bash
git add models/kermt/README.md
git commit -m "docs: record GROVER_base weight source + checksum"
```

Note: The weight file itself is gitignored; only the README is committed.

---

## Task 3: Prepare log2_fc pretrain CSVs

**Files:**
- Create: `track1_activity/scripts/prepare_kermt_pretrain_csv.py`
- Create (as output): `data/kermt/pretrain_train.csv`, `pretrain_val.csv`, `pretrain_all.csv`

- [ ] **Step 3.1: Write the CSV prep script**

Create `track1_activity/scripts/prepare_kermt_pretrain_csv.py`:

```python
"""Export compounds + log2_fc measurements to CSVs for KERMT pretrain.

Mirrors the SQL in run_chemprop_pretrain.py::load_pretrain_data.
Writes three CSVs:
  - pretrain_train.csv (90% random, seed=42): columns smiles, log2fc_8p25, log2fc_33
  - pretrain_val.csv (10%): same columns
  - pretrain_all.csv (all 13,136): columns smiles, compound_id
    (no labels; used for embed extraction only)

Usage:
    pixi run python track1_activity/scripts/prepare_kermt_pretrain_csv.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402

OUT_DIR = REPO_ROOT.joinpath("data", "kermt")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SQL = """
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


def main() -> None:
    with psycopg2.connect(**DB_PARAMS) as conn:
        df = pd.read_sql(SQL, conn)

    print(f"Loaded {len(df)} compounds")
    print(
        f"  log2fc_8p25 labeled: {df['log2fc_8p25'].notna().sum()}"
        f"  log2fc_33 labeled: {df['log2fc_33'].notna().sum()}"
    )

    rng = np.random.default_rng(42)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    split = int(len(idx) * 0.9)
    train_idx, val_idx = idx[:split], idx[split:]

    df_train = df.iloc[train_idx][["smiles", "log2fc_8p25", "log2fc_33"]]
    df_val = df.iloc[val_idx][["smiles", "log2fc_8p25", "log2fc_33"]]
    df_all = df[["compound_id", "smiles"]]

    df_train.to_csv(OUT_DIR.joinpath("pretrain_train.csv"), index=False)
    df_val.to_csv(OUT_DIR.joinpath("pretrain_val.csv"), index=False)
    df_all.to_csv(OUT_DIR.joinpath("pretrain_all.csv"), index=False)

    print(
        f"Wrote {OUT_DIR.joinpath('pretrain_train.csv')} ({len(df_train)} rows)"
    )
    print(f"Wrote {OUT_DIR.joinpath('pretrain_val.csv')} ({len(df_val)} rows)")
    print(f"Wrote {OUT_DIR.joinpath('pretrain_all.csv')} ({len(df_all)} rows)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.2: Run the script**

```bash
pixi run python track1_activity/scripts/prepare_kermt_pretrain_csv.py
```

Expected: prints "Loaded 13136 compounds", "log2fc_8p25 labeled: 10752", "log2fc_33 labeled: 9527" (or close values), writes three CSVs.

- [ ] **Step 3.3: Sanity check output**

```bash
wc -l data/kermt/*.csv
head -3 data/kermt/pretrain_train.csv
```

Expected: `pretrain_train.csv` ≈ 11823 lines, `pretrain_val.csv` ≈ 1314 lines, `pretrain_all.csv` 13137 lines (incl. header). `head` shows `smiles,log2fc_8p25,log2fc_33` header + data rows.

- [ ] **Step 3.4: Format + lint**

```bash
pixi run ruff format track1_activity/scripts/prepare_kermt_pretrain_csv.py
pixi run ruff check track1_activity/scripts/prepare_kermt_pretrain_csv.py
```

Expected: "1 file reformatted" (or "already formatted"), "All checks passed".

- [ ] **Step 3.5: Commit**

```bash
git add track1_activity/scripts/prepare_kermt_pretrain_csv.py
git commit -m "feat(kermt): CSV export for log2_fc continued-pretrain"
```

---

## Task 4: KERMT pretrain shell wrapper + smoke test

**Files:**
- Create: `track1_activity/scripts/run_kermt_pretrain.sh`

- [ ] **Step 4.1: Write the wrapper script**

Create `track1_activity/scripts/run_kermt_pretrain.sh`:

```bash
#!/bin/bash
# KERMT continued-pretrain on single_concentration log2_fc.
#
# Runs in the KERMT-local pixi env (separate from main PXR pixi env).
# Inputs (produced by prepare_kermt_pretrain_csv.py):
#   data/kermt/pretrain_train.csv
#   data/kermt/pretrain_val.csv
# Output:
#   models/kermt/pretrain/fold_0/model_0/model.pt  (KERMT's default layout)
#
# Pass --epochs 2 as first arg to run a smoke test.
set -euo pipefail

PXR_REPO="${PXR_REPO:-$HOME/pxr-iduction-challenge}"
KERMT_REPO="${KERMT_REPO:-$HOME/ghq/github.com/NVIDIA-Digital-Bio/KERMT}"
EPOCHS="${1:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"

cd "$KERMT_REPO/code"
export PYTHONPATH=$PWD
export CUBLAS_WORKSPACE_CONFIG=:4096:8

pixi run --manifest-path "$KERMT_REPO/pixi.toml" python main.py finetune \
    --data_path "$PXR_REPO/data/kermt/pretrain_train.csv" \
    --separate_val_path "$PXR_REPO/data/kermt/pretrain_val.csv" \
    --save_dir "$PXR_REPO/models/kermt/pretrain" \
    --checkpoint_path "$PXR_REPO/models/kermt/grover_base.pt" \
    --dataset_type regression \
    --split_type scaffold_balanced \
    --ensemble_size 1 \
    --num_folds 1 \
    --no_features_scaling \
    --ffn_hidden_size 256 \
    --ffn_num_layers 3 \
    --bond_drop_rate 0.1 \
    --epochs "$EPOCHS" \
    --metric mae \
    --self_attention \
    --dist_coff 0.15 \
    --max_lr 1e-4 \
    --final_lr 2e-5 \
    --dropout 0.1 \
    --batch_size "$BATCH_SIZE"
```

- [ ] **Step 4.2: chmod + ruff-check not applicable (shell)**

```bash
chmod +x track1_activity/scripts/run_kermt_pretrain.sh
```

(No ruff check for shell files.)

- [ ] **Step 4.3: Smoke test with 2 epochs**

```bash
bash track1_activity/scripts/run_kermt_pretrain.sh 2 2>&1 | tee /tmp/kermt_smoke.log | tail -50
```

Expected: KERMT builds the graph encoder, reports per-epoch train/val MAE, completes in < 15 min. If fails:
- Missing flag or flag name mismatch: read error, adjust the wrapper (possible KERMT renamed some flags vs original GROVER).
- DGL/torch CUDA mismatch: pin different versions in KERMT `pixi.toml` and `pixi install` again.

- [ ] **Step 4.4: Inspect smoke output**

```bash
ls -la models/kermt/pretrain/
find models/kermt/pretrain -name "model*.pt" -o -name "*.pt" | head
```

Expected: a `.pt` checkpoint exists under `fold_0/model_0/` or similar.

- [ ] **Step 4.5: Commit**

```bash
git add track1_activity/scripts/run_kermt_pretrain.sh
git commit -m "feat(kermt): pretrain shell wrapper (pixi-in-ghq isolation)"
```

---

## Task 5: Full pretrain run (30 epochs)

**Files:**
- No new files (runs the wrapper from Task 4).

- [ ] **Step 5.1: Launch full pretrain**

Preferably in a tmux session, since this takes ~1.5–3h:

```bash
tmux new -s kermt_pretrain
cd ~/pxr-iduction-challenge
bash track1_activity/scripts/run_kermt_pretrain.sh 30 2>&1 | tee logs/kermt_pretrain_30ep.log
```

Expected: training progresses, val MAE decreases from ~0.55 at epoch 1 to below 0.50 at epoch 30 (rough target; chemprop pretrain hit val MAE ~0.47).

- [ ] **Step 5.2: Monitor val MAE in a second pane**

```bash
grep -E "(Epoch [0-9]+|Validation.*mae|val_mae|save best)" logs/kermt_pretrain_30ep.log | tail -20
```

Expected: decreasing trend. If val MAE plateaus early or bounces, abort and tune (Step 5.3).

- [ ] **Step 5.3: (Contingency) If convergence fails**

If val MAE does not decrease after 5 epochs:
- Lower `--max_lr` to 5e-5 (edit wrapper, rerun)
- Increase `--dropout` to 0.2 to combat overfitting on weak labels
- Report to user before further scope changes

- [ ] **Step 5.4: Record final val MAE**

```bash
tail -30 logs/kermt_pretrain_30ep.log | grep -i "best\|val\|mae"
```

Record the best val MAE in the commit message for Task 5.7.

- [ ] **Step 5.5: Confirm checkpoint exists**

```bash
ls -la models/kermt/pretrain/fold_0/model_0/model.pt
```

Expected: file present, ~300 MB (base model + FFN head + optimizer state).

- [ ] **Step 5.6: No new code; commit log artifact only**

The run produces only gitignored artifacts. Write a short note to `models/kermt/README.md`:

```markdown

## Pretrain run (2026-04-21)
- Epochs: 30, batch_size 32, max_lr 1e-4, final_lr 2e-5
- Best val MAE: <record from Step 5.4>
- Checkpoint: models/kermt/pretrain/fold_0/model_0/model.pt
- Log: logs/kermt_pretrain_30ep.log (gitignored)
```

- [ ] **Step 5.7: Commit the pretrain-run README update**

```bash
git add models/kermt/README.md
git commit -m "docs(kermt): record pretrain run config + best val MAE"
```

---

## Task 6: Extract embeddings + convert to parquet

**Files:**
- Create: `track1_activity/scripts/run_kermt_embed_extract.sh`
- Create: `track1_activity/scripts/kermt_fingerprint_fallback.py` (only if `main.py fingerprint` is missing from KERMT)
- Create: `track1_activity/scripts/kermt_embed_npz_to_parquet.py`
- Create (output): `data/kermt_pretrain_embed.parquet`

- [ ] **Step 6.1: Confirm KERMT `main.py fingerprint` subcommand availability**

Recall result from Task 1 Step 1.7. If `fingerprint` was in the help output, proceed with the standard wrapper. If absent, skip to Step 6.3 (fallback).

- [ ] **Step 6.2: (If fingerprint exists) Write the embed extract wrapper**

Create `track1_activity/scripts/run_kermt_embed_extract.sh`:

```bash
#!/bin/bash
# Extract GROVER-style fingerprint (pre-FFN pooled graph representation)
# for all compounds in pretrain_all.csv.
#
# Output: data/kermt/embeddings.npz (KERMT native format) or
#         data/kermt/embeddings.csv (fallback)
set -euo pipefail

PXR_REPO="${PXR_REPO:-$HOME/pxr-iduction-challenge}"
KERMT_REPO="${KERMT_REPO:-$HOME/ghq/github.com/NVIDIA-Digital-Bio/KERMT}"
LIMIT="${LIMIT:-0}"  # 0 = all compounds; set positive for smoke test

INPUT_CSV="$PXR_REPO/data/kermt/pretrain_all.csv"
if [[ "$LIMIT" -gt 0 ]]; then
    TMP_CSV="$PXR_REPO/data/kermt/pretrain_all_head${LIMIT}.csv"
    head -n $((LIMIT + 1)) "$INPUT_CSV" > "$TMP_CSV"
    INPUT_CSV="$TMP_CSV"
fi

cd "$KERMT_REPO/code"
export PYTHONPATH=$PWD

pixi run --manifest-path "$KERMT_REPO/pixi.toml" python main.py fingerprint \
    --data_path "$INPUT_CSV" \
    --checkpoint_dir "$PXR_REPO/models/kermt/pretrain/fold_0/model_0" \
    --no_features_scaling \
    --output "$PXR_REPO/data/kermt/embeddings.npz"
```

chmod + smoke:
```bash
chmod +x track1_activity/scripts/run_kermt_embed_extract.sh
LIMIT=100 bash track1_activity/scripts/run_kermt_embed_extract.sh
ls -la data/kermt/embeddings.npz
```

Expected: an npz file is written; run completes in < 3 min.

- [ ] **Step 6.3: (If fingerprint is missing) Fallback extraction script**

Create `track1_activity/scripts/kermt_fingerprint_fallback.py` in the **KERMT repo** (so it can import KERMT internals). Path: `~/ghq/github.com/NVIDIA-Digital-Bio/KERMT/code/fingerprint_fallback.py`.

```python
"""Fallback fingerprint extractor — loads a KERMT finetune checkpoint
and returns the pre-FFN pooled graph representation for each SMILES.

Runs from inside the KERMT tree (imports KERMT modules directly).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# KERMT imports -- adjust if module layout differs
from grover.model.models import GROVEREmbedding  # type: ignore  # noqa: F401
from grover.data.molfeaturegenerator import get_features_generator  # type: ignore  # noqa: F401
from grover.util.utils import load_checkpoint  # type: ignore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint_path, device=device)
    model.eval()

    df = pd.read_csv(args.data_path)
    compound_ids = df["compound_id"].astype(int).to_numpy() if "compound_id" in df.columns else np.arange(len(df))
    smiles_list = df["smiles"].tolist()

    embs = []
    with torch.no_grad():
        for i in range(0, len(smiles_list), args.batch_size):
            batch = smiles_list[i : i + args.batch_size]
            # KERMT's featurize + forward until the readout layer -- exact
            # call depends on model class. The intent is: get the pooled
            # graph vector BEFORE the FFN head. Adapt if API differs.
            feats = model.featurize(batch).to(device)
            pooled = model.encoder(feats)  # (B, hidden) or (B, hidden*2)
            embs.append(pooled.cpu().numpy())

    emb_mat = np.concatenate(embs, axis=0).astype(np.float32)
    np.savez(args.output, compound_id=compound_ids, embedding=emb_mat)
    print(f"Wrote {args.output}  shape {emb_mat.shape}")


if __name__ == "__main__":
    main()
```

Then update the wrapper in Step 6.2 to call this script instead:

```bash
pixi run --manifest-path "$KERMT_REPO/pixi.toml" python "$KERMT_REPO/code/fingerprint_fallback.py" \
    --data_path "$INPUT_CSV" \
    --checkpoint_path "$PXR_REPO/models/kermt/pretrain/fold_0/model_0/model.pt" \
    --output "$PXR_REPO/data/kermt/embeddings.npz"
```

**If the exact KERMT module API differs**, read `~/ghq/github.com/NVIDIA-Digital-Bio/KERMT/code/main.py::predict` to see the load + forward convention, and adapt.

- [ ] **Step 6.4: Full embedding extraction (all 13,136)**

```bash
unset LIMIT  # or LIMIT=0
bash track1_activity/scripts/run_kermt_embed_extract.sh
ls -la data/kermt/embeddings.npz
```

Expected: completes in ~10–20 min; npz file is ~80–180 MB (13136 × 800 to 1600 floats × 4 bytes).

- [ ] **Step 6.5: Write the npz-to-parquet converter (main pixi)**

Create `track1_activity/scripts/kermt_embed_npz_to_parquet.py`:

```python
"""Convert KERMT fingerprint npz to the parquet format expected by
run_train.py --feature kermt_pretrain_embed.

Reads:  data/kermt/embeddings.npz  (keys: compound_id, embedding)
Writes: data/kermt_pretrain_embed.parquet
         (index = compound_id, columns = emb_0000..emb_N-1)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
IN_PATH = REPO_ROOT.joinpath("data", "kermt", "embeddings.npz")
OUT_PATH = REPO_ROOT.joinpath("data", "kermt_pretrain_embed.parquet")


def main() -> None:
    data = np.load(IN_PATH)
    compound_ids = data["compound_id"].astype(int)
    emb = data["embedding"].astype(np.float32)

    print(f"Loaded {IN_PATH}  compound_ids={compound_ids.shape}  emb={emb.shape}")

    cols = [f"emb_{i:04d}" for i in range(emb.shape[1])]
    df = pd.DataFrame(emb, index=pd.Index(compound_ids, name="compound_id"), columns=cols)
    df.to_parquet(OUT_PATH)
    print(f"Wrote {OUT_PATH}  shape {df.shape}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.6: Run the converter**

```bash
pixi run python track1_activity/scripts/kermt_embed_npz_to_parquet.py
```

Expected: prints input + output shapes. Output parquet file exists.

- [ ] **Step 6.7: Format + lint the main-pixi scripts**

```bash
pixi run ruff format track1_activity/scripts/kermt_embed_npz_to_parquet.py
pixi run ruff check track1_activity/scripts/kermt_embed_npz_to_parquet.py
```

Expected: formatted + all checks pass.

- [ ] **Step 6.8: Commit**

```bash
git add track1_activity/scripts/run_kermt_embed_extract.sh track1_activity/scripts/kermt_embed_npz_to_parquet.py
git commit -m "feat(kermt): embedding extraction wrapper + npz->parquet"
```

If fallback script was needed, also commit its reference (but the file itself lives in the external KERMT repo; document its existence in `models/kermt/README.md`):

```markdown
## Fingerprint extraction
- KERMT's `main.py fingerprint` was MISSING / PRESENT (circle one)
- Fallback script: `~/ghq/github.com/NVIDIA-Digital-Bio/KERMT/code/fingerprint_fallback.py`
```

---

## Task 7: Register `kermt_pretrain_embed` feature in run_train.py

**Files:**
- Modify: `track1_activity/scripts/run_train.py` (~line 435, mirroring the `molformer_c3_pretrain_embed` branch)

- [ ] **Step 7.1: Locate the insertion point**

Open `track1_activity/scripts/run_train.py`, find the `molformer_c3_pretrain_embed` branch (around line 412–434).

- [ ] **Step 7.2: Add the new feature branch**

Right after the `molformer_c3_pretrain_embed` block (after its `return`), insert:

```python
    if feature_name == "kermt_pretrain_embed":
        # 800d or 1600d per-compound graph embedding from KERMT
        # (GROVER_base) after continued-pretrain on single_concentration
        # log2_fc. See:
        #   track1_activity/scripts/run_kermt_pretrain.sh
        #   track1_activity/scripts/run_kermt_embed_extract.sh
        #   track1_activity/scripts/kermt_embed_npz_to_parquet.py
        # Buterez 2024 strategy-3 with a graph-transformer backbone
        # (parallel to chemprop_pretrain_embed = GNN,
        # molformer_c3_pretrain_embed = transformer).
        embed_path = REPO_ROOT.joinpath("data", "kermt_pretrain_embed.parquet")
        if not embed_path.exists():
            raise SystemExit(
                f"Missing {embed_path}. Run "
                f"track1_activity/scripts/kermt_embed_npz_to_parquet.py"
            )
        emb_df = pd.read_parquet(embed_path)
        X_train = emb_df.reindex(index=train_ids).to_numpy(dtype=np.float32).copy()
        X_test = emb_df.reindex(index=test_ids).to_numpy(dtype=np.float32).copy()
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        print(
            f"  kermt_pretrain_embed: {X_train.shape[1]} dims "
            f"(train {X_train.shape[0]} / test {X_test.shape[0]})"
        )
        return X_train, X_test
```

- [ ] **Step 7.3: Add the feature name to the VALID_FEATURES list**

Find the existing list (around line 1315 where `"molformer_c3_pretrain_embed"` appears) and append:

```python
            "kermt_pretrain_embed",
```

- [ ] **Step 7.4: Smoke test the feature loader**

```bash
pixi run python -c "
import sys
sys.path.insert(0, 'track1_activity/scripts')
from run_train import load_features
import numpy as np, psycopg2, pandas as pd
sys.path.insert(0, 'track1_activity/src')
from data import DB_PARAMS

with psycopg2.connect(**DB_PARAMS) as conn:
    tr = pd.read_sql('SELECT compound_id FROM train_activity ORDER BY compound_id', conn)['compound_id'].astype(int).tolist()
    te = pd.read_sql('SELECT compound_id FROM test_activity ORDER BY compound_id', conn)['compound_id'].astype(int).tolist()

Xtr, Xte = load_features('kermt_pretrain_embed', tr, te)
print('train', Xtr.shape, 'test', Xte.shape, 'NaN', np.isnan(Xtr).sum() + np.isnan(Xte).sum())
"
```

Expected: `train (4140, <dim>) test (513, <dim>) NaN 0`.

- [ ] **Step 7.5: Format + lint**

```bash
pixi run ruff format track1_activity/scripts/run_train.py
pixi run ruff check track1_activity/scripts/run_train.py
```

Expected: formatted, all checks pass.

- [ ] **Step 7.6: Commit**

```bash
git add track1_activity/scripts/run_train.py
git commit -m "feat(kermt): register kermt_pretrain_embed feature in run_train.py"
```

---

## Task 8: Train TabPFN on the embedding

**Files:**
- No new files; runs `run_train.py` with the new feature.

- [ ] **Step 8.1: Dry run (2-fold) sanity check**

```bash
pixi run python track1_activity/scripts/run_train.py \
    --model tabpfn \
    --feature kermt_pretrain_embed \
    --split umap_default \
    --trials 0 \
    --outer-folds 2 \
    --experiment-name tabpfn_kermt_pretrain_embed_smoke
```

Expected: completes in < 15 min, prints per-fold MAE.

- [ ] **Step 8.2: Full 5-fold run**

```bash
pixi run python track1_activity/scripts/run_train.py \
    --model tabpfn \
    --feature kermt_pretrain_embed \
    --split umap_default \
    --trials 0 \
    --experiment-name tabpfn_kermt_pretrain_embed_umap_default
```

Expected: 5-fold UMAP split, OOF MAE ≤ 0.48 (acceptance criterion 1). Runtime ~20–40 min.

- [ ] **Step 8.3: Verify OOF stored in DB**

```bash
pixi run python -c "
import psycopg2, sys
sys.path.insert(0, 'track1_activity/src')
from data import DB_PARAMS
with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
    cur.execute('''
        SELECT experiment_name, model_type, feature_set,
               mae_oof, rae_oof, spearman_oof, kendall_oof
        FROM experiment_summary
        WHERE experiment_name = 'tabpfn_kermt_pretrain_embed_umap_default'
    ''')
    for r in cur.fetchall():
        print(r)
"
```

Expected: one row with MAE ≤ 0.48.

- [ ] **Step 8.4: Spot-check acceptance criteria (1) and (4)**

Acceptance (1) OOF MAE ≤ 0.48: confirmed in Step 8.3 output.

Acceptance (4) OOF Pearson correlation < 0.96 with existing members:

```bash
pixi run python -c "
import psycopg2, sys, numpy as np
import pandas as pd
sys.path.insert(0, 'track1_activity/src')
from data import DB_PARAMS

names = [
    'tabpfn_chemprop_pretrain_embed',
    'tabpfn_molformer_c3_pretrain_embed_umap',
    'tabpfn_2d_full_boltz_log2fc_pred',
    'tabpfn_kermt_pretrain_embed_umap_default',
]
with psycopg2.connect(**DB_PARAMS) as conn:
    cur = conn.cursor()
    oofs = {}
    for n in names:
        cur.execute('''
            SELECT compound_id, oof_prediction
            FROM experiment_oof_predictions eop
            JOIN experiments e ON e.id = eop.experiment_id
            WHERE e.name = %s
            ORDER BY compound_id
        ''', (n,))
        rows = cur.fetchall()
        if rows:
            oofs[n] = pd.Series({r[0]: float(r[1]) for r in rows})
df = pd.DataFrame(oofs).dropna()
print(df.corr(method='pearson').round(3))
"
```

Expected: kermt's Pearson r < 0.96 with chemprop/molformer/log2fc_pred.

- [ ] **Step 8.5: Commit a note (no code change)**

If both acceptance (1) and (4) pass, nothing to commit yet — proceed to Task 9.
If either fails, **stop and report to the user** per the spec's Failure Handling.

---

## Task 9: Register in ENSEMBLE_MODELS + re-run caruana + calibrate

**Files:**
- Modify: `track1_activity/scripts/run_ensemble.py` (~line 151, after the `tabpfn_molformer_c3_pretrain_embed_umap` entry)

- [ ] **Step 9.1: Append to ENSEMBLE_MODELS tuple**

Edit `track1_activity/scripts/run_ensemble.py`, locate `"tabpfn_molformer_c3_pretrain_embed_umap",` (around line 151), and immediately after it insert:

```python
    # --- KERMT/GROVER pretrain embed + TabPFN ---
    # Graph-transformer family (continued-pretrain of GROVER_base on
    # single_concentration log2_fc, frozen, then TabPFN). Added as a
    # decorrelating 11th pool member. Previous pretrain-embed backbones:
    # chemprop (GNN) and MoLFormer-c3 (transformer). See
    # docs/superpowers/specs/2026-04-21-kermt-grover-pretrain-embed-design.md.
    "tabpfn_kermt_pretrain_embed_umap_default",
```

- [ ] **Step 9.2: Re-run caruana_bag20 ensemble**

```bash
pixi run python track1_activity/scripts/run_ensemble.py --strategy caruana_bag20 2>&1 | tee logs/ensemble_kermt.log | tail -60
```

Expected output: new 11-member pool, caruana_bag20 MAE printed, plus weights table. Record the MAE value.

- [ ] **Step 9.3: Verify acceptance (2) and (3)**

Acceptance (2) — caruana weight > 0 on `tabpfn_kermt_pretrain_embed_umap_default`:

```bash
grep "kermt" logs/ensemble_kermt.log | head -5
```

Expected: a weight entry like `tabpfn_kermt_pretrain_embed_umap_default: 0.0XX`.

Acceptance (3) — 11-pool caruana_bag20 OOF MAE ≤ 0.4314 (pre-KERMT baseline):

```bash
grep "caruana_bag20.*MAE\|bag20.*mae" logs/ensemble_kermt.log | tail -5
```

Expected: MAE ≤ 0.4314.

If either fails, **stop and report** per the failure handling policy. Do NOT silently submit.

- [ ] **Step 9.4: Re-run calibration**

```bash
pixi run python track1_activity/scripts/run_ensemble_calibrate.py 2>&1 | tee logs/calibrate_kermt.log | tail -40
```

Expected: 4-way nested CV across linear/linear_pos/spline_k5/isotonic; the best method (likely still linear_pos) printed; `track1_activity/submissions/ens_caruana_bag20_calibrated_best.csv` rewritten.

- [ ] **Step 9.5: Format + lint**

```bash
pixi run ruff format track1_activity/scripts/run_ensemble.py
pixi run ruff check track1_activity/scripts/run_ensemble.py
```

Expected: formatted, checks pass.

- [ ] **Step 9.6: Commit**

```bash
git add track1_activity/scripts/run_ensemble.py
git commit -m "feat(ens): add tabpfn_kermt_pretrain_embed to 11-pool caruana_bag20"
```

---

## Task 10: PR preparation + self-check

**Files:**
- None (documentation + git work).

- [ ] **Step 10.1: Push branch**

```bash
git push -u origin feature/kermt-grover-embed-pretrain
```

- [ ] **Step 10.2: Spot-check all new scripts lint clean**

```bash
pixi run ruff format --check track1_activity/scripts/prepare_kermt_pretrain_csv.py track1_activity/scripts/kermt_embed_npz_to_parquet.py
pixi run ruff check track1_activity/scripts/prepare_kermt_pretrain_csv.py track1_activity/scripts/kermt_embed_npz_to_parquet.py track1_activity/scripts/run_train.py track1_activity/scripts/run_ensemble.py
```

Expected: "X files already formatted" + "All checks passed".

- [ ] **Step 10.3: Verify the six acceptance criteria in a summary comment**

Draft for PR body (mirror the style of PR #98 / #101):

```
## Summary
- Adds an 11th ensemble pool member via the Buterez 2024 strategy-3
  recipe on GROVER_base (graph-transformer family; parallel to the
  existing chemprop=GNN and MoLFormer-c3=transformer members).
- Runs KERMT in an isolated pixi project inside the ghq clone.
- No DB schema changes; embedding stored as parquet (matches
  molformer_c3 convention).

## Acceptance results
1. Single-model OOF MAE: <fill from Task 8>  (target ≤ 0.48)
2. caruana_bag20 weight: <fill from Task 9>  (target > 0)
3. 11-pool caruana_bag20 OOF MAE: <fill from Task 9>  (target ≤ 0.4314)
4. Pearson r vs existing members: <fill from Task 8.4>  (target < 0.96)
5. ruff format + check: clean
6. DB idempotent: on_conflict_replace=True on all new rows

## Test plan
- [ ] Re-run `run_ensemble.py --strategy caruana_bag20` on a fresh pixi shell
- [ ] Verify calibrated submission CSV row count = 513
- [ ] Spot-check 3 compound_id predictions are finite floats in [3, 8]

## CI: N/A (no workflow)
```

- [ ] **Step 10.4: Open PR**

```bash
gh pr create --title "feat: KERMT/GROVER pretrain-embed as 11th caruana pool member" --body "$(cat <<'EOF'
## Summary
- Adds `tabpfn_kermt_pretrain_embed_umap_default` as the 11th member
  of `caruana_bag20`. Graph-transformer family (GROVER_base) closes
  the decorrelation gap vs the existing chemprop (GNN) and
  MoLFormer-c3 (transformer) pretrain-embed members.
- KERMT dep stack (DGL + pytorch-lightning 2.x) isolated via a
  **separate pixi project** inside the ghq clone. Embeddings cross
  the boundary as a parquet file; `run_train.py --feature kermt_pretrain_embed`
  consumes it identically to the existing pretrain-embed members.
- No DB schema changes. Gitignored artifacts: models/kermt/, data/kermt/,
  data/kermt_pretrain_embed.parquet.

## Acceptance results
1. Single-model OOF MAE: <fill>  (target ≤ 0.48)
2. caruana_bag20 weight on new member: <fill>  (target > 0)
3. 11-pool caruana_bag20 OOF MAE: <fill>  (target ≤ 0.4314 pre-KERMT baseline)
4. Pearson r with existing pool members: <fill>  (target < 0.96)
5. ruff format + check: clean on all modified/new main-pixi files
6. on_conflict_replace=True on DB writes

## Test plan
- [ ] Fresh pixi shell re-run of run_ensemble.py reproduces the pool MAE
- [ ] Calibrated CSV has 513 rows of finite predictions
- [ ] Pearson r of new member with chemprop/molformer/log2fc_pred < 0.96

## CI: N/A (no workflow, per CLAUDE.md)
EOF
)"
```

- [ ] **Step 10.5: Ask user before merging**

Once PR is open, paste the ask to the user:
> PR #<N> is ready. Acceptance summary: <fill values>. CI: N/A (no workflow). Shall I merge?

Wait for explicit approval. After merge, clean up the branch per `git-workflow-rules.md`.

---

## Self-review notes

Coverage verification (against the spec):

- **Environment isolation (spec §Architecture)** → Task 1
- **Weight acquisition (spec §Weight acquisition)** → Task 2
- **Pretrain data (spec §Pretrain data)** → Task 3
- **Pretrain configuration (spec §Pretrain configuration)** → Tasks 4–5
- **Embedding extraction (spec §Embedding extraction)** → Task 6
- **Feature plumbing** → Task 7
- **Downstream TabPFN (spec §Downstream)** → Task 8
- **Ensemble integration (spec §Ensemble integration)** → Task 9
- **Acceptance criteria 1–6** → Tasks 8, 9, 10 (with explicit stop-and-report on fail)
- **ruff format + check gate** → Tasks 3.4, 6.7, 7.5, 9.5, 10.2
- **Idempotent DB writes** → inherited from `run_train.py` / `run_ensemble.py` existing code paths

Placeholder scan: no TBD/TODO; all code blocks are complete; all file paths absolute where needed.

Type consistency: `load_features` signature reused as-is; `ENSEMBLE_MODELS` is a tuple (matches the existing pattern).

Known fluid points (flagged inline, not placeholders):
- **Task 1.4 version pins**: may need one tweak based on what environment.yml actually shows (Step 1.2 output); spec notes this.
- **Task 6.3 fallback**: only triggers if `main.py fingerprint` is missing. Script is complete as written; adaptation notes for edge cases in the embedded prose.
- **Task 5.4 val MAE**: recorded dynamically, not pre-specified (cannot predict exactly).
