# Uni-Mol v2 × ETKDG Pretrain-Embed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `tabpfn_unimol_v2_pretrain_embed_umap_default` as a 3D-structure-aware pool member. Pretrain Uni-Mol v2 on `single_concentration.log2_fc` using ETKDG conformers, extract CLS representation for all 13,136 compounds, feed to TabPFN.

**Architecture:** Path 2 (user-approved): ETKDG for BOTH pretrain and inference to avoid distribution shift with Uni-Mol's native RDKit-conformer training. Isolated pixi project for Uni-Mol env. CLS repr extracted via `unimol_tools.UniMolRepr.get_repr()`.

**Tech Stack:** `unimol_tools` (pip, Uni-Mol v2, via HuggingFace auto-download), RDKit (ETKDGv3), torch + CUDA, psycopg2, pandas (parquet), TabPFN v2.6 (main pixi env).

**Related spec:** `docs/superpowers/specs/2026-04-21-unimol-etkdg-pretrain-embed-design.md`

---

## File map

| Path | Create / Modify | Responsibility |
|---|---|---|
| `~/ghq/github.com/deepmodeling/Uni-Mol/unimol_tools/pixi.toml` | Create (external) | Uni-Mol isolated pixi env |
| `track1_activity/scripts/unimol/01_prepare_log2fc_data.py` | Create | Export 13,136 SMILES + log2_fc labels to CSV for unimol_tools |
| `track1_activity/scripts/unimol/02_pretrain_molv2.sh` | Create | Shell wrapper: `pixi run` in Uni-Mol env, call MolTrain on log2_fc |
| `track1_activity/scripts/unimol/03_extract_repr.sh` | Create | Shell wrapper: call UniMolRepr.get_repr() on 13,136 SMILES, dump npz |
| `track1_activity/scripts/unimol/04_npz_to_parquet.py` | Create | Main-pixi: convert CLS repr npz → parquet (indexed by compound_id) |
| `track1_activity/scripts/run_train.py` | Modify | Register `unimol_v2_pretrain_embed` feature + all_features entry |
| `track1_activity/scripts/run_ensemble.py` | Modify | Append or swap in ENSEMBLE_MODELS |
| `.gitignore` | Modify | Ensure `data/unimol_v2_pretrain_embed.parquet`, checkpoints dir |

---

## Conventions recap

- Branch: `feature/unimol-etkdg-pretrain-embed` (already checked out).
- No CI; ruff format + check gates.
- Commit per task.

---

## Task 1: Set up Uni-Mol isolated pixi env

**Files:**
- Create: `~/ghq/github.com/deepmodeling/Uni-Mol/unimol_tools/pixi.toml` (external)

- [ ] **Step 1.1: Verify Uni-Mol repo is cloned**

```bash
ls ~/ghq/github.com/deepmodeling/Uni-Mol
```

Expected: directory with `unimol_tools/`, `unimol2/`, etc. Already cloned (user confirmed 2026-04-21).

- [ ] **Step 1.2: Inspect unimol_tools requirements**

```bash
cat ~/ghq/github.com/deepmodeling/Uni-Mol/unimol_tools/requirements.txt
cat ~/ghq/github.com/deepmodeling/Uni-Mol/unimol_tools/setup.py | head -40
```

Note version pins for torch, rdkit, numpy.

- [ ] **Step 1.3: pixi init + port requirements**

```bash
cd ~/ghq/github.com/deepmodeling/Uni-Mol/unimol_tools
pixi init
```

Edit `pixi.toml` to the following (adjust pins based on Step 1.2):

```toml
[project]
name = "unimol_tools"
version = "0.1.0"
description = "Uni-Mol tools isolated env for PXR pretrain-embed"
authors = []
channels = ["conda-forge", "pytorch", "nvidia"]
platforms = ["linux-64"]

[system-requirements]
cuda = "12.9"

[dependencies]
python = "3.11.*"
pytorch-gpu = ">=2.1,<2.9"
rdkit = ">=2023.09"
numpy = "<2.0"
pandas = ">=2.0"
scipy = "*"
tqdm = "*"
scikit-learn = "*"

[pypi-dependencies]
unimol_tools = ">=0.1.3"
huggingface_hub = "*"
```

- [ ] **Step 1.4: pixi install**

```bash
cd ~/ghq/github.com/deepmodeling/Uni-Mol/unimol_tools
pixi install
```

Expected: resolves in a few minutes. If `pytorch-gpu` resolution fails, try `pytorch` + `pytorch-cuda` explicit pin.

- [ ] **Step 1.5: Smoke imports**

```bash
cd ~/ghq/github.com/deepmodeling/Uni-Mol/unimol_tools
pixi run python -c "
import torch, rdkit, unimol_tools
from unimol_tools import UniMolRepr, MolTrain
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
print('unimol_tools:', unimol_tools.__version__ if hasattr(unimol_tools, '__version__') else 'imported')
print('UniMolRepr:', UniMolRepr)
print('MolTrain:', MolTrain)
"
```

Expected: prints versions + class objects, `cuda True`.

- [ ] **Step 1.6: Download pretrained weights (test)**

Smoke the auto-download:

```bash
cd ~/ghq/github.com/deepmodeling/Uni-Mol/unimol_tools
pixi run python -c "
from unimol_tools import UniMolRepr
r = UniMolRepr(model_name='unimolv2', model_size='84m', use_cuda=True)
print('model loaded, embed_dim (from probe):')
out = r.get_repr(data=['CCO', 'c1ccccc1', 'CCN'])
import numpy as np
arr = np.array(out['cls_repr'])
print('  shape:', arr.shape)
"
```

Expected: prints `(3, embed_dim)`. First run downloads weights from HuggingFace (`dptech/Uni-Mol-Models`), takes ~2 min depending on model size.

Note: If download is slow, set `HF_ENDPOINT=https://hf-mirror.com`.

- [ ] **Step 1.7: No commit (external repo)**

---

## Task 2: Export log2_fc training CSV

**Files:**
- Create: `/home/nagaet/pxr-iduction-challenge/track1_activity/scripts/unimol/01_prepare_log2fc_data.py`
- Create (output, gitignored): `data/unimol/pretrain_all.csv`, `pretrain_labeled.csv`

- [ ] **Step 2.1: Write the exporter**

Create `/home/nagaet/pxr-iduction-challenge/track1_activity/scripts/unimol/01_prepare_log2fc_data.py`:

```python
"""Export log2_fc training data for unimol_tools.MolTrain.

Two CSVs:
  - pretrain_labeled.csv: compounds with at least one of log2fc_8p25/log2fc_33
    non-null (used for pretraining; NaN in other heads handled internally)
  - pretrain_all.csv: all 13,136 compounds (compound_id + smiles only, for
    repr extraction)

Column schema for pretrain_labeled.csv:
  SMILES, log2fc_8p25, log2fc_33

(unimol_tools expects a 'SMILES' column; other columns are targets.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402

OUT_DIR = REPO_ROOT.joinpath("data", "unimol")
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

    n8 = df["log2fc_8p25"].notna().sum()
    n33 = df["log2fc_33"].notna().sum()
    print(f"Total: {len(df)}, log2fc_8p25 labeled: {n8}, log2fc_33 labeled: {n33}")

    # Labeled subset (at least one of the two targets)
    labeled_mask = df["log2fc_8p25"].notna() | df["log2fc_33"].notna()
    df_lab = df[labeled_mask].rename(columns={"smiles": "SMILES"})[
        ["SMILES", "log2fc_8p25", "log2fc_33"]
    ]
    print(f"  labeled (at least one target): {len(df_lab)}")

    # All compounds (for repr extraction)
    df_all = df.rename(columns={"smiles": "SMILES"})[["compound_id", "SMILES"]]
    print(f"  all (for repr): {len(df_all)}")

    df_lab.to_csv(OUT_DIR.joinpath("pretrain_labeled.csv"), index=False)
    df_all.to_csv(OUT_DIR.joinpath("pretrain_all.csv"), index=False)
    print(f"Wrote {OUT_DIR}/pretrain_labeled.csv ({len(df_lab)} rows)")
    print(f"Wrote {OUT_DIR}/pretrain_all.csv ({len(df_all)} rows)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2.2: Run exporter**

```bash
cd /home/nagaet/pxr-iduction-challenge
mkdir -p track1_activity/scripts/unimol
pixi run python track1_activity/scripts/unimol/01_prepare_log2fc_data.py 2>&1 | tail -10
wc -l data/unimol/*.csv
head -3 data/unimol/pretrain_labeled.csv data/unimol/pretrain_all.csv
```

Expected: pretrain_labeled ~11k rows (subset with labels), pretrain_all 13,137 rows.

- [ ] **Step 2.3: Gitignore check**

```bash
git check-ignore -v data/unimol/pretrain_all.csv
```

If NOT ignored, append to `.gitignore`:
```
data/unimol/
data/unimol_v2_pretrain_embed.parquet
```

- [ ] **Step 2.4: Format + lint + commit**

```bash
pixi run ruff format track1_activity/scripts/unimol/01_prepare_log2fc_data.py
pixi run ruff check track1_activity/scripts/unimol/01_prepare_log2fc_data.py
git add track1_activity/scripts/unimol/01_prepare_log2fc_data.py
# + .gitignore if modified
git status
git commit -m "feat(unimol): log2_fc CSV export for MolTrain"
```

---

## Task 3: Uni-Mol MolTrain pretrain on log2_fc

**Files:**
- Create: `/home/nagaet/pxr-iduction-challenge/track1_activity/scripts/unimol/02_pretrain_molv2.sh`

- [ ] **Step 3.1: Write pretrain wrapper**

Create `/home/nagaet/pxr-iduction-challenge/track1_activity/scripts/unimol/02_pretrain_molv2.sh`:

```bash
#!/usr/bin/env bash
# Pretrain Uni-Mol v2 on log2_fc labels using unimol_tools.MolTrain.
# Runs in the isolated Uni-Mol pixi env.
#
# Inputs:
#   data/unimol/pretrain_labeled.csv (SMILES + log2fc_8p25 + log2fc_33)
# Output:
#   models/unimol_v2_log2fc/exp/<ts>/model_<seed>.pth  (checkpoint)
#
# Usage:
#   bash track1_activity/scripts/unimol/02_pretrain_molv2.sh [--smoke]
set -euo pipefail

PXR_REPO="${PXR_REPO:-$HOME/pxr-iduction-challenge}"
UNIMOL_REPO="${UNIMOL_REPO:-$HOME/ghq/github.com/deepmodeling/Uni-Mol/unimol_tools}"
MODEL_SIZE="${MODEL_SIZE:-84m}"   # 84m fits 16GB comfortably; upgrade to 164m if stable
EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-32}"

if [[ ! -f "$PXR_REPO/data/unimol/pretrain_labeled.csv" ]]; then
    echo "ERROR: $PXR_REPO/data/unimol/pretrain_labeled.csv not found. Run 01_prepare_log2fc_data.py first." >&2
    exit 1
fi

SMOKE=""
if [[ "${1:-}" == "--smoke" ]]; then
    SMOKE="--epochs 2"
fi

SAVE_DIR="$PXR_REPO/models/unimol_v2_log2fc"
mkdir -p "$SAVE_DIR"

cat > /tmp/unimol_pretrain_invoke.py <<PYEOF
from unimol_tools import MolTrain
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--save_dir', required=True)
ap.add_argument('--model_size', default='$MODEL_SIZE')
ap.add_argument('--epochs', type=int, default=$EPOCHS)
ap.add_argument('--batch', type=int, default=$BATCH)
args = ap.parse_args()

clf = MolTrain(
    task='regression',
    data_type='molecule',
    model_name='unimolv2',
    model_size=args.model_size,
    epochs=args.epochs,
    batch_size=args.batch,
    metrics='mae',
    save_path=args.save_dir,
    target_cols=['log2fc_8p25', 'log2fc_33'],
)
clf.fit(data=args.data)
print('MolTrain complete, checkpoint at:', args.save_dir)
PYEOF

cd "$UNIMOL_REPO"
pixi run --manifest-path "$UNIMOL_REPO/pixi.toml" python /tmp/unimol_pretrain_invoke.py \
    --data "$PXR_REPO/data/unimol/pretrain_labeled.csv" \
    --save_dir "$SAVE_DIR" \
    --model_size "$MODEL_SIZE" \
    --epochs "$EPOCHS" \
    --batch "$BATCH"
```

chmod +x this.

- [ ] **Step 3.2: Smoke test (2 epochs)**

```bash
cd /home/nagaet/pxr-iduction-challenge
EPOCHS=2 bash track1_activity/scripts/unimol/02_pretrain_molv2.sh 2>&1 | tee /tmp/unimol_pretrain_smoke.log | tail -40
ls -la models/unimol_v2_log2fc/
```

Expected: training runs 2 epochs, checkpoint emitted to `models/unimol_v2_log2fc/exp/<ts>/`. If `MolTrain` argument names differ from the wrapper, adjust (check `unimol_tools/train.py::MolTrain.__init__`).

- [ ] **Step 3.3: Full pretrain run**

```bash
cd /home/nagaet/pxr-iduction-challenge
nohup bash track1_activity/scripts/unimol/02_pretrain_molv2.sh \
    > /tmp/unimol_pretrain_full.log 2>&1 &
echo $! > /tmp/unimol_pretrain_pid.txt
disown
```

Monitor:
```bash
tail -30 /tmp/unimol_pretrain_full.log
```

Expected wall-clock: ~2-4h for 30 epochs. Track val_mae; should decrease monotone-ish.

- [ ] **Step 3.4: Verify checkpoint**

After completion:

```bash
find models/unimol_v2_log2fc -type f -name "*.pt*" | head
```

Note the checkpoint path (e.g., `models/unimol_v2_log2fc/exp/<ts>/model_0.pth`) — needed for Task 4.

- [ ] **Step 3.5: chmod + commit**

```bash
chmod +x track1_activity/scripts/unimol/02_pretrain_molv2.sh
git add track1_activity/scripts/unimol/02_pretrain_molv2.sh
git commit -m "feat(unimol): MolTrain pretrain wrapper for log2_fc"
```

---

## Task 4: Extract CLS representation (frozen)

**Files:**
- Create: `/home/nagaet/pxr-iduction-challenge/track1_activity/scripts/unimol/03_extract_repr.sh`
- Create: `/home/nagaet/pxr-iduction-challenge/track1_activity/scripts/unimol/04_npz_to_parquet.py`

- [ ] **Step 4.1: Write extractor wrapper**

Create `/home/nagaet/pxr-iduction-challenge/track1_activity/scripts/unimol/03_extract_repr.sh`:

```bash
#!/usr/bin/env bash
# Extract Uni-Mol v2 CLS representation for all 13,136 compounds using
# the pretrained-on-log2_fc checkpoint from Task 3.
#
# unimol_tools.UniMolRepr auto-loads weights from HuggingFace by default.
# We point it at our finetuned checkpoint via UNIMOL_WEIGHT_DIR env var
# (per unimol_tools README).
#
# Output: data/unimol/cls_repr.npz with keys compound_id, cls_repr
set -euo pipefail

PXR_REPO="${PXR_REPO:-$HOME/pxr-iduction-challenge}"
UNIMOL_REPO="${UNIMOL_REPO:-$HOME/ghq/github.com/deepmodeling/Uni-Mol/unimol_tools}"
CKPT_DIR="${CKPT_DIR:-$PXR_REPO/models/unimol_v2_log2fc}"
MODEL_SIZE="${MODEL_SIZE:-84m}"

if [[ ! -f "$PXR_REPO/data/unimol/pretrain_all.csv" ]]; then
    echo "ERROR: pretrain_all.csv not found. Run 01_prepare_log2fc_data.py." >&2
    exit 1
fi

# Point UNIMOL_WEIGHT_DIR at the finetuned checkpoint directory
export UNIMOL_WEIGHT_DIR="$CKPT_DIR"

cat > /tmp/unimol_extract_repr.py <<PYEOF
import argparse, numpy as np, pandas as pd
from unimol_tools import UniMolRepr
ap = argparse.ArgumentParser()
ap.add_argument('--csv', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--model_size', default='$MODEL_SIZE')
args = ap.parse_args()

df = pd.read_csv(args.csv)
assert 'SMILES' in df.columns
smiles = df['SMILES'].tolist()
cids = df['compound_id'].astype(int).tolist()

r = UniMolRepr(model_name='unimolv2', model_size=args.model_size, use_cuda=True)
out = r.get_repr(data=smiles)
cls = np.array(out['cls_repr'], dtype=np.float32)
print('cls shape:', cls.shape)
np.savez(args.out, compound_id=np.array(cids, dtype=np.int64), cls_repr=cls)
print('saved:', args.out)
PYEOF

cd "$UNIMOL_REPO"
pixi run --manifest-path "$UNIMOL_REPO/pixi.toml" python /tmp/unimol_extract_repr.py \
    --csv "$PXR_REPO/data/unimol/pretrain_all.csv" \
    --out "$PXR_REPO/data/unimol/cls_repr.npz" \
    --model_size "$MODEL_SIZE"
```

chmod +x.

- [ ] **Step 4.2: Write npz-to-parquet converter**

Create `/home/nagaet/pxr-iduction-challenge/track1_activity/scripts/unimol/04_npz_to_parquet.py`:

```python
"""Convert Uni-Mol CLS repr npz → parquet for run_train.py consumption.

Input:  data/unimol/cls_repr.npz (keys: compound_id, cls_repr)
Output: data/unimol_v2_pretrain_embed.parquet (index=compound_id,
        columns=emb_0000..emb_NNNN)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
IN_PATH = REPO_ROOT.joinpath("data", "unimol", "cls_repr.npz")
OUT_PATH = REPO_ROOT.joinpath("data", "unimol_v2_pretrain_embed.parquet")


def main() -> None:
    data = np.load(IN_PATH)
    cids = data["compound_id"].astype(int)
    cls = data["cls_repr"].astype(np.float32)
    print(f"Loaded npz: compound_id {cids.shape} cls_repr {cls.shape}")
    if cids.shape[0] != cls.shape[0]:
        raise SystemExit("row count mismatch")

    cols = [f"emb_{i:04d}" for i in range(cls.shape[1])]
    df = pd.DataFrame(cls, index=pd.Index(cids, name="compound_id"), columns=cols)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH)
    print(f"Wrote {OUT_PATH}  shape {df.shape}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.3: Run extractor (smoke first)**

Smoke with head 100:

```bash
head -101 data/unimol/pretrain_all.csv > data/unimol/pretrain_smoke.csv
CKPT_DIR=models/unimol_v2_log2fc \
bash track1_activity/scripts/unimol/03_extract_repr.sh
```

Check output npz:
```bash
pixi run python -c "
import numpy as np
d = np.load('data/unimol/cls_repr.npz')
print('keys:', list(d.keys()))
for k in d.keys():
    print(f'  {k}: shape {d[k].shape}')
"
```

Expected: cls_repr shape (13136, ~512). If 100 only (smoke), verify shape is (100, ~).

- [ ] **Step 4.4: Full extraction**

```bash
bash track1_activity/scripts/unimol/03_extract_repr.sh 2>&1 | tail -10
```

Expected: (13136, embed_dim) npz, ~10-20 min.

- [ ] **Step 4.5: Convert to parquet**

```bash
pixi run python track1_activity/scripts/unimol/04_npz_to_parquet.py 2>&1 | tail -5
pixi run python -c "
import pandas as pd
df = pd.read_parquet('data/unimol_v2_pretrain_embed.parquet')
print('shape:', df.shape)
print('index:', df.index.name, df.index.min(), df.index.max())
print('NaN:', int(df.isna().sum().sum()))
"
```

Expected: (13136, ~512), compound_id 1..13136, 0 NaN.

- [ ] **Step 4.6: Format + lint + commit**

```bash
pixi run ruff format track1_activity/scripts/unimol/04_npz_to_parquet.py
pixi run ruff check track1_activity/scripts/unimol/04_npz_to_parquet.py
chmod +x track1_activity/scripts/unimol/03_extract_repr.sh
git add track1_activity/scripts/unimol/03_extract_repr.sh track1_activity/scripts/unimol/04_npz_to_parquet.py
git commit -m "feat(unimol): CLS repr extraction + npz->parquet converter"
```

---

## Task 5: Register `unimol_v2_pretrain_embed` feature

**Files:**
- Modify: `/home/nagaet/pxr-iduction-challenge/track1_activity/scripts/run_train.py`

- [ ] **Step 5.1: Locate insertion point**

```bash
grep -n "gatedgcn_pretrain_embed" track1_activity/scripts/run_train.py
```

- [ ] **Step 5.2: Add feature branch**

After the `gatedgcn_pretrain_embed` branch's `return X_train, X_test`, insert:

```python
    if feature_name == "unimol_v2_pretrain_embed":
        # Uni-Mol v2 CLS representation, pretrained on single_concentration
        # log2_fc via unimol_tools.MolTrain with ETKDG conformers. See:
        #   track1_activity/scripts/unimol/02_pretrain_molv2.sh
        #   track1_activity/scripts/unimol/03_extract_repr.sh
        # First 3D-structure-aware pool member (protein-free, ligand-3D-only,
        # distinct from Boltz trunk which is protein-ligand joint).
        embed_path = REPO_ROOT.joinpath("data", "unimol_v2_pretrain_embed.parquet")
        if not embed_path.exists():
            raise SystemExit(
                f"Missing {embed_path}. Run "
                f"track1_activity/scripts/unimol/04_npz_to_parquet.py"
            )
        emb_df = pd.read_parquet(embed_path)
        X_train = emb_df.reindex(index=train_ids).to_numpy(dtype=np.float32).copy()
        X_test = emb_df.reindex(index=test_ids).to_numpy(dtype=np.float32).copy()
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        print(
            f"  unimol_v2_pretrain_embed: {X_train.shape[1]} dims "
            f"(train {X_train.shape[0]} / test {X_test.shape[0]})"
        )
        return X_train, X_test
```

- [ ] **Step 5.3: Add to all_features list**

After `"gatedgcn_pretrain_embed",` in the CLI choices list, add `"unimol_v2_pretrain_embed",`.

- [ ] **Step 5.4: Smoke test feature loader**

```bash
pixi run python -c "
import sys
sys.path.insert(0, 'track1_activity/scripts')
sys.path.insert(0, 'track1_activity/src')
from run_train import load_features
import psycopg2, pandas as pd, numpy as np
from data import DB_PARAMS
with psycopg2.connect(**DB_PARAMS) as conn:
    tr = pd.read_sql('SELECT compound_id FROM train_activity ORDER BY compound_id', conn)['compound_id'].astype(int).tolist()
    te = pd.read_sql('SELECT compound_id FROM test_activity ORDER BY compound_id', conn)['compound_id'].astype(int).tolist()
Xtr, Xte = load_features('unimol_v2_pretrain_embed', tr, te)
print('train', Xtr.shape, 'test', Xte.shape, 'NaN', int(np.isnan(Xtr).sum() + np.isnan(Xte).sum()))
"
```

Expected: `train (4140, embed_dim) test (513, embed_dim) NaN 0`.

- [ ] **Step 5.5: Format + lint + commit**

```bash
pixi run ruff format track1_activity/scripts/run_train.py
pixi run ruff check track1_activity/scripts/run_train.py
git add track1_activity/scripts/run_train.py
git commit -m "feat(unimol): register unimol_v2_pretrain_embed feature"
```

---

## Task 6: TabPFN 5-fold + acceptance

**Files:** none.

- [ ] **Step 6.1: Launch training**

```bash
cd /home/nagaet/pxr-iduction-challenge
nohup pixi run python track1_activity/scripts/run_train.py \
    --model tabpfn \
    --feature unimol_v2_pretrain_embed \
    --split umap \
    --trials 0 \
    > /tmp/tabpfn_unimol.log 2>&1 &
echo $! > /tmp/tabpfn_unimol_pid.txt
disown
```

Expected wall-clock: 20-30 min.

- [ ] **Step 6.2: Verify DB record + MAE**

```bash
pixi run python -c "
import psycopg2, sys
sys.path.insert(0, 'track1_activity/src')
from data import DB_PARAMS
with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
    cur.execute(\"SELECT name, mae_mean, rae_mean, spearman_mean, kendall_mean FROM experiment_summary WHERE name LIKE 'tabpfn_unimol_v2_pretrain_embed%' ORDER BY created_at DESC LIMIT 2\")
    for r in cur.fetchall():
        print(r)
"
```

**Gate**: OOF MAE ≤ 0.48 strict. If fails, DONE_WITH_CONCERNS but still proceed to pool bakeoff.

- [ ] **Step 6.3: Pearson r vs 8-pool**

```bash
pixi run python -c "
import psycopg2, sys, pandas as pd
sys.path.insert(0, 'track1_activity/src')
from data import DB_PARAMS

names = [
    'tabpfn_cheme_2d_full_boltz_log2fc_pred_umap_default',
    'tabpfn_chemprop_pretrain_embed_umap_default',
    'tabpfn_pooled_boltz_umap_default',
    'tabpfn_pooled_boltz_allpairs_umap_default',
    'tabpfn_molformer_c3_pretrain_embed_umap',
    'tabpfn_kermt_pretrain_embed_umap_default',
    'tabpfn_attentivefp_pretrain_embed_umap_default',
    'tabpfn_gatedgcn_pretrain_embed_umap_default',
    'tabpfn_unimol_v2_pretrain_embed_umap_default',
]
with psycopg2.connect(**DB_PARAMS) as conn:
    cur = conn.cursor()
    oofs = {}
    for n in names:
        cur.execute('SELECT eop.train_idx, eop.oof_prediction FROM experiment_oof_predictions eop JOIN experiments e ON e.id = eop.experiment_id WHERE e.name=%s ORDER BY eop.train_idx', (n,))
        rows = cur.fetchall()
        if rows:
            oofs[n] = pd.Series({int(r[0]): float(r[1]) for r in rows})
df = pd.DataFrame(oofs).dropna()
new = 'tabpfn_unimol_v2_pretrain_embed_umap_default'
print(f'shape: {df.shape}')
for other in df.columns:
    if other != new:
        r = df[new].corr(df[other])
        print(f'  unimol vs {other:<55} r={r:.4f}')
"
```

Gate: all Pearson r < 0.95 preferred (< 0.96 acceptable).

---

## Task 7: Ensemble bakeoff + integration

**Files:**
- Modify: `/home/nagaet/pxr-iduction-challenge/track1_activity/scripts/run_ensemble.py`

- [ ] **Step 7.1: Swap/add bakeoff**

```bash
pixi run python -c "
import sys, numpy as np, pandas as pd, psycopg2
from pathlib import Path
REPO_ROOT = Path('/home/nagaet/pxr-iduction-challenge')
sys.path.insert(0, str(REPO_ROOT.joinpath('track1_activity', 'src')))
sys.path.insert(0, str(REPO_ROOT.joinpath('track1_activity', 'scripts')))
from data import DB_PARAMS, load_train_smiles_target
from run_ensemble import optimize_caruana

BASE = [
    'tabpfn_cheme_2d_full_boltz_log2fc_pred_umap_default',
    'tabpfn_chemprop_pretrain_embed_umap_default',
    'tabpfn_pooled_boltz_umap_default',
    'tabpfn_pooled_boltz_allpairs_umap_default',
    'tabpfn_molformer_c3_pretrain_embed_umap',
    'tabpfn_kermt_pretrain_embed_umap_default',
    'tabpfn_attentivefp_pretrain_embed_umap_default',
    'tabpfn_gatedgcn_pretrain_embed_umap_default',
]
NEW = 'tabpfn_unimol_v2_pretrain_embed_umap_default'

y = load_train_smiles_target()['pec50'].to_numpy(dtype=np.float32)
with psycopg2.connect(**DB_PARAMS) as conn:
    cur = conn.cursor()
    oofs = {}
    for n in BASE + [NEW]:
        cur.execute('SELECT eop.train_idx, eop.oof_prediction FROM experiment_oof_predictions eop JOIN experiments e ON e.id = eop.experiment_id WHERE e.name=%s ORDER BY eop.train_idx', (n,))
        rows = cur.fetchall()
        v = np.full(len(y), np.nan, dtype=np.float32)
        for idx, pred in rows:
            v[int(idx)] = float(pred)
        oofs[n] = v

def run_caruana(members):
    mat = np.stack([oofs[m] for m in members], axis=1)
    mask = np.all(np.isfinite(mat), axis=1) & np.isfinite(y)
    w = optimize_caruana(mat[mask], y[mask])
    mae = float(np.mean(np.abs(mat[mask] @ w - y[mask])))
    return mae, {m: float(wi) for m, wi in zip(members, w)}

A_mae, _ = run_caruana(BASE)
add_mae, add_w = run_caruana(BASE + [NEW])
# Try 3 swap candidates — weakest members in current pool
for swap_target in ['tabpfn_gatedgcn_pretrain_embed_umap_default',
                    'tabpfn_attentivefp_pretrain_embed_umap_default',
                    'tabpfn_pooled_boltz_umap_default']:
    swap_mae, swap_w = run_caruana([m for m in BASE if m != swap_target] + [NEW])
    print(f'swap -{swap_target:<55} + unimol -> MAE {swap_mae:.4f}  Δ {swap_mae - A_mae:+.4f}  wt {swap_w[NEW]:.4f}')
print()
print(f'Baseline 8-pool MAE:   {A_mae:.4f}')
print(f'add-9 (keep 8 + unimol): MAE {add_mae:.4f}  Δ {add_mae - A_mae:+.4f}  wt {add_w[NEW]:.4f}')
"
```

Pick the config with best (lowest) MAE. If none < baseline, try DONE_WITH_CONCERNS add-9 (pool still grows).

- [ ] **Step 7.2: Edit run_ensemble.py**

Per chosen config, edit `ENSEMBLE_MODELS` in `track1_activity/scripts/run_ensemble.py`. Add a comment block mirroring the gatedgcn or attentivefp entries above.

- [ ] **Step 7.3: Run caruana + calibrate**

```bash
pixi run python track1_activity/scripts/run_ensemble.py 2>&1 | tee /tmp/ens_unimol.log | tail -20
pixi run python track1_activity/scripts/run_ensemble_calibrate.py 2>&1 | tee /tmp/cal_unimol.log | tail -15
wc -l track1_activity/submissions/ens_caruana_bag20_calibrated_best.csv
```

- [ ] **Step 7.4: Format + lint + commit**

```bash
pixi run ruff format track1_activity/scripts/run_ensemble.py
pixi run ruff check track1_activity/scripts/run_ensemble.py
git add track1_activity/scripts/run_ensemble.py
git commit -m "$(cat <<'COMMIT_EOF'
feat(ens): unimol_v2_pretrain_embed — 3D-aware pool member

First pure ligand-3D backbone in pool. Uni-Mol v2 (<SIZE>) pretrained
on log2_fc via unimol_tools.MolTrain (ETKDG conformers, <CHOSEN>
config), CLS repr extracted for all 13,136 compounds.

Task 6 single-model OOF:
- MAE <FILL>  (<gate status>)
- RAE <FILL>, Spearman <FILL>
- Pearson r vs existing 8-pool members: max <FILL> with <member>

Task 7 <swap|add-9> 8/9-pool caruana_bag20:
- MAE <FILL> (vs 8-pool baseline 0.4185, delta <FILL>)
- unimol weight <FILL>
COMMIT_EOF
)"
```

---

## Task 8: Push + PR + LB

- [ ] **Step 8.1: Push**

```bash
cd /home/nagaet/pxr-iduction-challenge
git push -u origin feature/unimol-etkdg-pretrain-embed
```

- [ ] **Step 8.2: Open PR**

```bash
gh pr create --title "feat: Uni-Mol v2 × ETKDG pretrain-embed (3D-aware pool member)" --body "$(cat <<'EOF'
## Summary
- Adds `tabpfn_unimol_v2_pretrain_embed_umap_default` as the first pure-ligand-3D pool member via `unimol_tools`.
- Path 2 (user-chosen): ETKDG for BOTH pretrain and inference, avoiding Uni-Mol's native RDKit-conformer distribution shift.
- Isolated pixi project at `~/ghq/github.com/deepmodeling/Uni-Mol/unimol_tools`; main PXR pixi env untouched.
- Full 13,136-compound coverage (no Boltz coverage restriction).

## Acceptance results
1. ETKDG generation: handled internally by unimol_tools, no failures.
2. Pretrain convergence: val MAE <FILL> after <EPOCHS> epochs.
3. Single-model OOF MAE: <FILL>.
4. caruana weight: <FILL>.
5. Pool MAE (swap or add-9): <FILL> vs 8-pool 0.4185, Δ <FILL>.
6. Pearson r with existing members: max <FILL> with <member>.
7. ruff clean.

## Files
### New
- track1_activity/scripts/unimol/01_prepare_log2fc_data.py
- track1_activity/scripts/unimol/02_pretrain_molv2.sh
- track1_activity/scripts/unimol/03_extract_repr.sh
- track1_activity/scripts/unimol/04_npz_to_parquet.py

### Modified
- track1_activity/scripts/run_train.py
- track1_activity/scripts/run_ensemble.py
- .gitignore

### External (NOT in PR)
- ~/ghq/github.com/deepmodeling/Uni-Mol/unimol_tools/pixi.toml (isolated env)
- models/unimol_v2_log2fc/ (checkpoints, gitignored)
- data/unimol/ (CSV + npz, gitignored)

## CI: N/A

## Related
- Spec: `docs/superpowers/specs/2026-04-21-unimol-etkdg-pretrain-embed-design.md`
- Plan: `docs/superpowers/plans/2026-04-21-unimol-etkdg-pretrain-embed.md`
- Memory: `project_protein_ligand_coupling_nn_weak` — this PR addresses the "Uni-Mol NOT covered" open item
EOF
)"
```

- [ ] **Step 8.3: LB plan**

After merge, use `scheduled_submit.sh`:

```bash
nohup bash track1_activity/scripts/scheduled_submit.sh \
    track1_activity/submissions/ens_caruana_bag20_calibrated_best.csv \
    --experiment ens_caruana_bag20_calibrated_best \
    --notes "Uni-Mol v2 pretrain-embed added (PR #<N>, 3D-aware). ..." \
    > /tmp/submit_unimol.log 2>&1 &
disown
```

Report PR URL + cooldown. Wait for user before merge.

---

## Self-review notes

Spec coverage:
- Phase 1 env setup → Task 1
- Phase 2 data prep → Task 2
- Phase 3 pretrain → Task 3
- Phase 4 extract → Task 4
- Phase 5 TabPFN + ensemble → Tasks 5-8

Placeholder scan:
- `<FILL>` / `<CHOSEN>` / `<SIZE>` / `<EPOCHS>` only in commit message and PR body templates — explicit "fill from task outputs" markers
- No placeholder code

Type consistency:
- `compound_id` column + int index across all parquets
- `cls_repr` key consistent between extraction script and converter
- `UniMolRepr` API calls match the actual class signature in `unimol_tools/predictor.py`

Fluid points:
- `MolTrain` exact `target_cols` kwarg name may differ from the actual `unimol_tools` version (verify via `MolTrain.__init__` signature during smoke)
- Model size `84m` is default; if 164m OOMs or is unavailable, stays at 84m
- ETKDG is handled by `unimol_tools` internally — Option B (hybrid cache reuse) is deferred as an optimization per spec
- Checkpoint loading convention for `UniMolRepr` with fine-tuned weights: via `UNIMOL_WEIGHT_DIR` env var; if API has changed, adapt
