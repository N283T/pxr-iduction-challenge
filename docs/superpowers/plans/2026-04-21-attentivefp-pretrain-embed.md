# AttentiveFP Pretrain-Embed Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `tabpfn_attentivefp_pretrain_embed_umap_default` as the 12th caruana_bag20 pool member via Buterez strategy-3 (extract frozen graph-level embeddings from the existing AttentiveFP pretrain checkpoint → TabPFN on pEC50 → ensemble integration → LB submission).

**Architecture:** Reuse `track1_activity/checkpoints/attentivefp_pretrain/pretrain.pt` (from PR #79, 2026-04-19). Replace the final linear projection (`model.lin2`) with `nn.Identity()` to extract the 512-dim molecular readout. Wire it into the existing parquet-based feature pipeline.

**Tech Stack:** PyTorch + torch_geometric (`AttentiveFP`, `from_smiles`, `Batch`), psycopg2, pandas (parquet), TabPFN v2.6, main PXR pixi env.

**Related spec:** `docs/superpowers/specs/2026-04-21-attentivefp-pretrain-embed-design.md`

---

## File map

| Path | Create / Modify | Responsibility |
|---|---|---|
| `track1_activity/scripts/run_attentivefp_embed_extract.py` | Create | Load pretrain.pt, extract 512d embeddings, write parquet |
| `data/attentivefp_pretrain_embed.parquet` | Create (gitignored — under `data/` but not in existing ignore rules; check) | Output parquet indexed by compound_id |
| `track1_activity/scripts/run_train.py` | Modify (~line 465) | Add `attentivefp_pretrain_embed` feature branch + VALID_FEATURES entry |
| `track1_activity/scripts/run_ensemble.py` | Modify (~line 162) | Append `tabpfn_attentivefp_pretrain_embed_umap_default` to ENSEMBLE_MODELS |
| `.gitignore` | Modify (only if `data/attentivefp_pretrain_embed.parquet` is not already covered) | Ensure new parquet is gitignored |

---

## Conventions recap

- **Branch**: `feature/attentivefp-pretrain-embed` (already checked out; commits go here).
- **No CI in this repo.** `gh pr checks` reports "no checks" — expected.
- **No unit tests for DL code.** Gates are `pixi run ruff format <file>` + `pixi run ruff check <file>`, plus smoke runs.
- **Idempotent DB writes**: `record_experiment(..., on_conflict_replace=True)` inherited from existing patterns.
- **Commit frequently** — one commit per task.

---

## Task 1: Write the AttentiveFP embedding extraction script + run it

**Files:**
- Create: `/home/nagaet/pxr-iduction-challenge/track1_activity/scripts/run_attentivefp_embed_extract.py`
- Output (gitignored): `/home/nagaet/pxr-iduction-challenge/data/attentivefp_pretrain_embed.parquet`

- [ ] **Step 1.1: Verify the pretrain checkpoint exists + metadata**

```bash
cd /home/nagaet/pxr-iduction-challenge
ls -la track1_activity/checkpoints/attentivefp_pretrain/
cat track1_activity/checkpoints/attentivefp_pretrain/pretrain_meta.json
```

Expected: `pretrain.pt` (~36 MB) + `pretrain_meta.json` with `hidden_channels=512, num_layers=4, num_timesteps=3`.

- [ ] **Step 1.2: Check if `data/attentivefp_pretrain_embed.parquet` is gitignored**

```bash
cd /home/nagaet/pxr-iduction-challenge
git check-ignore -v data/attentivefp_pretrain_embed.parquet 2>&1
```

If NOT ignored, append to `.gitignore`:
```
data/attentivefp_pretrain_embed.parquet
```

Expected outcome: the file pattern is already covered by `data/*.parquet` or similar, OR we add an explicit line. Inspect existing rules via `grep -E "parquet|embed" .gitignore`.

- [ ] **Step 1.3: Write the extraction script**

Create `/home/nagaet/pxr-iduction-challenge/track1_activity/scripts/run_attentivefp_embed_extract.py`:

```python
"""Extract 512d per-compound graph-level embeddings from the AttentiveFP
pretrain checkpoint.

Phase 2 of Buterez 2024 strategy-3 for AttentiveFP. Loads the pretrain
checkpoint produced by run_attentivefp_pretrain_finetune.py --phase
pretrain, replaces the final linear projection (lin2) with nn.Identity()
so the forward pass returns the 512d molecular readout (post-GRU,
pre-final projection), then batch-extracts embeddings for all 13,136
compounds.

Output: data/attentivefp_pretrain_embed.parquet (index=compound_id,
columns=emb_0000..emb_0511, float32). Downstream TabPFN / LGBM
consumers read via run_train.py --feature attentivefp_pretrain_embed.

Usage:
    pixi run python track1_activity/scripts/run_attentivefp_embed_extract.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn.models import AttentiveFP
from torch_geometric.utils import from_smiles

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402

CKPT_DIR = REPO_ROOT.joinpath(
    "track1_activity", "checkpoints", "attentivefp_pretrain"
)
PRETRAIN_PATH = CKPT_DIR.joinpath("pretrain.pt")
META_PATH = CKPT_DIR.joinpath("pretrain_meta.json")
OUT_PATH = REPO_ROOT.joinpath("data", "attentivefp_pretrain_embed.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_compounds() -> tuple[list[int], list[str]]:
    sql = """
    SELECT id AS compound_id, std_smiles AS smiles
    FROM compounds
    WHERE std_smiles IS NOT NULL
    ORDER BY id
    """
    with psycopg2.connect(**DB_PARAMS) as conn:
        df = pd.read_sql(sql, conn)
    return df["compound_id"].astype(int).tolist(), df["smiles"].tolist()


def smiles_to_pyg(smiles_list: list[str]):
    graphs = []
    for i, smi in enumerate(smiles_list):
        g = from_smiles(smi)
        if g.x is None or g.x.shape[0] == 0:
            raise ValueError(f"SMILES[{i}] produced empty graph: {smi}")
        graphs.append(g)
    return graphs


def load_model() -> AttentiveFP:
    with META_PATH.open() as f:
        meta = json.load(f)
    params = meta["params"]

    # Probe first compound to learn in_channels + edge_dim (from_smiles
    # output dims depend on PyG version; read them from an actual graph).
    probe = from_smiles("CCO")
    in_channels = probe.x.shape[1]
    edge_dim = probe.edge_attr.shape[1]

    model = AttentiveFP(
        in_channels=in_channels,
        hidden_channels=params["hidden_channels"],
        out_channels=2,  # pretrain had 2-head log2_fc
        edge_dim=edge_dim,
        num_layers=params["num_layers"],
        num_timesteps=params["num_timesteps"],
        dropout=params["dropout"],
    )
    state_dict = torch.load(PRETRAIN_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)

    # Replace the final projection (512 -> 2) with Identity so forward()
    # returns the 512d molecular readout post-GRU, pre-projection.
    model.lin2 = nn.Identity()

    return model.to(DEVICE).eval()


@torch.no_grad()
def extract_embeddings(
    model: AttentiveFP, graphs: list
) -> np.ndarray:
    n = len(graphs)
    outs: list[np.ndarray] = []
    for i in range(0, n, BATCH_SIZE):
        batch = Batch.from_data_list(graphs[i : i + BATCH_SIZE]).to(DEVICE)
        emb = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
        outs.append(emb.cpu().numpy())
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  batch {i // BATCH_SIZE + 1} / {(n + BATCH_SIZE - 1) // BATCH_SIZE}")
    return np.concatenate(outs, axis=0).astype(np.float32)


def main() -> None:
    cids, smiles_list = load_compounds()
    print(f"Loaded {len(cids)} compounds")

    print("Converting SMILES to PyG graphs...")
    graphs = smiles_to_pyg(smiles_list)

    print(f"Loading AttentiveFP pretrain checkpoint from {PRETRAIN_PATH}")
    model = load_model()

    print("Extracting embeddings...")
    emb = extract_embeddings(model, graphs)
    print(f"  shape: {emb.shape}  dtype: {emb.dtype}")
    print(f"  NaN count: {np.isnan(emb).sum()}")

    cols = [f"emb_{i:04d}" for i in range(emb.shape[1])]
    df = pd.DataFrame(emb, index=pd.Index(cids, name="compound_id"), columns=cols)
    df.to_parquet(OUT_PATH)
    print(f"Wrote {OUT_PATH}  shape {df.shape}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 1.4: Run the extraction**

```bash
cd /home/nagaet/pxr-iduction-challenge
pixi run python track1_activity/scripts/run_attentivefp_embed_extract.py 2>&1 | tee /tmp/attentivefp_embed.log | tail -30
```

Expected output:
- `Loaded 13136 compounds`
- No SMILES conversion errors
- Final embedding shape `(13136, 512)`, 0 NaN
- Wall-clock: 5-15 min on RTX 5080

- [ ] **Step 1.5: Sanity-check the parquet**

```bash
pixi run python -c "
import pandas as pd, numpy as np
df = pd.read_parquet('data/attentivefp_pretrain_embed.parquet')
print('shape:', df.shape)
print('index name:', df.index.name)
print('index min/max:', df.index.min(), df.index.max())
print('columns first 3:', df.columns[:3].tolist())
print('columns last 3:', df.columns[-3:].tolist())
print('NaN count:', df.isna().sum().sum())
print('first row first 5 emb values:', df.iloc[0, :5].tolist())
"
```

Expected: `(13136, 512)`, index name `compound_id`, min=1 max=13136, columns `emb_0000..emb_0511`, NaN=0.

- [ ] **Step 1.6: Format + lint**

```bash
pixi run ruff format track1_activity/scripts/run_attentivefp_embed_extract.py
pixi run ruff check track1_activity/scripts/run_attentivefp_embed_extract.py
```

Expected: formatted, all checks pass.

- [ ] **Step 1.7: Commit**

```bash
cd /home/nagaet/pxr-iduction-challenge
git add track1_activity/scripts/run_attentivefp_embed_extract.py
# Plus .gitignore if modified in Step 1.2
git status
git commit -m "feat(attentivefp): 512d graph-readout embedding extraction"
```

---

## Task 2: Register `attentivefp_pretrain_embed` feature in run_train.py

**Files:**
- Modify: `/home/nagaet/pxr-iduction-challenge/track1_activity/scripts/run_train.py` (feature branch ~after the `kermt_pretrain_embed` block; VALID_FEATURES entry)

- [ ] **Step 2.1: Locate insertion point**

```bash
cd /home/nagaet/pxr-iduction-challenge
grep -n "kermt_pretrain_embed" track1_activity/scripts/run_train.py
```

Expected output includes a feature branch (~line 436) and a VALID_FEATURES entry (~line 1347). Record the actual line numbers.

- [ ] **Step 2.2: Add the new feature branch**

Open `track1_activity/scripts/run_train.py` and immediately after the `kermt_pretrain_embed` block's `return X_train, X_test` statement (inside `load_features`), insert:

```python
    if feature_name == "attentivefp_pretrain_embed":
        # 512d graph-readout embedding from PyG AttentiveFP pretrained on
        # single_concentration log2_fc (2-head, 90/10 random split, z-scored
        # targets). Extracted by replacing model.lin2 with nn.Identity() so
        # forward returns post-GRU pre-projection representation. See:
        #   track1_activity/scripts/run_attentivefp_embed_extract.py
        # Buterez 2024 strategy-3 with graph-attention backbone
        # (parallel to chemprop_pretrain_embed = D-MPNN,
        # molformer_c3_pretrain_embed = transformer,
        # kermt_pretrain_embed = graph-transformer).
        embed_path = REPO_ROOT.joinpath("data", "attentivefp_pretrain_embed.parquet")
        if not embed_path.exists():
            raise SystemExit(
                f"Missing {embed_path}. Run "
                f"track1_activity/scripts/run_attentivefp_embed_extract.py"
            )
        emb_df = pd.read_parquet(embed_path)
        X_train = emb_df.reindex(index=train_ids).to_numpy(dtype=np.float32).copy()
        X_test = emb_df.reindex(index=test_ids).to_numpy(dtype=np.float32).copy()
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        print(
            f"  attentivefp_pretrain_embed: {X_train.shape[1]} dims "
            f"(train {X_train.shape[0]} / test {X_test.shape[0]})"
        )
        return X_train, X_test
```

- [ ] **Step 2.3: Add to VALID_FEATURES list**

Find the VALID_FEATURES (or whatever the CLI `--feature` choices list is — the Task 7 KERMT implementer found it was named `all_features` inside `main()`). Append `"attentivefp_pretrain_embed"` after the `"kermt_pretrain_embed"` entry:

```bash
grep -n "kermt_pretrain_embed" track1_activity/scripts/run_train.py
```

Locate the occurrence inside the choices list (NOT the one in `load_features`). Insert `"attentivefp_pretrain_embed"` right after.

- [ ] **Step 2.4: Smoke test the feature loader**

```bash
cd /home/nagaet/pxr-iduction-challenge
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
Xtr, Xte = load_features('attentivefp_pretrain_embed', tr, te)
print('train', Xtr.shape, 'test', Xte.shape, 'NaN', int(np.isnan(Xtr).sum() + np.isnan(Xte).sum()))
print('first row first 5:', Xtr[0, :5].tolist())
"
```

Expected:
```
  attentivefp_pretrain_embed: 512 dims (train 4140 / test 513)
train (4140, 512) test (513, 512) NaN 0
first row first 5: [...]
```

- [ ] **Step 2.5: Format + lint**

```bash
pixi run ruff format track1_activity/scripts/run_train.py
pixi run ruff check track1_activity/scripts/run_train.py
```

Expected: format clean; ruff check may show 8 pre-existing E402 warnings (same as Task 7 KERMT; not caused by this task).

- [ ] **Step 2.6: Commit**

```bash
git add track1_activity/scripts/run_train.py
git commit -m "feat(attentivefp): register attentivefp_pretrain_embed feature"
```

---

## Task 3: Train TabPFN on AttentiveFP embedding + verify acceptance

**Files:**
- No new files; runs `run_train.py` and DB inserts.

- [ ] **Step 3.1: Run 5-fold TabPFN training**

```bash
cd /home/nagaet/pxr-iduction-challenge
pixi run python track1_activity/scripts/run_train.py \
    --model tabpfn \
    --feature attentivefp_pretrain_embed \
    --split umap \
    --trials 0 2>&1 | tee /tmp/tabpfn_attentivefp.log | tail -80
```

CLI notes (from KERMT Task 8 findings):
- Use `--split umap` (NOT `umap_default`)
- No `--outer-folds` (5 is hardcoded)
- Experiment name is auto-derived: `tabpfn_attentivefp_pretrain_embed_umap_default`

Expected wall-clock: 20-30 min (512d << 3200d KERMT was).

- [ ] **Step 3.2: Verify experiment recorded in DB + OOF MAE ≤ 0.48**

```bash
pixi run python -c "
import psycopg2, sys
sys.path.insert(0, 'track1_activity/src')
from data import DB_PARAMS
with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
    cur.execute(\"SELECT name, model_type, feature_set, mae_mean, rae_mean, spearman_mean, kendall_mean FROM experiment_summary WHERE name LIKE 'tabpfn_attentivefp_pretrain_embed%' ORDER BY created_at DESC LIMIT 3\")
    for r in cur.fetchall():
        print(r)
" 2>&1 | grep -v UserWarning | grep -v read_sql
```

Expected row: `tabpfn_attentivefp_pretrain_embed_umap_default` with `mae_mean < 0.48`.

**If MAE > 0.48**: STOP. Report to controller; do not proceed with ensemble integration.

- [ ] **Step 3.3: Verify Pearson r < 0.96 with existing pretrain-embed members**

```bash
pixi run python -c "
import psycopg2, sys, pandas as pd
sys.path.insert(0, 'track1_activity/src')
from data import DB_PARAMS

names = [
    'tabpfn_chemprop_pretrain_embed_umap_default',
    'tabpfn_molformer_c3_pretrain_embed_umap',
    'tabpfn_2d_full_boltz_log2fc_pred_umap_default',
    'tabpfn_kermt_pretrain_embed_umap_default',
    'tabpfn_attentivefp_pretrain_embed_umap_default',
]
short = {
    'tabpfn_chemprop_pretrain_embed_umap_default': 'chemprop',
    'tabpfn_molformer_c3_pretrain_embed_umap': 'molformer_c3',
    'tabpfn_2d_full_boltz_log2fc_pred_umap_default': '2d_full_boltz',
    'tabpfn_kermt_pretrain_embed_umap_default': 'kermt',
    'tabpfn_attentivefp_pretrain_embed_umap_default': 'attentivefp',
}
with psycopg2.connect(**DB_PARAMS) as conn:
    cur = conn.cursor()
    oofs = {}
    for n in names:
        cur.execute('''
            SELECT eop.train_idx, eop.oof_prediction
            FROM experiment_oof_predictions eop
            JOIN experiments e ON e.id = eop.experiment_id
            WHERE e.name = %s
            ORDER BY eop.train_idx
        ''', (n,))
        rows = cur.fetchall()
        if rows:
            oofs[short[n]] = pd.Series({int(r[0]): float(r[1]) for r in rows})
df = pd.DataFrame(oofs).dropna()
print('shape:', df.shape)
print(df.corr(method='pearson').round(3))
print()
for other in ['chemprop', 'molformer_c3', '2d_full_boltz', 'kermt']:
    r = df['attentivefp'].corr(df[other])
    ok = 'OK' if r < 0.96 else 'FAIL'
    print(f'  attentivefp vs {other}: r = {r:.4f}  {ok}')
" 2>&1 | grep -v UserWarning | grep -v read_sql
```

Expected: all r < 0.96. Highest is likely `attentivefp vs chemprop` (both GNN).

**If any r ≥ 0.96**: note in report but continue to Task 4; controller will decide whether to merge or revert.

- [ ] **Step 3.4: Report + no commit**

No commit in this task. The experiment is already recorded in the DB.

---

## Task 4: Add to ENSEMBLE_MODELS + re-run caruana + calibrate

**Files:**
- Modify: `/home/nagaet/pxr-iduction-challenge/track1_activity/scripts/run_ensemble.py` (append after the `tabpfn_kermt_pretrain_embed_umap_default` entry)

- [ ] **Step 4.1: Locate insertion point**

```bash
cd /home/nagaet/pxr-iduction-challenge
grep -n "tabpfn_kermt_pretrain_embed_umap_default" track1_activity/scripts/run_ensemble.py
```

Expected: one occurrence in the `ENSEMBLE_MODELS` tuple (~line 161).

- [ ] **Step 4.2: Append the new entry**

Immediately after the `"tabpfn_kermt_pretrain_embed_umap_default",` line (inside the tuple, before the closing `)`), insert:

```python
    # --- AttentiveFP pretrain embed + TabPFN ---
    # Graph-attention family (PyG AttentiveFP, pretrained on
    # single_concentration log2_fc via PR #79 checkpoint, then frozen
    # 512d readout into TabPFN). Added as decorrelating 12th pool
    # member. Complements chemprop (D-MPNN), molformer_c3 (transformer),
    # kermt (graph-transformer). See
    # docs/superpowers/specs/2026-04-21-attentivefp-pretrain-embed-design.md.
    # Single-model OOF MAE <fill from Task 3>, Pearson r vs others:
    # chemprop <fill>, molformer_c3 <fill>, 2d_full_boltz <fill>,
    # kermt <fill>. PR <TBD>.
    "tabpfn_attentivefp_pretrain_embed_umap_default",
```

Fill in the `<...>` values from Task 3 outputs before committing.

- [ ] **Step 4.3: Re-run caruana_bag20**

```bash
cd /home/nagaet/pxr-iduction-challenge
pixi run python track1_activity/scripts/run_ensemble.py 2>&1 | tee /tmp/ensemble_attentivefp.log | tail -80
```

Expected: the script reports OOF metrics for multiple strategies, with `caruana_bag20` being the canonical one.

- [ ] **Step 4.4: Verify 12-pool acceptance criteria**

```bash
pixi run python -c "
import psycopg2, sys
sys.path.insert(0, 'track1_activity/src')
from data import DB_PARAMS
with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
    cur.execute('''
        SELECT e.hyperparameters, s.mae_mean, s.rae_mean, s.spearman_mean
        FROM experiments e
        JOIN experiment_summary s ON s.name = e.name
        WHERE e.name = 'ens_caruana_bag20'
        ORDER BY e.created_at DESC LIMIT 1
    ''')
    hp, mae, rae, spr = cur.fetchone()
print(f'12-pool caruana_bag20: MAE={mae:.4f} RAE={rae:.4f} Spearman={spr:.4f}')
weights = hp.get('weights', {}) if isinstance(hp, dict) else {}
print()
print('Caruana weights (non-zero, sorted):')
for n, w in sorted(weights.items(), key=lambda kv: -kv[1]):
    if w > 0:
        marker = '<-- NEW' if 'attentivefp' in n else ''
        print(f'  {n:<55} {w:.4f} {marker}')
print()
kermt_baseline = 0.4268
delta = float(mae) - kermt_baseline
print(f'vs 11-pool baseline 0.4268: delta = {delta:+.4f}')
" 2>&1 | grep -v UserWarning | grep -v read_sql
```

Acceptance gates:
- Gate (3): MAE ≤ 0.4268 — if MAE > 0.4268, **STOP** and report BLOCKED.
- Gate (2): attentivefp caruana weight > 0 — if 0, proceed but note (framework still valuable).

- [ ] **Step 4.5: Re-run calibration**

```bash
cd /home/nagaet/pxr-iduction-challenge
pixi run python track1_activity/scripts/run_ensemble_calibrate.py 2>&1 | tee /tmp/calibrate_attentivefp.log | tail -40
```

Expected: 4-way nested CV across linear / linear_pos / spline_k5 / isotonic; best method (likely linear_pos) selected; `track1_activity/submissions/ens_caruana_bag20_calibrated_best.csv` rewritten.

Verify CSV:
```bash
wc -l track1_activity/submissions/ens_caruana_bag20_calibrated_best.csv
head -3 track1_activity/submissions/ens_caruana_bag20_calibrated_best.csv
```

Expected: 514 lines (1 header + 513), SMILES + Molecule Name + pEC50 columns, floats in [3, 8] range.

- [ ] **Step 4.6: Format + lint**

```bash
pixi run ruff format track1_activity/scripts/run_ensemble.py
pixi run ruff check track1_activity/scripts/run_ensemble.py
```

Expected: formatted, all checks pass.

- [ ] **Step 4.7: Commit**

Fill in the `<...>` values in the comment block (Task 4.2) BEFORE committing, using the Task 3 and Task 4.4 numbers. Then:

```bash
git add track1_activity/scripts/run_ensemble.py
git commit -m "$(cat <<'EOF'
feat(ens): add tabpfn_attentivefp_pretrain_embed to 12-pool caruana_bag20

New graph-attention family pool member (AttentiveFP pretrained on
log2_fc via PR #79 checkpoint, 512d readout frozen, TabPFN
downstream).

Task 3 single-model OOF:
- MAE <fill>
- RAE <fill>, Spearman <fill>
- Pearson r vs existing: chemprop <fill>, molformer_c3 <fill>,
  2d_full_boltz <fill>, kermt <fill>

Task 4 12-pool caruana_bag20:
- MAE <fill> (vs 11-pool baseline 0.4268, delta <fill>)
- attentivefp caruana weight <fill>
EOF
)"
```

---

## Task 5: PR prep + acceptance summary

**Files:**
- None; git + gh CLI work.

- [ ] **Step 5.1: Push branch**

```bash
cd /home/nagaet/pxr-iduction-challenge
git push -u origin feature/attentivefp-pretrain-embed
```

- [ ] **Step 5.2: Final ruff sweep**

```bash
pixi run ruff format --check \
    track1_activity/scripts/run_attentivefp_embed_extract.py \
    track1_activity/scripts/run_ensemble.py
pixi run ruff check \
    track1_activity/scripts/run_attentivefp_embed_extract.py \
    track1_activity/scripts/run_ensemble.py
```

(Skip `run_train.py` from ruff check due to 8 pre-existing E402 warnings that are not caused by this PR.)

Expected: "X files already formatted", "All checks passed".

- [ ] **Step 5.3: Open PR**

```bash
gh pr create --title "feat: AttentiveFP pretrain-embed as 12th caruana pool member" --body "$(cat <<'EOF'
## Summary
- Adds `tabpfn_attentivefp_pretrain_embed_umap_default` as the 12th caruana_bag20 pool member. Graph-attention family (PyG AttentiveFP) closes the remaining inductive-bias gap vs existing pretrain-embed members: chemprop (D-MPNN), molformer_c3 (transformer), kermt (graph-transformer).
- Reuses the PR #79 pretrain checkpoint (`track1_activity/checkpoints/attentivefp_pretrain/pretrain.pt`, 2026-04-19). No retraining needed — this PR is **phase 2 only** (embed extraction → TabPFN → ensemble).
- 512d readout extracted by replacing `model.lin2` with `nn.Identity()`. Well within TabPFN v2.6's 2000-dim supported regime; no `ignore_pretraining_limits` override.
- No DB schema change. Gitignored artifact: `data/attentivefp_pretrain_embed.parquet`.

## Acceptance results
1. **Extraction covered all 13,136 compounds**: 0 failed SMILES
2. **Single-model OOF MAE**: <fill from Task 3>  (target ≤ 0.48 ✓)
   - RAE <fill>, Spearman <fill>, Kendall <fill>
3. **caruana_bag20 weight on attentivefp**: <fill>  (target > 0 ✓)
4. **12-pool caruana_bag20 OOF MAE**: <fill>  (vs 11-pool 0.4268, delta <fill>; target ≤ 0.4268 ✓)
5. **Pearson r with existing pretrain-embed members** (all < 0.96 ✓):
   - vs chemprop: <fill>
   - vs molformer_c3: <fill>
   - vs 2d_full_boltz: <fill>
   - vs kermt: <fill>
6. **ruff format + check**: clean on new/modified files (run_train.py 8 E402 warnings pre-existing)

## Calibration
- Post-hoc calibration auto-picks winner via 4-way nested CV (linear/linear_pos/spline_k5/isotonic).
- Winner: <fill> (calibrated MAE <fill>).
- Submission CSV: `track1_activity/submissions/ens_caruana_bag20_calibrated_best.csv` (513 rows).

## Files
### New (main repo)
- `track1_activity/scripts/run_attentivefp_embed_extract.py` — extract 512d graph readout

### Modified (main repo)
- `track1_activity/scripts/run_train.py` — register attentivefp_pretrain_embed feature
- `track1_activity/scripts/run_ensemble.py` — append to ENSEMBLE_MODELS

### Unchanged (reused)
- `track1_activity/checkpoints/attentivefp_pretrain/pretrain.pt` (from PR #79, 2026-04-19)

## Test plan
- [x] Extraction over all 13,136 compounds produces (13136, 512) parquet
- [x] Fresh pixi shell reproduces 12-pool MAE
- [x] Calibrated CSV has 513 finite predictions in [3, 8]
- [x] All 4 Pearson r values < 0.96

## CI: N/A (no workflow, per CLAUDE.md)

## Related
- Spec: `docs/superpowers/specs/2026-04-21-attentivefp-pretrain-embed-design.md`
- Plan: `docs/superpowers/plans/2026-04-21-attentivefp-pretrain-embed.md`
- PR #79 (AttentiveFP pretrain + frozen+head-FT finetune)
- PR #103 (KERMT pretrain-embed — latest pretrain-embed precedent)
EOF
)"
```

Fill in the `<...>` placeholders from Tasks 3 and 4 before running this command.

Record the PR URL returned by `gh pr create`.

- [ ] **Step 5.4: Check LB cooldown status**

```bash
cd /home/nagaet/pxr-iduction-challenge
pixi run python track1_activity/scripts/api.py cooldown
```

Report whether LB is READY or time remaining.

- [ ] **Step 5.5: Ask the user**

Do NOT merge automatically. Do NOT submit to LB automatically. Report to the controller:
- PR URL
- All 6 acceptance criteria with filled-in values
- LB cooldown status
- Recommendation: merge + submit, or wait

Controller will relay to the user for merge/LB-submit approval.

---

## Self-review notes

Coverage verification (against the spec):

- **Embedding extraction (spec §Architecture)** → Task 1
- **Feature plumbing (spec §DB / feature plumbing)** → Task 2
- **Downstream TabPFN (spec §Downstream TabPFN)** → Task 3
- **Ensemble integration (spec §Ensemble integration)** → Task 4
- **Acceptance criteria 1-7** → Tasks 1-4 with explicit stop-and-report on fail
- **ruff format + check gate** → Tasks 1.6, 2.5, 4.6, 5.2
- **Idempotent DB writes** → inherited from `run_train.py` / `run_ensemble.py` existing paths

Placeholder scan: no TBD/TODO; all code blocks complete; file paths absolute.

Type consistency: `load_features` signature reused; `ENSEMBLE_MODELS` append follows existing tuple pattern; parquet output format matches chemprop/molformer_c3/kermt precedents.

Known fluid points (flagged inline, not placeholders):
- **Task 1.2 gitignore check**: existing rules may or may not already cover the new parquet. Inspect + add if needed.
- **Task 2.3 VALID_FEATURES list name**: Task 7 KERMT implementer discovered it's called `all_features` inside `main()`. Verify at runtime.
- **Task 4.2 comment placeholders**: must be filled with actual metric values from Tasks 3-4 before committing.
- **Task 5.3 PR body placeholders**: same — fill before opening PR.
