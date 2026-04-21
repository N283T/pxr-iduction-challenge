# GatedGCN Pretrain-Embed Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `tabpfn_gatedgcn_pretrain_embed_umap_default` as the 13th caruana_bag20 pool member. Extract 128d graph-pooled embeddings from the existing GatedGCN pretrain checkpoint → TabPFN → ensemble integration → LB submission.

**Architecture:** Reuse `track1_activity/checkpoints/gatedgcn_pretrain/pretrain.pt` (from PR #79). Replace `GatedGCNModel.ffn` with `nn.Identity()` to expose the 128d `global_mean_pool` output. Wire into the existing parquet-based feature pipeline.

**Tech Stack:** PyTorch + torch_geometric (`ResGatedGraphConv`, `from_smiles`, `Batch`, `global_mean_pool`), psycopg2, pandas (parquet), TabPFN v2.6, main PXR pixi env.

**Related spec:** `docs/superpowers/specs/2026-04-21-gatedgcn-pretrain-embed-design.md`

---

## File map

| Path | Create / Modify | Responsibility |
|---|---|---|
| `track1_activity/scripts/run_gatedgcn_embed_extract.py` | Create | Load pretrain.pt, replace ffn with Identity, extract 128d readout |
| `data/gatedgcn_pretrain_embed.parquet` | Create (gitignored via `data/*.parquet`) | Output parquet |
| `track1_activity/scripts/run_train.py` | Modify (~line 497) | Add `gatedgcn_pretrain_embed` feature branch + all_features entry |
| `track1_activity/scripts/run_ensemble.py` | Modify (~line 174) | Append to ENSEMBLE_MODELS after attentivefp entry |

---

## Conventions recap

- **Branch**: `feature/gatedgcn-pretrain-embed` (already checked out).
- No CI; ruff is the gate.
- Commit per task. `on_conflict_replace=True` for DB writes (inherited).

---

## Task 1: Write + run GatedGCN embed extraction

**Files:**
- Create: `/home/nagaet/pxr-iduction-challenge/track1_activity/scripts/run_gatedgcn_embed_extract.py`
- Output (gitignored): `/home/nagaet/pxr-iduction-challenge/data/gatedgcn_pretrain_embed.parquet`

- [ ] **Step 1.1: Verify checkpoint + metadata**

```bash
cd /home/nagaet/pxr-iduction-challenge
ls -la track1_activity/checkpoints/gatedgcn_pretrain/
cat track1_activity/checkpoints/gatedgcn_pretrain/pretrain_meta.json
```

Expected: `pretrain.pt` (~1.9 MB) + meta with `hidden_dim=128, num_layers=4, dropout=0.05`.

- [ ] **Step 1.2: Gitignore coverage**

```bash
git check-ignore -v data/gatedgcn_pretrain_embed.parquet
```

Expected: matches `data/*.parquet`. No .gitignore change required.

- [ ] **Step 1.3: Write extraction script**

Create `/home/nagaet/pxr-iduction-challenge/track1_activity/scripts/run_gatedgcn_embed_extract.py`:

```python
"""Extract 128d per-compound graph-pooled embeddings from the GatedGCN
pretrain checkpoint.

Phase 2 of Buterez 2024 strategy-3 for GatedGCN. Loads the pretrain
checkpoint produced by run_gatedgcn_pretrain_finetune.py --phase
pretrain, replaces the FFN head with nn.Identity() so the forward
pass returns the 128d global_mean_pool output (post-conv stack,
pre-FFN), then batch-extracts embeddings for all 13,136 compounds.

Output: data/gatedgcn_pretrain_embed.parquet (index=compound_id,
columns=emb_0000..emb_0127, float32).

Usage:
    pixi run python track1_activity/scripts/run_gatedgcn_embed_extract.py
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
from torch_geometric.utils import from_smiles

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from data import DB_PARAMS  # noqa: E402
from run_gatedgcn_pretrain_finetune import GatedGCNModel  # noqa: E402

CKPT_DIR = REPO_ROOT.joinpath(
    "track1_activity", "checkpoints", "gatedgcn_pretrain"
)
PRETRAIN_PATH = CKPT_DIR.joinpath("pretrain.pt")
META_PATH = CKPT_DIR.joinpath("pretrain_meta.json")
OUT_PATH = REPO_ROOT.joinpath("data", "gatedgcn_pretrain_embed.parquet")
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


def load_model() -> GatedGCNModel:
    with META_PATH.open() as f:
        meta = json.load(f)
    params = meta["params"]

    probe = from_smiles("CCO")
    in_dim = probe.x.shape[1]
    edge_dim = probe.edge_attr.shape[1]

    model = GatedGCNModel(
        in_dim=in_dim,
        edge_dim=edge_dim,
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        out_dim=2,  # pretrain had 2-head log2_fc
    )
    ckpt = torch.load(PRETRAIN_PATH, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        print(
            f"load_state_dict: missing={result.missing_keys} "
            f"unexpected={result.unexpected_keys}"
        )

    # Replace the FFN head with Identity so forward returns the 128d
    # global_mean_pool output (post-conv stack, pre-FFN projection).
    model.ffn = nn.Identity()

    return model.to(DEVICE).eval()


@torch.no_grad()
def extract_embeddings(model: GatedGCNModel, graphs: list) -> np.ndarray:
    n = len(graphs)
    outs: list[np.ndarray] = []
    for i in range(0, n, BATCH_SIZE):
        batch = Batch.from_data_list(graphs[i : i + BATCH_SIZE]).to(DEVICE)
        emb = model(
            batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch
        )
        outs.append(emb.cpu().numpy())
        if (i // BATCH_SIZE) % 10 == 0:
            print(
                f"  batch {i // BATCH_SIZE + 1} / {(n + BATCH_SIZE - 1) // BATCH_SIZE}"
            )
    return np.concatenate(outs, axis=0).astype(np.float32)


def main() -> None:
    cids, smiles_list = load_compounds()
    print(f"Loaded {len(cids)} compounds")

    print("Converting SMILES to PyG graphs...")
    graphs = smiles_to_pyg(smiles_list)

    print(f"Loading GatedGCN pretrain checkpoint from {PRETRAIN_PATH}")
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

- [ ] **Step 1.4: Run extraction**

```bash
cd /home/nagaet/pxr-iduction-challenge
pixi run python track1_activity/scripts/run_gatedgcn_embed_extract.py 2>&1 | tee /tmp/gatedgcn_embed.log | tail -20
```

Expected: 13,136 compounds processed, shape (13136, 128), 0 NaN, wall-clock < 5 min.

- [ ] **Step 1.5: Sanity check parquet**

```bash
pixi run python -c "
import pandas as pd
df = pd.read_parquet('data/gatedgcn_pretrain_embed.parquet')
print('shape:', df.shape)
print('index:', df.index.name, df.index.min(), df.index.max())
print('cols:', df.columns[0], '..', df.columns[-1])
print('NaN:', int(df.isna().sum().sum()))
print('first row first 5:', df.iloc[0, :5].tolist())
"
```

Expected: (13136, 128), index `compound_id` 1..13136, cols `emb_0000..emb_0127`, NaN=0.

- [ ] **Step 1.6: Format + lint**

```bash
pixi run ruff format track1_activity/scripts/run_gatedgcn_embed_extract.py
pixi run ruff check track1_activity/scripts/run_gatedgcn_embed_extract.py
```

- [ ] **Step 1.7: Commit**

```bash
cd /home/nagaet/pxr-iduction-challenge
git add track1_activity/scripts/run_gatedgcn_embed_extract.py
git commit -m "feat(gatedgcn): 128d graph-pooled embedding extraction"
```

---

## Task 2: Register `gatedgcn_pretrain_embed` feature in run_train.py

**Files:**
- Modify: `/home/nagaet/pxr-iduction-challenge/track1_activity/scripts/run_train.py`

- [ ] **Step 2.1: Locate insertion points**

```bash
grep -n "attentivefp_pretrain_embed" track1_activity/scripts/run_train.py
```

Expected: 2 occurrences — (a) feature branch in `load_features()`, (b) entry in CLI `all_features` choices list.

- [ ] **Step 2.2: Add feature branch**

Insert after the `attentivefp_pretrain_embed` branch's `return X_train, X_test`:

```python
    if feature_name == "gatedgcn_pretrain_embed":
        # 128d graph-pooled embedding from PyG ResGatedGraphConv stack
        # pretrained on single_concentration log2_fc (2-head, z-scored
        # targets). Extracted by replacing GatedGCNModel.ffn with
        # nn.Identity() so forward returns the global_mean_pool output.
        # See: track1_activity/scripts/run_gatedgcn_embed_extract.py
        # Buterez 2024 strategy-3 with gated edge-conditioned message
        # passing backbone (fifth pretrain-embed family member).
        embed_path = REPO_ROOT.joinpath("data", "gatedgcn_pretrain_embed.parquet")
        if not embed_path.exists():
            raise SystemExit(
                f"Missing {embed_path}. Run "
                f"track1_activity/scripts/run_gatedgcn_embed_extract.py"
            )
        emb_df = pd.read_parquet(embed_path)
        X_train = emb_df.reindex(index=train_ids).to_numpy(dtype=np.float32).copy()
        X_test = emb_df.reindex(index=test_ids).to_numpy(dtype=np.float32).copy()
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        print(
            f"  gatedgcn_pretrain_embed: {X_train.shape[1]} dims "
            f"(train {X_train.shape[0]} / test {X_test.shape[0]})"
        )
        return X_train, X_test
```

- [ ] **Step 2.3: Add to all_features list**

Find the `attentivefp_pretrain_embed` entry in the `all_features` choices list (inside `main()`) and append `"gatedgcn_pretrain_embed"` after it.

- [ ] **Step 2.4: Smoke test**

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
Xtr, Xte = load_features('gatedgcn_pretrain_embed', tr, te)
print('train', Xtr.shape, 'test', Xte.shape, 'NaN', int(np.isnan(Xtr).sum() + np.isnan(Xte).sum()))
print('first row first 5:', Xtr[0, :5].tolist())
"
```

Expected: `train (4140, 128) test (513, 128) NaN 0`.

- [ ] **Step 2.5: Format + lint**

```bash
pixi run ruff format track1_activity/scripts/run_train.py
pixi run ruff check track1_activity/scripts/run_train.py
```

8 pre-existing E402 warnings expected (unchanged).

- [ ] **Step 2.6: Commit**

```bash
git add track1_activity/scripts/run_train.py
git commit -m "feat(gatedgcn): register gatedgcn_pretrain_embed feature"
```

---

## Task 3: Train TabPFN + acceptance check

**Files:**
- No new files.

- [ ] **Step 3.1: Run 5-fold TabPFN**

```bash
cd /home/nagaet/pxr-iduction-challenge
nohup pixi run python track1_activity/scripts/run_train.py \
    --model tabpfn \
    --feature gatedgcn_pretrain_embed \
    --split umap \
    --trials 0 \
    > /tmp/tabpfn_gatedgcn.log 2>&1 &
echo $! > /tmp/tabpfn_gatedgcn_pid.txt
disown
echo "Started PID $(cat /tmp/tabpfn_gatedgcn_pid.txt)"
```

Expected wall-clock: 15-25 min (128d is small).

- [ ] **Step 3.2: Monitor**

Use Monitor:
```
tail -F /tmp/tabpfn_gatedgcn.log 2>&1 | grep --line-buffered -E "(Fold [0-9]+|MAE|RAE|Spearman|Kendall|Overall|complete|finished|saved|Recorded|Done|Traceback|Error|FAILED|OOM|Killed)"
```

- [ ] **Step 3.3: Verify OOF MAE ≤ 0.48 (or within fold-std)**

```bash
pixi run python -c "
import psycopg2, sys
sys.path.insert(0, 'track1_activity/src')
from data import DB_PARAMS
with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
    cur.execute(\"SELECT name, mae_mean, rae_mean, spearman_mean, kendall_mean FROM experiment_summary WHERE name LIKE 'tabpfn_gatedgcn_pretrain_embed%' ORDER BY created_at DESC LIMIT 3\")
    for r in cur.fetchall():
        print(r)
" 2>&1 | grep -v UserWarning | grep -v read_sql
```

- Gate (2): MAE ≤ 0.48 strict-PASS. If MAE > 0.48 by ≤ fold-std: DONE_WITH_CONCERNS (AttentiveFP precedent). If > 0.50: BLOCKED.

- [ ] **Step 3.4: Pearson r vs 5 existing pretrain-embed members**

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
    'tabpfn_gatedgcn_pretrain_embed_umap_default',
]
short = {
    'tabpfn_chemprop_pretrain_embed_umap_default': 'chemprop',
    'tabpfn_molformer_c3_pretrain_embed_umap': 'molformer_c3',
    'tabpfn_2d_full_boltz_log2fc_pred_umap_default': '2d_full_boltz',
    'tabpfn_kermt_pretrain_embed_umap_default': 'kermt',
    'tabpfn_attentivefp_pretrain_embed_umap_default': 'attentivefp',
    'tabpfn_gatedgcn_pretrain_embed_umap_default': 'gatedgcn',
}
with psycopg2.connect(**DB_PARAMS) as conn:
    cur = conn.cursor()
    oofs = {}
    for n in names:
        cur.execute('SELECT eop.train_idx, eop.oof_prediction FROM experiment_oof_predictions eop JOIN experiments e ON e.id = eop.experiment_id WHERE e.name = %s ORDER BY eop.train_idx', (n,))
        rows = cur.fetchall()
        if rows:
            oofs[short[n]] = pd.Series({int(r[0]): float(r[1]) for r in rows})
df = pd.DataFrame(oofs).dropna()
print('shape:', df.shape)
print(df.corr(method='pearson').round(3))
print()
print('gatedgcn pairwise (< 0.96 gate):')
for other in ['chemprop', 'molformer_c3', '2d_full_boltz', 'kermt', 'attentivefp']:
    r = df['gatedgcn'].corr(df[other])
    ok = 'OK' if r < 0.96 else 'FAIL'
    print(f'  gatedgcn vs {other}: r = {r:.4f}  {ok}')
" 2>&1 | grep -v UserWarning | grep -v read_sql
```

- [ ] **Step 3.5: No commit. Report.**

---

## Task 4: ENSEMBLE_MODELS + 13-pool caruana + calibrate

**Files:**
- Modify: `/home/nagaet/pxr-iduction-challenge/track1_activity/scripts/run_ensemble.py`

- [ ] **Step 4.1: Locate insertion point**

```bash
grep -n "tabpfn_attentivefp_pretrain_embed_umap_default" track1_activity/scripts/run_ensemble.py
```

Expected: one occurrence in ENSEMBLE_MODELS tuple.

- [ ] **Step 4.2: Append new entry**

After the `"tabpfn_attentivefp_pretrain_embed_umap_default",` line, insert:

```python
    # --- GatedGCN pretrain embed + TabPFN ---
    # Gated edge-conditioned message-passing family (PyG ResGatedGraphConv
    # pretrained on single_concentration log2_fc via PR #79 checkpoint,
    # then 128d global_mean_pool output frozen into TabPFN). 13th pool
    # member — completes the "all-available-pretrain-checkpoints" sweep
    # alongside chemprop (D-MPNN), molformer_c3 (transformer), kermt
    # (graph-transformer), attentivefp (graph-attention). See
    # docs/superpowers/specs/2026-04-21-gatedgcn-pretrain-embed-design.md.
    # Single-model OOF MAE <fill>, Pearson r vs existing: chemprop <fill>,
    # molformer_c3 <fill>, 2d_full_boltz <fill>, kermt <fill>,
    # attentivefp <fill>. PR <TBD>.
    "tabpfn_gatedgcn_pretrain_embed_umap_default",
```

- [ ] **Step 4.3: Re-run caruana**

```bash
cd /home/nagaet/pxr-iduction-challenge
pixi run python track1_activity/scripts/run_ensemble.py 2>&1 | tee /tmp/ensemble_gatedgcn.log | tail -80
```

- [ ] **Step 4.4: Verify 13-pool metrics**

```bash
pixi run python -c "
import psycopg2, sys
sys.path.insert(0, 'track1_activity/src')
from data import DB_PARAMS
with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
    cur.execute('''
        SELECT e.hyperparameters, s.mae_mean, s.rae_mean, s.spearman_mean
        FROM experiments e JOIN experiment_summary s ON s.name = e.name
        WHERE e.name = 'ens_caruana_bag20'
        ORDER BY e.created_at DESC LIMIT 1
    ''')
    hp, mae, rae, spr = cur.fetchone()
print(f'13-pool: MAE={mae:.4f} RAE={rae:.4f} Spearman={spr:.4f}')
weights = hp.get('weights', {})
print()
for n, w in sorted(weights.items(), key=lambda kv: -kv[1]):
    marker = ' <-- NEW' if 'gatedgcn_pretrain_embed' in n else ''
    print(f'  {n:<55} {w:.4f}{marker}')
print()
delta = float(mae) - 0.4242
print(f'vs 12-pool baseline 0.4242: delta = {delta:+.4f}')
" 2>&1 | grep -v UserWarning | grep -v read_sql
```

Gate (4): MAE ≤ 0.4242 → PASS.

If FAIL: STOP, BLOCKED, do not calibrate or commit.

- [ ] **Step 4.5: Re-run calibration**

```bash
pixi run python track1_activity/scripts/run_ensemble_calibrate.py 2>&1 | tee /tmp/calibrate_gatedgcn.log | tail -40
wc -l track1_activity/submissions/ens_caruana_bag20_calibrated_best.csv
```

Expected 514 lines.

- [ ] **Step 4.6: Format + lint**

```bash
pixi run ruff format track1_activity/scripts/run_ensemble.py
pixi run ruff check track1_activity/scripts/run_ensemble.py
```

- [ ] **Step 4.7: Commit** (replace `<fill>` placeholders first)

```bash
git add track1_activity/scripts/run_ensemble.py
git commit -m "$(cat <<'COMMIT_EOF'
feat(ens): add tabpfn_gatedgcn_pretrain_embed to 13-pool caruana_bag20

Fifth and final pretrain-embed member (GatedGCN — gated edge-
conditioned message passing). Completes the all-available-pretrain
sweep (chemprop, molformer_c3, kermt, attentivefp, gatedgcn).

Single-model OOF MAE <fill>, Pearson r vs existing all < 0.96.
13-pool caruana_bag20 MAE <fill> (vs 12-pool 0.4242, delta <fill>),
gatedgcn weight <fill>.
COMMIT_EOF
)"
```

---

## Task 5: Push + PR

**Files:** none.

- [ ] **Step 5.1: Push**

```bash
cd /home/nagaet/pxr-iduction-challenge
git log --oneline origin/main..HEAD
git push -u origin feature/gatedgcn-pretrain-embed
```

- [ ] **Step 5.2: Final ruff**

```bash
pixi run ruff format --check \
    track1_activity/scripts/run_gatedgcn_embed_extract.py \
    track1_activity/scripts/run_ensemble.py
pixi run ruff check \
    track1_activity/scripts/run_gatedgcn_embed_extract.py \
    track1_activity/scripts/run_ensemble.py
```

- [ ] **Step 5.3: Open PR with filled metrics**

```bash
gh pr create --title "feat: GatedGCN pretrain-embed as 13th caruana pool member" --body "$(cat <<'BODY_EOF'
## Summary
- Adds `tabpfn_gatedgcn_pretrain_embed_umap_default` as the 13th pool member. **Fifth and final** pretrain-embed member, completing the sweep alongside chemprop (D-MPNN), molformer_c3 (transformer), kermt (graph-transformer), attentivefp (graph-attention).
- Reuses PR #79 pretrain checkpoint (`track1_activity/checkpoints/gatedgcn_pretrain/pretrain.pt`, 1.9 MB). No retraining.
- 128d global_mean_pool output extracted by replacing `model.ffn` with `nn.Identity()`. Smallest embed dim in the pool (chemprop 256, attentivefp 512, molformer_c3 768, kermt 3200).

## Acceptance results
1. **Extraction**: 13,136 / 13,136 compounds, 0 NaN ✓
2. **Single-model OOF MAE**: <fill>
   - RAE <fill>, Spearman <fill>, Kendall <fill>
3. **caruana weight**: <fill> (target > 0)
4. **13-pool MAE**: <fill> (vs 12-pool 0.4242, delta <fill>)
5. **Pearson r** (all < 0.96 ✓):
   - vs chemprop: <fill>
   - vs molformer_c3: <fill>
   - vs 2d_full_boltz: <fill>
   - vs kermt: <fill>
   - vs attentivefp: <fill>
6. **ruff clean**

## Calibration
- Winner: <fill>, calibrated MAE <fill>.
- Submission CSV: 513 rows.

## Files
### New
- `track1_activity/scripts/run_gatedgcn_embed_extract.py`

### Modified
- `track1_activity/scripts/run_train.py`
- `track1_activity/scripts/run_ensemble.py`

### Reused
- `track1_activity/checkpoints/gatedgcn_pretrain/pretrain.pt` (PR #79, 2026-04-19)

## Test plan
- [x] Parquet (13136, 128), 0 NaN
- [x] Feature loader returns (4140, 128) / (513, 128)
- [x] 13-pool MAE improves (or ties) 12-pool 0.4242
- [x] All Pearson r < 0.96

## CI: N/A (no workflow)

## Related
- Spec: `docs/superpowers/specs/2026-04-21-gatedgcn-pretrain-embed-design.md`
- Plan: `docs/superpowers/plans/2026-04-21-gatedgcn-pretrain-embed.md`
- PR #79 (GatedGCN pretrain + frozen+head-FT)
- PR #104 (AttentiveFP pretrain-embed — direct predecessor)
BODY_EOF
)"
```

- [ ] **Step 5.4: LB cooldown + ask user**

```bash
pixi run python track1_activity/scripts/api.py cooldown
```

Report PR URL + cooldown status. Do NOT merge or submit.

---

## Self-review notes

Spec coverage:
- Extraction (spec §Architecture) → Task 1
- Feature plumbing (spec §DB/feature plumbing) → Task 2
- TabPFN (spec §Downstream TabPFN) → Task 3
- Ensemble (spec §Ensemble integration) → Task 4
- Acceptance 1-6 → Tasks 1-4
- ruff gate → Tasks 1.6, 2.5, 4.6, 5.2

Placeholder scan: no TBD/TODO; all code blocks complete.

Type consistency: `GatedGCNModel` imported from existing pretrain-finetune script; signature `(in_dim, edge_dim, hidden_dim, num_layers, dropout, out_dim)` matches the constructor at `run_gatedgcn_pretrain_finetune.py:84`.

Fluid points (flagged inline):
- Task 2.3: `all_features` choices list location — use grep if line moves.
- Task 4.2 / 5.3: `<fill>` placeholders must be replaced with actual Task 3-4 metric values before commit / PR.
