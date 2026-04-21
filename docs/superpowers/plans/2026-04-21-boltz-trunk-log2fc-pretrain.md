# Boltz-2 Trunk × log2_fc Pretrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `tabpfn_boltz_trunk_pretrain_embed_umap_default` as a strategy-3 pool member by extending Boltz-2 trunk to all 13,136 compounds (fast mode rcycle=1 for new 8,483) and pretraining an MLP head on log2_fc.

**Architecture:** Phase 1 dominates wall-clock (12-24h unattended fast trunk extraction). Phases 2-5 (DB upsert, MLP pretrain, extract, TabPFN+ensemble) run in a single half-day. Reuses existing `boltz_affhead/01b_pool_allpairs.py` pooling, existing `run_train.py` TabPFN harness, existing `run_ensemble.py` caruana integration.

**Tech Stack:** Boltz-2 fork at `~/ghq/github.com/N283T/boltz` (uv tool install), pytorch (main pixi env for MLP), psycopg2, pandas (parquet), TabPFN v2.6.

**Related spec:** `docs/superpowers/specs/2026-04-21-boltz-trunk-log2fc-pretrain-design.md`

---

## File map

| Path | Create / Modify | Responsibility |
|---|---|---|
| `track2_structure/scripts/boltz2_build_inputs_fast.py` | Create | Query 13k compounds - 4653 covered = 8483 missing, emit YAMLs for Boltz trunk-only run |
| `track2_structure/scripts/boltz2_fast_embeddings_run.sh` | Create | Mirror of `boltz2_embeddings_run.sh` but with `--embeddings_only --recycling_steps 1` for the 8483-subset inputs dir |
| `track1_activity/scripts/boltz_affhead/08_pool_fast_embeddings.py` | Create | Apply allpairs pooling to 8483 new embeddings, write parquet |
| `db/compound_boltz2_trunk_fast_schema.sql` | Create | New table schema |
| `track1_activity/scripts/boltz_affhead/09_upsert_trunk_fast.py` | Create | Upsert existing 4653 (rcycle=3) + new 8483 (rcycle=1) into `compound_boltz2_trunk_fast` |
| `track1_activity/scripts/boltz_affhead/10_train_mlp_head.py` | Create | MLP pretrain on log2_fc, 3 variants (A/B/C), select best |
| `track1_activity/scripts/boltz_affhead/11_extract_embedding.py` | Create | Forward 13,136 trunks through chosen MLP, extract penultimate hidden, save parquet |
| `track1_activity/scripts/run_train.py` | Modify | Register `boltz_trunk_pretrain_embed` feature branch + all_features entry |
| `track1_activity/scripts/run_ensemble.py` | Modify | Append `tabpfn_boltz_trunk_pretrain_embed_umap_default` OR swap `tabpfn_pooled_boltz_umap_default` |
| `.gitignore` | Modify | Ensure `data/boltz_trunk_pretrain_embed.parquet` is ignored |

---

## Conventions recap

- Branch: `feature/boltz-trunk-log2fc-pretrain` (already checked out, spec commits c203fd4 / 21fc27b present).
- No CI; ruff format + check are the gates.
- `on_conflict_replace=True` for DB writes (inherited).
- Commit frequently — one commit per task.

---

## Task 1: Build YAML inputs for 8,483 uncovered compounds

**Files:**
- Create: `/home/nagaet/pxr-iduction-challenge/track2_structure/scripts/boltz2_build_inputs_fast.py`

- [ ] **Step 1.1: Identify the 8,483 target compounds**

```bash
cd /home/nagaet/pxr-iduction-challenge
pixi run python -c "
import psycopg2, sys
sys.path.insert(0, 'track1_activity/src')
from data import DB_PARAMS
with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
    cur.execute(\"\"\"
        SELECT c.id FROM compounds c
        WHERE c.std_smiles IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM compound_boltz2 b WHERE b.compound_id = c.id)
        ORDER BY c.id LIMIT 5
    \"\"\")
    print('first 5:', [r[0] for r in cur.fetchall()])
    cur.execute('SELECT COUNT(*) FROM compounds WHERE std_smiles IS NOT NULL')
    print('total compounds with std_smiles:', cur.fetchone()[0])
    cur.execute('SELECT COUNT(*) FROM compound_boltz2')
    print('existing boltz2 rows:', cur.fetchone()[0])
" 2>&1 | grep -v "UserWarning\|read_sql"
```

Expected: total ~13,136, existing ~4,653, delta ~8,483.

- [ ] **Step 1.2: Write the YAML builder**

Create `/home/nagaet/pxr-iduction-challenge/track2_structure/scripts/boltz2_build_inputs_fast.py`:

```python
"""Generate Boltz-2 input YAMLs for the 8,483 compounds not yet covered
by the main compound_boltz2 table.

Used for the "fast trunk" extension (strategy-3 Boltz pretrain): we only
want trunk s/z embeddings, no diffusion/confidence/affinity. See
track2_structure/scripts/boltz2_fast_embeddings_run.sh for the
embeddings-only run script that consumes the emitted YAMLs.

Usage:
    pixi run python track2_structure/scripts/boltz2_build_inputs_fast.py
    pixi run python track2_structure/scripts/boltz2_build_inputs_fast.py --smoke  # 10 compounds only
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import psycopg2

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.joinpath("src")))

from boltz2.input_builder import build_yaml, write_yaml  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402

INPUTS_DIR = REPO_ROOT.joinpath("structures", "boltz2", "inputs_fast")
MANIFEST_PATH = REPO_ROOT.joinpath("structures", "boltz2", "manifest_fast.csv")


MISSING_QUERY = """
    SELECT c.id, c.std_smiles
    FROM compounds c
    WHERE c.std_smiles IS NOT NULL
      AND NOT EXISTS (
          SELECT 1 FROM compound_boltz2 b WHERE b.compound_id = c.id
      )
    ORDER BY c.id
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true", help="Only emit YAMLs for 10 compounds"
    )
    args = parser.parse_args()

    with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
        cur.execute(MISSING_QUERY)
        rows = cur.fetchall()
    if args.smoke:
        rows = rows[:10]
    print(f"Found {len(rows)} compounds without compound_boltz2 row")

    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for compound_id, smiles in rows:
        yaml_name = f"{compound_id:05d}.yaml"
        yaml_path = INPUTS_DIR.joinpath(yaml_name)
        y = build_yaml(compound_id=compound_id, smiles=smiles)
        write_yaml(y, yaml_path)
        manifest_rows.append({
            "compound_id": compound_id,
            "smiles": smiles,
            "yaml_path": str(yaml_path.relative_to(REPO_ROOT)),
        })

    with MANIFEST_PATH.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["compound_id", "smiles", "yaml_path"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Wrote manifest: {MANIFEST_PATH} ({len(manifest_rows)} rows)")
    print(f"YAMLs at: {INPUTS_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 1.3: Smoke test**

```bash
cd /home/nagaet/pxr-iduction-challenge
pixi run python track2_structure/scripts/boltz2_build_inputs_fast.py --smoke 2>&1 | tail -10
ls structures/boltz2/inputs_fast/ | head -5
wc -l structures/boltz2/manifest_fast.csv
```

Expected: 10 YAMLs generated + 11-line manifest (header + 10 rows).

- [ ] **Step 1.4: Full YAML generation**

```bash
# Clean smoke-mode files first (or let them be overwritten by full run)
rm -rf structures/boltz2/inputs_fast
pixi run python track2_structure/scripts/boltz2_build_inputs_fast.py 2>&1 | tail -5
ls structures/boltz2/inputs_fast/ | wc -l
```

Expected: ~8,483 YAML files.

- [ ] **Step 1.5: Format + lint**

```bash
pixi run ruff format track2_structure/scripts/boltz2_build_inputs_fast.py
pixi run ruff check track2_structure/scripts/boltz2_build_inputs_fast.py
```

- [ ] **Step 1.6: Add to gitignore (structures/boltz2 is likely already ignored)**

```bash
git check-ignore -v structures/boltz2/inputs_fast/00001.yaml
```

If NOT ignored, edit `.gitignore` to add `structures/boltz2/inputs_fast/` and `structures/boltz2/manifest_fast.csv`.

- [ ] **Step 1.7: Commit**

```bash
git add track2_structure/scripts/boltz2_build_inputs_fast.py
# Also .gitignore if modified
git status
git commit -m "feat(boltz): YAML builder for 8483 compounds uncovered by main run"
```

---

## Task 2: Run fast trunk extraction (background, 12-24h)

**Files:**
- Create: `/home/nagaet/pxr-iduction-challenge/track2_structure/scripts/boltz2_fast_embeddings_run.sh`

- [ ] **Step 2.1: Write the shell wrapper**

Create `/home/nagaet/pxr-iduction-challenge/track2_structure/scripts/boltz2_fast_embeddings_run.sh`:

```bash
#!/usr/bin/env bash
# Fast trunk embedding extraction for the 8,483 uncovered compounds.
# Uses user's Boltz fork's --embeddings_only flag + --recycling_steps 1
# to minimize compute (vs default rcycle=3 + full diffusion + affinity).
#
# Outputs land in structures/boltz2/outputs_fast/predictions/<id>/
# as embeddings_<id>.npz (s + z arrays).
#
# Requires the patched boltz fork installed via:
#   uv tool install --python 3.12 --reinstall --force --editable \
#       "$HOME/ghq/github.com/N283T/boltz[cuda]"
#
# Resume-safe: --embeddings_only causes the filter to treat per-compound
# folders as "done" only when embeddings_<id>.npz exists.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

NVIDIA_CU13_LIB_DIR="$HOME/.local/share/uv/tools/boltz/lib/python3.12/site-packages/nvidia/cu13/lib"
if [[ -d "$NVIDIA_CU13_LIB_DIR" ]]; then
    export LD_LIBRARY_PATH="${NVIDIA_CU13_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

INPUTS_DIR="$REPO_ROOT/structures/boltz2/inputs_fast"
OUTPUT_DIR="$REPO_ROOT/structures/boltz2/outputs_fast"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
LOG="$LOG_DIR/boltz2_fast_embeddings_$(date +%Y%m%d_%H%M%S).log"

if [[ ! -d "$INPUTS_DIR" ]]; then
    echo "ERROR: inputs dir not found: $INPUTS_DIR" >&2
    echo "Run track2_structure/scripts/boltz2_build_inputs_fast.py first." >&2
    exit 1
fi

echo "=== Fast trunk run ===" | tee "$LOG"
date | tee -a "$LOG"
echo "inputs:  $INPUTS_DIR" | tee -a "$LOG"
echo "outputs: $OUTPUT_DIR" | tee -a "$LOG"

boltz predict "$INPUTS_DIR" \
    --embeddings_only \
    --recycling_steps 1 \
    --out_dir "$OUTPUT_DIR" \
    --accelerator gpu \
    2>&1 | tee -a "$LOG"

echo "=== Done at $(date) ===" | tee -a "$LOG"
```

`chmod +x` the script.

- [ ] **Step 2.2: Smoke test (10 compounds)**

Prerequisite: Task 1 smoke emitted 10 YAMLs into `inputs_fast`.

```bash
bash track2_structure/scripts/boltz2_fast_embeddings_run.sh 2>&1 | tail -30
ls structures/boltz2/outputs_fast/boltz_results_inputs_fast/predictions/*/embeddings_*.npz 2>/dev/null | head
```

Expected: 10 embeddings_*.npz files produced in ~1-3 min.

- [ ] **Step 2.3: Inspect one npz shape**

```bash
pixi run python -c "
import numpy as np, glob
path = glob.glob('structures/boltz2/outputs_fast/boltz_results_*/predictions/*/embeddings_*.npz')[0]
print('path:', path)
d = np.load(path)
print('keys:', list(d.keys()))
for k in d.keys():
    print(f'  {k}: shape {d[k].shape} dtype {d[k].dtype}')
"
```

Expected: `s` (N_tokens, 384), `z` (N_tokens, N_tokens, 128), or similar Boltz-2 schema.

- [ ] **Step 2.4: Commit wrapper**

```bash
git add track2_structure/scripts/boltz2_fast_embeddings_run.sh
git commit -m "feat(boltz): fast trunk embedding run wrapper (rcycle=1)"
```

- [ ] **Step 2.5: Full launch in background**

After Task 1's full run generated all 8,483 YAMLs:

```bash
cd /home/nagaet/pxr-iduction-challenge
nohup bash track2_structure/scripts/boltz2_fast_embeddings_run.sh > /tmp/boltz_fast_status.log 2>&1 &
echo $! > /tmp/boltz_fast_pid.txt
disown
ps -p $(cat /tmp/boltz_fast_pid.txt) -o pid,etime,stat
```

Monitor progress periodically:

```bash
ls structures/boltz2/outputs_fast/boltz_results_*/predictions/*/embeddings_*.npz 2>/dev/null | wc -l
```

Expected final count: ~8,483 (minus any per-compound failures). Expected wall-clock: 12-24h.

- [ ] **Step 2.6: No commit on completion; failed compounds recorded**

After background run completes, record any failures:

```bash
pixi run python -c "
import glob
done = set()
for p in glob.glob('structures/boltz2/outputs_fast/boltz_results_*/predictions/*/embeddings_*.npz'):
    cid = int(p.split('/')[-1].replace('embeddings_', '').replace('.npz', ''))
    done.add(cid)
import psycopg2, sys
sys.path.insert(0, 'track1_activity/src')
from data import DB_PARAMS
with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
    cur.execute(\"\"\"
        SELECT c.id FROM compounds c
        WHERE c.std_smiles IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM compound_boltz2 b WHERE b.compound_id = c.id)
    \"\"\")
    expected = {r[0] for r in cur.fetchall()}
print('expected:', len(expected), 'done:', len(done), 'missing:', len(expected - done))
" 2>&1 | grep -v "UserWarning\|read_sql"
```

Expected: missing < 100. Document any in commit message for Task 3.

---

## Task 3: Pool fast embeddings (allpairs)

**Files:**
- Create: `/home/nagaet/pxr-iduction-challenge/track1_activity/scripts/boltz_affhead/08_pool_fast_embeddings.py`

- [ ] **Step 3.1: Inspect existing allpairs pooler as template**

```bash
head -80 /home/nagaet/pxr-iduction-challenge/track1_activity/scripts/boltz_affhead/01b_pool_allpairs.py
```

Note the pool function signatures (e.g. `pool_s(s, ...)`, `pool_z(z, ...)`, output 1024d).

- [ ] **Step 3.2: Write the fast pooler**

Create `/home/nagaet/pxr-iduction-challenge/track1_activity/scripts/boltz_affhead/08_pool_fast_embeddings.py`:

```python
"""Pool fast trunk embeddings into the same 1024d allpairs schema as
boltz_affhead/01b_pool_allpairs.py, but reads from
structures/boltz2/outputs_fast (rcycle=1) instead of the full run.

Output: data/boltz_affhead/pooled_fast.parquet (index=compound_id,
columns s_prot_mean_000..s_lig_mean_..z_if_mean_..z_if_max_383)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

EMBED_GLOB = "structures/boltz2/outputs_fast/boltz_results_*/predictions/*/embeddings_*.npz"
OUT_PATH = REPO_ROOT.joinpath("data", "boltz_affhead", "pooled_fast.parquet")


def pool_one(npz_path: Path) -> np.ndarray:
    """Return 1024d vector = [s_prot_mean 384 | s_lig_mean 384 | z_if_mean 128 | z_if_max 128]."""
    d = np.load(npz_path)
    # Schema (boltz-2 convention): s is (tokens, 384), z is (tokens, tokens, 128).
    # Token IDs 0..N_prot-1 are protein residues (PXR LBD ≈ 434 residues),
    # N_prot..N-1 are ligand atoms.
    s = d["s"]  # (N, 384)
    z = d["z"]  # (N, N, 128)
    N = s.shape[0]
    # PXR LBD token count = 434 (from compound_boltz2 extraction convention).
    # Verify at runtime: the first 434 should be protein, rest ligand.
    n_prot = 434
    if N <= n_prot:
        raise ValueError(f"Unexpected token count {N} (expected > {n_prot}) for {npz_path}")
    s_prot = s[:n_prot]
    s_lig = s[n_prot:]
    z_if = z[:n_prot, n_prot:]  # (n_prot, n_lig, 128) -- all-pairs protein x ligand

    s_prot_mean = s_prot.mean(axis=0)                       # (384,)
    s_lig_mean = s_lig.mean(axis=0)                         # (384,)
    z_if_mean = z_if.reshape(-1, z_if.shape[-1]).mean(axis=0)  # (128,)
    z_if_max = z_if.reshape(-1, z_if.shape[-1]).max(axis=0)    # (128,)
    return np.concatenate([s_prot_mean, s_lig_mean, z_if_mean, z_if_max]).astype(np.float32)


def main() -> None:
    paths = sorted(REPO_ROOT.glob(EMBED_GLOB))
    print(f"Found {len(paths)} fast embeddings")

    rows = {}
    for i, p in enumerate(paths):
        cid = int(p.name.replace("embeddings_", "").replace(".npz", ""))
        try:
            rows[cid] = pool_one(p)
        except Exception as e:
            print(f"  skip {cid}: {e}")
            continue
        if i % 500 == 0 and i:
            print(f"  pooled {i} / {len(paths)}")

    print(f"Pooled {len(rows)} compounds")
    cols = (
        [f"s_prot_mean_{i:03d}" for i in range(384)]
        + [f"s_lig_mean_{i:03d}" for i in range(384)]
        + [f"z_if_mean_{i:03d}" for i in range(128)]
        + [f"z_if_max_{i:03d}" for i in range(128)]
    )
    df = pd.DataFrame(
        np.stack([rows[cid] for cid in sorted(rows)], axis=0),
        index=pd.Index(sorted(rows), name="compound_id"),
        columns=cols,
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH)
    print(f"Wrote {OUT_PATH}  shape {df.shape}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.3: Smoke on 10-compound output**

If the smoke run in Task 2 produced 10 embeddings, run pooler first on those to verify.

```bash
pixi run python track1_activity/scripts/boltz_affhead/08_pool_fast_embeddings.py 2>&1 | tail -5
pixi run python -c "
import pandas as pd
df = pd.read_parquet('data/boltz_affhead/pooled_fast.parquet')
print('shape:', df.shape)
print('cols first/last:', df.columns[0], df.columns[-1])
print('index:', df.index.name, df.index.min(), df.index.max())
"
```

Expected: (~10 or 8483, 1024), cols `s_prot_mean_000..z_if_max_127`.

- [ ] **Step 3.4: Format + lint**

```bash
pixi run ruff format track1_activity/scripts/boltz_affhead/08_pool_fast_embeddings.py
pixi run ruff check track1_activity/scripts/boltz_affhead/08_pool_fast_embeddings.py
```

- [ ] **Step 3.5: Commit**

```bash
git add track1_activity/scripts/boltz_affhead/08_pool_fast_embeddings.py
git commit -m "feat(boltz): allpairs pooling for fast trunk output (1024d)"
```

---

## Task 4: DB schema + upsert

**Files:**
- Create: `/home/nagaet/pxr-iduction-challenge/db/compound_boltz2_trunk_fast_schema.sql`
- Create: `/home/nagaet/pxr-iduction-challenge/track1_activity/scripts/boltz_affhead/09_upsert_trunk_fast.py`

- [ ] **Step 4.1: Write schema**

```sql
-- /home/nagaet/pxr-iduction-challenge/db/compound_boltz2_trunk_fast_schema.sql
-- Boltz-2 trunk pooled embeddings for the "fast" 13k pretrain corpus.
-- Combines 4653 compounds at rcycle=3 (from existing compound_boltz2 + pooled.parquet)
-- with 8483 new compounds at rcycle=1 (from outputs_fast).
-- recycling_steps column documents the provenance.

CREATE TABLE IF NOT EXISTS compound_boltz2_trunk_fast (
    compound_id INTEGER PRIMARY KEY REFERENCES compounds(id),
    s_prot_mean FLOAT[] NOT NULL,  -- 384d
    s_lig_mean FLOAT[] NOT NULL,   -- 384d
    z_if_mean FLOAT[] NOT NULL,    -- 128d
    z_if_max FLOAT[] NOT NULL,     -- 128d
    recycling_steps INTEGER NOT NULL,
    source_npz_path TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS compound_boltz2_trunk_fast_rcycle_idx
  ON compound_boltz2_trunk_fast(recycling_steps);
```

- [ ] **Step 4.2: Apply schema**

```bash
pixi run psql -h /tmp -p 5433 -d pxr_challenge -f db/compound_boltz2_trunk_fast_schema.sql
pixi run psql -h /tmp -p 5433 -d pxr_challenge -c "\\d compound_boltz2_trunk_fast"
```

- [ ] **Step 4.3: Write upsert script**

Create `/home/nagaet/pxr-iduction-challenge/track1_activity/scripts/boltz_affhead/09_upsert_trunk_fast.py`:

```python
"""Upsert compound_boltz2_trunk_fast from 2 sources:
1. Existing 4653 compounds: pull from data/boltz_affhead/pooled.parquet (rcycle=3)
2. New 8483 compounds: from data/boltz_affhead/pooled_fast.parquet (rcycle=1)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402

POOLED_FULL = REPO_ROOT.joinpath("data", "boltz_affhead", "pooled.parquet")
POOLED_FAST = REPO_ROOT.joinpath("data", "boltz_affhead", "pooled_fast.parquet")

UPSERT_SQL = """
INSERT INTO compound_boltz2_trunk_fast
    (compound_id, s_prot_mean, s_lig_mean, z_if_mean, z_if_max, recycling_steps)
VALUES %s
ON CONFLICT (compound_id) DO UPDATE SET
    s_prot_mean = EXCLUDED.s_prot_mean,
    s_lig_mean = EXCLUDED.s_lig_mean,
    z_if_mean = EXCLUDED.z_if_mean,
    z_if_max = EXCLUDED.z_if_max,
    recycling_steps = EXCLUDED.recycling_steps,
    created_at = NOW();
"""


def row_from_pooled(compound_id: int, row: pd.Series, rcycle: int) -> tuple:
    s_prot = row.iloc[:384].to_numpy(dtype=np.float32)
    s_lig = row.iloc[384:768].to_numpy(dtype=np.float32)
    z_mean = row.iloc[768:896].to_numpy(dtype=np.float32)
    z_max = row.iloc[896:1024].to_numpy(dtype=np.float32)
    return (compound_id, s_prot.tolist(), s_lig.tolist(), z_mean.tolist(), z_max.tolist(), rcycle)


def main() -> None:
    if not POOLED_FULL.exists():
        raise SystemExit(f"Missing {POOLED_FULL}")
    if not POOLED_FAST.exists():
        raise SystemExit(f"Missing {POOLED_FAST}")

    df_full = pd.read_parquet(POOLED_FULL)
    df_fast = pd.read_parquet(POOLED_FAST)
    print(f"pooled.parquet: {df_full.shape} (rcycle=3)")
    print(f"pooled_fast.parquet: {df_fast.shape} (rcycle=1)")

    rows = []
    for cid, r in df_full.iterrows():
        rows.append(row_from_pooled(int(cid), r, rcycle=3))
    for cid, r in df_fast.iterrows():
        if int(cid) not in df_full.index:
            rows.append(row_from_pooled(int(cid), r, rcycle=1))

    print(f"upserting {len(rows)} rows")
    with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
        execute_values(cur, UPSERT_SQL, rows, page_size=100)
        conn.commit()
        cur.execute(
            "SELECT recycling_steps, COUNT(*) FROM compound_boltz2_trunk_fast "
            "GROUP BY recycling_steps ORDER BY recycling_steps"
        )
        for r in cur.fetchall():
            print(f"  rcycle={r[0]}: {r[1]} rows")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.4: Run upsert**

```bash
pixi run python track1_activity/scripts/boltz_affhead/09_upsert_trunk_fast.py 2>&1 | tail -5
```

Expected: two row-count lines, one for rcycle=3 (~4653), one for rcycle=1 (~8483). Total ~13,136.

- [ ] **Step 4.5: Format + lint + commit**

```bash
pixi run ruff format track1_activity/scripts/boltz_affhead/09_upsert_trunk_fast.py
pixi run ruff check track1_activity/scripts/boltz_affhead/09_upsert_trunk_fast.py
git add db/compound_boltz2_trunk_fast_schema.sql \
        track1_activity/scripts/boltz_affhead/09_upsert_trunk_fast.py
git commit -m "feat(boltz): trunk_fast DB table + upsert (mixed rcycle 1+3)"
```

---

## Task 5: MLP pretrain on log2_fc

**Files:**
- Create: `/home/nagaet/pxr-iduction-challenge/track1_activity/scripts/boltz_affhead/10_train_mlp_head.py`

- [ ] **Step 5.1: Write the MLP pretrain script**

Create `/home/nagaet/pxr-iduction-challenge/track1_activity/scripts/boltz_affhead/10_train_mlp_head.py`:

```python
"""MLP pretrain on Boltz-2 trunk (1024d) → log2_fc (2 heads).

3 architectures A/B/C. Pick best by val_mae. Saves:
  models/boltz_mlp_head/{variant}/pretrain.pt
  models/boltz_mlp_head/{variant}/meta.json
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402

CKPT_ROOT = REPO_ROOT.joinpath("models", "boltz_mlp_head")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trunk_and_labels() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (trunk 1024d, targets 2d, compound_ids) for all 13k compounds
    with std_smiles + trunk_fast row. Missing log2_fc entries are NaN."""
    sql = """
    SELECT c.id,
           b.s_prot_mean, b.s_lig_mean, b.z_if_mean, b.z_if_max,
           agg.log2fc_8p25,
           agg.log2fc_33
    FROM compounds c
    JOIN compound_boltz2_trunk_fast b ON b.compound_id = c.id
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
    with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    cids = [r[0] for r in rows]
    trunk = np.array(
        [list(r[1]) + list(r[2]) + list(r[3]) + list(r[4]) for r in rows],
        dtype=np.float32,
    )
    targets = np.array([[r[5] if r[5] is not None else np.nan,
                         r[6] if r[6] is not None else np.nan] for r in rows],
                       dtype=np.float32)
    return trunk, targets, np.array(cids, dtype=np.int64)


class MLPVariantA(nn.Module):
    """Simple: Linear 1024 -> 256 -> 2."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        h = torch.nn.functional.gelu(self.fc1(x))
        h = self.dropout(h)
        return self.fc2(h), h  # (out, hidden 256)


class MLPVariantB(nn.Module):
    """Wider: 1024 -> 512 -> 256 -> 2."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        h = torch.nn.functional.gelu(self.fc1(x))
        h = self.dropout(h)
        h = torch.nn.functional.gelu(self.fc2(h))
        h = self.dropout(h)
        return self.fc3(h), h  # (out, hidden 256)


class MLPVariantC(nn.Module):
    """Attention: 1024 -> 256 -> LN + 2-layer Transformer -> 256 -> 2."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1024, 256)
        self.ln = nn.LayerNorm(256)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=4, dim_feedforward=512, dropout=0.1, batch_first=True
        )
        self.attn = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.fc_out = nn.Linear(256, 2)

    def forward(self, x):
        h = self.proj(x)  # (B, 256)
        h = self.ln(h)
        # Treat each compound as a length-1 sequence; self-attention is a
        # simple block on the single token — essentially LN + FFN residual.
        h_seq = h.unsqueeze(1)  # (B, 1, 256)
        h_attn = self.attn(h_seq).squeeze(1)  # (B, 256)
        return self.fc_out(h_attn), h_attn


def masked_mse(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(y)
    diff = (pred - torch.nan_to_num(y, nan=0.0)) ** 2
    return (diff * mask.float()).sum() / mask.float().sum().clamp(min=1.0)


def train_variant(variant: str, trunk: np.ndarray, targets: np.ndarray, epochs: int, lr: float, batch: int, seed: int) -> tuple[float, dict]:
    """Train one variant, return (best_val_mae, meta)."""
    torch.manual_seed(seed)
    n = len(trunk)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(n * 0.1)
    va_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    # z-score targets (NaN-safe)
    means = np.array([
        np.nanmean(targets[:, 0]), np.nanmean(targets[:, 1])
    ], dtype=np.float32)
    stds = np.array([
        np.nanstd(targets[:, 0]), np.nanstd(targets[:, 1])
    ], dtype=np.float32).clip(min=1e-6)
    targets_z = (targets - means) / stds

    if variant == "A":
        model = MLPVariantA().to(DEVICE)
    elif variant == "B":
        model = MLPVariantB().to(DEVICE)
    elif variant == "C":
        model = MLPVariantC().to(DEVICE)
    else:
        raise ValueError(variant)

    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 20)

    Xt = torch.from_numpy(trunk).to(DEVICE)
    Yt = torch.from_numpy(targets_z).to(DEVICE)

    best_val = float("inf")
    best_state = None
    patience = 10
    bad = 0

    for epoch in range(epochs):
        # Shuffled minibatches on train indices
        model.train()
        perm_tr = np.random.permutation(len(tr_idx))
        tr_losses = []
        for i in range(0, len(tr_idx), batch):
            sel = tr_idx[perm_tr[i : i + batch]]
            xb = Xt[sel]
            yb = Yt[sel]
            opt.zero_grad()
            out, _ = model(xb)
            loss = masked_mse(out, yb)
            loss.backward()
            opt.step()
            tr_losses.append(float(loss))
        sched.step()

        model.eval()
        with torch.no_grad():
            out_val, _ = model(Xt[va_idx])
            val_loss = float(masked_mse(out_val, Yt[va_idx]))

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        print(f"  [variant {variant}] epoch {epoch:02d} tr {np.mean(tr_losses):.4f} val {val_loss:.4f} best {best_val:.4f} bad {bad}")
        if bad >= patience:
            print("  early stop")
            break

    ckpt_dir = CKPT_ROOT.joinpath(variant)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir.joinpath("pretrain.pt")
    torch.save({"state_dict": best_state, "variant": variant, "means": means.tolist(), "stds": stds.tolist()}, ckpt_path)
    print(f"  saved {ckpt_path}")
    meta = {"best_val_loss": best_val, "means": means.tolist(), "stds": stds.tolist()}
    (ckpt_dir.joinpath("meta.json")).write_text(json.dumps(meta, indent=2))
    return best_val, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="A,B,C", help="Comma-separated")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Loading trunk + labels from DB...")
    trunk, targets, cids = load_trunk_and_labels()
    n_v8 = int(np.isfinite(targets[:, 0]).sum())
    n_v33 = int(np.isfinite(targets[:, 1]).sum())
    print(f"  trunk {trunk.shape}  targets {targets.shape}  labeled: 8p25={n_v8} 33={n_v33}")

    results = {}
    for v in args.variants.split(","):
        v = v.strip().upper()
        lr = 3e-4 if v == "C" else args.lr
        print(f"\n=== Variant {v}  epochs={args.epochs} lr={lr} batch={args.batch} ===")
        bv, meta = train_variant(v, trunk, targets, args.epochs, lr, args.batch, args.seed)
        results[v] = bv

    print("\n=== Summary ===")
    for v, bv in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"  {v}: best val MSE {bv:.4f}")
    print("Best variant:", min(results.items(), key=lambda kv: kv[1])[0])


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.2: Run pretrain for all 3 variants**

```bash
cd /home/nagaet/pxr-iduction-challenge
pixi run python track1_activity/scripts/boltz_affhead/10_train_mlp_head.py 2>&1 | tee /tmp/boltz_mlp_pretrain.log | tail -40
```

Expected: each variant converges in 15-40 epochs, total wall-clock ~30-60 min. Records best variant by val MSE.

- [ ] **Step 5.3: Inspect checkpoints**

```bash
ls -la models/boltz_mlp_head/{A,B,C}/
cat models/boltz_mlp_head/{A,B,C}/meta.json
```

Note the best-variant name for Task 6.

- [ ] **Step 5.4: Format + lint + commit**

```bash
pixi run ruff format track1_activity/scripts/boltz_affhead/10_train_mlp_head.py
pixi run ruff check track1_activity/scripts/boltz_affhead/10_train_mlp_head.py
# checkpoints should be gitignored under models/
git check-ignore -v models/boltz_mlp_head/A/pretrain.pt
git add track1_activity/scripts/boltz_affhead/10_train_mlp_head.py
git commit -m "feat(boltz): MLP pretrain on log2_fc (variants A/B/C)"
```

If `models/boltz_mlp_head/` NOT gitignored, add `models/` or explicit pattern to `.gitignore` and include in commit.

---

## Task 6: Extract frozen penultimate embedding

**Files:**
- Create: `/home/nagaet/pxr-iduction-challenge/track1_activity/scripts/boltz_affhead/11_extract_embedding.py`

- [ ] **Step 6.1: Write extractor**

Create `/home/nagaet/pxr-iduction-challenge/track1_activity/scripts/boltz_affhead/11_extract_embedding.py`:

```python
"""Extract penultimate 256d embedding from a chosen MLP variant.

Output: data/boltz_trunk_pretrain_embed.parquet (13136 × 256)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts", "boltz_affhead")))

from data import DB_PARAMS  # noqa: E402

# Reuse same model defs from training script.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "boltz_mlp_head_train",
    REPO_ROOT.joinpath("track1_activity", "scripts", "boltz_affhead", "10_train_mlp_head.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MLPVariantA, MLPVariantB, MLPVariantC, load_trunk_and_labels = (
    _mod.MLPVariantA, _mod.MLPVariantB, _mod.MLPVariantC, _mod.load_trunk_and_labels
)

CKPT_ROOT = REPO_ROOT.joinpath("models", "boltz_mlp_head")
OUT_PATH = REPO_ROOT.joinpath("data", "boltz_trunk_pretrain_embed.parquet")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_variant(variant: str):
    ckpt = torch.load(
        CKPT_ROOT.joinpath(variant, "pretrain.pt"), map_location="cpu", weights_only=False
    )
    if variant == "A":
        m = MLPVariantA()
    elif variant == "B":
        m = MLPVariantB()
    elif variant == "C":
        m = MLPVariantC()
    else:
        raise ValueError(variant)
    m.load_state_dict(ckpt["state_dict"])
    return m.to(DEVICE).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="A, B, or C (chosen best from Task 5)")
    args = ap.parse_args()

    print("Loading trunk from DB...")
    trunk, _, cids = load_trunk_and_labels()
    print(f"  trunk {trunk.shape}  cids {len(cids)}")

    model = load_variant(args.variant)
    Xt = torch.from_numpy(trunk).to(DEVICE)
    with torch.no_grad():
        _, hidden = model(Xt)
        emb = hidden.cpu().numpy().astype(np.float32)
    print(f"  embedding shape {emb.shape}")

    cols = [f"emb_{i:04d}" for i in range(emb.shape[1])]
    df = pd.DataFrame(emb, index=pd.Index(cids, name="compound_id"), columns=cols)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH)
    print(f"Wrote {OUT_PATH} shape {df.shape}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.2: Run extraction**

```bash
# Replace <BEST> with the best variant reported in Task 5
pixi run python track1_activity/scripts/boltz_affhead/11_extract_embedding.py --variant <BEST> 2>&1 | tail -10
```

Expected: ~256d embedding parquet, 13,136 rows.

- [ ] **Step 6.3: Sanity check**

```bash
pixi run python -c "
import pandas as pd
df = pd.read_parquet('data/boltz_trunk_pretrain_embed.parquet')
print('shape:', df.shape)
print('index min/max:', df.index.min(), df.index.max())
print('cols first/last:', df.columns[0], df.columns[-1])
print('NaN:', int(df.isna().sum().sum()))
"
```

Expected: (13136, 256), index `compound_id` 1..13136, NaN=0.

- [ ] **Step 6.4: Commit extractor**

```bash
pixi run ruff format track1_activity/scripts/boltz_affhead/11_extract_embedding.py
pixi run ruff check track1_activity/scripts/boltz_affhead/11_extract_embedding.py
git add track1_activity/scripts/boltz_affhead/11_extract_embedding.py
git commit -m "feat(boltz): extract penultimate 256d from pretrain MLP"
```

---

## Task 7: Register `boltz_trunk_pretrain_embed` feature in run_train.py

**Files:**
- Modify: `/home/nagaet/pxr-iduction-challenge/track1_activity/scripts/run_train.py`

- [ ] **Step 7.1: Locate insertion point**

```bash
grep -n "gatedgcn_pretrain_embed" track1_activity/scripts/run_train.py
```

Note the two occurrences (feature branch + all_features list).

- [ ] **Step 7.2: Add feature branch**

After the `gatedgcn_pretrain_embed` branch's `return X_train, X_test`, insert:

```python
    if feature_name == "boltz_trunk_pretrain_embed":
        # 256d penultimate embedding from MLP pretrained on log2_fc over
        # Boltz-2 trunk (1024d, mix of rcycle=3 for 4653 compounds and
        # rcycle=1 for 8483). See
        # track1_activity/scripts/boltz_affhead/10_train_mlp_head.py +
        # 11_extract_embedding.py.
        # Buterez 2024 strategy-3 applied to Boltz-2 trunk — the only
        # protein-ligand-aware backbone in the pool.
        embed_path = REPO_ROOT.joinpath("data", "boltz_trunk_pretrain_embed.parquet")
        if not embed_path.exists():
            raise SystemExit(
                f"Missing {embed_path}. Run "
                f"track1_activity/scripts/boltz_affhead/11_extract_embedding.py"
            )
        emb_df = pd.read_parquet(embed_path)
        X_train = emb_df.reindex(index=train_ids).to_numpy(dtype=np.float32).copy()
        X_test = emb_df.reindex(index=test_ids).to_numpy(dtype=np.float32).copy()
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        print(
            f"  boltz_trunk_pretrain_embed: {X_train.shape[1]} dims "
            f"(train {X_train.shape[0]} / test {X_test.shape[0]})"
        )
        return X_train, X_test
```

- [ ] **Step 7.3: Add to all_features list**

Find the occurrence of `"gatedgcn_pretrain_embed"` inside the CLI choices list (`all_features`). Append `"boltz_trunk_pretrain_embed"` right after.

- [ ] **Step 7.4: Smoke test the feature loader**

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
Xtr, Xte = load_features('boltz_trunk_pretrain_embed', tr, te)
print('train', Xtr.shape, 'test', Xte.shape, 'NaN', int(np.isnan(Xtr).sum() + np.isnan(Xte).sum()))
"
```

Expected: `train (4140, 256) test (513, 256) NaN 0`.

- [ ] **Step 7.5: Format + lint + commit**

```bash
pixi run ruff format track1_activity/scripts/run_train.py
pixi run ruff check track1_activity/scripts/run_train.py
git add track1_activity/scripts/run_train.py
git commit -m "feat(boltz): register boltz_trunk_pretrain_embed feature"
```

(The 8 pre-existing E402 warnings are unchanged.)

---

## Task 8: TabPFN 5-fold + acceptance gates

**Files:** none (runs `run_train.py`)

- [ ] **Step 8.1: Launch training in background**

```bash
cd /home/nagaet/pxr-iduction-challenge
nohup pixi run python track1_activity/scripts/run_train.py \
    --model tabpfn \
    --feature boltz_trunk_pretrain_embed \
    --split umap \
    --trials 0 \
    > /tmp/tabpfn_boltz_trunk.log 2>&1 &
echo $! > /tmp/tabpfn_boltz_trunk_pid.txt
disown
```

Expected wall-clock: 20-30 min.

- [ ] **Step 8.2: Monitor completion + verify MAE**

When complete, verify:

```bash
pixi run python -c "
import psycopg2, sys
sys.path.insert(0, 'track1_activity/src')
from data import DB_PARAMS
with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
    cur.execute(\"SELECT name, mae_mean, rae_mean, spearman_mean, kendall_mean FROM experiment_summary WHERE name LIKE 'tabpfn_boltz_trunk_pretrain_embed%' ORDER BY created_at DESC LIMIT 2\")
    for r in cur.fetchall():
        print(r)
"
```

Acceptance gate: OOF MAE ≤ 0.48. If fails, STOP and report.

- [ ] **Step 8.3: Pearson r vs current 8-pool**

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
    'tabpfn_boltz_trunk_pretrain_embed_umap_default',
]
with psycopg2.connect(**DB_PARAMS) as conn:
    cur = conn.cursor()
    oofs = {}
    for n in names:
        cur.execute('SELECT eop.train_idx, eop.oof_prediction FROM experiment_oof_predictions eop JOIN experiments e ON e.id = eop.experiment_id WHERE e.name = %s ORDER BY eop.train_idx', (n,))
        rows = cur.fetchall()
        if rows:
            oofs[n] = pd.Series({int(r[0]): float(r[1]) for r in rows})
df = pd.DataFrame(oofs).dropna()
print('shape:', df.shape)
new = 'tabpfn_boltz_trunk_pretrain_embed_umap_default'
print('pairwise r (< 0.95 preferred):')
for other in df.columns:
    if other != new:
        r = df[new].corr(df[other])
        ok = 'OK' if r < 0.95 else 'HIGH'
        print(f'  boltz_trunk_PE vs {other:<50} r={r:.4f} {ok}')
"
```

- [ ] **Step 8.4: Report — no commit**

Record OOF MAE + Pearson r values for Task 9's bakeoff step.

---

## Task 9: Ensemble swap/add bakeoff + commit

**Files:**
- Modify: `/home/nagaet/pxr-iduction-challenge/track1_activity/scripts/run_ensemble.py`

- [ ] **Step 9.1: Bakeoff: test swap vs add-9**

```bash
cd /home/nagaet/pxr-iduction-challenge
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
NEW = 'tabpfn_boltz_trunk_pretrain_embed_umap_default'
SWAP_TARGET = 'tabpfn_pooled_boltz_umap_default'  # the raw-trunk candidate that this PR is supposed to supersede

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

def run(members):
    mat = np.stack([oofs[m] for m in members], axis=1)
    mask = np.all(np.isfinite(mat), axis=1) & np.isfinite(y)
    w = optimize_caruana(mat[mask], y[mask])
    mae = float(np.mean(np.abs(mat[mask] @ w - y[mask])))
    return mae, {m: float(wi) for m, wi in zip(members, w)}

# Configurations
A_mae, A_w = run(BASE)
add_mae, add_w = run(BASE + [NEW])
swap_members = [m for m in BASE if m != SWAP_TARGET] + [NEW]
swap_mae, swap_w = run(swap_members)
print(f'Baseline 8-pool MAE: {A_mae:.4f}')
print(f'add-9 (base + boltz_trunk_PE) MAE: {add_mae:.4f}  Δ {add_mae - A_mae:+.4f}  new wt {add_w[NEW]:.4f}')
print(f'swap (-pooled_boltz +boltz_trunk_PE) MAE: {swap_mae:.4f}  Δ {swap_mae - A_mae:+.4f}  new wt {swap_w[NEW]:.4f}')
"
```

Decision:
- If **swap < baseline** (ideally < 0.4185): prefer swap
- Elif **add-9 < baseline**: add-9
- Else: STOP and report BLOCKED, drop from allow-list

- [ ] **Step 9.2: Edit run_ensemble.py per chosen action**

If **swap**: replace `"tabpfn_pooled_boltz_umap_default"` → `"tabpfn_boltz_trunk_pretrain_embed_umap_default"` in ENSEMBLE_MODELS, update the comment block.

If **add-9**: append `"tabpfn_boltz_trunk_pretrain_embed_umap_default"` after `"tabpfn_gatedgcn_pretrain_embed_umap_default"` with a new comment block (see PR #107 for format).

- [ ] **Step 9.3: Run caruana + calibrate**

```bash
pixi run python track1_activity/scripts/run_ensemble.py 2>&1 | tee /tmp/ens_boltz_trunk.log | tail -30
pixi run python track1_activity/scripts/run_ensemble_calibrate.py 2>&1 | tee /tmp/cal_boltz_trunk.log | tail -15
wc -l track1_activity/submissions/ens_caruana_bag20_calibrated_best.csv
```

Expected: 514 lines, calibrated MAE reported, linear_pos winner.

- [ ] **Step 9.4: Format + lint + commit**

```bash
pixi run ruff format track1_activity/scripts/run_ensemble.py
pixi run ruff check track1_activity/scripts/run_ensemble.py
git add track1_activity/scripts/run_ensemble.py
git commit -m "$(cat <<'COMMIT_EOF'
feat(ens): boltz_trunk_pretrain_embed — strategy-3 Boltz backbone

Sixth strategy-3 pool member (chemprop / molformer_c3 / kermt /
attentivefp / gatedgcn / boltz_trunk). Boltz-2 trunk 1024d extended
to all 13k via rcycle=1 fast mode (existing 4653 kept at rcycle=3),
MLP variant <CHOSEN> pretrained on log2_fc (val MSE <fill>),
penultimate 256d extracted as downstream feature.

Task 8 single-model OOF:
- MAE <fill>
- RAE <fill>, Spearman <fill>
- Pearson r vs existing: <fill> (max with <closest_member>)

Task 9 <swap|add-9> 8/9-pool caruana_bag20:
- MAE <fill> (vs 8-pool baseline 0.4185, delta <fill>)
- boltz_trunk weight <fill>
COMMIT_EOF
)"
```

---

## Task 10: Push + PR + LB submission

- [ ] **Step 10.1: Push**

```bash
cd /home/nagaet/pxr-iduction-challenge
git push -u origin feature/boltz-trunk-log2fc-pretrain
```

- [ ] **Step 10.2: Open PR**

```bash
gh pr create --title "feat: Boltz-2 trunk × log2_fc pretrain (strategy-3 Boltz)" --body "$(cat <<'EOF'
## Summary
- Completes Buterez strategy-3 sweep on Boltz-2 trunk. Extended trunk to all 13,136 compounds via user's fork's `--embeddings_only --recycling_steps 1` fast mode (8,483 new compounds at rcycle=1; existing 4,653 kept at rcycle=3).
- MLP pretrained on log2_fc (variants A/B/C, chose <CHOSEN>, val MSE <FILL>).
- Penultimate 256d embedding → TabPFN 5-fold.

## Acceptance results
1. Trunk extension: <N> / 8,483 compounds covered
2. Single-model OOF MAE: <FILL> (target ≤ 0.48)
3. caruana weight: <FILL>
4. Pool MAE (swap or add-9): <FILL> (vs 0.4185 baseline, Δ <FILL>)
5. Pearson r with existing 8-pool: all < 0.95 (max: <value> vs <name>)
6. ruff clean

## Files
### New
- track2_structure/scripts/boltz2_build_inputs_fast.py
- track2_structure/scripts/boltz2_fast_embeddings_run.sh
- track1_activity/scripts/boltz_affhead/08_pool_fast_embeddings.py
- track1_activity/scripts/boltz_affhead/09_upsert_trunk_fast.py
- track1_activity/scripts/boltz_affhead/10_train_mlp_head.py
- track1_activity/scripts/boltz_affhead/11_extract_embedding.py
- db/compound_boltz2_trunk_fast_schema.sql

### Modified
- track1_activity/scripts/run_train.py (register feature)
- track1_activity/scripts/run_ensemble.py (swap or add-9)

## CI: N/A

## Related
- Spec: `docs/superpowers/specs/2026-04-21-boltz-trunk-log2fc-pretrain-design.md`
- Paper reading notes: `docs/papers/boltz2_affinity_notes.md`
EOF
)"
```

- [ ] **Step 10.3: LB plan**

After merge, use `track1_activity/scripts/scheduled_submit.sh` for background LB submission:

```bash
nohup bash track1_activity/scripts/scheduled_submit.sh \
    track1_activity/submissions/ens_caruana_bag20_calibrated_best.csv \
    --experiment ens_caruana_bag20_calibrated_best \
    --notes "8-pool with boltz_trunk_pretrain_embed (strategy-3 Boltz, PR #<N>). ..." \
    > /tmp/submit_status.log 2>&1 &
disown
```

Report PR URL + cooldown status. Wait for user before merge.

---

## Self-review notes

Spec coverage:
- Phase 1 trunk extension → Tasks 1-2
- Phase 2 DB upsert → Tasks 3-4
- Phase 3 MLP pretrain → Task 5
- Phase 4 extract → Task 6
- Phase 5 TabPFN + ensemble → Tasks 7-10
- Acceptance criteria 1-6 → Tasks 3 (extraction count), 8 (MAE, Pearson), 9 (pool MAE)

Placeholder scan: `<CHOSEN>`, `<FILL>`, `<N>` placeholders appear only in commit message / PR body templates and are explicitly noted as "fill during task". No mid-task placeholder code.

Type consistency:
- `load_trunk_and_labels()` is used in both Task 5 (pretrain) and Task 6 (extract)
- `MLPVariantA/B/C` imported via importlib in Task 6 to avoid code duplication
- Parquet schema matches across phases (`compound_id` index, `emb_NNNN` columns)

Fluid points:
- Task 1.1 compound count may vary (smoke reduces to 10); Task 2.6 failure count documented inline
- Task 5 variant selection is an empirical check; Task 6 runs once with the chosen variant
- Task 9 swap-vs-add decision is data-driven; commit message + PR body fill in the chosen path
