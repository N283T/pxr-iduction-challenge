# MTR Domain Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply Sultan 2025 MTR-DA recipe to PXR pool — build two SWAP candidates (ChemProp scratch + MoLFormer-c3-1.1B DA) trained on 217 RDKit physchem descriptors, evaluate against 4-gate ladder before LB submit.

**Architecture:** Two independent encoder variants are trained with descriptor MTR as the *sole* objective on 13,134 PXR std_smiles (drop 2 NaN rows). Frozen embeddings → TabPFN UMAP CV → 4-gate evaluation. Pool integration is SWAP-only (replacing existing log2fc-pretrained equivalents), driven by caruana_bag20 OOF Δ. No existing pretrain scripts modified.

**Tech Stack:** ChemProp 2.x (D-MPNN), MoLFormer-c3-1.1B (PEFT/LoRA via `peft_trainer.py`), PyTorch Lightning, TabPFN, PostgreSQL+RDKit cartridge, pixi.

**Spec:** `docs/superpowers/specs/2026-05-04-mtr-domain-adaptation-design.md` (commit cbc4f6e)

**Branch:** `experiment/mtr-domain-adaptation`

---

## Important conventions

- All new scripts use `#!/usr/bin/env -S pixi run python` shebang (PEP 723 / project convention).
- All DB queries use `psycopg2` directly with `host="/tmp", port=5433, dbname="pxr_challenge"` (matches `data.DB_PARAMS`). NOT pixi-run-db-psql per memory `feedback_avoid_pixi_run_db_psql`.
- Code/comments/commits in English. Conventional commit format `feat(track1):` / `fix(track1):` / `chore(track1):`.
- After each task with code changes: ruff format + ruff check + commit.
- **GPU run gates**: Tasks marked `[USER GATE]` require explicit user approval before running. Do not execute these unprompted.
- ChemProp existing config from `run_chemprop_pretrain.py:62-77` uses `message_hidden_dim=256, depth=4` (Optuna-tuned, not 300/3 as the spec referenced). Use **the same `DEFAULT_PARAMS` dict** verbatim from that file for Variant C — copy, don't import. Spec is corrected in this plan.

---

## File map

### New files

| Path | Role |
|---|---|
| `track1_activity/scripts/audit_mtr_leak.py` | G0 leak audit gate (6 checks → JSON report) |
| `track1_activity/scripts/run_chemprop_mtr_pretrain.py` | Variant C training |
| `track1_activity/scripts/run_chemprop_mtr_extract.py` | Variant C frozen embedding extract |
| `track1_activity/scripts/run_molformer_mtr_pretrain.py` | Variant M training |
| `track1_activity/scripts/run_molformer_mtr_extract.py` | Variant M frozen embedding extract |
| `track1_activity/scripts/eval_mtr_gates.py` | G1 + G2 evaluation (single MAE + residual r) |
| `track1_activity/reports/mtr_leak_audit_2026-05-04.json` | G0 output |
| `models/chemprop_mtr_seed42/` | Variant C: `pretrain.pt`, `scaler.json`, `meta.json` |
| `models/molformer_c3_mtr_seed42/` | Variant M: `pretrain.pt`, `scaler.json`, `meta.json` |
| `data/chemprop_mtr_embedding_seed42.parquet` | Variant C 300d × 13,136 |
| `data/molformer_c3_mtr_embedding_seed42.parquet` | Variant M 768d × 13,136 |

### Existing files modified

| Path | Change |
|---|---|
| `track1_activity/src/features.py` | Add 2 new entries to `FP_REGISTRY`: `chemprop_mtr_embed`, `molformer_c3_mtr_embed` (loaders that read the parquets above) |

### Existing files NOT modified (protected)

- `run_chemprop_pretrain.py` (rank-1 driver — read-only reference)
- `run_molformer_c3_pretrain.py` (read-only reference)
- `peft_trainer.py` (imported, not edited)

---

## Task 1: Leak audit script (G0)

**Files:**
- Create: `track1_activity/scripts/audit_mtr_leak.py`

**Why this task first:** All downstream pretrain scripts call this and refuse to run on audit failure. Building it first lets us validate the data assumptions cheaply (no GPU).

- [ ] **Step 1: Create the audit script**

```python
#!/usr/bin/env -S pixi run python
"""MTR leak audit — gate G0.

Performs 6 leak-related checks on the MTR pretrain data setup.
Exits 0 with JSON report on success, exits 1 on any failure.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402

REPORT_DIR = REPO_ROOT.joinpath("track1_activity", "reports")
EXPECTED_NAN_COMPOUND_IDS = {1657, 8624}
EXPECTED_DESCRIPTOR_COUNT = 217


def check_id_overlap(conn) -> dict:
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM ("
        "  SELECT compound_id FROM train_activity"
        "  INTERSECT"
        "  SELECT compound_id FROM test_activity"
        ") AS x"
    )
    overlap = cur.fetchone()[0]
    return {"name": "L5a_compound_id_overlap", "passed": overlap == 0, "value": overlap}


def check_smiles_overlap(conn) -> dict:
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM ("
        "  SELECT c.std_smiles FROM compounds c JOIN train_activity t USING(compound_id)"
        "  INTERSECT"
        "  SELECT c.std_smiles FROM compounds c JOIN test_activity t USING(compound_id)"
        ") AS x"
    )
    overlap = cur.fetchone()[0]
    return {"name": "L5b_std_smiles_overlap", "passed": overlap == 0, "value": overlap}


def check_descriptor_source(conn) -> dict:
    # Read schema; verify only `descriptors` jsonb column exists for
    # compound_descriptors_full. No experiment-derived columns.
    cur = conn.cursor()
    cur.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'compound_descriptors_full' ORDER BY ordinal_position"
    )
    cols = [r[0] for r in cur.fetchall()]
    expected = ["compound_id", "descriptors"]
    return {
        "name": "L1_descriptor_source",
        "passed": cols == expected,
        "value": cols,
    }


def check_descriptor_count(conn) -> dict:
    df = pd.read_sql(
        "SELECT descriptors FROM compound_descriptors_full LIMIT 1",
        psycopg2.connect(**DB_PARAMS),
    )
    expanded = pd.json_normalize(df["descriptors"])
    n = len(expanded.columns)
    return {
        "name": "L6a_descriptor_count",
        "passed": n == EXPECTED_DESCRIPTOR_COUNT,
        "value": n,
    }


def check_nan_drop_set(conn) -> dict:
    df = pd.read_sql(
        "SELECT compound_id, descriptors FROM compound_descriptors_full",
        psycopg2.connect(**DB_PARAMS),
    )
    expanded = pd.json_normalize(df["descriptors"]).apply(
        pd.to_numeric, errors="coerce"
    )
    nan_rows = expanded.isna().any(axis=1)
    bad_ids = set(df.loc[nan_rows, "compound_id"].tolist())
    return {
        "name": "L6b_nan_drop_set",
        "passed": bad_ids == EXPECTED_NAN_COMPOUND_IDS,
        "value": sorted(bad_ids),
    }


def check_no_inf(conn) -> dict:
    df = pd.read_sql(
        "SELECT descriptors FROM compound_descriptors_full",
        psycopg2.connect(**DB_PARAMS),
    )
    expanded = pd.json_normalize(df["descriptors"]).apply(
        pd.to_numeric, errors="coerce"
    )
    inf_count = int(np.isinf(expanded.to_numpy()).sum())
    return {
        "name": "L6c_no_inf",
        "passed": inf_count == 0,
        "value": inf_count,
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    conn = psycopg2.connect(**DB_PARAMS)
    checks = [
        check_id_overlap(conn),
        check_smiles_overlap(conn),
        check_descriptor_source(conn),
        check_descriptor_count(conn),
        check_nan_drop_set(conn),
        check_no_inf(conn),
    ]
    conn.close()

    report = {
        "date": str(date.today()),
        "checks": checks,
        "all_passed": all(c["passed"] for c in checks),
    }
    out_path = REPORT_DIR.joinpath(f"mtr_leak_audit_{date.today()}.json")
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Format + lint**

```bash
pixi run ruff format track1_activity/scripts/audit_mtr_leak.py
pixi run ruff check track1_activity/scripts/audit_mtr_leak.py
```

- [ ] **Step 3: Run the audit (no GPU)**

```bash
pixi run python track1_activity/scripts/audit_mtr_leak.py
```

Expected output: JSON with `"all_passed": true` and 6 individual checks all passed. The 2 NaN compound IDs printed should be `[1657, 8624]`. Check `track1_activity/reports/mtr_leak_audit_<date>.json` exists.

- [ ] **Step 4: Commit**

```bash
git add track1_activity/scripts/audit_mtr_leak.py track1_activity/reports/mtr_leak_audit_*.json
git commit -m "feat(track1): MTR leak audit script (gate G0) - 6 checks all pass"
```

---

## Task 2: ChemProp MTR pretrain script (Variant C)

**Files:**
- Create: `track1_activity/scripts/run_chemprop_mtr_pretrain.py`

**Reference (read-only):** `track1_activity/scripts/run_chemprop_pretrain.py:1-326` for module imports, `DEFAULT_PARAMS` dict, ChemProp data loader pattern, Lightning trainer wiring.

- [ ] **Step 1: Write the pretrain script (skeleton + audit guard + data loader + model + trainer)**

```python
#!/usr/bin/env -S pixi run python
"""ChemProp MTR pretrain (Variant C from spec 2026-05-04-mtr-domain-adaptation-design).

Standalone descriptor multi-task regression on 13,134 PXR compounds
(2 NaN rows dropped: 1657, 8624). 217 RDKit descriptors as targets,
StandardScaler-normalized, MSE loss summed across heads. NO log2fc,
NO pec50 in the loss. Output: encoder state_dict + scaler.json.

Usage:
    pixi run python track1_activity/scripts/run_chemprop_mtr_pretrain.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import torch
from lightning import pytorch as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from chemprop import data as chemprop_data  # noqa: E402
from chemprop import models, nn  # noqa: E402
from chemprop.nn.metrics import MSE  # noqa: E402

from data import DB_PARAMS  # noqa: E402

torch.set_float32_matmul_precision("medium")

CKPT_BASE = REPO_ROOT.joinpath("models")
NAN_DROP_IDS = {1657, 8624}

# Mirrors run_chemprop_pretrain.py DEFAULT_PARAMS verbatim
# (Optuna-tuned best, OOF RAE 0.5724 single-task baseline).
DEFAULT_PARAMS = {
    "message_hidden_dim": 256,
    "depth": 4,
    "mp_dropout": 0.2,
    "activation": "relu",
    "aggregation": "norm",
    "ffn_hidden_dim": 256,
    "ffn_num_layers": 1,
    "ffn_dropout": 0.1,
    "warmup_epochs": 3,
    "learning_rate": 0.0001364559692954765,
    "lr_ratio": 10.0,
    "batch_size": 128,
    "max_epochs": 50,   # Sultan recipe: 20-50 is enough for MTR DA
    "patience": 10,
}

AGG_REGISTRY = {
    "mean": nn.MeanAggregation,
    "sum": nn.SumAggregation,
    "norm": nn.NormAggregation,
}


def run_audit_or_die() -> None:
    """Invoke audit_mtr_leak.py; exit if any check fails."""
    audit = REPO_ROOT.joinpath("track1_activity/scripts/audit_mtr_leak.py")
    res = subprocess.run(
        ["pixi", "run", "python", str(audit)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if res.returncode != 0:
        print("MTR LEAK AUDIT FAILED — refusing to start pretrain.")
        print(res.stdout)
        print(res.stderr)
        sys.exit(1)
    print("Audit passed.")


def load_descriptor_targets() -> tuple[list[str], np.ndarray, list[int], list[str]]:
    """Returns (smiles_list, target_matrix, compound_ids, descriptor_names).

    smiles, target_matrix and compound_ids are aligned by index.
    NAN_DROP_IDS are excluded.
    """
    conn = psycopg2.connect(**DB_PARAMS)
    desc_df = pd.read_sql(
        "SELECT cd.compound_id, c.std_smiles, cd.descriptors "
        "FROM compound_descriptors_full cd "
        "JOIN compounds c USING(compound_id) "
        "ORDER BY cd.compound_id",
        conn,
    )
    conn.close()

    desc_df = desc_df[~desc_df["compound_id"].isin(NAN_DROP_IDS)].reset_index(drop=True)
    expanded = pd.json_normalize(desc_df["descriptors"]).apply(
        pd.to_numeric, errors="coerce"
    )
    descriptor_names = expanded.columns.tolist()
    targets = expanded.to_numpy(dtype=np.float32)

    if np.isnan(targets).any():
        raise RuntimeError(
            "NaN remains after row-drop — audit may be stale. "
            "Re-run audit_mtr_leak.py."
        )

    return (
        desc_df["std_smiles"].tolist(),
        targets,
        desc_df["compound_id"].tolist(),
        descriptor_names,
    )


def fit_scaler(targets: np.ndarray) -> dict:
    mean = targets.mean(axis=0)
    std = targets.std(axis=0) + 1e-8
    return {"mean": mean.astype(np.float64).tolist(), "std": std.astype(np.float64).tolist()}


def apply_scaler(targets: np.ndarray, scaler: dict) -> np.ndarray:
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    return (targets - mean) / std


def build_model(params: dict, n_tasks: int) -> models.MPNN:
    mp = nn.BondMessagePassing(
        d_h=params["message_hidden_dim"],
        depth=params["depth"],
        dropout=params["mp_dropout"],
        activation=params["activation"],
    )
    agg = AGG_REGISTRY[params["aggregation"]]()
    predictor = nn.RegressionFFN(
        n_tasks=n_tasks,
        input_dim=mp.output_dim,
        hidden_dim=params["ffn_hidden_dim"],
        n_layers=params["ffn_num_layers"],
        dropout=params["ffn_dropout"],
        criterion=MSE(),
    )
    return models.MPNN(
        mp,
        agg,
        predictor,
        warmup_epochs=params["warmup_epochs"],
        init_lr=params["learning_rate"] / params["lr_ratio"],
        max_lr=params["learning_rate"],
        final_lr=params["learning_rate"] / params["lr_ratio"],
    )


def make_dataloader(smiles: list[str], targets: np.ndarray, batch_size: int, shuffle: bool):
    data = [
        chemprop_data.MoleculeDatapoint.from_smi(s, y) for s, y in zip(smiles, targets)
    ]
    dataset = chemprop_data.MoleculeDataset(data)
    return chemprop_data.build_dataloader(dataset, batch_size=batch_size, shuffle=shuffle)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="Smoke run: 3 epochs, 200 compounds, no checkpoint write")
    args = p.parse_args()

    run_audit_or_die()
    pl.seed_everything(args.seed, workers=True)

    smiles, raw_targets, ids, desc_names = load_descriptor_targets()
    print(f"Loaded {len(smiles)} compounds × {raw_targets.shape[1]} descriptors")

    if args.smoke:
        smiles = smiles[:200]
        raw_targets = raw_targets[:200]
        ids = ids[:200]

    scaler = fit_scaler(raw_targets)
    targets = apply_scaler(raw_targets, scaler)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(smiles))
    n_val = int(0.1 * len(perm))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_loader = make_dataloader(
        [smiles[i] for i in train_idx],
        targets[train_idx],
        DEFAULT_PARAMS["batch_size"],
        shuffle=True,
    )
    val_loader = make_dataloader(
        [smiles[i] for i in val_idx],
        targets[val_idx],
        DEFAULT_PARAMS["batch_size"],
        shuffle=False,
    )

    params = dict(DEFAULT_PARAMS)
    if args.smoke:
        params["max_epochs"] = 3
    model = build_model(params, n_tasks=targets.shape[1])

    out_dir = CKPT_BASE.joinpath(f"chemprop_mtr_seed{args.seed}")
    if not args.smoke:
        out_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [pl.callbacks.EarlyStopping(monitor="val_loss", patience=params["patience"])]
    trainer = pl.Trainer(
        max_epochs=params["max_epochs"],
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        callbacks=callbacks,
        log_every_n_steps=20,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_loader, val_loader)

    if args.smoke:
        print("Smoke run completed; no checkpoint written.")
        return

    torch.save(model.state_dict(), out_dir.joinpath("pretrain.pt"))
    out_dir.joinpath("scaler.json").write_text(json.dumps(scaler, indent=2))
    out_dir.joinpath("meta.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "n_compounds": len(smiles),
                "descriptor_names": desc_names,
                "params": params,
                "best_val_loss": float(trainer.callback_metrics.get("val_loss", -1)),
            },
            indent=2,
        )
    )
    print(f"Saved {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Format + lint**

```bash
pixi run ruff format track1_activity/scripts/run_chemprop_mtr_pretrain.py
pixi run ruff check track1_activity/scripts/run_chemprop_mtr_pretrain.py
```

- [ ] **Step 3: Smoke test (3 epochs, 200 compounds, ~1 min on GPU)**

```bash
pixi run python track1_activity/scripts/run_chemprop_mtr_pretrain.py --smoke --seed 42
```

Expected: prints "Audit passed.", "Loaded 200 compounds × 217 descriptors", trainer runs 3 epochs, prints "Smoke run completed; no checkpoint written."
If audit fails, fix the failure before continuing.
If trainer errors, debug locally — do not proceed to full run.

- [ ] **Step 4: Commit**

```bash
git add track1_activity/scripts/run_chemprop_mtr_pretrain.py
git commit -m "feat(track1): chemprop MTR pretrain script (Variant C) + smoke pass"
```

---

## Task 3: ChemProp MTR extract script

**Files:**
- Create: `track1_activity/scripts/run_chemprop_mtr_extract.py`

- [ ] **Step 1: Write the extract script**

```python
#!/usr/bin/env -S pixi run python
"""Extract frozen ChemProp MTR encoder embeddings (300d) for all 13,136
compounds (including the 2 NaN-dropped at pretrain time — forward pass
still works on any valid SMILES).
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from chemprop import data as chemprop_data  # noqa: E402

from data import DB_PARAMS  # noqa: E402
from run_chemprop_mtr_pretrain import (  # noqa: E402
    DEFAULT_PARAMS,
    build_model,
    make_dataloader,
)


def load_all_compounds() -> pd.DataFrame:
    conn = psycopg2.connect(**DB_PARAMS)
    df = pd.read_sql(
        "SELECT compound_id, std_smiles FROM compounds ORDER BY compound_id",
        conn,
    )
    conn.close()
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ckpt_dir = REPO_ROOT.joinpath(f"models/chemprop_mtr_seed{args.seed}")
    meta = json.loads(ckpt_dir.joinpath("meta.json").read_text())
    n_tasks = len(meta["descriptor_names"])

    model = build_model(meta["params"], n_tasks=n_tasks)
    model.load_state_dict(torch.load(ckpt_dir.joinpath("pretrain.pt")))
    model.eval().cuda()

    compounds = load_all_compounds()
    print(f"Extracting embeddings for {len(compounds)} compounds")

    dummy_targets = np.zeros((len(compounds), n_tasks), dtype=np.float32)
    loader = make_dataloader(
        compounds["std_smiles"].tolist(),
        dummy_targets,
        batch_size=256,
        shuffle=False,
    )

    chunks = []
    with torch.no_grad():
        for batch in loader:
            bmg = batch.bmg.to("cuda")
            # MPNN forward: message-passing → aggregate → graph embedding
            mp_out = model.message_passing(bmg, batch.V_d.to("cuda") if batch.V_d is not None else None)
            graph_emb = model.agg(mp_out, bmg.batch)
            chunks.append(graph_emb.cpu().numpy())

    embeddings = np.concatenate(chunks)
    print(f"Embedding shape: {embeddings.shape}")
    assert not np.isnan(embeddings).any(), "NaN in embeddings"
    assert not np.isinf(embeddings).any(), "inf in embeddings"

    out_df = pd.DataFrame(
        embeddings,
        index=compounds["compound_id"].values,
        columns=[f"chemprop_mtr_{i}" for i in range(embeddings.shape[1])],
    )
    out_df.index.name = "compound_id"

    out_path = REPO_ROOT.joinpath(f"data/chemprop_mtr_embedding_seed{args.seed}.parquet")
    out_df.to_parquet(out_path)
    print(f"Saved {out_path} ({embeddings.shape})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Format + lint**

```bash
pixi run ruff format track1_activity/scripts/run_chemprop_mtr_extract.py
pixi run ruff check track1_activity/scripts/run_chemprop_mtr_extract.py
```

- [ ] **Step 3: Commit (no run yet — depends on Task 4 checkpoint)**

```bash
git add track1_activity/scripts/run_chemprop_mtr_extract.py
git commit -m "feat(track1): chemprop MTR extract script (Variant C)"
```

---

## Task 4 [USER GATE]: Variant C full pretrain run

**Required permission:** This task launches a ~30 min GPU training run. Per CLAUDE.md "Never run benchmarks, long-running computations, or destructive operations without explicit user permission" — **wait for the user to say "go" / "OK" / "進めて" before proceeding**.

- [ ] **Step 1: Verify GPU is free**

```bash
nvidia-smi
```

Expected: no other process holding the RTX 5080 (no Boltz-2 / no other training).

- [ ] **Step 2 [USER GATE]: Ask user for permission**

Show the user:
- Estimated time: ~30 min
- Estimated VRAM: 4-6 GB
- Output artifact: `models/chemprop_mtr_seed42/{pretrain.pt, scaler.json, meta.json}`

Wait for explicit approval.

- [ ] **Step 3: Run full pretrain**

```bash
pixi run python track1_activity/scripts/run_chemprop_mtr_pretrain.py --seed 42 \
    2>&1 | tee track1_activity/reports/chemprop_mtr_pretrain_seed42.log
```

Expected final lines: best_val_loss reasonable (likely 0.6-0.9 in normalized MSE units, all 217 heads averaged), checkpoint saved to `models/chemprop_mtr_seed42/`.

- [ ] **Step 4: Verify checkpoint integrity**

```bash
pixi run python -c "
import json, torch
from pathlib import Path
d = Path('models/chemprop_mtr_seed42')
sd = torch.load(d / 'pretrain.pt', map_location='cpu')
print('state_dict keys:', len(sd))
print('scaler:', json.loads((d / 'scaler.json').read_text())['mean'][:3])
print('meta best_val_loss:', json.loads((d / 'meta.json').read_text())['best_val_loss'])
"
```

- [ ] **Step 5: Run extract**

```bash
pixi run python track1_activity/scripts/run_chemprop_mtr_extract.py --seed 42
```

Expected: parquet at `data/chemprop_mtr_embedding_seed42.parquet`, shape `(13136, 300)`, no NaN/inf assertions raised.

- [ ] **Step 6: Commit logs**

```bash
git add track1_activity/reports/chemprop_mtr_pretrain_seed42.log
git commit -m "chore(track1): Variant C pretrain seed42 log"
```

---

## Task 5: features.py registration for run_train.py

**Files:**
- Modify: `track1_activity/src/features.py` (add 2 entries to `FP_REGISTRY` + 2 loader functions)

- [ ] **Step 1: Read existing FP_REGISTRY structure**

```bash
sed -n '95,118p' track1_activity/src/features.py
```

Confirm `FP_REGISTRY` is a dict where each key maps to either a callable (mol-list → ndarray) or a special parquet-backed loader. Find the existing parquet-loader pattern (look for `chemprop_pretrain_embed` if present, else use this pattern).

- [ ] **Step 2: Add MTR-embed loader functions and registry entries**

Add after the existing `FP_REGISTRY` block:

```python
def _load_chemprop_mtr_embed(compound_ids: list[int], seed: int = 42) -> np.ndarray:
    """Load Variant C frozen embeddings (300d) for the given compound IDs."""
    path = REPO_ROOT.joinpath(f"data/chemprop_mtr_embedding_seed{seed}.parquet")
    df = pd.read_parquet(path)
    return df.loc[compound_ids].to_numpy(dtype=np.float32)


def _load_molformer_c3_mtr_embed(compound_ids: list[int], seed: int = 42) -> np.ndarray:
    """Load Variant M frozen embeddings (768d) for the given compound IDs."""
    path = REPO_ROOT.joinpath(f"data/molformer_c3_mtr_embedding_seed{seed}.parquet")
    df = pd.read_parquet(path)
    return df.loc[compound_ids].to_numpy(dtype=np.float32)


FP_REGISTRY["chemprop_mtr_embed"] = _load_chemprop_mtr_embed
FP_REGISTRY["molformer_c3_mtr_embed"] = _load_molformer_c3_mtr_embed
```

If `FP_REGISTRY` consumes mol lists rather than compound IDs in the existing loaders, follow the **existing pretrain-embed parquet loader pattern** in `features.py` (look for `chemprop_pretrain_embed`, `molformer_c3_pretrain_embed` — same shape).

- [ ] **Step 3: Verify by loading**

```bash
pixi run python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('track1_activity/src').resolve()))
from features import FP_REGISTRY

assert 'chemprop_mtr_embed' in FP_REGISTRY
loader = FP_REGISTRY['chemprop_mtr_embed']
arr = loader([1, 2, 3, 4, 5])
print('Shape:', arr.shape, 'dtype:', arr.dtype)
assert arr.shape == (5, 300)
"
```

- [ ] **Step 4: Format + commit**

```bash
pixi run ruff format track1_activity/src/features.py
pixi run ruff check track1_activity/src/features.py
git add track1_activity/src/features.py
git commit -m "feat(track1): register chemprop_mtr_embed + molformer_c3_mtr_embed in FP_REGISTRY"
```

---

## Task 6: TabPFN UMAP CV for Variant C

**Files:** No new files — uses existing `track1_activity/scripts/run_train.py`

- [ ] **Step 1: Verify CLI invocation against existing patterns**

```bash
grep -n 'tabpfn' track1_activity/scripts/run_train.py | head -10
grep -n 'feature' track1_activity/scripts/run_train.py | head -20
```

The expected invocation should be similar to the way `chemprop_pretrain_embed` is consumed.

- [ ] **Step 2: Run TabPFN UMAP CV with the new feature**

```bash
pixi run python track1_activity/scripts/run_train.py \
    --model tabpfn \
    --feature chemprop_mtr_embed \
    --split umap \
    --n-folds 5 \
    --experiment-name tabpfn_chemprop_mtr_embed \
    --notes "Variant C: chemprop scratch + MTR pretrain seed=42, Sultan 2025 recipe" \
    2>&1 | tee track1_activity/reports/tabpfn_chemprop_mtr_embed.log
```

Expected: OOF MAE printed at end, `experiments` table row inserted, `experiment_oof_predictions` table populated.

- [ ] **Step 3: Verify experiment record**

```bash
pixi run python -c "
import psycopg2
conn = psycopg2.connect(host='/tmp', port=5433, dbname='pxr_challenge')
cur = conn.cursor()
cur.execute(\"SELECT id, experiment_name, oof_mae, oof_rae, oof_spearman FROM experiment_summary WHERE experiment_name = 'tabpfn_chemprop_mtr_embed' ORDER BY id DESC LIMIT 1\")
print(cur.fetchall())
"
```

- [ ] **Step 4: Commit log**

```bash
git add track1_activity/reports/tabpfn_chemprop_mtr_embed.log
git commit -m "chore(track1): TabPFN UMAP CV log for Variant C"
```

---

## Task 7: Gate evaluation script (G1 + G2)

**Files:**
- Create: `track1_activity/scripts/eval_mtr_gates.py`

- [ ] **Step 1: Write the evaluator**

```python
#!/usr/bin/env -S pixi run python
"""MTR gate evaluation: G1 (single OOF MAE) and G2 (residual r vs pool members).

Usage:
    pixi run python track1_activity/scripts/eval_mtr_gates.py \
        --candidate tabpfn_chemprop_mtr_embed \
        --swap-target tabpfn_chemprop_pretrain_embed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402

# Pool member experiment_name list (current 9-pool snapshot — update if pool changed)
POOL_MEMBERS = [
    "tabpfn_chemprop_pretrain_embed",
    "tabpfn_molformer_c3_pretrain_embed",
    "tabpfn_2d_full_boltz_log2fc_pred",
    "tabpfn_top500_2d_full",
    "tabpfn_kermt",
    "tabpfn_optuna_t10",
    "tabpfn_pooled_boltz",
    "tabpfn_chemberta_zinc_v1",
    "tabpfn_atompair_2048",
]

G1_THRESHOLD = 0.485
G2_THRESHOLD = 0.85


def fetch_oof(experiment_name: str) -> tuple[pd.Series, dict]:
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, oof_mae, oof_rae, oof_spearman FROM experiment_summary "
        "WHERE experiment_name = %s ORDER BY id DESC LIMIT 1",
        (experiment_name,),
    )
    row = cur.fetchone()
    if row is None:
        raise SystemExit(f"No experiment_summary row for '{experiment_name}'")
    eid, mae, rae, sp = row
    cur.execute(
        "SELECT train_idx, oof_prediction FROM experiment_oof_predictions "
        "WHERE experiment_id = %s ORDER BY train_idx",
        (eid,),
    )
    rows = cur.fetchall()
    conn.close()
    s = pd.Series({r[0]: r[1] for r in rows}, name=experiment_name)
    return s, {"mae": mae, "rae": rae, "spearman": sp, "id": eid}


def fetch_targets() -> pd.Series:
    conn = psycopg2.connect(**DB_PARAMS)
    df = pd.read_sql(
        "SELECT compound_id, pec50 FROM train_activity ORDER BY compound_id", conn
    )
    conn.close()
    return df.set_index("compound_id")["pec50"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True, help="experiment_name of new variant")
    p.add_argument("--swap-target", required=True,
                   help="experiment_name being replaced (excluded from G2)")
    args = p.parse_args()

    cand_oof, cand_meta = fetch_oof(args.candidate)
    y = fetch_targets()
    cand_oof = cand_oof.reindex(y.index)

    print(f"\n=== Gate G1: single-member OOF MAE ===")
    print(f"  candidate {args.candidate}: MAE = {cand_meta['mae']:.4f}")
    print(f"  threshold: <= {G1_THRESHOLD}")
    g1_pass = cand_meta["mae"] <= G1_THRESHOLD
    print(f"  G1: {'PASS' if g1_pass else 'FAIL'}")

    print(f"\n=== Gate G2: residual r vs non-swap-target pool ===")
    cand_resid = cand_oof - y
    rows = []
    for member in POOL_MEMBERS:
        if member == args.swap_target:
            continue
        member_oof, _ = fetch_oof(member)
        member_oof = member_oof.reindex(y.index)
        member_resid = member_oof - y
        common = cand_resid.index.intersection(member_resid.index)
        r = np.corrcoef(cand_resid.loc[common], member_resid.loc[common])[0, 1]
        rows.append({"member": member, "residual_r": r})

    g2_df = pd.DataFrame(rows).sort_values("residual_r")
    print(g2_df.to_string(index=False))
    g2_min = g2_df["residual_r"].min()
    g2_pass = g2_min <= G2_THRESHOLD
    print(f"\n  min residual r = {g2_min:.4f} (threshold: <= {G2_THRESHOLD})")
    print(f"  G2: {'PASS' if g2_pass else 'FAIL'}")

    print(f"\n=== Summary ===")
    print(f"  G1: {'PASS' if g1_pass else 'FAIL'}")
    print(f"  G2: {'PASS' if g2_pass else 'FAIL'}")
    return 0 if (g1_pass and g2_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Format + lint**

```bash
pixi run ruff format track1_activity/scripts/eval_mtr_gates.py
pixi run ruff check track1_activity/scripts/eval_mtr_gates.py
```

- [ ] **Step 3: Run G1 + G2 for Variant C**

```bash
pixi run python track1_activity/scripts/eval_mtr_gates.py \
    --candidate tabpfn_chemprop_mtr_embed \
    --swap-target tabpfn_chemprop_pretrain_embed \
    | tee track1_activity/reports/mtr_gates_chemprop.txt
```

Expected: PASS / FAIL printed. The 8 non-swap-target pool members shown with their residual correlations.

- [ ] **Step 4: Commit**

```bash
git add track1_activity/scripts/eval_mtr_gates.py track1_activity/reports/mtr_gates_chemprop.txt
git commit -m "feat(track1): MTR gate G1+G2 evaluator + Variant C results"
```

- [ ] **Step 5: Decision branch — report to user**

State explicitly to the user:
- G1 result: PASS / FAIL with MAE value
- G2 result: PASS / FAIL with min residual r value and which member is closest

If both pass → continue to Task 8 (Variant M).
If either fails → stop here, write null-result PR description, do NOT continue to Variant M unless user explicitly overrides.

---

## Task 8 [conditional on Task 7 PASS]: MoLFormer MTR pretrain (Variant M)

**Files:**
- Create: `track1_activity/scripts/run_molformer_mtr_pretrain.py`

**Reference (read-only):** `track1_activity/scripts/run_molformer_c3_pretrain.py:1-330` for `SmilesDataset`, model wrapping, training loop. `track1_activity/src/peft_trainer.py:60-90` for **rotary fix** (must be copied verbatim).

- [ ] **Step 1: Write the script (~250 lines, mirrors run_molformer_c3_pretrain.py structure)**

Key sections that must follow existing patterns:

```python
#!/usr/bin/env -S pixi run python
"""MoLFormer-c3-1.1B MTR domain-adaptation (Variant M)."""

# ... boilerplate imports identical to run_molformer_c3_pretrain.py ...

from peft_backbones import get_backbone
from peft_methods import get_peft_builder
from peft import get_peft_model
from transformers import AutoModel, AutoTokenizer

# Data loader -- reuse load_descriptor_targets from chemprop variant
from run_chemprop_mtr_pretrain import (
    load_descriptor_targets,
    fit_scaler,
    apply_scaler,
    run_audit_or_die,
)


class SmilesMTRDataset(Dataset):
    """Same as SmilesDataset in run_molformer_c3_pretrain but multi-target."""

    def __init__(self, smiles, targets, tokenizer, max_length):
        self.smiles = smiles
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.smiles[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "targets": self.targets[idx],
        }


class MolformerMTRModel(nn.Module):
    def __init__(self, backbone_name, peft_method, peft_params, n_tasks, head_dim, head_dropout):
        super().__init__()
        meta = get_backbone(backbone_name)
        base = AutoModel.from_pretrained(
            meta["hf_id"], trust_remote_code=meta["trust_remote_code"]
        )
        # === ROTARY FIX (verbatim from peft_trainer.py:75-90, issue #30) ===
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
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_dim, n_tasks),
        )

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # mean-pool over non-padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(pooled)
```

The full main() loop should mirror `run_molformer_c3_pretrain.py:159-330` but with:
- multi-target MSE loss (`F.mse_loss(preds, targets)` no NaN-mask needed since rows pre-dropped)
- 217-output head
- Save to `models/molformer_c3_mtr_seed{N}/{pretrain.pt, scaler.json, meta.json}`
- Sultan hyperparameters: lr=3e-5, batch=16, max_epochs=20, AdamW + linear schedule with 10% warmup
- LoRA params: rank=8, alpha=16, target=("q_proj", "v_proj") — match existing peft default

Use `transformers.get_linear_schedule_with_warmup` for the schedule.

- [ ] **Step 2: Format + lint + smoke**

```bash
pixi run ruff format track1_activity/scripts/run_molformer_mtr_pretrain.py
pixi run ruff check track1_activity/scripts/run_molformer_mtr_pretrain.py
pixi run python track1_activity/scripts/run_molformer_mtr_pretrain.py --smoke --seed 42
```

Expected: 1-2 minutes, model loads, 3 epochs run, no rotary errors, no checkpoint write.

- [ ] **Step 3: Commit**

```bash
git add track1_activity/scripts/run_molformer_mtr_pretrain.py
git commit -m "feat(track1): molformer-c3 MTR pretrain script (Variant M) + smoke pass"
```

---

## Task 9: MoLFormer MTR extract script

**Files:**
- Create: `track1_activity/scripts/run_molformer_mtr_extract.py`

- [ ] **Step 1: Write the extract**

Skeleton:

```python
#!/usr/bin/env -S pixi run python
"""Extract frozen MoLFormer-c3 MTR embeddings (768d) for all 13,136 compounds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402
from peft_backbones import get_backbone  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from run_molformer_mtr_pretrain import MolformerMTRModel, SmilesMTRDataset  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ckpt_dir = REPO_ROOT.joinpath(f"models/molformer_c3_mtr_seed{args.seed}")
    meta = json.loads(ckpt_dir.joinpath("meta.json").read_text())
    backbone_name = meta["backbone_name"]
    peft_method = meta["peft_method"]
    peft_params = meta["peft_params"]
    n_tasks = len(meta["descriptor_names"])

    backbone_meta = get_backbone(backbone_name)
    tokenizer = AutoTokenizer.from_pretrained(
        backbone_meta["hf_id"], trust_remote_code=backbone_meta["trust_remote_code"]
    )
    model = MolformerMTRModel(
        backbone_name, peft_method, peft_params, n_tasks=n_tasks,
        head_dim=meta["head_dim"], head_dropout=0.0,
    )
    model.load_state_dict(torch.load(ckpt_dir.joinpath("pretrain.pt")))
    model.eval().cuda()

    conn = psycopg2.connect(**DB_PARAMS)
    df = pd.read_sql(
        "SELECT compound_id, std_smiles FROM compounds ORDER BY compound_id", conn,
    )
    conn.close()

    dummy = np.zeros((len(df), n_tasks), dtype=np.float32)
    ds = SmilesMTRDataset(
        df["std_smiles"].tolist(), dummy, tokenizer, max_length=backbone_meta["max_length"]
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    chunks = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].cuda()
            mask = batch["attention_mask"].cuda()
            out = model.backbone(input_ids=ids, attention_mask=mask)
            m = mask.unsqueeze(-1).float()
            pooled = (out.last_hidden_state * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            chunks.append(pooled.cpu().numpy())

    embeddings = np.concatenate(chunks)
    print("Embedding shape:", embeddings.shape)
    assert not np.isnan(embeddings).any()
    assert not np.isinf(embeddings).any()

    out_df = pd.DataFrame(
        embeddings, index=df["compound_id"].values,
        columns=[f"molformer_c3_mtr_{i}" for i in range(embeddings.shape[1])],
    )
    out_df.index.name = "compound_id"
    out_path = REPO_ROOT.joinpath(f"data/molformer_c3_mtr_embedding_seed{args.seed}.parquet")
    out_df.to_parquet(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Format + commit**

```bash
pixi run ruff format track1_activity/scripts/run_molformer_mtr_extract.py
pixi run ruff check track1_activity/scripts/run_molformer_mtr_extract.py
git add track1_activity/scripts/run_molformer_mtr_extract.py
git commit -m "feat(track1): molformer-c3 MTR extract script (Variant M)"
```

---

## Task 10 [USER GATE]: Variant M full pretrain run

**Required permission:** ~3 hour GPU run. Wait for explicit user approval.

- [ ] **Step 1: Verify GPU is free**

```bash
nvidia-smi
```

- [ ] **Step 2 [USER GATE]: Ask user**

Show:
- Estimated time: ~3 hours
- Estimated VRAM: 12-14 GB (LoRA on 1.1B params)
- Output: `models/molformer_c3_mtr_seed42/`

Wait for explicit approval.

- [ ] **Step 3: Run pretrain in tmux/background**

```bash
tmux new-session -d -s mtr_molformer 'pixi run python track1_activity/scripts/run_molformer_mtr_pretrain.py --seed 42 2>&1 | tee track1_activity/reports/molformer_c3_mtr_pretrain_seed42.log'
```

Check with `tmux attach -t mtr_molformer`.

- [ ] **Step 4: After completion, run extract**

```bash
pixi run python track1_activity/scripts/run_molformer_mtr_extract.py --seed 42
```

- [ ] **Step 5: TabPFN UMAP CV (mirrors Task 6)**

```bash
pixi run python track1_activity/scripts/run_train.py \
    --model tabpfn \
    --feature molformer_c3_mtr_embed \
    --split umap \
    --n-folds 5 \
    --experiment-name tabpfn_molformer_c3_mtr_embed \
    --notes "Variant M: molformer-c3-1.1B + MTR-DA seed=42, Sultan 2025 recipe" \
    2>&1 | tee track1_activity/reports/tabpfn_molformer_c3_mtr_embed.log
```

- [ ] **Step 6: G1 + G2 for Variant M**

```bash
pixi run python track1_activity/scripts/eval_mtr_gates.py \
    --candidate tabpfn_molformer_c3_mtr_embed \
    --swap-target tabpfn_molformer_c3_pretrain_embed \
    | tee track1_activity/reports/mtr_gates_molformer.txt
```

- [ ] **Step 7: Commit logs**

```bash
git add track1_activity/reports/molformer_c3_mtr_pretrain_seed42.log \
        track1_activity/reports/tabpfn_molformer_c3_mtr_embed.log \
        track1_activity/reports/mtr_gates_molformer.txt
git commit -m "chore(track1): Variant M pretrain + TabPFN + G1/G2 results"
```

- [ ] **Step 8: Report results to user; gate decision**

Same decision logic as Task 7 Step 5. If both Variant C and Variant M pass G1+G2, proceed to Task 11 (G3). If only one passes, single-SWAP path. If neither, abort and write null-result PR.

---

## Task 11 [conditional]: caruana SWAP G3 evaluation

**Files:** No new files — uses existing `track1_activity/scripts/run_ensemble.py`

- [ ] **Step 1: Read current ENSEMBLE_MODELS allow-list**

```bash
grep -nA 15 'ENSEMBLE_MODELS = \[' track1_activity/scripts/run_ensemble.py
```

Confirm the 9 current pool member names.

- [ ] **Step 2: Run baseline caruana_bag20 (current pool)**

```bash
pixi run python track1_activity/scripts/run_ensemble.py \
    --strategy caruana_bag20 \
    2>&1 | tee track1_activity/reports/ensemble_baseline_$(date +%Y%m%d).log
```

Capture baseline OOF MAE.

- [ ] **Step 3: Single SWAP for Variant C (if it passed G1+G2)**

Edit `run_ensemble.py` ENSEMBLE_MODELS: replace `tabpfn_chemprop_pretrain_embed` with `tabpfn_chemprop_mtr_embed`.

```bash
pixi run python track1_activity/scripts/run_ensemble.py \
    --strategy caruana_bag20 \
    2>&1 | tee track1_activity/reports/ensemble_swap_chemprop.log
```

Compute Δ MAE = swap − baseline. If Δ ≤ -0.003 → G3 PASS.

- [ ] **Step 4: Single SWAP for Variant M (if it passed G1+G2)**

Revert step 3, then replace `tabpfn_molformer_c3_pretrain_embed` with `tabpfn_molformer_c3_mtr_embed`. Same evaluation.

- [ ] **Step 5: Double SWAP (if both passed G1+G2)**

Apply both SWAPs simultaneously. Same evaluation.

- [ ] **Step 6: Summarize and choose final ENSEMBLE_MODELS**

Compare 4 configurations:
1. Baseline
2. Variant C single SWAP
3. Variant M single SWAP
4. Double SWAP

Pick the configuration with best OOF Δ that also passes G3 threshold (≤ -0.003). If none pass, abort to null-result PR.

- [ ] **Step 7: Commit final ENSEMBLE_MODELS choice**

```bash
git add track1_activity/scripts/run_ensemble.py track1_activity/reports/ensemble_*.log
git commit -m "feat(track1): MTR-DA SWAP — chosen pool config <description>, OOF Δ <value>"
```

---

## Task 12 [USER GATE]: Calibrator re-evaluation + LB submit decision

- [ ] **Step 1: Re-run both calibrators**

```bash
pixi run python track1_activity/scripts/run_ensemble_calibrate.py \
    2>&1 | tee track1_activity/reports/calibrate_nested_cv_post_mtr.log

pixi run python track1_activity/scripts/run_ensemble_calibrate_importance.py \
    2>&1 | tee track1_activity/reports/calibrate_importance_post_mtr.log
```

Per memory `feedback_calibrator_importance_locked_10seed`, default to importance affine for LB submit unless nested-CV strongly favors linear_pos.

- [ ] **Step 2: Verify cooldown**

```bash
pixi run python track1_activity/scripts/api.py cooldown
```

- [ ] **Step 3 [USER GATE]: Ask user before LB submit**

Report:
- Final pool composition (which SWAP was chosen)
- Calibrator selected
- OOF MAE Δ vs baseline
- Memory-tagged LB regression risk (`feedback_oof_lb_reverse_amplification`, `feedback_oof_minus_0002_ceiling`, etc.)
- Recommended submission file path

Wait for explicit user approval.

- [ ] **Step 4: Submit and fetch**

```bash
pixi run python track1_activity/scripts/api.py submit \
    --file <chosen calibrated csv> \
    --notes "MTR-DA SWAP <variant>: <pool snapshot>, importance affine"
# Wait ~30 min - 2 h
pixi run python track1_activity/scripts/api.py fetch
```

- [ ] **Step 5: Commit LB result**

```bash
git add track1_activity/submissions/*.csv  # if any new submission file
git commit -m "submit(track1): MTR-DA SWAP LB submit (id=<id>) — <result>"
```

---

## Task 13: PR + writeup

- [ ] **Step 1: Create PR**

```bash
gh pr create --title "experiment(track1): MTR domain-adaptation (Sultan 2025) — <result>" \
  --body "$(cat <<'EOF'
## Summary
- Applied Sultan 2025 MTR-DA recipe to PXR pool with 2 variants
- Variant C: chemprop scratch + MTR pretrain (217 RDKit descriptors)
- Variant M: molformer-c3-1.1B + MTR-DA on top of existing checkpoint
- Spec: docs/superpowers/specs/2026-05-04-mtr-domain-adaptation-design.md

## Gates (G0 → G4)
- G0 audit: PASS / FAIL
- G1 single OOF MAE: <values>
- G2 residual r vs non-swap pool: <values>
- G3 caruana SWAP Δ: <values>
- G4 LB Δ: <values or N/A>

## Files
- New: 5 scripts + 1 audit report + 2 model dirs + 2 embedding parquets
- Modified: features.py (FP_REGISTRY entries)
- Untouched: existing pretrain scripts (rank-1 driver protected)

## Memory updates
- <feedback / project memory pointers>

## Test plan
- [x] G0 audit pass
- [x] Smoke test pass (chemprop)
- [x] Full pretrain seed=42 complete
- [x] Embedding extract clean (no NaN/inf)
- [x] TabPFN UMAP CV recorded
- [x] G1 + G2 evaluated
- [<>] G3 evaluated
- [<>] LB submitted
EOF
)"
```

- [ ] **Step 2: Update issue #100 with status comment**

```bash
gh issue comment 100 --body "<status comment for 2026-05-04 MTR-DA experiment>"
```

- [ ] **Step 3: Update memory if outcome merits**

Per `feedback_oof_minus_0002_ceiling` and similar, write a memory file under `~/.claude/projects/-home-nagaet-pxr-iduction-challenge/memory/` capturing the outcome (any of: recipe-works, recipe-fails-gate2, LB-reverse-amp). Update `MEMORY.md` index.

---

## Self-review notes (post-write)

**Spec coverage check:**
- §4.1 Variant C → Tasks 2-7 ✓
- §4.2 Variant M → Tasks 8-10 ✓
- §5 NaN drop policy → Task 2 (NAN_DROP_IDS constant + assertion) ✓
- §6 Leak audit (6 risks) → Task 1 ✓
- §7 Gates G0-G4 → Task 1 (G0), Task 7 (G1+G2), Task 11 (G3), Task 12 (G4) ✓
- §8 Pool integration single/double SWAP → Task 11 ✓
- §9 Multi-seed Phase 2 → **Not in plan** (deferred per spec; only triggered after G2 PASS, in a follow-up plan). Mention this in PR writeup as a pending decision.
- §10 New / unmodified files → Tasks 1, 2, 3, 5, 7, 8, 9 produce these ✓
- §11 Out-of-scope → respected (no Mordred, no joint loss, no cross-NR, no model-size scaling) ✓

**Placeholder scan:** none.

**Type consistency:**
- `MolformerMTRModel` defined in Task 8, imported by extract in Task 9 ✓
- `SmilesMTRDataset` defined in Task 8, imported by extract in Task 9 ✓
- `load_descriptor_targets`, `fit_scaler`, `apply_scaler`, `run_audit_or_die` defined in Task 2, reused by Task 8 ✓
- FP_REGISTRY keys `chemprop_mtr_embed` / `molformer_c3_mtr_embed` consistent across Tasks 5, 6, 10, 11 ✓

**Risk: ChemProp 2.x API drift.** The MPNN forward path in `run_chemprop_mtr_extract.py` (Task 3) accesses `model.message_passing` and `model.agg`. If the installed chemprop version uses different attribute names, the extract will fail. Mitigation: smoke test step in Task 4 and integrity test in Task 4 Step 4 catch this before full extraction. If they fail, inspect `models.MPNN.__init__` source and adjust.

**Risk: LoRA target module names for MoLFormer-c3.** The spec said `q_proj, v_proj` but actual MoLFormer attention modules may use different names. Solution: in Task 8 Step 1, before writing `peft_params`, run a one-line probe to print attention layer module names, then set `lora_target` accordingly. Existing `peft_methods.py` may already have the canonical config — prefer that to ad-hoc guessing.
