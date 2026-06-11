#!/usr/bin/env -S pixi run python
"""ChemProp pretrain on log2fc plus HTChem corrected pEC50.

This is a conservative HTChem use: keep the strong single-concentration
log2fc heads, add HTChem corrected pEC50 as a small third auxiliary head, and
use the resulting frozen encoder fingerprint as a downstream feature.
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
from lightning import pytorch as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "track1_activity" / "src"))
sys.path.insert(0, str(REPO_ROOT / "track1_activity" / "scripts"))

from chemprop import data as chemprop_data  # noqa: E402

import run_chemprop_pretrain as pretrain_mod  # noqa: E402
from data import DB_PARAMS  # noqa: E402

torch.set_float32_matmul_precision("medium")

DEFAULT_CKPT_DIR = (
    REPO_ROOT / "track1_activity" / "checkpoints" / "chemprop_log2fc_htchem_pretrain"
)
DEFAULT_EMBED_PATH = (
    REPO_ROOT / "data" / "chemprop_log2fc_htchem_pretrain_embed.parquet"
)
TARGET_NAMES = ["log2fc_8p25", "log2fc_33", "htchem_corrected_pec50"]


def load_pretrain_data() -> tuple[pd.DataFrame, np.ndarray]:
    sql = """
    SELECT
        c.id AS compound_id,
        COALESCE(c.std_smiles, c.smiles) AS smiles,
        sc.log2fc_8p25,
        sc.log2fc_33,
        ht.htchem_corrected_pec50
    FROM compounds c
    LEFT JOIN (
      SELECT compound_id,
        AVG(CASE WHEN concentration_m BETWEEN 8.2e-6 AND 8.3e-6
                 THEN log2_fc_estimate END) AS log2fc_8p25,
        AVG(CASE WHEN concentration_m BETWEEN 3.28e-5 AND 3.32e-5
                 THEN log2_fc_estimate END) AS log2fc_33
      FROM single_concentration
      GROUP BY compound_id
    ) sc ON sc.compound_id = c.id
    LEFT JOIN (
      SELECT compound_id,
             AVG(corrected_pec50) AS htchem_corrected_pec50
      FROM htchem_activity
      WHERE corrected_pec50 IS NOT NULL
      GROUP BY compound_id
    ) ht ON ht.compound_id = c.id
    WHERE COALESCE(c.std_smiles, c.smiles) IS NOT NULL
    ORDER BY c.id
    """
    with psycopg2.connect(**DB_PARAMS) as conn:
        df = pd.read_sql(sql, conn)
    targets = df[TARGET_NAMES].to_numpy(dtype=np.float32)
    return df, targets


def make_dataloader(
    smiles: list[str], targets: np.ndarray, batch_size: int, shuffle: bool
):
    pts = [
        chemprop_data.MoleculeDatapoint.from_smi(smi, np.asarray(y, dtype=np.float32))
        for smi, y in zip(smiles, targets)
    ]
    return chemprop_data.build_dataloader(
        chemprop_data.MoleculeDataset(pts),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def train(args: argparse.Namespace) -> Path:
    params = pretrain_mod.DEFAULT_PARAMS.copy()
    params["max_epochs"] = args.max_epochs
    params["patience"] = args.patience
    params["batch_size"] = args.batch_size
    ckpt_dir: Path = args.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    df, targets = load_pretrain_data()
    means = np.zeros(targets.shape[1], dtype=np.float32)
    stds = np.ones(targets.shape[1], dtype=np.float32)
    valid_counts: dict[str, int] = {}
    for i, name in enumerate(TARGET_NAMES):
        valid = np.isfinite(targets[:, i])
        valid_counts[name] = int(valid.sum())
        means[i] = float(np.mean(targets[valid, i]))
        stds[i] = float(np.std(targets[valid, i]))
        if stds[i] < 1e-6:
            stds[i] = 1.0
        print(
            f"target {name}: n={valid_counts[name]} mean={means[i]:+.4f} std={stds[i]:.4f}"
        )
    targets_z = (targets - means) / stds

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_val = int(len(df) * args.val_frac)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    train_loader = make_dataloader(
        df["smiles"].iloc[train_idx].tolist(),
        targets_z[train_idx],
        params["batch_size"],
        shuffle=True,
    )
    val_loader = make_dataloader(
        df["smiles"].iloc[val_idx].tolist(),
        targets_z[val_idx],
        params["batch_size"],
        shuffle=False,
    )

    task_weights_np = np.asarray(
        [args.w_8p25, args.w_33, args.w_htchem], dtype=np.float32
    )
    task_weights = torch.tensor(task_weights_np, dtype=torch.float32)
    print(
        "task weights: "
        + ", ".join(
            f"{name}={weight:.3f}"
            for name, weight in zip(TARGET_NAMES, task_weights_np)
        )
    )

    model = pretrain_mod.build_pretrain_model(params, task_weights)
    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=params["patience"], mode="min"
    )
    best_ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="pretrain_best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    trainer = pl.Trainer(
        max_epochs=params["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
        callbacks=[early_stop, best_ckpt],
    )
    trainer.fit(model, train_loader, val_loader)

    state_path = ckpt_dir / "pretrain.pt"
    payload = {
        "state_dict": model.state_dict(),
        "params": params,
        "target_names": TARGET_NAMES,
        "target_means": means.tolist(),
        "target_stds": stds.tolist(),
        "task_weights": task_weights_np.tolist(),
        "valid_counts": valid_counts,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "final_val_loss": float(trainer.callback_metrics.get("val_loss", -1)),
        "best_ckpt_path": str(best_ckpt.best_model_path),
    }
    torch.save(payload, state_path)
    meta = {key: value for key, value in payload.items() if key != "state_dict"}
    (ckpt_dir / "pretrain_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved checkpoint: {state_path}")
    return state_path


def load_target_compounds() -> pd.DataFrame:
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


def extract_embeddings(ckpt_path: Path, out_path: Path, batch_size: int) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    params = ckpt["params"]
    task_weights = torch.ones(len(ckpt["target_names"]), dtype=torch.float32)
    model = pretrain_mod.build_pretrain_model(params, task_weights)
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    df = load_target_compounds()
    pts = [
        chemprop_data.MoleculeDatapoint.from_smi(
            smi, np.zeros(len(TARGET_NAMES), dtype=np.float32)
        )
        for smi in df["smiles"]
    ]
    loader = chemprop_data.build_dataloader(
        chemprop_data.MoleculeDataset(pts),
        batch_size=batch_size,
        shuffle=False,
    )
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            bmg = batch.bmg
            bmg.to(device)
            v_d = batch.V_d.to(device) if batch.V_d is not None else None
            x_d = batch.X_d.to(device) if batch.X_d is not None else None
            chunks.append(model.fingerprint(bmg, v_d, x_d).detach().cpu().numpy())
    emb = np.concatenate(chunks, axis=0)
    out = pd.DataFrame(emb, columns=[f"emb_{i:03d}" for i in range(emb.shape[1])])
    out.insert(0, "compound_id", df["compound_id"].to_numpy(dtype=int))
    out = out.set_index("compound_id")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    print(f"saved embeddings: {out_path} shape={out.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--embed-out", type=Path, default=DEFAULT_EMBED_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--w-8p25", type=float, default=1.0)
    parser.add_argument("--w-33", type=float, default=0.5)
    parser.add_argument("--w-htchem", type=float, default=0.25)
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()

    ckpt_path = args.ckpt_dir / "pretrain.pt"
    if not args.extract_only:
        ckpt_path = train(args)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")
    extract_embeddings(ckpt_path, args.embed_out, args.batch_size)


if __name__ == "__main__":
    main()
