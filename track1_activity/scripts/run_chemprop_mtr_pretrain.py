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
    "max_epochs": 50,  # Sultan recipe: 20-50 is enough for MTR DA
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
    with psycopg2.connect(**DB_PARAMS) as conn:
        desc_df = pd.read_sql(
            "SELECT cd.compound_id, c.std_smiles, cd.descriptors "
            "FROM compound_descriptors_full cd "
            "JOIN compounds c ON c.id = cd.compound_id "
            "ORDER BY cd.compound_id",
            conn,
        )

    desc_df = desc_df[~desc_df["compound_id"].isin(NAN_DROP_IDS)].reset_index(drop=True)
    expanded = pd.json_normalize(desc_df["descriptors"]).apply(
        pd.to_numeric, errors="coerce"
    )
    descriptor_names = expanded.columns.tolist()
    targets = expanded.to_numpy(dtype=np.float32)

    if np.isnan(targets).any():
        raise RuntimeError(
            "NaN remains after row-drop — audit may be stale. Re-run audit_mtr_leak.py."
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
    return {
        "mean": mean.astype(np.float64).tolist(),
        "std": std.astype(np.float64).tolist(),
    }


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
    criterion = MSE()
    ffn = nn.RegressionFFN(
        n_tasks=n_tasks,
        input_dim=mp.output_dim,
        hidden_dim=params["ffn_hidden_dim"],
        n_layers=params["ffn_num_layers"],
        dropout=params["ffn_dropout"],
        criterion=criterion,
    )
    return models.MPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        batch_norm=True,
        warmup_epochs=params["warmup_epochs"],
        init_lr=params["learning_rate"],
        max_lr=params["learning_rate"] * params["lr_ratio"],
        final_lr=params["learning_rate"] * 0.1,
    )


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke run: 3 epochs, 200 compounds, no checkpoint write",
    )
    args = p.parse_args()

    run_audit_or_die()
    pl.seed_everything(args.seed, workers=True)

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

    early_stop_cb = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=params["patience"], mode="min"
    )
    if not args.smoke:
        best_cb = pl.callbacks.ModelCheckpoint(
            dirpath=str(out_dir),
            filename="best_val",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
        callbacks = [early_stop_cb, best_cb]
    else:
        callbacks = [early_stop_cb]

    trainer = pl.Trainer(
        max_epochs=params["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="bf16-mixed",
        callbacks=callbacks,
        log_every_n_steps=20,
        enable_checkpointing=not args.smoke,
    )
    trainer.fit(model, train_loader, val_loader)

    if args.smoke:
        print("Smoke run completed; no checkpoint written.")
        return

    best_path = best_cb.best_model_path
    ckpt = torch.load(best_path, map_location="cpu")
    torch.save(ckpt["state_dict"], out_dir.joinpath("pretrain.pt"))
    Path(best_path).unlink()

    out_dir.joinpath("scaler.json").write_text(json.dumps(scaler, indent=2))
    best_val = best_cb.best_model_score
    out_dir.joinpath("meta.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "n_compounds": len(smiles),
                "descriptor_names": desc_names,
                "params": params,
                "best_val_loss": float(
                    best_val.item() if hasattr(best_val, "item") else best_val
                ),
                "final_val_loss": float(trainer.callback_metrics.get("val_loss", -1)),
            },
            indent=2,
        )
    )
    print(f"Saved {out_dir}")


if __name__ == "__main__":
    main()
