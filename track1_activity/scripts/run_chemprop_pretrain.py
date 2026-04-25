#!/usr/bin/env -S pixi run python
"""ChemProp backbone pretraining on single-concentration log2_fc targets.

Phase 1 of the pretrain + fine-tune pipeline:
  * Uses all 13,136 compounds (compounds.std_smiles). test SMILES are
    included transductively -- they contribute no gradient (aux = NaN)
    but the encoder sees their structures during message passing.
  * Two regression heads:
      1. log2_fc @ 8.25 uM (10,752 labels, r=0.72 with pEC50)
      2. log2_fc @ 33 uM   (9,527 labels,  r=0.50 with pEC50)
  * NaN on either target is masked by ChemProp's training_step via
    ``targets.isfinite()`` (same mechanism as run_chemprop_multitask.py).
  * 90/10 random split within pretrain for early stopping and best-
    checkpoint selection.

Output: state_dict of the trained MPNN saved to
``track1_activity/checkpoints/chemprop_pretrain/pretrain.pt`` plus a
metadata JSON with pretrain architecture + target statistics. The
fine-tune script (run_chemprop_finetune.py) consumes these.

Usage:
    pixi run python track1_activity/scripts/run_chemprop_pretrain.py

Design rationale: Joint multitask WITH pEC50 failed (#82 extension --
gradient competition starves pEC50 head). Pretrain-first separates the
two phases so the encoder can commit to "what makes PXR compounds
activate" via log2_fc, then the fine-tune phase learns pEC50 on a
warm-start backbone without the aux-gradient interference.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from chemprop import data as chemprop_data  # noqa: E402
from chemprop import models, nn  # noqa: E402
from chemprop.nn.metrics import MSE  # noqa: E402

from data import DB_PARAMS  # noqa: E402

torch.set_float32_matmul_precision("medium")


CKPT_BASE = REPO_ROOT.joinpath("track1_activity", "checkpoints")


# Default architecture mirrors run_chemprop_multitask's TUNED_PARAMS
# (best single-task chemprop Optuna config, OOF RAE 0.5724).
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
    "max_epochs": 150,
    "patience": 15,
}


AGG_REGISTRY = {
    "mean": nn.MeanAggregation,
    "sum": nn.SumAggregation,
    "norm": nn.NormAggregation,
}


def load_pretrain_data() -> tuple[list[str], np.ndarray, list[int]]:
    """Fetch (SMILES, targets, compound_ids) for all 13k compounds.

    Targets have 2 columns: [log2fc_8p25, log2fc_33]. NaN where the
    compound has no measurement at that concentration.
    """
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

    smiles = df["smiles"].tolist()
    targets = df[["log2fc_8p25", "log2fc_33"]].to_numpy(dtype=np.float32)
    cids = df["compound_id"].astype(int).tolist()
    return smiles, targets, cids


def build_pretrain_model(params: dict, task_weights: torch.Tensor):
    mp = nn.BondMessagePassing(
        d_h=params["message_hidden_dim"],
        depth=params["depth"],
        dropout=params["mp_dropout"],
        activation=params["activation"],
    )
    agg = AGG_REGISTRY[params["aggregation"]]()
    criterion = MSE(task_weights=task_weights)
    ffn = nn.RegressionFFN(
        n_tasks=int(task_weights.numel()),
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


def make_dataloader(smiles, targets, batch_size, shuffle):
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
    parser = argparse.ArgumentParser(description="ChemProp backbone pretrain")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument(
        "--val-frac", type=float, default=0.1, help="fraction for pretrain val"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--w-8p25",
        type=float,
        default=1.0,
        help="Loss weight on 8.25uM head (primary, r=0.72 with pEC50)",
    )
    parser.add_argument(
        "--w-33",
        type=float,
        default=0.5,
        help="Loss weight on 33uM head (secondary, r=0.50)",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=None,
        help="Checkpoint output directory. Default: chemprop_pretrain (seed 42) "
        "or chemprop_pretrain_seed{N} for non-default seeds. Use this for "
        "multi-seed ensembles (Plan A).",
    )
    args = parser.parse_args()

    params = DEFAULT_PARAMS.copy()
    if args.max_epochs is not None:
        params["max_epochs"] = args.max_epochs

    # Resolve ckpt directory: explicit override, or seed-derived default
    if args.ckpt_dir is not None:
        ckpt_dir = args.ckpt_dir
    elif args.seed == 42:
        ckpt_dir = CKPT_BASE.joinpath("chemprop_pretrain")
    else:
        ckpt_dir = CKPT_BASE.joinpath(f"chemprop_pretrain_seed{args.seed}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ckpt_dir: {ckpt_dir}")

    print("ChemProp pretrain (13k compounds + single_conc 2-head)")
    print(f"  params: {params}")

    smiles, targets, cids = load_pretrain_data()
    n_total = len(smiles)
    n_valid_8 = int(np.isfinite(targets[:, 0]).sum())
    n_valid_33 = int(np.isfinite(targets[:, 1]).sum())
    print(
        f"  data: {n_total} compounds, "
        f"log2fc_8p25 valid={n_valid_8}, log2fc_33 valid={n_valid_33}"
    )

    # Z-score each target column (mask NaN during computation)
    means = np.zeros(2, dtype=np.float32)
    stds = np.ones(2, dtype=np.float32)
    for i in range(2):
        valid = np.isfinite(targets[:, i])
        means[i] = float(np.mean(targets[valid, i]))
        stds[i] = float(np.std(targets[valid, i]))
        if stds[i] < 1e-6:
            stds[i] = 1.0
        print(
            f"  target[{i}] (concentration "
            f"{'8.25uM' if i == 0 else '33uM'}): "
            f"mean={means[i]:+.3f}, std={stds[i]:.3f}"
        )

    # Standardise targets in place (NaN stays NaN)
    targets_z = (targets - means) / stds

    # Random val split (val_frac of n_total)
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

    # Task weights: main 8.25uM + secondary 33uM
    task_weights_np = np.asarray([args.w_8p25, args.w_33], dtype=np.float32)
    task_weights = torch.tensor(task_weights_np, dtype=torch.float32)
    print(f"  task_weights: 8.25uM={args.w_8p25:.3f}  33uM={args.w_33:.3f}")

    model = build_pretrain_model(params, task_weights)

    train_loader = make_dataloader(tr_smi, tr_y, params["batch_size"], shuffle=True)
    val_loader = make_dataloader(va_smi, va_y, params["batch_size"], shuffle=False)

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
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[early_stop, best_ckpt],
    )
    trainer.fit(model, train_loader, val_loader)

    # Persist the final model state_dict (flat .pt) plus metadata.
    # We save the FULL state_dict; the fine-tune loader picks out the
    # encoder keys (message_passing.*, agg.*, batch_norm-related) and
    # discards the predictor FFN (single-task head will be fresh).
    state_path = ckpt_dir.joinpath("pretrain.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "params": params,
            "target_means": means.tolist(),
            "target_stds": stds.tolist(),
            "task_weights": task_weights_np.tolist(),
            "n_train": len(tr_idx),
            "n_val": len(val_idx),
            "final_val_loss": float(trainer.callback_metrics.get("val_loss", -1)),
        },
        state_path,
    )
    print(f"  saved state_dict: {state_path}")

    # Separate JSON for quick inspection
    meta_path = ckpt_dir.joinpath("pretrain_meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "params": params,
                "target_means": means.tolist(),
                "target_stds": stds.tolist(),
                "task_weights_8p25_33": [args.w_8p25, args.w_33],
                "n_train_compounds": len(tr_idx),
                "n_val_compounds": len(val_idx),
                "n_target_valid": {
                    "8.25uM": n_valid_8,
                    "33uM": n_valid_33,
                },
                "final_val_loss": float(trainer.callback_metrics.get("val_loss", -1)),
                "best_ckpt_path": str(best_ckpt.best_model_path),
            },
            indent=2,
        )
    )
    print(f"  saved meta: {meta_path}")

    print("\n=== Pretrain done ===")
    print(f"  final val_loss: {trainer.callback_metrics.get('val_loss')}")
    print(f"  best ckpt (Lightning): {best_ckpt.best_model_path}")


if __name__ == "__main__":
    main()
