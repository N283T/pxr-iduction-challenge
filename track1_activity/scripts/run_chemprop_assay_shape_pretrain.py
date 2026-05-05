#!/usr/bin/env -S pixi run python
"""ChemProp pretraining on masked assay-shape auxiliary targets.

This is the heavier successor to the 2-head single-concentration pretrain.
Default targets intentionally exclude PXR pEC50 and PXR-counter deltas so that
frozen embeddings do not directly distill the supervised Track 1 label into OOF
features. Those label-derived heads can be enabled for diagnostic experiments
with ``--include-pxr-targets`` but should not be used for leaderboard OOF claims
without cross-fit pretraining.
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
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from chemprop import data as chemprop_data  # noqa: E402
from chemprop import models, nn  # noqa: E402
from chemprop.nn.metrics import MSE  # noqa: E402

from data import DB_PARAMS  # noqa: E402

from track1_activity.scripts.build_dose_response_latent import (  # noqa: E402
    build_assay_matrix,
)

torch.set_float32_matmul_precision("medium")

CKPT_BASE = REPO_ROOT.joinpath("track1_activity", "checkpoints")

ALL_TASKS = [
    "pxr_pec50",
    "pxr_emax",
    "pxr_emax_vs_pos_ctrl",
    "counter_present",
    "counter_pec50",
    "counter_emax",
    "counter_emax_vs_pos_ctrl",
    "pxr_minus_counter",
    "log2fc_8p25",
    "log2fc_33",
    "log2fc_99",
]

DEFAULT_TASKS = [
    "counter_present",
    "counter_pec50",
    "counter_emax",
    "counter_emax_vs_pos_ctrl",
    "log2fc_8p25",
    "log2fc_33",
    "log2fc_99",
]

COUNTER_EMAX_TASKS = [
    "counter_present",
    "counter_pec50",
    "counter_emax",
    "counter_emax_vs_pos_ctrl",
    "pxr_emax",
    "pxr_emax_vs_pos_ctrl",
]

DR_LATENT_PREFIX = "drlatent"

# Exported for tests. This includes all possible columns; the training CLI can
# select a subset.
TASKS = ALL_TASKS

DEFAULT_TASK_WEIGHTS = {
    "pxr_pec50": 0.0,
    "pxr_emax": 0.2,
    "pxr_emax_vs_pos_ctrl": 0.2,
    "counter_present": 0.2,
    "counter_pec50": 0.5,
    "counter_emax": 0.25,
    "counter_emax_vs_pos_ctrl": 0.25,
    "pxr_minus_counter": 0.0,
    "log2fc_8p25": 1.0,
    "log2fc_33": 0.5,
    "log2fc_99": 0.15,
}

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


def build_target_matrix(
    df: pd.DataFrame, tasks: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return target matrix and finite-observation mask for assay-shape tasks."""
    selected = TASKS if tasks is None else tasks
    work = pd.DataFrame(index=df.index)
    work["pxr_pec50"] = df.get("pxr_pec50", np.nan)
    work["pxr_emax"] = df.get("pxr_emax", np.nan)
    work["pxr_emax_vs_pos_ctrl"] = df.get("pxr_emax_vs_pos_ctrl", np.nan)
    work["counter_present"] = df.get("counter_present", np.nan)
    work["counter_pec50"] = df.get("counter_pec50", np.nan)
    work["counter_emax"] = df.get("counter_emax", np.nan)
    work["counter_emax_vs_pos_ctrl"] = df.get("counter_emax_vs_pos_ctrl", np.nan)
    work["pxr_minus_counter"] = work["pxr_pec50"] - work["counter_pec50"]
    work["log2fc_8p25"] = df.get("log2fc_8p25", np.nan)
    work["log2fc_33"] = df.get("log2fc_33", np.nan)
    work["log2fc_99"] = df.get("log2fc_99", np.nan)
    targets = work[selected].to_numpy(dtype=np.float32)
    return targets, np.isfinite(targets)


def standardize_targets(
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Z-score each task over observed rows while preserving NaNs."""
    means = np.zeros(targets.shape[1], dtype=np.float32)
    stds = np.ones(targets.shape[1], dtype=np.float32)
    counts = np.isfinite(targets).sum(axis=0).astype(np.int64)
    z = targets.copy()
    for i in range(targets.shape[1]):
        valid = np.isfinite(targets[:, i])
        if not valid.any():
            continue
        means[i] = float(np.mean(targets[valid, i]))
        std = float(np.std(targets[valid, i]))
        stds[i] = std if std >= 1e-6 else 1.0
        z[valid, i] = (targets[valid, i] - means[i]) / stds[i]
    return z, means, stds, counts


def build_dose_response_latent_targets(
    df: pd.DataFrame,
    n_components: int,
    min_observed: int,
    seed: int,
) -> tuple[np.ndarray, list[str], dict]:
    """Build observed dose-response latent targets from internal assay labels.

    The latent fit excludes PXR pEC50 by construction through
    build_assay_matrix(). Rows with too few observed auxiliary assay values are
    left as NaN so ChemProp's masked MSE ignores them.
    """
    assay = build_assay_matrix(df)
    values = assay.to_numpy(dtype=np.float32)
    counts = np.isfinite(values).sum(axis=1)
    fit_mask = counts >= min_observed
    n_fit = int(fit_mask.sum())
    n_components_eff = min(int(n_components), values.shape[1], n_fit - 1)
    if n_components_eff < 1:
        raise ValueError(
            f"Need at least two rows with >= {min_observed} observed assay values"
        )

    pipe = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
        PCA(n_components=n_components_eff, random_state=seed),
    )
    latent_fit = pipe.fit_transform(values[fit_mask]).astype(np.float32)
    targets = np.full((len(df), n_components_eff), np.nan, dtype=np.float32)
    targets[fit_mask] = latent_fit
    tasks = [f"{DR_LATENT_PREFIX}_{i:02d}" for i in range(n_components_eff)]
    meta = {
        "assay_columns": assay.columns.tolist(),
        "fit_rows": n_fit,
        "min_observed": int(min_observed),
        "n_components_requested": int(n_components),
        "n_components": int(n_components_eff),
        "explained_variance": [
            float(v) for v in pipe.named_steps["pca"].explained_variance_ratio_
        ],
    }
    return targets, tasks, meta


def load_assay_shape_data() -> pd.DataFrame:
    """Load all standardized compounds with optional assay-shape targets."""
    sql = """
    WITH sc AS (
      SELECT compound_id,
        AVG(CASE WHEN concentration_m BETWEEN 8.2e-6 AND 8.3e-6
                 THEN log2_fc_estimate END) AS log2fc_8p25,
        AVG(CASE WHEN concentration_m BETWEEN 3.28e-5 AND 3.32e-5
                 THEN log2_fc_estimate END) AS log2fc_33,
        AVG(CASE WHEN concentration_m BETWEEN 9.85e-5 AND 9.95e-5
                 THEN log2_fc_estimate END) AS log2fc_99
      FROM single_concentration
      GROUP BY compound_id
    )
    SELECT
      c.id AS compound_id,
      c.std_smiles AS smiles,
      t.pec50 AS pxr_pec50,
      t.emax_estimate AS pxr_emax,
      t.emax_vs_pos_ctrl AS pxr_emax_vs_pos_ctrl,
      CASE WHEN ca.compound_id IS NULL THEN NULL ELSE 1.0 END AS counter_present,
      ca.pec50 AS counter_pec50,
      ca.emax_estimate AS counter_emax,
      ca.emax_vs_pos_ctrl AS counter_emax_vs_pos_ctrl,
      sc.log2fc_8p25,
      sc.log2fc_33,
      sc.log2fc_99
    FROM compounds c
    LEFT JOIN train_activity t ON t.compound_id = c.id
    LEFT JOIN counter_assay ca ON ca.compound_id = c.id
    LEFT JOIN sc ON sc.compound_id = c.id
    WHERE c.std_smiles IS NOT NULL
    ORDER BY c.id
    """
    with psycopg2.connect(**DB_PARAMS) as conn:
        return pd.read_sql(sql, conn)


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


def make_dataloader(smiles, targets, batch_size: int, shuffle: bool):
    pts = [
        chemprop_data.MoleculeDatapoint.from_smi(smi, np.asarray(y, dtype=np.float32))
        for smi, y in zip(smiles, targets)
    ]
    return chemprop_data.build_dataloader(
        chemprop_data.MoleculeDataset(pts),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def resolve_tasks(
    include_pxr_targets: bool, counter_emax_only: bool, dose_response_latent: bool
) -> list[str]:
    if dose_response_latent:
        return []
    if counter_emax_only:
        return COUNTER_EMAX_TASKS
    if include_pxr_targets:
        return ALL_TASKS
    return DEFAULT_TASKS


def main() -> None:
    parser = argparse.ArgumentParser(description="ChemProp assay-shape pretrain")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--include-pxr-targets", action="store_true")
    parser.add_argument(
        "--counter-emax-only",
        action="store_true",
        help="Exclude log2fc heads and pretrain on counter/Emax assay-shape only.",
    )
    parser.add_argument(
        "--dose-response-latent",
        action="store_true",
        help="Pretrain on PCA latent targets from internal assay-shape labels.",
    )
    parser.add_argument(
        "--append-dose-response-latent",
        action="store_true",
        help="Append dose-response latent heads to the selected assay-shape heads.",
    )
    parser.add_argument("--latent-components", type=int, default=6)
    parser.add_argument("--latent-min-observed", type=int, default=2)
    parser.add_argument("--latent-weight", type=float, default=0.5)
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.counter_emax_only and args.dose_response_latent:
        raise SystemExit("--counter-emax-only and --dose-response-latent are exclusive")
    if args.include_pxr_targets and args.dose_response_latent:
        raise SystemExit(
            "--include-pxr-targets and --dose-response-latent are exclusive"
        )
    if args.dose_response_latent and args.append_dose_response_latent:
        raise SystemExit(
            "--dose-response-latent and --append-dose-response-latent are exclusive"
        )

    params = DEFAULT_PARAMS.copy()
    if args.max_epochs is not None:
        params["max_epochs"] = args.max_epochs

    tasks = resolve_tasks(
        args.include_pxr_targets, args.counter_emax_only, args.dose_response_latent
    )
    ckpt_dir = args.ckpt_dir or CKPT_BASE.joinpath(
        (
            f"chemprop_dose_response_latent_pretrain_seed{args.seed}"
            if args.dose_response_latent
            else (
                f"chemprop_assay_shape_drlatent_pretrain_seed{args.seed}"
                if args.append_dose_response_latent
                else f"chemprop_assay_shape_pretrain_seed{args.seed}"
            )
        )
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    df = load_assay_shape_data()
    latent_meta = None
    if args.dose_response_latent:
        targets, tasks, latent_meta = build_dose_response_latent_targets(
            df=df,
            n_components=args.latent_components,
            min_observed=args.latent_min_observed,
            seed=args.seed,
        )
        weights = np.ones(len(tasks), dtype=np.float32)
    else:
        targets, _mask = build_target_matrix(df, tasks)
        weights = np.asarray([DEFAULT_TASK_WEIGHTS[t] for t in tasks], dtype=np.float32)
        if args.append_dose_response_latent:
            latent_targets, latent_tasks, latent_meta = (
                build_dose_response_latent_targets(
                    df=df,
                    n_components=args.latent_components,
                    min_observed=args.latent_min_observed,
                    seed=args.seed,
                )
            )
            targets = np.concatenate([targets, latent_targets], axis=1)
            tasks = tasks + latent_tasks
            latent_weights = np.full(
                len(latent_tasks), float(args.latent_weight), dtype=np.float32
            )
            weights = np.concatenate([weights, latent_weights]).astype(np.float32)
    targets_z, means, stds, counts = standardize_targets(targets)
    if not args.include_pxr_targets and not args.counter_emax_only:
        if any(t.startswith("pxr_") or t == "pxr_minus_counter" for t in tasks):
            raise RuntimeError("default task set unexpectedly contains PXR targets")

    valid_rows = np.isfinite(targets_z).any(axis=1)
    smiles = df.loc[valid_rows, "smiles"].tolist()
    compound_ids = df.loc[valid_rows, "compound_id"].astype(int).tolist()
    targets_z = targets_z[valid_rows]

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(smiles))
    rng.shuffle(idx)
    n_val = int(len(idx) * args.val_frac)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_loader = make_dataloader(
        [smiles[i] for i in train_idx],
        targets_z[train_idx],
        params["batch_size"],
        shuffle=True,
    )
    val_loader = make_dataloader(
        [smiles[i] for i in val_idx],
        targets_z[val_idx],
        params["batch_size"],
        shuffle=False,
    )

    print("ChemProp assay-shape pretrain")
    print(f"  ckpt_dir: {ckpt_dir}")
    print(f"  tasks: {tasks}")
    if latent_meta is not None:
        print(f"  latent_meta: {latent_meta}")
    print(f"  valid rows: {len(smiles)} / {len(df)}")
    for task, count, mean, std, weight in zip(tasks, counts, means, stds, weights):
        print(
            f"  {task:24s} n={int(count):5d} mean={mean:+.4f} "
            f"std={std:.4f} weight={weight:.3f}"
        )

    model = build_pretrain_model(params, torch.tensor(weights, dtype=torch.float32))
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor="val_loss", patience=params["patience"], mode="min"
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="pretrain_best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
    ]
    trainer = pl.Trainer(
        max_epochs=params["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=callbacks,
    )
    trainer.fit(model, train_loader, val_loader)

    state_path = ckpt_dir.joinpath("pretrain.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "params": params,
            "tasks": tasks,
            "target_means": means.tolist(),
            "target_stds": stds.tolist(),
            "target_counts": counts.tolist(),
            "task_weights": weights.tolist(),
            "compound_ids": compound_ids,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "include_pxr_targets": bool(args.include_pxr_targets),
            "counter_emax_only": bool(args.counter_emax_only),
            "dose_response_latent": bool(args.dose_response_latent),
            "append_dose_response_latent": bool(args.append_dose_response_latent),
            "latent_meta": latent_meta,
            "final_val_loss": float(trainer.callback_metrics.get("val_loss", -1)),
        },
        state_path,
    )
    ckpt_callback = callbacks[1]
    meta = {
        "params": params,
        "tasks": tasks,
        "target_means": means.tolist(),
        "target_stds": stds.tolist(),
        "target_counts": counts.tolist(),
        "task_weights": dict(zip(tasks, weights.astype(float).tolist())),
        "n_valid_rows": len(smiles),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "include_pxr_targets": bool(args.include_pxr_targets),
        "counter_emax_only": bool(args.counter_emax_only),
        "dose_response_latent": bool(args.dose_response_latent),
        "append_dose_response_latent": bool(args.append_dose_response_latent),
        "latent_meta": latent_meta,
        "final_val_loss": float(trainer.callback_metrics.get("val_loss", -1)),
        "best_ckpt_path": str(ckpt_callback.best_model_path),
    }
    ckpt_dir.joinpath("pretrain_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved: {state_path}")


if __name__ == "__main__":
    main()
