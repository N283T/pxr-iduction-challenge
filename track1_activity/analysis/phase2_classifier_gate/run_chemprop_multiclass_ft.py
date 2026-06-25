#!/usr/bin/env -S pixi run python
"""Fine-tune a ChemProp multiclass classifier for Phase 2 tail gates.

This is an experiment-only companion to ``run_multiclass_gate_probe.py``.
It trains a 5-class activity-bin ChemProp classifier on the train+AS1 labeled
pool using the existing Phase 2 folds, then reuses the same gate diagnostics.
Outputs stay under the ignored ``outputs/`` directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from sklearn.metrics import accuracy_score

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chemprop import data as chemprop_data  # noqa: E402
from chemprop import models, nn  # noqa: E402
from run_chemprop_finetune import (  # noqa: E402
    AGG_REGISTRY,
    PRETRAIN_PATH,
    freeze_encoder,
    load_pretrain_encoder_weights,
)
from run_multiclass_gate_probe import (  # noqa: E402
    BIN_LABELS,
    OUT_ROOT,
    label_arrays,
    load_pool_with_folds,
    load_test_frame,
    scan_continuous_gates,
    scan_gates,
    summarize_classifier,
)

torch.set_float32_matmul_precision("medium")


def load_params(args: argparse.Namespace) -> dict:
    if args.no_pretrain:
        params = {
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
            "batch_size": 64,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
        }
    else:
        if not PRETRAIN_PATH.exists():
            raise FileNotFoundError(f"Missing pretrain checkpoint: {PRETRAIN_PATH}")
        ckpt = torch.load(PRETRAIN_PATH, map_location="cpu", weights_only=False)
        params = dict(ckpt["params"])
        params["batch_size"] = 64
        params["max_epochs"] = args.max_epochs
        params["patience"] = args.patience

    if args.learning_rate is not None:
        params["learning_rate"] = args.learning_rate
    if args.lr_ratio is not None:
        params["lr_ratio"] = args.lr_ratio
    if args.warmup_epochs is not None:
        params["warmup_epochs"] = args.warmup_epochs
    return params


def build_classifier_model(params: dict, n_classes: int) -> models.MPNN:
    mp = nn.BondMessagePassing(
        d_h=params["message_hidden_dim"],
        depth=params["depth"],
        dropout=params["mp_dropout"],
        activation=params["activation"],
    )
    agg = AGG_REGISTRY[params["aggregation"]]()
    ffn = nn.MulticlassClassificationFFN(
        n_classes=n_classes,
        n_tasks=1,
        input_dim=mp.output_dim,
        hidden_dim=params["ffn_hidden_dim"],
        n_layers=params["ffn_num_layers"],
        dropout=params["ffn_dropout"],
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


def oversample_indices(y: np.ndarray, seed: int) -> np.ndarray:
    counts = Counter(int(v) for v in y)
    max_count = max(counts.values())
    rng = np.random.default_rng(seed)
    out = []
    for cls in sorted(counts):
        idx = np.flatnonzero(y == cls)
        sampled = rng.choice(idx, size=max_count, replace=True)
        out.append(sampled)
    merged = np.concatenate(out).astype(np.int64)
    rng.shuffle(merged)
    return merged


def make_points(
    smiles: list[str], y: np.ndarray
) -> list[chemprop_data.MoleculeDatapoint]:
    return [
        chemprop_data.MoleculeDatapoint.from_smi(
            smi, np.asarray([cls], dtype=np.float32)
        )
        for smi, cls in zip(smiles, y)
    ]


def make_dataloader(
    smiles: list[str],
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    drop_last: bool = False,
) -> torch.utils.data.DataLoader:
    dataset = chemprop_data.MoleculeDataset(make_points(smiles, y))
    return chemprop_data.build_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed if shuffle else None,
        drop_last=drop_last,
    )


def predict_proba(
    trainer: pl.Trainer,
    model: models.MPNN,
    smiles: list[str],
    batch_size: int,
    n_classes: int,
) -> np.ndarray:
    dummy_y = np.zeros(len(smiles), dtype=np.float32)
    loader = make_dataloader(
        smiles, dummy_y, batch_size, shuffle=False, seed=0, drop_last=False
    )
    preds = trainer.predict(model, loader)
    arr = np.concatenate([p.detach().cpu().numpy() for p in preds], axis=0)
    arr = arr.reshape(len(smiles), -1, n_classes)
    arr = arr[:, 0, :]
    row_sum = arr.sum(axis=1, keepdims=True)
    return np.divide(
        arr, row_sum, out=np.full_like(arr, 1.0 / n_classes), where=row_sum > 0
    )


def fit_one(
    args: argparse.Namespace,
    params: dict,
    tr_smiles: list[str],
    tr_y: np.ndarray,
    va_smiles: list[str],
    va_y: np.ndarray,
    seed: int,
) -> tuple[models.MPNN, pl.Trainer]:
    if args.oversample:
        idx = oversample_indices(tr_y, seed)
        tr_smiles = [tr_smiles[int(i)] for i in idx]
        tr_y = tr_y[idx]

    model = build_classifier_model(params, len(BIN_LABELS))
    if not args.no_pretrain:
        load_pretrain_encoder_weights(model, PRETRAIN_PATH)
    if args.freeze_encoder:
        freeze_encoder(model)

    train_loader = make_dataloader(
        tr_smiles,
        tr_y,
        params["batch_size"],
        shuffle=True,
        seed=seed,
        drop_last=False,
    )
    val_loader = make_dataloader(
        va_smiles,
        va_y,
        params["batch_size"],
        shuffle=False,
        seed=seed,
        drop_last=False,
    )
    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=params["patience"], mode="min"
    )
    trainer = pl.Trainer(
        max_epochs=params["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
        callbacks=[early_stop],
    )
    trainer.fit(model, train_loader, val_loader)
    return model, trainer


def run_oof(
    args: argparse.Namespace,
    params: dict,
    pool: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    n_classes = len(BIN_LABELS)
    oof = np.zeros((len(pool), n_classes), dtype=np.float64)
    test_fold_probas = []
    rows = []
    folds = sorted(pool["fold"].unique())
    if args.fold_limit is not None:
        folds = folds[: args.fold_limit]

    smiles = pool["smiles"].tolist()
    test_smiles = test["smiles"].tolist()
    for fold in folds:
        fold = int(fold)
        val_idx = pool.index[pool["fold"].eq(fold)].to_numpy(dtype=np.int64)
        train_idx = pool.index[~pool["fold"].eq(fold)].to_numpy(dtype=np.int64)
        tr_smiles = [smiles[int(i)] for i in train_idx]
        va_smiles = [smiles[int(i)] for i in val_idx]
        model = trainer = None
        try:
            model, trainer = fit_one(
                args,
                params,
                tr_smiles,
                y[train_idx],
                va_smiles,
                y[val_idx],
                args.seed + fold,
            )
            oof[val_idx] = predict_proba(
                trainer, model, va_smiles, params["batch_size"], n_classes
            )
            test_fold_probas.append(
                predict_proba(
                    trainer, model, test_smiles, params["batch_size"], n_classes
                )
            )
            rows.append(
                {
                    "fold": fold,
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                    "val_acc": float(
                        accuracy_score(y[val_idx], oof[val_idx].argmax(axis=1))
                    ),
                }
            )
            print(f"fold={fold} n_val={len(val_idx)} val_acc={rows[-1]['val_acc']:.4f}")
        finally:
            del model, trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.fold_limit is not None:
        missing = np.isclose(oof.sum(axis=1), 0.0)
        oof[missing] = 1.0 / n_classes
    return oof, np.mean(np.stack(test_fold_probas), axis=0), pd.DataFrame(rows)


def train_final_test_proba(
    args: argparse.Namespace,
    params: dict,
    pool: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
) -> np.ndarray:
    model = trainer = None
    try:
        model, trainer = fit_one(
            args,
            params,
            pool["smiles"].tolist(),
            y,
            pool["smiles"].tolist(),
            y,
            args.seed,
        )
        return predict_proba(
            trainer,
            model,
            test["smiles"].tolist(),
            params["batch_size"],
            len(BIN_LABELS),
        )
    finally:
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def config_name(args: argparse.Namespace) -> str:
    parts = ["chemprop_multiclass_ft"]
    parts.append("scratch" if args.no_pretrain else "pretrain")
    parts.append("frozen" if args.freeze_encoder else "full")
    if args.oversample:
        parts.append("oversample")
    parts.append(f"e{args.max_epochs}")
    if args.fold_limit is not None:
        parts.append(f"{args.fold_limit}fold_smoke")
    if args.run_name:
        parts.append(args.run_name)
    return "_".join(parts)


def write_report(
    out_dir: Path,
    name: str,
    args: argparse.Namespace,
    clf_summary: pd.DataFrame,
    by_class: pd.DataFrame,
    auc: pd.DataFrame,
    gate_scan: pd.DataFrame,
    continuous_gate_scan: pd.DataFrame,
    fold_df: pd.DataFrame,
) -> None:
    lines = [
        "# ChemProp multiclass FT gate probe",
        "",
        f"- Config: `{name}`",
        f"- Pretrain: `{not args.no_pretrain}`",
        f"- Freeze encoder: `{args.freeze_encoder}`",
        f"- Oversample classes: `{args.oversample}`",
        f"- Max epochs: `{args.max_epochs}`",
        f"- Fold limit: `{args.fold_limit}`",
        "",
        "## Fold summary",
        "",
        fold_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Classifier OOF summary",
        "",
        clf_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Class metrics",
        "",
        by_class.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Tail binary AUC",
        "",
        auc.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Best gate sweeps by id55 AS1 replay",
        "",
        gate_scan.head(12).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Best continuous probability gates",
        "",
        continuous_gate_scan.head(8).to_markdown(index=False, floatfmt=".4f"),
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold-limit", type=int, default=None)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--no-pretrain", action="store_true")
    parser.add_argument("--no-oversample", dest="oversample", action="store_false")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--lr-ratio", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--run-name", default="")
    parser.set_defaults(oversample=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = load_params(args)
    name = config_name(args)
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = load_pool_with_folds()
    test = load_test_frame()
    y, label_to_id, _id_to_label = label_arrays(pool)
    print(
        f"config={name} n_pool={len(pool)} n_test={len(test)} "
        f"classes={dict(zip(*np.unique(y, return_counts=True)))}"
    )
    print(f"params={params}")

    oof_proba, test_proba_cv, fold_df = run_oof(args, params, pool, test, y)
    final_test_proba = train_final_test_proba(args, params, pool, test, y)
    clf_summary, by_class, auc, cm = summarize_classifier(
        pool, y, oof_proba, label_to_id
    )
    gate_scan = scan_gates(pool, y, label_to_id, oof_proba, final_test_proba, test)
    continuous_gate_scan = scan_continuous_gates(pool, y, label_to_id, oof_proba)

    proba_cols = [f"p_{label}" for label in BIN_LABELS]
    oof_df = pool[
        [
            "pool_idx",
            "compound_id",
            "molecule_name",
            "pec50",
            "source",
            "true_bin",
            "fold",
        ]
    ].copy()
    for idx, col in enumerate(proba_cols):
        oof_df[col] = oof_proba[:, idx]
    oof_df["pred_bin"] = [BIN_LABELS[i] for i in oof_proba.argmax(axis=1)]
    test_df = test[["test_id", "compound_id", "molecule_name", "as1_pec50"]].copy()
    test_df["split"] = np.where(test_df["as1_pec50"].notna(), "AS1", "AS2")
    for idx, col in enumerate(proba_cols):
        test_df[f"cv_{col}"] = test_proba_cv[:, idx]
        test_df[f"final_{col}"] = final_test_proba[:, idx]
    test_df["final_pred_bin"] = [BIN_LABELS[i] for i in final_test_proba.argmax(axis=1)]

    oof_df.to_csv(out_dir / "oof_class_probabilities.csv", index=False)
    test_df.to_csv(out_dir / "test_class_probabilities.csv", index=False)
    fold_df.to_csv(out_dir / "fold_summary.csv", index=False)
    clf_summary.to_csv(out_dir / "classifier_summary.csv", index=False)
    by_class.to_csv(out_dir / "class_metrics.csv", index=False)
    auc.to_csv(out_dir / "tail_binary_auc.csv", index=False)
    cm.to_csv(out_dir / "confusion_matrix.csv")
    gate_scan.to_csv(out_dir / "gate_scan.csv", index=False)
    continuous_gate_scan.to_csv(out_dir / "continuous_gate_scan.csv", index=False)
    (out_dir / "metadata.json").write_text(
        json.dumps({"args": vars(args), "params": params}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    write_report(
        out_dir,
        name,
        args,
        clf_summary,
        by_class,
        auc,
        gate_scan,
        continuous_gate_scan,
        fold_df,
    )

    print("\nClassifier summary")
    print(clf_summary.to_string(index=False))
    print("\nClass metrics")
    print(by_class.to_string(index=False))
    print("\nTail AUC")
    print(auc.to_string(index=False))
    print("\nBest gate rows")
    print(gate_scan.head(10).to_string(index=False))
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
