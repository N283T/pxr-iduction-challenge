#!/usr/bin/env python
"""Buterez 2024 Strategy 6 probe: frozen ChemProp + adaptive readout.

Strategy 6 freezes the low-fidelity-pretrained ChemProp message-passing encoder
and fine-tunes only an adaptive graph readout on the high-fidelity pEC50 task.
This differs from the existing Strategy 3 member, which exports ChemProp's fixed
molecule fingerprint and lets TabPFN do the high-fidelity regression.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from chemprop import data as chemprop_data  # noqa: E402
from chemprop_strategy6 import (  # noqa: E402
    ChempropNodeEncoder,
    ChempropStrategy6Regressor,
    freeze_all,
    pad_node_embeddings,
)
from data import load_test_smiles, load_train_smiles_target  # noqa: E402
from evaluate import (  # noqa: E402
    compute_metrics,
    print_fold_summary,
    print_metrics,
    record_experiment,
    save_oof_predictions,
)
from run_chemprop_embed_extract import (  # noqa: E402
    DEFAULT_CKPT_PATH,
    build_pretrain_model,
)
from splits import umap_split_indices  # noqa: E402

SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
OUTPUT_ROOT = REPO_ROOT.joinpath(
    "track1_activity", "analysis", "strategy6_chemprop", "outputs"
)


def iter_index_batches(indices: np.ndarray, batch_size: int, shuffle: bool, seed: int):
    idx = np.array(indices, copy=True)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        yield idx[start : start + batch_size]


def make_points(smiles: list[str]) -> list[chemprop_data.MoleculeDatapoint]:
    dummy = np.zeros(1, dtype=np.float32)
    return [chemprop_data.MoleculeDatapoint.from_smi(smi, dummy) for smi in smiles]


def make_batch(
    points: list[chemprop_data.MoleculeDatapoint], indices: np.ndarray
) -> chemprop_data.TrainingBatch:
    dataset = chemprop_data.MoleculeDataset([points[int(i)] for i in indices])
    loader = chemprop_data.build_dataloader(
        dataset, batch_size=len(indices), shuffle=False
    )
    return next(iter(loader))


def move_batch(batch: chemprop_data.TrainingBatch, device: torch.device):
    bmg = batch.bmg
    bmg.to(device)
    V_d = batch.V_d.to(device) if batch.V_d is not None else None
    return bmg, V_d


def load_pretrained_strategy6_model(
    ckpt_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ChempropStrategy6Regressor, dict, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    params = dict(ckpt["params"])
    base = build_pretrain_model(params)
    load_result = base.load_state_dict(ckpt["state_dict"], strict=True)
    node_encoder = ChempropNodeEncoder(base.message_passing)
    model = ChempropStrategy6Regressor(
        node_encoder=node_encoder,
        input_dim=int(params["message_hidden_dim"]),
        hidden_dim=args.readout_dim,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        num_seeds=args.num_seeds,
        dropout=args.dropout,
    ).to(device)
    frozen_count = freeze_all(model.node_encoder)
    model.node_encoder.eval()
    summary = {
        "frozen_param_tensors": frozen_count,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "pretrain_val_loss": float(ckpt.get("final_val_loss", -1.0)),
    }
    return model, params, summary


def predict_indices(
    points: list[chemprop_data.MoleculeDatapoint],
    indices: np.ndarray,
    model: ChempropStrategy6Regressor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    preds: list[np.ndarray] = []
    model.eval()
    model.node_encoder.eval()
    with torch.no_grad():
        for sel in iter_index_batches(indices, batch_size, shuffle=False, seed=0):
            batch = make_batch(points, sel)
            bmg, V_d = move_batch(batch, device)
            pred = model(bmg, V_d)
            preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds).astype(np.float64)


def train_fold(
    points: list[chemprop_data.MoleculeDatapoint],
    y_all: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    ckpt_path: Path,
    args: argparse.Namespace,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    torch.manual_seed(args.seed + fold)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + fold)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, params, pretrain_summary = load_pretrained_strategy6_model(
        ckpt_path, args, device
    )
    optimizer = AdamW(model.readout.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.min_lr)
    criterion = nn.MSELoss()
    y_mean = float(np.mean(y_all[train_idx])) if args.target_standardize else 0.0
    y_std = float(np.std(y_all[train_idx])) if args.target_standardize else 1.0
    if y_std < 1e-6:
        y_std = 1.0

    best_state = None
    best_val_mae = float("inf")
    patience = 0
    loss_start = None
    loss_end = None

    for epoch in range(args.max_epochs):
        model.readout.train()
        model.node_encoder.eval()
        epoch_losses = []
        for sel in iter_index_batches(
            train_idx,
            args.batch_size,
            shuffle=True,
            seed=args.seed + fold * 1000 + epoch,
        ):
            batch = make_batch(points, sel)
            bmg, V_d = move_batch(batch, device)
            target_np = (y_all[sel] - y_mean) / y_std
            target = torch.tensor(target_np, dtype=torch.float32, device=device)
            with torch.no_grad():
                nodes = model.node_encoder(bmg, V_d)
                padded, mask = pad_node_embeddings(nodes, bmg.batch)
            optimizer.zero_grad(set_to_none=True)
            pred = model.readout(padded, mask)
            loss = criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.readout.parameters(), args.grad_clip)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        epoch_loss = float(np.mean(epoch_losses))
        if loss_start is None:
            loss_start = epoch_loss
        loss_end = epoch_loss

        val_pred_z = predict_indices(points, val_idx, model, args.batch_size, device)
        val_pred = val_pred_z * y_std + y_mean
        val_mae = float(np.mean(np.abs(y_all[val_idx] - val_pred)))
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.readout.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

    if best_state is None:
        raise RuntimeError(f"fold {fold}: no best readout state")
    model.readout.load_state_dict(best_state)
    val_pred = predict_indices(points, val_idx, model, args.batch_size, device) * y_std + y_mean
    test_pred = predict_indices(points, test_idx, model, args.batch_size, device) * y_std + y_mean
    info = {
        "fold": fold,
        "epochs_run": epoch + 1,
        "best_val_mae": best_val_mae,
        "loss_start": loss_start,
        "loss_end": loss_end,
        "pretrain_val_loss": pretrain_summary["pretrain_val_loss"],
        "frozen_param_tensors": pretrain_summary["frozen_param_tensors"],
        "encoder_hidden_dim": params["message_hidden_dim"],
        "target_mean": y_mean,
        "target_std": y_std,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return val_pred, test_pred, info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="initial")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--readout-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-blocks", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fold-limit", type=int, default=None)
    parser.add_argument("--target-standardize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-record", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.ckpt.exists():
        raise FileNotFoundError(f"Missing ChemProp pretrain checkpoint: {args.ckpt}")
    run_dir = OUTPUT_ROOT.joinpath(args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y = train_df["pec50"].to_numpy(dtype=np.float32)
    train_smiles = train_df["smiles"].tolist()
    test_smiles = test_df["smiles"].tolist()
    all_smiles = train_smiles + test_smiles
    points = make_points(all_smiles)
    folds = umap_split_indices(train_smiles)
    if args.fold_limit is not None:
        folds = folds[: args.fold_limit]

    oof = np.full(len(train_df), np.nan, dtype=np.float64)
    test_preds_per_fold = []
    fold_metrics = []
    fold_rows = []
    test_idx = np.arange(len(train_df), len(all_smiles), dtype=np.int64)
    y_all = np.concatenate([y, np.zeros(len(test_df), dtype=np.float32)])

    print("ChemProp Strategy 6 adaptive readout")
    print(f"  run_name={args.run_name}")
    print(f"  ckpt={args.ckpt}")
    print(f"  train={len(train_df)} test={len(test_df)} folds={len(folds)}")
    print(
        f"  readout_dim={args.readout_dim} blocks={args.num_blocks} "
        f"dropout={args.dropout} lr={args.lr}"
    )

    for fold, (tr_idx, va_idx) in enumerate(folds):
        print(f"\n[Fold {fold}] train={len(tr_idx)} val={len(va_idx)}")
        val_pred, test_pred, info = train_fold(
            points=points,
            y_all=y_all,
            train_idx=tr_idx,
            val_idx=va_idx,
            test_idx=test_idx,
            ckpt_path=args.ckpt,
            args=args,
            fold=fold,
        )
        oof[va_idx] = val_pred
        test_preds_per_fold.append(test_pred)
        metrics = compute_metrics(y[va_idx], val_pred)
        fold_metrics.append(metrics)
        fold_rows.append({**info, **{f"val_{k}": v for k, v in metrics.items()}})
        print_metrics(metrics, label=f"Fold {fold}")

    covered = np.isfinite(oof)
    if not covered.all():
        print(f"  WARNING: fold_limit covered {covered.sum()} / {len(oof)} train rows")
    oof_metrics = compute_metrics(y[covered], oof[covered])
    test_pred = np.mean(test_preds_per_fold, axis=0)
    print("\nOverall OOF:")
    print_metrics(oof_metrics)
    print_fold_summary(fold_metrics)
    print(
        f"\nTest preds: mean={test_pred.mean():.4f} "
        f"std={test_pred.std():.4f}"
    )

    exp_name = f"chemprop_strategy6_adaptive_readout_{args.run_name}_umap"
    sub = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": test_pred,
        }
    )
    sub_path = SUBMISSION_DIR.joinpath(f"{exp_name}.csv")
    sub.to_csv(sub_path, index=False)

    residual_r = float("nan")
    if covered.all():
        try:
            from run_ensemble_calibrate_importance import load_caruana_oof_and_test

            anchor_oof, _anchor_test, _anchor_df = load_caruana_oof_and_test()
            residual_r = float(np.corrcoef(oof - y, anchor_oof - y)[0, 1])
        except Exception as exc:  # noqa: BLE001 - diagnostic only
            print(f"Could not compute residual correlation vs current ensemble: {exc}")

    summary = pd.DataFrame(
        [
            {
                "name": exp_name,
                **oof_metrics,
                "residual_r_vs_ens_caruana_bag20": residual_r,
                "oof_coverage": int(covered.sum()),
                "test_mean": float(test_pred.mean()),
                "test_std": float(test_pred.std()),
                "submission_path": str(sub_path.relative_to(REPO_ROOT)),
            }
        ]
    )
    summary.to_csv(run_dir.joinpath("summary.csv"), index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir.joinpath("fold_metrics.csv"), index=False)
    pd.DataFrame({"train_idx": np.arange(len(oof)), "pec50": y, "oof_prediction": oof}).to_csv(
        run_dir.joinpath("oof_predictions.csv"), index=False
    )
    pd.DataFrame({
        "SMILES": test_df["smiles"],
        "Molecule Name": test_df["molecule_name"],
        "pEC50": test_pred,
    }).to_csv(run_dir.joinpath("test_predictions.csv"), index=False)

    final_read = (
        "Do not submit directly unless this beats the current ChemProp Strategy 3 sibling "
        "or earns non-trivial Caruana weight with low residual correlation."
    )
    if covered.all() and oof_metrics["MAE"] <= 0.48:
        final_read = (
            "Single-model gate passed. Record OOF and run Caruana ADD/SWAP bakeoff "
            "before considering a leaderboard submission."
        )
    report = "\n".join(
        [
            "# Buterez Strategy 6 ChemProp Report",
            "",
            f"Run name: `{args.run_name}`",
            f"Experiment: `{exp_name}`",
            f"Pretrain checkpoint: `{args.ckpt}`",
            "",
            "## OOF Metrics",
            "",
            f"Coverage: `{int(covered.sum())} / {len(oof)}`",
            f"MAE: `{oof_metrics['MAE']:.6f}`",
            f"RAE: `{oof_metrics['RAE']:.6f}`",
            f"Spearman: `{oof_metrics['Spearman_R']:.6f}`",
            f"Residual r vs ens_caruana_bag20: `{residual_r:.6f}`",
            "",
            "## Fold Metrics",
            "",
            pd.DataFrame(fold_rows).to_markdown(index=False),
            "",
            "## Decision Gate",
            "",
            "Direct-submit gate: MAE <= 0.48 and no Spearman collapse; otherwise only test via Caruana ADD if decorrelated.",
            "",
            "## Final Read",
            "",
            final_read,
        ]
    )
    report_path = run_dir.joinpath("report.md")
    report_path.write_text(report + "\n")

    if not args.no_record and covered.all():
        args_for_json = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        exp_id = record_experiment(
            name=exp_name,
            description="Buterez 2024 Strategy 6: frozen ChemProp LF encoder + adaptive readout on pEC50",
            model_type="chemprop_strategy6",
            feature_set="smiles_adaptive_readout",
            hyperparameters={
                "pretrain_path": str(args.ckpt),
                "args": args_for_json,
            },
            fold_metrics=fold_metrics,
            submission_path=str(sub_path.relative_to(REPO_ROOT)),
            notes=f"OOF MAE={oof_metrics['MAE']:.4f}, Strategy 6 adaptive readout",
            on_conflict_replace=True,
        )
        save_oof_predictions(exp_id, oof)

    print(f"Saved report: {report_path}")
    print(f"Saved submission: {sub_path}")


if __name__ == "__main__":
    main()
