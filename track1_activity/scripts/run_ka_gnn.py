#!/usr/bin/env python
"""Train a PyG-native KA-GNN probe on PXR pEC50.

The model ports LongLee220/KA-GNN's Fourier KAN message-passing idea into the
repo's PyG graph-training stack: fold-local feature normalization, KAN Fourier
message transforms, residual aggregation, graph pooling, and KAN readout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Batch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import load_test_smiles, load_train_smiles_target  # noqa: E402
from evaluate import (  # noqa: E402
    compute_metrics,
    print_fold_summary,
    print_metrics,
    record_experiment,
    save_oof_predictions,
)
from ka_gnn import FourierKAGNNModel, PykanSAGEModel, load_pretrained_encoder_state  # noqa: E402
from pyg_training import smiles_to_pyg_list  # noqa: E402
from splits import umap_split_indices  # noqa: E402

SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
OUTPUT_ROOT = REPO_ROOT.joinpath("track1_activity", "analysis", "ka_gnn", "outputs")


def iter_batches(indices: np.ndarray, batch_size: int, shuffle: bool, seed: int):
    idx = np.array(indices, copy=True)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        yield idx[start : start + batch_size]


def compute_feature_stats(graphs: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.cat([g.x.float() for g in graphs], dim=0)
    edge = torch.cat([g.edge_attr.float() for g in graphs], dim=0)
    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0)
    edge_mean = edge.mean(dim=0)
    edge_std = edge.std(dim=0)
    x_std = torch.clamp(x_std, min=1.0)
    edge_std = torch.clamp(edge_std, min=1.0)
    return x_mean, x_std, edge_mean, edge_std


def build_model(args: argparse.Namespace, in_dim: int, edge_dim: int, device: torch.device):
    if args.model_type == "pykan_sage":
        return PykanSAGEModel(
            in_dim=in_dim,
            edge_dim=edge_dim,
            hidden_dim=args.hidden_dim,
            out_dim=args.out_dim,
            grid_size=args.grid_size,
            num_layers=args.num_layers,
            pooling=args.pooling,
            dropout=args.dropout,
            kan_bottleneck=args.kan_bottleneck,
            spline_order=args.spline_order,
            seed=args.seed,
        ).to(device)
    return FourierKAGNNModel(
        in_dim=in_dim,
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        grid_size=args.grid_size,
        num_layers=args.num_layers,
        pooling=args.pooling,
        dropout=args.dropout,
        use_bias=True,
        aggr=args.aggr,
    ).to(device)


def make_batch(graphs: list, indices: np.ndarray, device: torch.device) -> Batch:
    return Batch.from_data_list([graphs[int(i)] for i in indices]).to(device)


def predict_indices(
    graphs: list,
    indices: np.ndarray,
    model: FourierKAGNNModel,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for sel in iter_batches(indices, batch_size, shuffle=False, seed=0):
            batch = make_batch(graphs, sel, device)
            out = model(
                batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch
            ).squeeze(-1)
            preds.append(out.detach().cpu().numpy())
    return np.concatenate(preds).astype(np.float32)


def train_fold(
    graphs: list,
    y_all: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    torch.manual_seed(args.seed + fold)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + fold)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    in_dim = int(graphs[0].x.shape[1])
    edge_dim = int(graphs[0].edge_attr.shape[1])
    model = build_model(args, in_dim, edge_dim, device)
    pretrained_tensors = 0
    if args.pretrained_encoder is not None:
        ckpt = torch.load(args.pretrained_encoder, map_location="cpu", weights_only=False)
        encoder_state = ckpt.get("encoder_state_dict", ckpt.get("state_dict", ckpt))
        pretrained_tensors = load_pretrained_encoder_state(model, encoder_state)
        if pretrained_tensors <= 0:
            raise RuntimeError(f"no compatible encoder tensors loaded from {args.pretrained_encoder}")
    stats = compute_feature_stats([graphs[int(i)] for i in train_idx])
    model.set_feature_standardization(*(s.to(device) for s in stats))

    y_mean = float(np.mean(y_all[train_idx]))
    y_std = float(np.std(y_all[train_idx]))
    if y_std < 1e-6:
        y_std = 1.0

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.min_lr)
    criterion = torch.nn.MSELoss()
    best_state = None
    best_val_mae = float("inf")
    patience = 0
    loss_start = None
    loss_end = None

    for epoch in range(args.max_epochs):
        model.train()
        epoch_losses = []
        for sel in iter_batches(
            train_idx,
            args.batch_size,
            shuffle=True,
            seed=args.seed + fold * 1000 + epoch,
        ):
            batch = make_batch(graphs, sel, device)
            target_np = (y_all[sel] - y_mean) / y_std
            target = torch.tensor(target_np, dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(
                batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch
            ).squeeze(-1)
            loss = criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        epoch_loss = float(np.mean(epoch_losses))
        if loss_start is None:
            loss_start = epoch_loss
        loss_end = epoch_loss

        val_pred_z = predict_indices(graphs, val_idx, model, args.batch_size, device)
        val_pred = val_pred_z * y_std + y_mean
        val_mae = float(np.mean(np.abs(y_all[val_idx] - val_pred)))
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

    if best_state is None:
        raise RuntimeError(f"fold {fold}: no best model state")
    model.load_state_dict(best_state)
    val_pred = predict_indices(graphs, val_idx, model, args.batch_size, device) * y_std + y_mean
    test_pred = predict_indices(graphs, test_idx, model, args.batch_size, device) * y_std + y_mean
    info = {
        "fold": fold,
        "epochs_run": epoch + 1,
        "best_val_mae": best_val_mae,
        "loss_start": loss_start,
        "loss_end": loss_end,
        "target_mean": y_mean,
        "target_std": y_std,
        "node_dim": in_dim,
        "edge_dim": edge_dim,
        "pretrained_tensors": pretrained_tensors,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return val_pred.astype(np.float64), test_pred.astype(np.float64), info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="fourier_h128_l3_g3")
    parser.add_argument("--model-type", choices=["fourier", "pykan_sage"], default="fourier")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--out-dim", type=int, default=64)
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--spline-order", type=int, default=3)
    parser.add_argument("--kan-bottleneck", type=int, default=5)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--pooling", choices=["mean", "sum", "max"], default="mean")
    parser.add_argument("--aggr", choices=["sum", "mean"], default="sum")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fold-limit", type=int, default=None)
    parser.add_argument("--pretrained-encoder", type=Path, default=None)
    parser.add_argument("--no-record", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = OUTPUT_ROOT.joinpath(args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y = train_df["pec50"].to_numpy(dtype=np.float32)
    train_smiles = train_df["smiles"].tolist()
    test_smiles = test_df["smiles"].tolist()
    all_smiles = train_smiles + test_smiles
    print("Converting SMILES to PyG graphs...")
    graphs = smiles_to_pyg_list(all_smiles)
    folds = umap_split_indices(train_smiles)
    if args.fold_limit is not None:
        folds = folds[: args.fold_limit]
    test_idx = np.arange(len(train_smiles), len(all_smiles), dtype=np.int64)
    y_all = np.concatenate([y, np.zeros(len(test_smiles), dtype=np.float32)])

    print("KA-GNN Track 1 probe")
    print(f"  run_name={args.run_name}")
    print(f"  train={len(train_smiles)} test={len(test_smiles)} folds={len(folds)}")
    print(
        f"  model_type={args.model_type} hidden={args.hidden_dim} out={args.out_dim} layers={args.num_layers} "
        f"grid={args.grid_size} pooling={args.pooling} aggr={args.aggr} lr={args.lr}"
    )
    if args.pretrained_encoder is not None:
        print(f"  pretrained_encoder={args.pretrained_encoder}")

    oof = np.full(len(train_smiles), np.nan, dtype=np.float64)
    test_preds_per_fold = []
    fold_metrics = []
    fold_rows = []
    for fold, (tr_idx, va_idx) in enumerate(folds):
        print(f"\n[Fold {fold}] train={len(tr_idx)} val={len(va_idx)}")
        val_pred, test_pred, info = train_fold(
            graphs=graphs,
            y_all=y_all,
            train_idx=tr_idx,
            val_idx=va_idx,
            test_idx=test_idx,
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
        print(f"  WARNING: fold_limit covered {covered.sum()} / {len(oof)} rows")
    oof_metrics = compute_metrics(y[covered], oof[covered])
    test_pred = np.mean(test_preds_per_fold, axis=0)
    print("\nOverall OOF:")
    print_metrics(oof_metrics)
    print_fold_summary(fold_metrics)
    print(f"\nTest preds: mean={test_pred.mean():.4f} std={test_pred.std():.4f}")

    exp_name = f"ka_gnn_{args.run_name}_umap"
    sub = pd.DataFrame(
        {"SMILES": test_df["smiles"], "Molecule Name": test_df["molecule_name"], "pEC50": test_pred}
    )
    sub_path = SUBMISSION_DIR.joinpath(f"{exp_name}.csv")
    sub.to_csv(sub_path, index=False)

    residual_r = float("nan")
    if covered.all():
        try:
            sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
            from run_ensemble_calibrate_importance import load_caruana_oof_and_test

            anchor_oof, _anchor_test, _anchor_df = load_caruana_oof_and_test()
            residual_r = float(np.corrcoef(oof - y, anchor_oof - y)[0, 1])
        except Exception as exc:  # noqa: BLE001
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
    sub.to_csv(run_dir.joinpath("test_predictions.csv"), index=False)

    final_read = "Do not submit directly; compare Caruana ADD and correlations first."
    if covered.all() and oof_metrics["MAE"] <= 0.49 and residual_r < 0.85:
        final_read = "Potentially useful: passes loose MAE/decorrelation gate; run ADD/SWAP preflight."
    report = "\n".join(
        [
            "# KA-GNN Report",
            "",
            f"Run name: `{args.run_name}`",
            f"Experiment: `{exp_name}`",
            "Reference implementation: `https://github.com/LongLee220/KA-GNN`",
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
            "Continue only if MAE <= 0.49 and residual correlation/candidate correlations show a new axis.",
            "",
            "## Final Read",
            "",
            final_read,
        ]
    )
    report_path = run_dir.joinpath("report.md")
    report_path.write_text(report + "\n")

    if not args.no_record and covered.all():
        exp_id = record_experiment(
            name=exp_name,
            description=f"PyG port of KA-GNN ({args.model_type}) for direct pEC50 graph regression",
            model_type="ka_gnn",
            feature_set="molecular_graph_fourier_kan",
            hyperparameters={
                **vars(args),
                "pretrained_encoder": str(args.pretrained_encoder) if args.pretrained_encoder else None,
            },
            fold_metrics=fold_metrics,
            submission_path=str(sub_path.relative_to(REPO_ROOT)),
            notes=f"OOF MAE={oof_metrics['MAE']:.4f}, KA-GNN {args.model_type}",
            on_conflict_replace=True,
        )
        save_oof_predictions(exp_id, oof)

    print(f"Saved report: {report_path}")
    print(f"Saved submission: {sub_path}")


if __name__ == "__main__":
    main()
