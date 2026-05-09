#!/usr/bin/env python
"""Probe pykan/KAN as a regressor on frozen molecular embeddings.

This tests whether KAN can replace the TabPFN/MLP-style high-fidelity regressor
on existing low-fidelity-pretrained embeddings. It is intentionally a probe: no
leaderboard submission should be made directly from the output without a pool
bakeoff and preflight.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from kan import KAN
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import get_engine, load_test_smiles, load_train_smiles_target  # noqa: E402
from evaluate import (  # noqa: E402
    compute_metrics,
    print_fold_summary,
    print_metrics,
    record_experiment,
    save_oof_predictions,
)
from kan_embed import FoldStandardizer, build_kan_width, fit_pca_if_needed  # noqa: E402
from splits import umap_split_indices  # noqa: E402

SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
OUTPUT_ROOT = REPO_ROOT.joinpath("track1_activity", "analysis", "kan_embed", "outputs")
DEFAULT_EMBED_PATH = REPO_ROOT.joinpath("data", "chemprop_pretrain_embed.parquet")


def load_train_test_ids() -> tuple[list[int], list[int]]:
    engine = get_engine()
    train_ids = pd.read_sql(
        "SELECT compound_id FROM train_activity ORDER BY id", engine
    )["compound_id"].astype(int).tolist()
    test_ids = pd.read_sql(
        "SELECT compound_id FROM test_activity ORDER BY id", engine
    )["compound_id"].astype(int).tolist()
    return train_ids, test_ids


def load_embedding_matrix(embed_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not embed_path.exists():
        raise FileNotFoundError(f"Missing embedding parquet: {embed_path}")
    train_ids, test_ids = load_train_test_ids()
    emb = pd.read_parquet(embed_path)
    if "compound_id" in emb.columns:
        emb = emb.set_index("compound_id")
    emb.index = emb.index.astype(int)
    X_train = emb.reindex(train_ids).to_numpy(dtype=np.float32)
    X_test = emb.reindex(test_ids).to_numpy(dtype=np.float32)
    if np.isnan(X_train).any() or np.isnan(X_test).any():
        raise ValueError(f"Embedding coverage has NaN after reindex: {embed_path}")
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    return X_train.astype(np.float32), X_test.astype(np.float32)


def iter_batches(n: int, batch_size: int, shuffle: bool, seed: int):
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        yield idx[start : start + batch_size]


def make_model(input_dim: int, args: argparse.Namespace, fold: int, device: torch.device):
    width = build_kan_width(input_dim, args.hidden_dim, args.second_hidden_dim)
    return KAN(
        width=width,
        grid=args.grid,
        k=args.spline_order,
        seed=args.seed + fold,
        device=device,
        symbolic_enabled=False,
        auto_save=False,
        save_act=False,
    )


def predict(model, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for idx in iter_batches(len(X), batch_size, shuffle=False, seed=0):
            xb = torch.tensor(X[idx], dtype=torch.float32, device=device)
            out = model(xb)
            preds.append(out[:, 0].detach().cpu().numpy())
    return np.concatenate(preds).astype(np.float32)


def train_fold(
    X_all: np.ndarray,
    y: np.ndarray,
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

    X_tr_raw = X_all[train_idx]
    X_va_raw = X_all[val_idx]
    X_te_raw = X_all[test_idx]
    X_tr_pca, X_va_pca, pca = fit_pca_if_needed(
        X_tr_raw, X_va_raw, max_dim=args.max_dim, seed=args.seed + fold
    )
    if pca is None:
        X_te_pca = X_te_raw.astype(np.float32, copy=True)
        pca_var = float("nan")
    else:
        X_te_pca = pca.transform(X_te_raw).astype(np.float32)
        pca_var = float(np.sum(pca.explained_variance_ratio_))

    scaler = FoldStandardizer.fit(X_tr_pca, y[train_idx])
    X_tr = scaler.transform_x(X_tr_pca)
    X_va = scaler.transform_x(X_va_pca)
    X_te = scaler.transform_x(X_te_pca)
    y_tr = scaler.transform_y(y[train_idx])

    model = make_model(X_tr.shape[1], args, fold, device)
    if args.optimizer == "adam":
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.min_lr)
    else:
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=args.lr,
            max_iter=1,
            history_size=args.lbfgs_history_size,
            line_search_fn="strong_wolfe",
        )
        scheduler = None
    criterion = torch.nn.MSELoss()

    best_state = None
    best_val_mae = float("inf")
    patience = 0
    loss_start = None
    loss_end = None

    X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_tensor = torch.tensor(y_tr, dtype=torch.float32, device=device)

    for epoch in range(args.max_epochs):
        model.train()
        epoch_losses = []
        if args.optimizer == "lbfgs":
            def closure(net=model):
                optimizer.zero_grad(set_to_none=True)
                pred_full = net(X_tr_tensor)[:, 0]
                loss_full = criterion(pred_full, y_tr_tensor)
                loss_full.backward()
                return loss_full

            loss = optimizer.step(closure)
            epoch_losses.append(float(loss.detach().cpu()))
        else:
            for batch_idx in iter_batches(
                len(X_tr), args.batch_size, shuffle=True, seed=args.seed + fold * 1000 + epoch
            ):
                xb = torch.tensor(X_tr[batch_idx], dtype=torch.float32, device=device)
                yb = torch.tensor(y_tr[batch_idx], dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)[:, 0]
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            scheduler.step()
        epoch_loss = float(np.mean(epoch_losses))
        if loss_start is None:
            loss_start = epoch_loss
        loss_end = epoch_loss

        val_pred_z = predict(model, X_va, args.predict_batch_size, device)
        val_pred = scaler.inverse_y(val_pred_z)
        val_mae = float(np.mean(np.abs(y[val_idx] - val_pred)))
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

    if best_state is None:
        raise RuntimeError(f"fold {fold}: no best KAN state")
    model.load_state_dict(best_state)
    val_pred = scaler.inverse_y(predict(model, X_va, args.predict_batch_size, device))
    test_pred = scaler.inverse_y(predict(model, X_te, args.predict_batch_size, device))
    info = {
        "fold": fold,
        "epochs_run": epoch + 1,
        "best_val_mae": best_val_mae,
        "loss_start": loss_start,
        "loss_end": loss_end,
        "input_dim": int(X_tr.shape[1]),
        "pca_variance": pca_var,
        "target_mean": scaler.y_mean,
        "target_std": scaler.y_std,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return val_pred.astype(np.float64), test_pred.astype(np.float64), info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="chemprop_pca64_h16")
    parser.add_argument("--embed-path", type=Path, default=DEFAULT_EMBED_PATH)
    parser.add_argument("--feature-name", default="chemprop_pretrain_embed")
    parser.add_argument("--max-dim", type=int, default=64, help="PCA cap; <=0 disables")
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--second-hidden-dim", type=int, default=None)
    parser.add_argument("--grid", type=int, default=3)
    parser.add_argument("--spline-order", type=int, default=3)
    parser.add_argument("--optimizer", choices=["adam", "lbfgs"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lbfgs-history-size", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--predict-batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fold-limit", type=int, default=None)
    parser.add_argument("--no-record", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_dim is not None and args.max_dim <= 0:
        args.max_dim = None
    run_dir = OUTPUT_ROOT.joinpath(args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y = train_df["pec50"].to_numpy(dtype=np.float32)
    X_train, X_test = load_embedding_matrix(args.embed_path)
    X_all = np.vstack([X_train, X_test]).astype(np.float32)
    folds = umap_split_indices(train_df["smiles"].tolist())
    if args.fold_limit is not None:
        folds = folds[: args.fold_limit]
    test_idx = np.arange(len(X_train), len(X_all), dtype=np.int64)

    print("KAN on frozen embedding")
    print(f"  run_name={args.run_name}")
    print(f"  embed_path={args.embed_path}")
    print(f"  train={X_train.shape} test={X_test.shape} folds={len(folds)}")
    print(
        f"  max_dim={args.max_dim} width=(*,{args.hidden_dim},{args.second_hidden_dim},1) "
        f"grid={args.grid} k={args.spline_order} lr={args.lr}"
    )

    oof = np.full(len(X_train), np.nan, dtype=np.float64)
    test_preds_per_fold = []
    fold_metrics = []
    fold_rows = []
    for fold, (tr_idx, va_idx) in enumerate(folds):
        print(f"\n[Fold {fold}] train={len(tr_idx)} val={len(va_idx)}")
        val_pred, test_pred, info = train_fold(
            X_all=X_all,
            y=y,
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
        print(f"  WARNING: fold_limit covered {covered.sum()} / {len(oof)} train rows")
    oof_metrics = compute_metrics(y[covered], oof[covered])
    test_pred = np.mean(test_preds_per_fold, axis=0)
    print("\nOverall OOF:")
    print_metrics(oof_metrics)
    print_fold_summary(fold_metrics)
    print(f"\nTest preds: mean={test_pred.mean():.4f} std={test_pred.std():.4f}")

    exp_name = f"kan_{args.feature_name}_{args.run_name}_umap"
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
            sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
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
    pd.DataFrame(
        {"train_idx": np.arange(len(oof)), "pec50": y, "oof_prediction": oof}
    ).to_csv(run_dir.joinpath("oof_predictions.csv"), index=False)
    sub.to_csv(run_dir.joinpath("test_predictions.csv"), index=False)

    final_read = (
        "KAN is below the current ChemProp embedding + TabPFN sibling; do not submit directly."
    )
    if covered.all() and oof_metrics["MAE"] <= 0.47:
        final_read = (
            "KAN is competitive enough for a Caruana ADD/SWAP bakeoff before deciding."
        )
    report = "\n".join(
        [
            "# KAN on Frozen Embedding Report",
            "",
            f"Run name: `{args.run_name}`",
            f"Experiment: `{exp_name}`",
            f"Embedding: `{args.embed_path}`",
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
            "Use for deeper KAN/GNN-KAN only if it reaches MAE <= 0.47 or is clearly decorrelated with acceptable Spearman.",
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
            description="pykan/KAN regressor on frozen low-fidelity pretrained molecular embeddings",
            model_type="kan",
            feature_set=args.feature_name,
            hyperparameters={"args": args_for_json},
            fold_metrics=fold_metrics,
            submission_path=str(sub_path.relative_to(REPO_ROOT)),
            notes=f"OOF MAE={oof_metrics['MAE']:.4f}, KAN on {args.feature_name}",
            on_conflict_replace=True,
        )
        save_oof_predictions(exp_id, oof)

    print(f"Saved report: {report_path}")
    print(f"Saved submission: {sub_path}")


if __name__ == "__main__":
    main()
