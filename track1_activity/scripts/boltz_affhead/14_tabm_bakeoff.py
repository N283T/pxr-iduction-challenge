"""TabM bakeoff on PXR features. Parallel to the TabPFN pool members but
different model family (MLP + weight-shared k-ensemble, ICLR 2025).

Motivation: pool currently 100% TabPFN v7 for tabular members. TabM
claims to match/beat TabPFN on d>500 benchmarks via parameter-efficient
ensembling (k=32 MLPs with shared weights). Pool family diversity could
add decorrelating signal for caruana.

This script runs UMAP 5-fold CV on a configurable feature set and reports
OOF MAE vs the TabPFN baseline recorded in experiment_summary.

Usage:
    pixi run python track1_activity/scripts/boltz_affhead/14_tabm_bakeoff.py --feature cheme_2d_full_boltz_log2fc_pred
    pixi run python track1_activity/scripts/boltz_affhead/14_tabm_bakeoff.py --feature pooled_boltz_allpairs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import torch
import torch.nn as nn
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from data import DB_PARAMS  # noqa: E402
from evaluate import record_experiment, save_oof_predictions  # noqa: E402
from splits import umap_split_indices  # noqa: E402

SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")


def rae(y_true, y_pred):
    num = np.sum(np.abs(y_true - y_pred))
    den = np.sum(np.abs(y_true - np.mean(y_true)))
    return float(num / den) if den > 0 else float("nan")


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RAE": rae(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman_R": float(spearmanr(y_pred, y_true).statistic),
        "Kendall_Tau": float(kendalltau(y_pred, y_true).statistic),
    }


def train_tabm_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    *,
    device: torch.device,
    k: int = 32,
    max_epochs: int = 100,
    patience: int = 15,
    batch_size: int = 256,
    lr: float = 2e-3,
    weight_decay: float = 3e-4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit a TabM on (X_tr, y_tr), early-stop on (X_va, y_va), return
    (val_preds, test_preds, best_epoch)."""
    from tabm import TabM

    torch.manual_seed(seed)
    np.random.seed(seed)

    d_in = X_tr.shape[1]
    model = TabM.make(n_num_features=d_in, cat_cardinalities=None, d_out=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_tr_t = torch.from_numpy(X_tr).float()
    y_tr_t = torch.from_numpy(y_tr).float()
    X_va_t = torch.from_numpy(X_va).float().to(device)
    y_va_t = torch.from_numpy(y_va).float()
    X_te_t = torch.from_numpy(X_te).float().to(device)

    loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )

    def predict(x_t: torch.Tensor) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            out = model(x_t)  # (B, k, 1)
        return out.squeeze(-1).mean(dim=1).cpu().numpy()

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)  # (B, k, 1)
            pred = pred.squeeze(-1).flatten(0, 1)  # (B*k,)
            yb_rep = yb.repeat_interleave(model.backbone.k)
            loss = nn.functional.mse_loss(pred, yb_rep)
            loss.backward()
            opt.step()

        val_pred = predict(X_va_t)
        val_mae = float(np.mean(np.abs(y_va_t.numpy() - val_pred)))
        if val_mae < best_val - 1e-5:
            best_val = val_mae
            best_epoch = epoch
            best_state = {k_: v.cpu().clone() for k_, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_pred = predict(X_va_t)
    test_pred = predict(X_te_t)
    return val_pred, test_pred, best_epoch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--feature",
        type=str,
        default="cheme_2d_full_boltz_log2fc_pred",
        help="Feature name matching run_train.load_features",
    )
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import run_train  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Feature: {args.feature}  device: {device}")

    with psycopg2.connect(**DB_PARAMS) as conn:
        tr_df = pd.read_sql(
            """SELECT t.id AS compound_id, t.pec50, c.std_smiles AS smiles, c.molecule_name
                 FROM train_activity t JOIN compounds c ON c.id = t.id
                 ORDER BY t.id""",
            conn,
        )
        te_df = pd.read_sql(
            """SELECT t.id AS compound_id, c.std_smiles AS smiles, c.molecule_name
                 FROM test_activity t JOIN compounds c ON c.id = t.id
                 ORDER BY t.id""",
            conn,
        )

    X_tr, X_te = run_train.load_features(args.feature, tr_df, te_df)
    y = tr_df["pec50"].to_numpy(dtype=np.float32)
    print(f"  X_train {X_tr.shape}, X_test {X_te.shape}")

    # Sanitize + scale (TabM / MLPs need standardized input)
    col_mean = np.nanmean(X_tr, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    X_tr = np.where(np.isfinite(X_tr), X_tr, col_mean).astype(np.float32)
    X_te = np.where(np.isfinite(X_te), X_te, col_mean).astype(np.float32)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr).astype(np.float32)
    X_te_scaled = scaler.transform(X_te).astype(np.float32)

    splits = umap_split_indices(
        tr_df["smiles"].tolist(), n_splits=5, n_clusters=50, seed=args.seed
    )

    oof = np.zeros(len(y), dtype=np.float32)
    test_preds_per_fold: list[np.ndarray] = []
    fold_metrics: list[dict] = []

    for fold, (tr_idx, va_idx) in enumerate(splits):
        print(f"\n[Fold {fold}] train={len(tr_idx)}, val={len(va_idx)}")
        val_pred, test_pred, best_epoch = train_tabm_fold(
            X_tr_scaled[tr_idx],
            y[tr_idx],
            X_tr_scaled[va_idx],
            y[va_idx],
            X_te_scaled,
            device=device,
            k=args.k,
            max_epochs=args.max_epochs,
            patience=args.patience,
            lr=args.lr,
            seed=args.seed + fold,
        )
        oof[va_idx] = val_pred
        test_preds_per_fold.append(test_pred)
        m = compute_metrics(y[va_idx], val_pred)
        fold_metrics.append(m)
        print(
            f"  Fold {fold} best_epoch={best_epoch} MAE={m['MAE']:.4f} "
            f"RAE={m['RAE']:.4f} Sp={m['Spearman_R']:.4f}"
        )

    overall = compute_metrics(y, oof)
    print("\nOverall OOF:")
    for k_, v in overall.items():
        print(f"  {k_}={v:.4f}")

    # Compare with TabPFN baseline on same feature
    ref_name = f"tabpfn_{args.feature}_umap_default"
    print("\nReference (TabPFN on same feature):")
    with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT mae_mean, rae_mean, spearman_mean FROM experiment_summary WHERE name=%s",
            (ref_name,),
        )
        row = cur.fetchone()
        if row:
            print(f"  {ref_name}: MAE={row[0]:.4f} RAE={row[1]:.4f} Sp={row[2]:.4f}")
            print(
                f"  Δ (TabM - TabPFN): MAE={overall['MAE'] - row[0]:+.4f} "
                f"RAE={overall['RAE'] - row[1]:+.4f} Sp={overall['Spearman_R'] - row[2]:+.4f}"
            )
        else:
            print(f"  {ref_name}: not found in experiment_summary")

    # Save submission
    test_preds_mean = np.mean(np.stack(test_preds_per_fold), axis=0)
    exp_name = f"tabm_{args.feature}_umap"
    sub = pd.DataFrame(
        {
            "SMILES": te_df["smiles"],
            "Molecule Name": te_df["molecule_name"],
            "pEC50": test_preds_mean,
        }
    )
    sub_path = SUBMISSION_DIR.joinpath(f"{exp_name}.csv")
    sub.to_csv(sub_path, index=False)
    print(f"\nSaved: {sub_path}")

    exp_id = record_experiment(
        name=exp_name,
        description=f"TabM (k={args.k}) on {args.feature}, UMAP 5-fold CV.",
        model_type="tabm",
        feature_set=args.feature,
        hyperparameters={
            "k": args.k,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "lr": args.lr,
            "weight_decay": 3e-4,
            "batch_size": 256,
            "seed": args.seed,
        },
        fold_metrics=fold_metrics,
        submission_path=f"track1_activity/submissions/{exp_name}.csv",
        notes=f"TabM pool-diversity test. OOF MAE={overall['MAE']:.4f} RAE={overall['RAE']:.4f}",
    )
    save_oof_predictions(exp_id, oof)
    print(f"Recorded experiment id={exp_id}: {exp_name}")


if __name__ == "__main__":
    main()
