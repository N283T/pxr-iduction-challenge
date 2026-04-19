"""MLP head on pooled Boltz-2 trunk embeddings (Phase C of #74).

Mirrors a simplified version of Boltz-2's own AffinityHeadsTransformer
(affinity.py:142-223): pooled z (mean over cross pairs) -> 2-stage MLP
bottleneck -> prediction head. We skip Boltz's upstream pairformer +
distogram injection because our input is already a post-trunk pair
embedding summary.

Architecture (mirrors Boltz scales):
  Input: 1024 (s_prot_mean + s_lig_mean + z_if_mean + z_if_max)
  Bottleneck: Linear(1024 -> 256) -> GELU -> Dropout(0.1)
              Linear(256  -> 128) -> GELU -> Dropout(0.1)
  Head:       Linear(128  -> 1)

Total ~300K params. Trained with AdamW, MSE loss, cosine LR schedule,
early stopping on val MAE. Seed=42 for reproducibility.

5-fold UMAP CV. Saves OOF to experiment `mlp_pooled_boltz_umap`
via evaluate.record_experiment + save_oof_predictions.
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
import torch.nn.functional as F
from scipy import stats
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
from data import DB_PARAMS  # noqa: E402
from evaluate import (  # noqa: E402
    compute_metrics,
    record_experiment,
    save_oof_predictions,
)
from splits import umap_split_indices  # noqa: E402


POOLED_PATH = REPO_ROOT.joinpath("data", "boltz_affhead", "pooled.parquet")


class AffinityMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        hidden1: int = 256,
        hidden2: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_one_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    *,
    max_epoch: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    patience: int,
    device: str,
    seed: int,
) -> tuple[np.ndarray, int]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AffinityMLP(input_dim=X_tr.shape[1], dropout=dropout).to(device)

    # Normalize each feature column using train stats (avoid leaking val)
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0) + 1e-6
    X_tr_n = ((X_tr - mean) / std).astype(np.float32)
    X_va_n = ((X_va - mean) / std).astype(np.float32)

    t_tr = torch.from_numpy(X_tr_n).to(device)
    y_tr_t = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    t_va = torch.from_numpy(X_va_n).to(device)
    y_va_t = torch.from_numpy(y_va.astype(np.float32)).to(device)

    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = CosineAnnealingLR(opt, T_max=max_epoch, eta_min=lr * 0.02)

    best_val = float("inf")
    best_epoch = -1
    best_pred = None
    bad_streak = 0
    n = len(t_tr)
    idx = np.arange(n)

    for epoch in range(max_epoch):
        model.train()
        np.random.shuffle(idx)
        for start in range(0, n, batch_size):
            batch = idx[start : start + batch_size]
            opt.zero_grad()
            pred = model(t_tr[batch])
            loss = F.mse_loss(pred, y_tr_t[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(t_va).cpu().numpy()
        val_mae = float(np.mean(np.abs(val_pred - y_va)))
        if val_mae < best_val - 1e-5:
            best_val = val_mae
            best_epoch = epoch
            best_pred = val_pred.copy()
            bad_streak = 0
        else:
            bad_streak += 1
            if bad_streak >= patience:
                break

    assert best_pred is not None
    return best_pred, best_epoch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--experiment-name", default="mlp_pooled_boltz_umap")
    ap.add_argument("--max-epoch", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    conn = psycopg2.connect(**DB_PARAMS)
    ta = pd.read_sql(
        "SELECT ta.compound_id, c.std_smiles, ta.pec50 "
        "FROM train_activity ta JOIN compounds c ON c.id = ta.compound_id "
        "ORDER BY ta.id",
        conn,
    )
    conn.close()
    train_ids = ta["compound_id"].tolist()
    y = ta["pec50"].to_numpy(dtype=np.float64)
    smiles = ta["std_smiles"].tolist()

    pool_df = pd.read_parquet(POOLED_PATH).set_index("compound_id")
    X = pool_df.reindex(index=train_ids).to_numpy(dtype=np.float32).copy()
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    print(f"Feature matrix: {X.shape}")

    splits = umap_split_indices(smiles, n_splits=5, seed=42)
    oof = np.full(len(y), np.nan)
    fold_metrics: list[dict] = []

    for k, (tr, va) in enumerate(splits):
        pred, best_epoch = train_one_fold(
            X[tr],
            y[tr],
            X[va],
            y[va],
            max_epoch=args.max_epoch,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            patience=args.patience,
            device=device,
            seed=args.seed + k,
        )
        oof[va] = pred
        m = compute_metrics(y[va], pred)
        fold_metrics.append(m)
        print(
            f"  fold{k}: MAE={m['MAE']:.4f}  RAE={m['RAE']:.4f}  "
            f"best_epoch={best_epoch}"
        )

    oof_m = compute_metrics(y, oof)
    avg_fold_mae = float(np.mean([fm["MAE"] for fm in fold_metrics]))
    print(
        f"\nAVG fold MAE={avg_fold_mae:.4f}  "
        f"OOF MAE={oof_m['MAE']:.4f}  RAE={oof_m['RAE']:.4f}  "
        f"Spearman={oof_m['Spearman_R']:+.4f}"
    )

    if args.register:
        exp_id = record_experiment(
            name=args.experiment_name,
            description="MLP head on pooled Boltz-2 trunk embeddings (#74 Phase C)",
            model_type="mlp",
            feature_set="pooled_boltz",
            hyperparameters={
                "max_epoch": args.max_epoch,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "dropout": args.dropout,
                "patience": args.patience,
                "architecture": "Linear(1024->256)->GELU->Drop->Linear(256->128)->GELU->Drop->Linear(128->1)",
                "seed": args.seed,
            },
            fold_metrics=fold_metrics,
            notes=(
                f"MLP head on Boltz-2 pooled trunk embeddings "
                f"(Phase C of issue #74); OOF MAE={oof_m['MAE']:.4f} "
                f"RAE={oof_m['RAE']:.4f}"
            ),
        )
        save_oof_predictions(exp_id, oof)


if __name__ == "__main__":
    main()
