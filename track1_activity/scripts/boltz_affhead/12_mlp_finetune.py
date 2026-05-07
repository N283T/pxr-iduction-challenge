"""α'' — Fine-tune the Phase 3 C-concat pretrained encoder on PXR pEC50.

Two modes:
  head-only: freeze the pretrained encoder, train only the new regression
             head (replaces the log2_fc 2-head with a pEC50 1-head MLP).
  full-ft:   unfreeze encoder + train new head end-to-end at a lower lr.

Conceptually matches the paper's "trunk gradient-detached, affinity
module updated" design: the pretrained encoder (our 4-token-transformer,
trained on 13134 × log2_fc) plays the role of trunk; the new head plays
the role of affinity module; PXR pEC50 is the fine-tune target.

Training: UMAP 5-fold CV (seed=42, k=50) matching run_train.py's default.
Output: experiment row `boltz_trunk_pretrain_c_concat_<mode>_umap` with
OOF predictions + test submission CSV.

Usage:
    pixi run python track1_activity/scripts/boltz_affhead/12_mlp_finetune.py --mode head-only
    pixi run python track1_activity/scripts/boltz_affhead/12_mlp_finetune.py --mode full-ft
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import torch
import torch.nn as nn
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from data import DB_PARAMS  # noqa: E402
from evaluate import record_experiment, save_oof_predictions  # noqa: E402
from splits import umap_split_indices  # noqa: E402

CKPT_DIR = REPO_ROOT.joinpath(
    "track1_activity", "checkpoints", "boltz_trunk_pretrain_c_concat"
)
SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")


def load_pretrain_module():
    spec = importlib.util.spec_from_file_location(
        "_mlp_pretrain",
        REPO_ROOT.joinpath(
            "track1_activity", "scripts", "boltz_affhead", "09_mlp_pretrain.py"
        ),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_mlp_pretrain"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_trunk_1024(conn, compound_ids: list[int]) -> np.ndarray:
    """Fetch all_pairs pooled 1024d trunk for given compound_ids (rcycle=3)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT compound_id, s_prot_mean, s_lig_mean, z_if_mean, z_if_max
              FROM compound_boltz2_trunk_fast
             WHERE recycling_steps = 3
               AND compound_id = ANY(%s)
            """,
            (compound_ids,),
        )
        rows = {r[0]: r[1:] for r in cur.fetchall()}

    out = np.zeros((len(compound_ids), 1024), dtype=np.float32)
    missing: list[int] = []
    for i, cid in enumerate(compound_ids):
        if cid not in rows:
            missing.append(cid)
            continue
        s_prot, s_lig, z_mean, z_max = rows[cid]
        out[i, :384] = s_prot
        out[i, 384:768] = s_lig
        out[i, 768:896] = z_mean
        out[i, 896:1024] = z_max
    if missing:
        # Fill with column means so training is well-defined
        col_mean = out.mean(axis=0)
        for i, cid in enumerate(compound_ids):
            if cid in missing:
                out[i] = col_mean
        print(
            f"  WARNING: {len(missing)} compounds missing trunk, filled with column mean"
        )
    return out


def load_train_ids_and_y(conn) -> tuple[list[int], np.ndarray, list[str], list[str]]:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT t.id, t.pec50, c.std_smiles, c.molecule_name
                 FROM train_activity t JOIN compounds c ON c.id = t.compound_id
                 ORDER BY t.id"""
        )
        rows = cur.fetchall()
    ids = [r[0] for r in rows]
    y = np.asarray([r[1] for r in rows], dtype=np.float32)
    smiles = [r[2] for r in rows]
    names = [r[3] for r in rows]
    return ids, y, smiles, names


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


def build_model(mod, head_hidden: int = 256) -> nn.Module:
    """Load pretrained C-concat encoder and attach a fresh regression head."""
    encoder = mod.MLPVariantC(pool="concat", hidden_dim=None)
    # HIDDEN == 1024 for concat pool; load the pretrained weights
    state = torch.load(
        CKPT_DIR.joinpath("pretrain.pt"), map_location="cpu", weights_only=True
    )
    encoder.load_state_dict(state)

    # Swap the 1024 -> 2 head (log2_fc) with a fresh 1024 -> hidden -> 1 head
    encoder.head = nn.Sequential(
        nn.Linear(1024, head_hidden),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(head_hidden, 1),
    )
    return encoder


def freeze_encoder(model: nn.Module) -> int:
    """Freeze every parameter except those under `.head.`. Returns n frozen."""
    n = 0
    for name, p in model.named_parameters():
        if not name.startswith("head."):
            p.requires_grad = False
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["head-only", "full-ft"], required=True)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument(
        "--encoder-lr",
        type=float,
        default=1e-5,
        help="Used only in --mode full-ft; head keeps --head-lr.",
    )
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--head-hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mod = load_pretrain_module()

    print("Loading train + test data ...")
    conn = psycopg2.connect(**DB_PARAMS)
    try:
        train_ids, y, train_smiles, train_names = load_train_ids_and_y(conn)
        with conn.cursor() as cur:
            cur.execute(
                """SELECT t.id, c.std_smiles, c.molecule_name
                     FROM test_activity t JOIN compounds c ON c.id = t.compound_id
                     ORDER BY t.id"""
            )
            test_rows = cur.fetchall()
        test_df = pd.DataFrame(
            test_rows, columns=["compound_id", "smiles", "molecule_name"]
        )
        test_ids = test_df["compound_id"].tolist()

        print(f"  train={len(train_ids)}, test={len(test_ids)}")
        X_train = load_trunk_1024(conn, train_ids)
        X_test = load_trunk_1024(conn, test_ids)
    finally:
        conn.close()

    print(f"  X_train {X_train.shape}, X_test {X_test.shape}")

    outer = umap_split_indices(train_smiles, n_splits=5, n_clusters=50, seed=42)

    exp_name = f"boltz_trunk_pretrain_c_concat_{args.mode.replace('-', '_')}_umap"
    print(f"\nExperiment: {exp_name}")

    oof = np.zeros(len(y), dtype=np.float32)
    fold_metrics: list[dict] = []
    test_preds_per_fold: list[np.ndarray] = []

    X_train_t = torch.from_numpy(X_train)
    y_t = torch.from_numpy(y)
    X_test_t = torch.from_numpy(X_test).to(device)

    for fold, (tr_idx, va_idx) in enumerate(outer):
        print(f"\n[Fold {fold}] train={len(tr_idx)}, val={len(va_idx)}")

        model = build_model(mod, head_hidden=args.head_hidden).to(device)

        if args.mode == "head-only":
            n_frozen = freeze_encoder(model)
            if fold == 0:
                total = sum(1 for _ in model.parameters())
                n_trainable = total - n_frozen
                print(f"  Frozen {n_frozen} param tensors, {n_trainable} trainable")
            trainable = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(
                trainable, lr=args.head_lr, weight_decay=args.weight_decay
            )
        else:
            encoder_params = [
                p for n, p in model.named_parameters() if not n.startswith("head.")
            ]
            head_params = [
                p for n, p in model.named_parameters() if n.startswith("head.")
            ]
            if fold == 0:
                print(
                    f"  Full-FT: encoder params={sum(p.numel() for p in encoder_params):,}, "
                    f"head params={sum(p.numel() for p in head_params):,}"
                )
            opt = torch.optim.AdamW(
                [
                    {"params": encoder_params, "lr": args.encoder_lr},
                    {"params": head_params, "lr": args.head_lr},
                ],
                weight_decay=args.weight_decay,
            )

        criterion = nn.MSELoss()

        tr_loader = DataLoader(
            TensorDataset(X_train_t[tr_idx], y_t[tr_idx]),
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=(device.type == "cuda"),
        )
        va_loader = DataLoader(
            TensorDataset(X_train_t[va_idx], y_t[va_idx]),
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=(device.type == "cuda"),
        )

        best_val_mae = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, args.max_epochs + 1):
            model.train()
            tr_losses: list[float] = []
            for xb, yb in tr_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                pred = model(xb).squeeze(-1)
                loss = criterion(pred, yb)
                loss.backward()
                opt.step()
                tr_losses.append(float(loss.detach()))

            model.eval()
            val_preds: list[np.ndarray] = []
            with torch.no_grad():
                for xb, _ in va_loader:
                    xb = xb.to(device, non_blocking=True)
                    val_preds.append(model(xb).squeeze(-1).cpu().numpy())
            val_pred = np.concatenate(val_preds)
            val_mae = float(np.mean(np.abs(y[va_idx] - val_pred)))

            if val_mae < best_val_mae - 1e-5:
                best_val_mae = val_mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  early stop at epoch {epoch}, best val MAE={best_val_mae:.4f}")
                break

        if best_state is None:
            raise RuntimeError(f"Fold {fold}: never improved")

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            oof[va_idx] = model(X_train_t[va_idx].to(device)).squeeze(-1).cpu().numpy()
            test_preds_per_fold.append(model(X_test_t).squeeze(-1).cpu().numpy())

        fold_m = compute_metrics(y[va_idx], oof[va_idx])
        fold_metrics.append(fold_m)
        print(
            f"  Fold {fold} MAE={fold_m['MAE']:.4f} RAE={fold_m['RAE']:.4f} "
            f"Sp={fold_m['Spearman_R']:.4f}"
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Aggregate
    overall = compute_metrics(y, oof)
    print("\nOverall OOF:")
    for k, v in overall.items():
        print(f"  {k}={v:.4f}")
    print("\nFold summary:")
    for k in ["MAE", "RAE", "Spearman_R"]:
        vals = np.asarray([m[k] for m in fold_metrics])
        print(f"  {k}: {vals.mean():.4f} ± {vals.std():.4f}")

    # Submission
    test_preds_mean = np.mean(np.stack(test_preds_per_fold), axis=0)
    sub = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df.get(
                "molecule_name",
                test_df.get("name", ["OADMET-UNKNOWN"] * len(test_df)),
            ),
            "pEC50": test_preds_mean,
        }
    )
    sub_path = SUBMISSION_DIR.joinpath(f"{exp_name}.csv")
    sub.to_csv(sub_path, index=False)
    print(f"\nSaved: {sub_path}")

    exp_id = record_experiment(
        name=exp_name,
        description=(
            f"α'' {args.mode}: Phase 3 C-concat pretrained encoder (log2_fc) "
            f"+ new head fine-tuned on PXR pEC50. UMAP 5-fold CV."
        ),
        model_type="boltz_trunk_pretrain_mlp_ft",
        feature_set="boltz_trunk_1024_pretrained",
        hyperparameters={
            "mode": args.mode,
            "head_hidden": args.head_hidden,
            "head_lr": args.head_lr,
            "encoder_lr": args.encoder_lr if args.mode == "full-ft" else None,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "pretrain_checkpoint": str(CKPT_DIR.joinpath("pretrain.pt")),
        },
        fold_metrics=fold_metrics,
        submission_path=f"track1_activity/submissions/{exp_name}.csv",
        notes=f"mode={args.mode} OOF MAE={overall['MAE']:.4f} RAE={overall['RAE']:.4f}",
    )
    save_oof_predictions(exp_id, oof)
    print(f"\nRecorded experiment id={exp_id}: {exp_name}")


if __name__ == "__main__":
    main()
