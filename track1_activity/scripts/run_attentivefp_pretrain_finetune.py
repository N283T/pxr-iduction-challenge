#!/usr/bin/env -S pixi run python
"""AttentiveFP pretrain + fine-tune on single_conc log2_fc -> pEC50.

Companion to run_chemprop_pretrain.py / run_chemprop_finetune.py. The
method that made chemprop_pretrain_finetune_frozen_umap the new pool
leader applies the same pretrain+frozen recipe to the existing
AttentiveFP member which currently carries weight ~0 in the ensemble
(OOF RAE 0.5803, zero contribution in vanilla weighting).

Phases:
  * ``--phase pretrain``: 13,136 compounds, 2-output-channel AttentiveFP
    trained on [log2_fc @ 8.25 uM, log2_fc @ 33 uM] with masked MSE.
    90/10 random val split, save state_dict.
  * ``--phase finetune``: 4,140 pEC50, 1-output-channel AttentiveFP.
    Encoder weights loaded from pretrain, output layer fresh. UMAP
    5-fold CV. ``--freeze-encoder`` locks everything but the final
    output linear (linear-probe style). ``--no-pretrain`` ablation
    trains from scratch.

Architecture reuses the Optuna-tuned hyperparams already in DB for
``attentivefp_optuna_umap`` (hidden 512, 4 layers, 3 timesteps,
dropout 0.1, lr 3.3e-4, batch 32). Pretrain overrides batch_size to
128 and max_epochs/patience so the larger 13k dataset fits in memory
and the run finishes in reasonable time.

Usage:
    pixi run python track1_activity/scripts/run_attentivefp_pretrain_finetune.py \\
        --phase pretrain
    pixi run python track1_activity/scripts/run_attentivefp_pretrain_finetune.py \\
        --phase finetune --freeze-encoder
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
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Batch
from torch_geometric.nn.models import AttentiveFP
from torch_geometric.utils import from_smiles

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS, load_test_smiles, load_train_smiles_target  # noqa: E402
from evaluate import (  # noqa: E402
    compute_metrics,
    print_fold_summary,
    print_metrics,
    record_experiment,
    save_oof_predictions,
)
from splits import umap_split_indices  # noqa: E402


SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

CKPT_DIR = REPO_ROOT.joinpath(
    "track1_activity", "checkpoints", "attentivefp_pretrain"
)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
PRETRAIN_PATH = CKPT_DIR.joinpath("pretrain.pt")


# Matches DB row 'attentivefp_optuna_umap' (50-trial Optuna best)
TUNED_PARAMS = {
    "hidden_channels": 512,
    "num_layers": 4,
    "num_timesteps": 3,
    "dropout": 0.1,
    "learning_rate": 0.00033232775773741117,
    "batch_size": 32,
    "max_epochs": 150,
    "patience": 15,
}

# Pretrain-specific overrides: bigger batch for 13k, slightly fewer epochs.
PRETRAIN_OVERRIDES = {
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 12,
}


def smiles_to_pyg(smiles_list: list[str]):
    out = []
    for i, smi in enumerate(smiles_list):
        g = from_smiles(smi)
        if g.x is None or g.x.shape[0] == 0:
            raise ValueError(f"SMILES[{i}] produced empty graph: {smi}")
        out.append(g)
    return out


def iter_batches(graphs, targets, batch_size, shuffle, rng=None):
    n = len(graphs)
    idx = np.arange(n)
    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        rng.shuffle(idx)
    for i in range(0, n, batch_size):
        sel = idx[i : i + batch_size]
        batch = Batch.from_data_list([graphs[j] for j in sel])
        if targets is not None:
            y = torch.tensor(
                np.asarray([targets[j] for j in sel], dtype=np.float32)
            )
            yield batch, y, sel
        else:
            yield batch, None, sel


def build_model(n_out: int, params: dict, in_channels: int, edge_dim: int):
    return AttentiveFP(
        in_channels=in_channels,
        hidden_channels=params["hidden_channels"],
        out_channels=n_out,
        edge_dim=edge_dim,
        num_layers=params["num_layers"],
        num_timesteps=params["num_timesteps"],
        dropout=params["dropout"],
    )


# ---------------------------------------------------------------------------
# Pretrain phase
# ---------------------------------------------------------------------------


def load_pretrain_data() -> tuple[list[str], np.ndarray, list[int]]:
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


def pretrain(args):
    params = {**TUNED_PARAMS, **PRETRAIN_OVERRIDES}
    if args.max_epochs is not None:
        params["max_epochs"] = args.max_epochs

    print("AttentiveFP pretrain (13k + single_conc 2-head)")
    print(f"  params: {params}")

    smiles, targets, _ = load_pretrain_data()
    n = len(smiles)
    n_v8 = int(np.isfinite(targets[:, 0]).sum())
    n_v33 = int(np.isfinite(targets[:, 1]).sum())
    print(f"  data: {n} compounds, 8.25uM valid={n_v8}, 33uM valid={n_v33}")

    # z-score (NaN-safe)
    means = np.zeros(2, dtype=np.float32)
    stds = np.ones(2, dtype=np.float32)
    for i in range(2):
        valid = np.isfinite(targets[:, i])
        means[i] = float(np.mean(targets[valid, i]))
        stds[i] = float(np.std(targets[valid, i]))
        if stds[i] < 1e-6:
            stds[i] = 1.0
    targets_z = (targets - means) / stds

    rng = np.random.default_rng(args.seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(n * args.val_frac)
    va_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    tr_smi = [smiles[i] for i in tr_idx]
    va_smi = [smiles[i] for i in va_idx]
    tr_y = targets_z[tr_idx]
    va_y = targets_z[va_idx]
    print(f"  split: train={len(tr_idx)}, val={len(va_idx)}")

    # Build PyG graphs once
    print("  building PyG graphs...")
    tr_g = smiles_to_pyg(tr_smi)
    va_g = smiles_to_pyg(va_smi)
    in_channels = tr_g[0].x.shape[1]
    edge_dim = tr_g[0].edge_attr.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(n_out=2, params=params, in_channels=in_channels, edge_dim=edge_dim).to(device)

    opt = Adam(model.parameters(), lr=params["learning_rate"])
    sched = CosineAnnealingLR(opt, T_max=params["max_epochs"], eta_min=1e-7)

    # Task weights for weighted-MSE (main 8.25uM=1.0, aux 33uM=0.5)
    task_w = torch.tensor([args.w_8p25, args.w_33], dtype=torch.float32, device=device)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    batch_rng = np.random.default_rng(args.seed)

    for epoch in range(params["max_epochs"]):
        model.train()
        tr_losses = []
        for batch, y, _ in iter_batches(tr_g, tr_y, params["batch_size"], shuffle=True, rng=batch_rng):
            batch = batch.to(device)
            y = y.to(device)  # (B, 2)
            opt.zero_grad()
            out = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)  # (B, 2)
            mask = torch.isfinite(y)
            diff = (out - torch.nan_to_num(y, nan=0.0)) ** 2
            # Apply mask + task weights
            weighted = diff * mask.float() * task_w.view(1, -1)
            # Mean over valid entries (not full product), per-task avg then sum
            valid_per_task = mask.float().sum(dim=0).clamp(min=1.0)
            per_task_loss = weighted.sum(dim=0) / valid_per_task
            loss = per_task_loss.sum()
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tr_losses.append(float(loss))
        sched.step()

        # Validation
        model.eval()
        va_losses = []
        with torch.no_grad():
            for batch, y, _ in iter_batches(va_g, va_y, params["batch_size"], shuffle=False):
                batch = batch.to(device)
                y = y.to(device)
                out = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
                mask = torch.isfinite(y)
                diff = (out - torch.nan_to_num(y, nan=0.0)) ** 2
                weighted = diff * mask.float() * task_w.view(1, -1)
                valid_per_task = mask.float().sum(dim=0).clamp(min=1.0)
                per_task_loss = weighted.sum(dim=0) / valid_per_task
                va_losses.append(float(per_task_loss.sum()))
        val_loss = float(np.mean(va_losses))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or patience_counter >= params["patience"]:
            print(
                f"  epoch {epoch:>3d}  tr_loss={np.mean(tr_losses):.4f}  "
                f"val_loss={val_loss:.4f}  best={best_val_loss:.4f}"
            )
        if patience_counter >= params["patience"]:
            print(f"  early stop at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("pretrain never improved")

    # Save
    torch.save(
        {
            "state_dict": best_state,
            "params": params,
            "target_means": means.tolist(),
            "target_stds": stds.tolist(),
            "task_weights": [args.w_8p25, args.w_33],
            "n_train": len(tr_idx),
            "n_val": len(va_idx),
            "in_channels": in_channels,
            "edge_dim": edge_dim,
            "best_val_loss": best_val_loss,
        },
        PRETRAIN_PATH,
    )
    meta_path = CKPT_DIR.joinpath("pretrain_meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "params": params,
                "target_means": means.tolist(),
                "target_stds": stds.tolist(),
                "task_weights_8p25_33": [args.w_8p25, args.w_33],
                "n_train": len(tr_idx),
                "n_val": len(va_idx),
                "best_val_loss": best_val_loss,
            },
            indent=2,
        )
    )
    print(f"\n  saved: {PRETRAIN_PATH}")
    print(f"  meta:  {meta_path}")
    print(f"  best val_loss = {best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# Fine-tune phase
# ---------------------------------------------------------------------------


def load_pretrain_encoder(model: AttentiveFP, ckpt_path: Path) -> dict:
    """Load AttentiveFP encoder weights, skipping the final output layer.

    The pretrain model has out_channels=2; the fine-tune model has
    out_channels=1. All weights except those feeding the final output
    can be reused. In PyG's AttentiveFP the output linear is named
    ``lin2`` (projection from mol_gru output to out_channels).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]

    # lin2 is the final out_channels projection in PyG AttentiveFP
    skip_prefixes = ("lin2.",)
    encoder_state = {
        k: v for k, v in state.items() if not any(k.startswith(p) for p in skip_prefixes)
    }
    skipped = [k for k in state if any(k.startswith(p) for p in skip_prefixes)]

    result = model.load_state_dict(encoder_state, strict=False)
    return {
        "loaded": len(encoder_state),
        "skipped": skipped,
        "missing": list(result.missing_keys),
        "unexpected": list(result.unexpected_keys),
        "pretrain_best_val_loss": ckpt.get("best_val_loss", -1),
    }


def freeze_encoder(model: AttentiveFP) -> int:
    """Freeze all params except the final output lin (lin2)."""
    frozen = 0
    for name, p in model.named_parameters():
        if not name.startswith("lin2."):
            p.requires_grad = False
            frozen += 1
    return frozen


def finetune(args):
    if not PRETRAIN_PATH.exists() and not args.no_pretrain:
        raise FileNotFoundError(
            f"Missing {PRETRAIN_PATH}. Run with --phase pretrain first."
        )

    params = TUNED_PARAMS.copy()
    if args.max_epochs is not None:
        params["max_epochs"] = args.max_epochs
    print(
        f"AttentiveFP fine-tune | no_pretrain={args.no_pretrain} | "
        f"freeze={args.freeze_encoder} | max_epochs={params['max_epochs']}"
    )

    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y = train_df["pec50"].to_numpy(dtype=np.float32)
    print(f"  train: {len(train_df)}, test: {len(test_df)}")

    print("  building PyG graphs...")
    train_g = smiles_to_pyg(train_df["smiles"].tolist())
    test_g = smiles_to_pyg(test_df["smiles"].tolist())
    in_channels = train_g[0].x.shape[1]
    edge_dim = train_g[0].edge_attr.shape[1]

    outer = umap_split_indices(train_df["smiles"].tolist(), n_splits=5, n_clusters=50, seed=42)

    exp_name = "attentivefp_pretrain_finetune"
    if args.no_pretrain:
        exp_name = "attentivefp_finetune_nopretrain_ablation"
    elif args.freeze_encoder:
        exp_name += "_frozen"
    exp_name += "_umap"
    print(f"  Experiment: {exp_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oof = np.zeros(len(train_df), dtype=np.float32)
    fold_metrics = []
    test_preds_per_fold = []

    for fold, (tr_idx, va_idx) in enumerate(outer):
        print(f"\n[Fold {fold}] train={len(tr_idx)}, val={len(va_idx)}")
        tr_g = [train_g[i] for i in tr_idx]
        va_g = [train_g[i] for i in va_idx]
        tr_y = y[tr_idx]
        va_y = y[va_idx]

        model = build_model(n_out=1, params=params, in_channels=in_channels, edge_dim=edge_dim).to(device)

        if not args.no_pretrain:
            summary = load_pretrain_encoder(model, PRETRAIN_PATH)
            if fold == 0:
                print(f"  Loaded encoder: {summary}")

        if args.freeze_encoder:
            n_frozen = freeze_encoder(model)
            if fold == 0:
                print(f"  Frozen {n_frozen} param tensors (all except lin2)")

        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = Adam(trainable, lr=params["learning_rate"])
        sched = CosineAnnealingLR(opt, T_max=params["max_epochs"], eta_min=1e-7)
        criterion = nn.MSELoss()

        best_val_mae = float("inf")
        best_state = None
        patience_counter = 0
        batch_rng = np.random.default_rng(42 + fold)

        for epoch in range(params["max_epochs"]):
            model.train()
            for batch, ys, _ in iter_batches(tr_g, tr_y, params["batch_size"], shuffle=True, rng=batch_rng):
                batch = batch.to(device)
                ys = ys.to(device)
                opt.zero_grad()
                out = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch).squeeze(-1)
                loss = criterion(out, ys)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
            sched.step()

            model.eval()
            preds = []
            with torch.no_grad():
                for batch, _, _ in iter_batches(va_g, va_y, params["batch_size"], shuffle=False):
                    batch = batch.to(device)
                    out = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch).squeeze(-1)
                    preds.append(out.cpu().numpy())
            val_preds = np.concatenate(preds)
            val_mae = float(np.mean(np.abs(va_y - val_preds)))

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= params["patience"]:
                break

        if best_state is None:
            raise RuntimeError(f"Fold {fold}: never improved")

        model.load_state_dict(best_state)
        model.eval()
        preds = []
        with torch.no_grad():
            for batch, _, _ in iter_batches(va_g, va_y, params["batch_size"], shuffle=False):
                batch = batch.to(device)
                out = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch).squeeze(-1)
                preds.append(out.cpu().numpy())
        val_preds = np.concatenate(preds)
        oof[va_idx] = val_preds

        # Test predictions for this fold
        tpreds = []
        with torch.no_grad():
            for batch, _, _ in iter_batches(test_g, None, params["batch_size"], shuffle=False):
                batch = batch.to(device)
                out = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch).squeeze(-1)
                tpreds.append(out.cpu().numpy())
        test_preds_per_fold.append(np.concatenate(tpreds))

        m = compute_metrics(va_y, val_preds)
        fold_metrics.append(m)
        print_metrics(m, label=f"Fold {fold}")

        del model
        torch.cuda.empty_cache()

    oof_metrics = compute_metrics(y, oof)
    print("\n  Overall OOF:")
    print_metrics(oof_metrics)
    print_fold_summary(fold_metrics)

    test_preds_mean = np.mean(test_preds_per_fold, axis=0)
    sub = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": test_preds_mean,
        }
    )
    sub_path = SUBMISSION_DIR.joinpath(f"{exp_name}.csv")
    sub_tmp = sub_path.with_suffix(sub_path.suffix + ".tmp")
    sub.to_csv(sub_tmp, index=False)

    try:
        exp_id = record_experiment(
            name=exp_name,
            description=(
                f"AttentiveFP fine-tune on pEC50 with "
                f"{'scratch' if args.no_pretrain else 'pretrained'} encoder "
                f"({'frozen' if args.freeze_encoder else 'full'} fine-tune), "
                f"UMAP 5-fold CV"
            ),
            model_type="attentivefp",
            feature_set="molecular_graph",
            hyperparameters={
                **params,
                "pretrain_path": (
                    str(PRETRAIN_PATH) if not args.no_pretrain else None
                ),
                "freeze_encoder": args.freeze_encoder,
                "no_pretrain": args.no_pretrain,
            },
            fold_metrics=fold_metrics,
            submission_path=f"track1_activity/submissions/{exp_name}.csv",
            notes=(
                f"OOF RAE={oof_metrics['RAE']:.4f}, "
                f"pretrain={'off' if args.no_pretrain else 'on'}, "
                f"freeze={args.freeze_encoder}"
            ),
        )
    except Exception:
        sub_tmp.unlink(missing_ok=True)
        raise

    sub_tmp.replace(sub_path)
    print(f"  Saved submission: {sub_path}")
    save_oof_predictions(exp_id, oof)
    print(f"\n  Done: {exp_name} -> RAE={oof_metrics['RAE']:.4f}")


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="AttentiveFP pretrain+finetune")
    parser.add_argument(
        "--phase", choices=["pretrain", "finetune"], required=True
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w-8p25", type=float, default=1.0)
    parser.add_argument("--w-33", type=float, default=0.5)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--no-pretrain", action="store_true")
    args = parser.parse_args()

    if args.phase == "pretrain":
        pretrain(args)
    else:
        finetune(args)


if __name__ == "__main__":
    main()
