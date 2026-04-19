#!/usr/bin/env -S pixi run python
"""ChemProp fine-tune on pEC50 using a pretrained encoder.

Phase 2 of the pretrain + fine-tune pipeline. Consumes the state_dict
produced by ``run_chemprop_pretrain.py`` (at
``track1_activity/checkpoints/chemprop_pretrain/pretrain.pt``), builds
a fresh single-task MPNN with the same encoder architecture, loads
only the encoder weights (message_passing + agg), and trains a new
pEC50 head with 5-fold UMAP CV.

Key design:
  * Encoder weights loaded from pretrain; predictor (FFN) is fresh.
  * No frozen layers -- full end-to-end fine-tune. Optuna defaults
    (depth 4, hidden 256, dropout 0.2, lr 1.36e-4 * 10) work well for
    single-task chemprop per history, and warm-start should speed
    convergence + improve generalization.
  * NaN-mask: pEC50 is never missing in train_activity, so no mask
    needed for the main task.
  * OOF predictions saved to DB; test predictions averaged across
    folds for a fresh submission CSV.

Usage:
    pixi run python track1_activity/scripts/run_chemprop_finetune.py
    pixi run python track1_activity/scripts/run_chemprop_finetune.py \\
        --freeze-encoder    # linear-probe style, only FFN trains
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from chemprop import data as chemprop_data  # noqa: E402
from chemprop import models, nn  # noqa: E402
from chemprop.nn.metrics import MSE  # noqa: E402

from data import load_test_smiles, load_train_smiles_target  # noqa: E402
from evaluate import (  # noqa: E402
    compute_metrics,
    print_fold_summary,
    print_metrics,
    record_experiment,
    save_oof_predictions,
)
from splits import scaffold_split_indices, umap_split_indices  # noqa: E402

torch.set_float32_matmul_precision("medium")

SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

CKPT_DIR = REPO_ROOT.joinpath("track1_activity", "checkpoints", "chemprop_pretrain")
PRETRAIN_PATH = CKPT_DIR.joinpath("pretrain.pt")


AGG_REGISTRY = {
    "mean": nn.MeanAggregation,
    "sum": nn.SumAggregation,
    "norm": nn.NormAggregation,
}


def build_finetune_model(params: dict):
    mp = nn.BondMessagePassing(
        d_h=params["message_hidden_dim"],
        depth=params["depth"],
        dropout=params["mp_dropout"],
        activation=params["activation"],
    )
    agg = AGG_REGISTRY[params["aggregation"]]()
    criterion = MSE(task_weights=torch.tensor([1.0], dtype=torch.float32))
    ffn = nn.RegressionFFN(
        n_tasks=1,
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


def load_pretrain_encoder_weights(model, pretrain_path: Path) -> dict:
    """Load message_passing + agg weights from pretrain checkpoint.

    Discards ``predictor.*`` keys (2-task vs 1-task shape mismatch) and
    any buffers that Lightning adds but we don't need. Returns a
    summary dict for logging.
    """
    ckpt = torch.load(pretrain_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]

    # Skip predictor.* (FFN head, 2-task vs 1-task shape mismatch) and
    # metrics.* (chemprop metric modules also carry per-task weight buffers).
    encoder_state = {
        k: v
        for k, v in state.items()
        if not (k.startswith("predictor.") or k.startswith("metrics."))
    }
    skipped = [
        k for k in state if k.startswith("predictor.") or k.startswith("metrics.")
    ]

    result = model.load_state_dict(encoder_state, strict=False)
    return {
        "loaded": len(encoder_state),
        "skipped_predictor": len(skipped),
        "missing_in_model": list(result.missing_keys),
        "unexpected_in_ckpt": list(result.unexpected_keys),
        "pretrain_val_loss": ckpt.get("final_val_loss", -1),
    }


def make_dataloader(smiles, targets, batch_size, shuffle):
    pts = [
        chemprop_data.MoleculeDatapoint.from_smi(smi, np.asarray([y], dtype=np.float32))
        for smi, y in zip(smiles, targets)
    ]
    return chemprop_data.build_dataloader(
        chemprop_data.MoleculeDataset(pts),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def predict_main(trainer, model, smiles, batch_size) -> np.ndarray:
    pts = [
        chemprop_data.MoleculeDatapoint.from_smi(
            smi, np.asarray([np.nan], dtype=np.float32)
        )
        for smi in smiles
    ]
    loader = chemprop_data.build_dataloader(
        chemprop_data.MoleculeDataset(pts),
        batch_size=batch_size,
        shuffle=False,
    )
    preds = trainer.predict(model, loader)
    arr = np.concatenate([p.numpy() for p in preds], axis=0)
    if arr.ndim == 2:
        arr = arr[:, 0]
    return arr


def freeze_encoder(model) -> int:
    """Freeze message_passing + agg parameters. Returns frozen count."""
    frozen = 0
    for name, p in model.named_parameters():
        if name.startswith("message_passing.") or name.startswith("agg."):
            p.requires_grad = False
            frozen += 1
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser(description="ChemProp pEC50 fine-tune")
    parser.add_argument("--split", choices=["umap", "scaffold"], default="umap")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze message_passing + agg (linear-probe style).",
    )
    parser.add_argument(
        "--no-pretrain",
        action="store_true",
        help="Train from scratch -- ablation control.",
    )
    parser.add_argument(
        "--exp-suffix",
        default="",
        help="Extra suffix on experiment name (e.g. 'frozen', 'ablation').",
    )
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=None,
        help="Override finetune learning rate (default = pretrain-saved LR).",
    )
    parser.add_argument(
        "--finetune-lr-ratio",
        type=float,
        default=None,
        help="Override finetune lr_ratio (max_lr/init_lr). Default = 10.0.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=None,
        help="Override warmup_epochs (default from pretrain params).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Override patience (default 20).",
    )
    args = parser.parse_args()

    if not PRETRAIN_PATH.exists() and not args.no_pretrain:
        raise FileNotFoundError(
            f"Pretrain checkpoint missing: {PRETRAIN_PATH}. Run "
            f"run_chemprop_pretrain.py first."
        )

    # Pull pretrain params so architecture matches exactly
    if args.no_pretrain:
        # Fallback to canonical tuned params
        params = dict(
            message_hidden_dim=256,
            depth=4,
            mp_dropout=0.2,
            activation="relu",
            aggregation="norm",
            ffn_hidden_dim=256,
            ffn_num_layers=1,
            ffn_dropout=0.1,
            warmup_epochs=3,
            learning_rate=0.0001364559692954765,
            lr_ratio=10.0,
            batch_size=64,
            max_epochs=200,
            patience=20,
        )
    else:
        ckpt = torch.load(PRETRAIN_PATH, map_location="cpu", weights_only=False)
        params = dict(ckpt["params"])
        # Fine-tune uses smaller batch / more epochs / tighter patience
        # Since train is only 4140 compounds the pretrain batch_size=128
        # leaves too few steps per epoch; revert to single-task defaults.
        params["batch_size"] = 64
        params["max_epochs"] = 200
        params["patience"] = 20

    # Finetune-side hparam overrides (for pretrain tune experiments)
    if args.finetune_lr is not None:
        params["learning_rate"] = args.finetune_lr
    if args.finetune_lr_ratio is not None:
        params["lr_ratio"] = args.finetune_lr_ratio
    if args.warmup_epochs is not None:
        params["warmup_epochs"] = args.warmup_epochs
    if args.patience is not None:
        params["patience"] = args.patience

    if args.max_epochs is not None:
        params["max_epochs"] = args.max_epochs

    print(
        f"ChemProp fine-tune | split={args.split} | "
        f"no_pretrain={args.no_pretrain} | freeze={args.freeze_encoder} "
        f"| max_epochs={params['max_epochs']}"
    )
    print(f"  params: {params}")

    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    smiles = train_df["smiles"].tolist()
    y = train_df["pec50"].to_numpy(dtype=np.float32)
    print(f"  train: {len(smiles)}, test: {len(test_df)}")

    if np.isnan(y).any():
        raise ValueError("pEC50 contains NaN -- refusing to train")

    if args.split == "scaffold":
        outer = scaffold_split_indices(smiles, n_splits=5, seed=42)
    else:
        outer = umap_split_indices(smiles, n_splits=5, n_clusters=50, seed=42)

    exp_name = "chemprop_pretrain_finetune"
    if args.no_pretrain:
        exp_name = "chemprop_finetune_nopretrain_ablation"
    elif args.freeze_encoder:
        exp_name += "_frozen"
    if args.exp_suffix:
        exp_name += f"_{args.exp_suffix}"
    exp_name += f"_{args.split}"
    print(f"  Experiment: {exp_name}")

    oof_preds = np.zeros(len(smiles), dtype=np.float32)
    fold_metrics = []
    test_pred_per_fold = []

    for fold, (tr_idx, va_idx) in enumerate(outer):
        print(f"\n[Fold {fold}] train={len(tr_idx)}, val={len(va_idx)}")
        tr_smi = [smiles[i] for i in tr_idx]
        va_smi = [smiles[i] for i in va_idx]
        tr_y = y[tr_idx]
        va_y = y[va_idx]

        model = trainer = None
        try:
            model = build_finetune_model(params)

            if not args.no_pretrain:
                summary = load_pretrain_encoder_weights(model, PRETRAIN_PATH)
                if fold == 0:
                    print(f"  Loaded encoder weights: {summary}")

            if args.freeze_encoder:
                n_frozen = freeze_encoder(model)
                if fold == 0:
                    print(f"  Frozen {n_frozen} encoder parameters")

            train_loader = make_dataloader(
                tr_smi, tr_y, params["batch_size"], shuffle=True
            )
            val_loader = make_dataloader(
                va_smi, va_y, params["batch_size"], shuffle=False
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
                logger=False,
                callbacks=[early_stop],
            )
            trainer.fit(model, train_loader, val_loader)

            val_preds = predict_main(trainer, model, va_smi, params["batch_size"])
            if not np.isfinite(val_preds).all():
                raise RuntimeError(f"Fold {fold}: val_preds contain NaN/Inf")
            oof_preds[va_idx] = val_preds

            metrics = compute_metrics(va_y, val_preds)
            fold_metrics.append(metrics)
            print_metrics(metrics, label=f"Fold {fold}")

            test_preds = predict_main(
                trainer, model, test_df["smiles"].tolist(), params["batch_size"]
            )
            if not np.isfinite(test_preds).all():
                raise RuntimeError(f"Fold {fold}: test_preds contain NaN/Inf")
            test_pred_per_fold.append(test_preds)
        finally:
            del model, trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    oof_metrics = compute_metrics(y, oof_preds)
    print("\n  Overall OOF:")
    print_metrics(oof_metrics)
    print_fold_summary(fold_metrics)

    test_preds_mean = np.mean(test_pred_per_fold, axis=0)
    print(
        f"\n  Test preds: mean={test_preds_mean.mean():.3f}, "
        f"std={test_preds_mean.std():.3f}"
    )

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
                f"ChemProp fine-tune on pEC50 with "
                f"{'scratch encoder (ablation)' if args.no_pretrain else 'pretrained encoder (13k + log2_fc)'}, "
                f"{'frozen' if args.freeze_encoder else 'full'} fine-tune, "
                f"{args.split} 5-fold CV"
            ),
            model_type="chemprop",
            feature_set="smiles",
            hyperparameters={
                **params,
                "pretrain_path": (str(PRETRAIN_PATH) if not args.no_pretrain else None),
                "freeze_encoder": args.freeze_encoder,
                "no_pretrain": args.no_pretrain,
            },
            fold_metrics=fold_metrics,
            submission_path=f"track1_activity/submissions/{exp_name}.csv",
            num_boost_rounds=[0] * len(fold_metrics),
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
    save_oof_predictions(exp_id, oof_preds)
    print(f"\n  Done: {exp_name} -> RAE={oof_metrics['RAE']:.4f}")


if __name__ == "__main__":
    main()
