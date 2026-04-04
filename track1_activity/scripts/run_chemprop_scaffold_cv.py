"""ChemProp and CheMeleon fine-tuning with scaffold split CV.

Trains D-MPNN models with proper scaffold splits for ensemble candidates.
Saves OOF predictions to DB.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl

from chemprop import data as chemprop_data
from chemprop import models, nn

from data import load_train_smiles_target, load_test_smiles
from evaluate import (
    compute_metrics,
    print_metrics,
    print_fold_summary,
    record_experiment,
    save_oof_predictions,
)
from splits import scaffold_split_indices

SUBMISSION_DIR = Path(__file__).resolve().parent.parent.joinpath("submissions")

torch.set_float32_matmul_precision("medium")

# Model configs to try
MODEL_CONFIGS = {
    "chemprop_scaffold": {
        "description": "ChemProp D-MPNN with scaffold split CV",
        "feature_set": "molecular_graph",
        "from_foundation": False,
        "params": {
            "message_hidden_dim": 300,
            "depth": 3,
            "ffn_hidden_dim": 300,
            "ffn_num_layers": 2,
            "dropout": 0.1,
            "batch_size": 64,
            "max_epochs": 80,
            "learning_rate": 1e-4,
            "warmup_epochs": 2,
        },
    },
    "chemeleon_finetune": {
        "description": "CheMeleon foundation model fine-tuned with scaffold split CV",
        "feature_set": "chemeleon_foundation",
        "from_foundation": True,
        "params": {
            "ffn_hidden_dim": 300,
            "ffn_num_layers": 2,
            "dropout": 0.1,
            "batch_size": 64,
            "max_epochs": 50,
            "learning_rate": 5e-5,
            "warmup_epochs": 2,
        },
    },
}


def load_chemeleon_mp():
    """Load CheMeleon pretrained message passing weights."""
    ckpt_path = Path.home().joinpath(".chemprop", "chemeleon_mp.pt")
    if not ckpt_path.exists():
        from urllib.request import urlretrieve

        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        print("  Downloading CheMeleon checkpoint...")
        urlretrieve(
            "https://zenodo.org/records/15460715/files/chemeleon_mp.pt", ckpt_path
        )
    return torch.load(ckpt_path, map_location="cpu", weights_only=True)


def build_model(config: dict):
    """Build ChemProp MPNN model from config."""
    params = config["params"]

    if config["from_foundation"]:
        # CheMeleon: load pretrained MP weights
        ckpt = load_chemeleon_mp()
        mp = nn.BondMessagePassing(**ckpt["hyper_parameters"])
        mp.load_state_dict(ckpt["state_dict"])
    else:
        mp = nn.BondMessagePassing(
            d_h=params["message_hidden_dim"],
            depth=params["depth"],
            dropout=params["dropout"],
        )

    agg = nn.MeanAggregation()
    ffn = nn.RegressionFFN(
        input_dim=mp.output_dim,
        hidden_dim=params["ffn_hidden_dim"],
        n_layers=params["ffn_num_layers"],
        dropout=params["dropout"],
    )

    model = models.MPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        batch_norm=True,
        warmup_epochs=params["warmup_epochs"],
        init_lr=params["learning_rate"],
        max_lr=params["learning_rate"] * 10,
        final_lr=params["learning_rate"],
    )
    return model


def train_and_predict(config, train_smiles, train_targets, val_smiles, val_targets, test_smiles):
    """Train model, return val and test predictions."""
    params = config["params"]

    train_data = [
        chemprop_data.MoleculeDatapoint.from_smi(smi, [y])
        for smi, y in zip(train_smiles, train_targets)
    ]
    val_data = [
        chemprop_data.MoleculeDatapoint.from_smi(smi, [y])
        for smi, y in zip(val_smiles, val_targets)
    ]
    test_data = [chemprop_data.MoleculeDatapoint.from_smi(smi) for smi in test_smiles]

    train_loader = chemprop_data.build_dataloader(
        chemprop_data.MoleculeDataset(train_data),
        batch_size=params["batch_size"],
        shuffle=True,
    )
    val_loader = chemprop_data.build_dataloader(
        chemprop_data.MoleculeDataset(val_data),
        batch_size=params["batch_size"],
        shuffle=False,
    )
    test_loader = chemprop_data.build_dataloader(
        chemprop_data.MoleculeDataset(test_data),
        batch_size=params["batch_size"],
        shuffle=False,
    )

    model = build_model(config)

    trainer = pl.Trainer(
        max_epochs=params["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )

    trainer.fit(model, train_loader, val_loader)

    val_preds = trainer.predict(model, val_loader)
    val_preds = np.concatenate([p.numpy().flatten() for p in val_preds])

    test_preds = trainer.predict(model, test_loader)
    test_preds = np.concatenate([p.numpy().flatten() for p in test_preds])

    return val_preds, test_preds


def run_model(name, config, train_smiles, y_train, test_smiles, test_df, outer_splits):
    """Run scaffold split CV for a model config."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"  {config['description']}")
    print(f"{'=' * 60}")

    oof_preds = np.zeros(len(y_train))
    test_preds_all = np.zeros((len(outer_splits), len(test_smiles)))
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(outer_splits):
        print(f"\n  --- Fold {fold} ---")
        tr_smi = [train_smiles[i] for i in train_idx]
        va_smi = [train_smiles[i] for i in val_idx]
        tr_y = y_train[train_idx]
        va_y = y_train[val_idx]

        val_pred, test_pred = train_and_predict(
            config, tr_smi, tr_y, va_smi, va_y, test_smiles
        )

        oof_preds[val_idx] = val_pred
        test_preds_all[fold] = test_pred

        metrics = compute_metrics(va_y, val_pred)
        fold_metrics.append(metrics)
        print_metrics(metrics, label=f"Fold {fold}")

    oof_metrics = compute_metrics(y_train, oof_preds)
    print("\n  Overall OOF:")
    print_metrics(oof_metrics)
    print_fold_summary(fold_metrics)

    # Average test predictions across folds
    test_preds_avg = test_preds_all.mean(axis=0)
    print(f"\n  Test preds: mean={test_preds_avg.mean():.3f}, std={test_preds_avg.std():.3f}")

    submission = pd.DataFrame({
        "SMILES": test_df["smiles"],
        "Molecule Name": test_df["molecule_name"],
        "pEC50": test_preds_avg,
    })
    sub_path = SUBMISSION_DIR.joinpath(f"{name}.csv")
    submission.to_csv(sub_path, index=False)

    exp_id = record_experiment(
        name=name,
        description=config["description"],
        model_type="chemprop" if not config["from_foundation"] else "chemeleon_finetune",
        feature_set=config["feature_set"],
        hyperparameters=config["params"],
        fold_metrics=fold_metrics,
        submission_path=f"track1_activity/submissions/{name}.csv",
        notes=f"OOF RAE={oof_metrics['RAE']:.4f}, scaffold_split, GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
    )
    save_oof_predictions(exp_id, oof_preds)

    return oof_metrics


def main():
    pl.seed_everything(42)

    print("Loading data...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    train_smiles = train_df["smiles"].tolist()
    y_train = train_df["pec50"].values
    test_smiles = test_df["smiles"].tolist()

    print(f"Train: {len(train_smiles)}, Test: {len(test_smiles)}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    outer_splits = scaffold_split_indices(train_smiles, n_splits=5, seed=42)

    results = {}
    for name, config in MODEL_CONFIGS.items():
        m = run_model(name, config, train_smiles, y_train, test_smiles, test_df, outer_splits)
        results[name] = m

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for name, m in sorted(results.items(), key=lambda x: x[1]["RAE"]):
        print(f"  {name:<30} RAE={m['RAE']:.4f}  R2={m['R2']:.4f}  Spearman={m['Spearman_R']:.4f}")


if __name__ == "__main__":
    main()
