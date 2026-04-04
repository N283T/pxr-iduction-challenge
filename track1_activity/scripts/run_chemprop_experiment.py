"""Run ChemProp GNN experiments for Track 1 Activity Prediction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from sklearn.model_selection import KFold

from chemprop import data as chemprop_data
from chemprop import models, nn

from data import load_train_smiles_target, load_test_smiles
from evaluate import compute_metrics, print_metrics, print_fold_summary, record_experiment

SUBMISSION_DIR = Path(__file__).resolve().parent.parent.joinpath("submissions")

CHEMPROP_PARAMS = {
    "message_hidden_dim": 300,
    "depth": 3,
    "ffn_hidden_dim": 300,
    "ffn_num_layers": 2,
    "dropout": 0.0,
    "batch_size": 64,
    "max_epochs": 50,
    "learning_rate": 1e-4,
    "warmup_epochs": 2,
}


def build_chemprop_datapoints(smiles_list, targets=None):
    """Build ChemProp MoleculeDatapoint list."""
    datapoints = []
    for i, smi in enumerate(smiles_list):
        y = [targets[i]] if targets is not None else None
        datapoints.append(chemprop_data.MoleculeDatapoint(chemprop_data.MoleculeDatapoint.from_smi(smi).mol, y))
    return datapoints


def train_and_predict(
    train_smiles, train_targets, val_smiles, val_targets, test_smiles, params
):
    """Train ChemProp model and return val/test predictions."""
    # Build datasets
    train_data = [
        chemprop_data.MoleculeDatapoint.from_smi(smi, [y])
        for smi, y in zip(train_smiles, train_targets)
    ]
    val_data = [
        chemprop_data.MoleculeDatapoint.from_smi(smi, [y])
        for smi, y in zip(val_smiles, val_targets)
    ]
    test_data = [
        chemprop_data.MoleculeDatapoint.from_smi(smi)
        for smi in test_smiles
    ]

    train_dataset = chemprop_data.MoleculeDataset(train_data)
    val_dataset = chemprop_data.MoleculeDataset(val_data)
    test_dataset = chemprop_data.MoleculeDataset(test_data)

    train_loader = chemprop_data.build_dataloader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    val_loader = chemprop_data.build_dataloader(val_dataset, batch_size=params["batch_size"], shuffle=False)
    test_loader = chemprop_data.build_dataloader(test_dataset, batch_size=params["batch_size"], shuffle=False)

    # Build model
    mp = nn.BondMessagePassing(
        d_h=params["message_hidden_dim"],
        depth=params["depth"],
        dropout=params["dropout"],
    )
    agg = nn.MeanAggregation()
    ffn = nn.RegressionFFN(
        input_dim=params["message_hidden_dim"],
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

    trainer = pl.Trainer(
        max_epochs=params["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )

    trainer.fit(model, train_loader, val_loader)

    # Predict
    val_preds = trainer.predict(model, val_loader)
    val_preds = np.concatenate([p.numpy().flatten() for p in val_preds])

    test_preds = trainer.predict(model, test_loader)
    test_preds = np.concatenate([p.numpy().flatten() for p in test_preds])

    return val_preds, test_preds


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

    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y_train))
    test_preds_all = np.zeros((5, len(test_smiles)))
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_smiles)):
        print(f"\n--- Fold {fold} ---")
        tr_smi = [train_smiles[i] for i in train_idx]
        val_smi = [train_smiles[i] for i in val_idx]
        tr_y = y_train[train_idx]
        val_y = y_train[val_idx]

        val_pred, test_pred = train_and_predict(
            tr_smi, tr_y, val_smi, val_y, test_smiles, CHEMPROP_PARAMS
        )

        oof_preds[val_idx] = val_pred
        test_preds_all[fold] = test_pred

        metrics = compute_metrics(val_y, val_pred)
        fold_metrics.append(metrics)
        print_metrics(metrics, label=f"Fold {fold}")

    # Overall OOF
    print("\n  Overall OOF:")
    oof_metrics = compute_metrics(y_train, oof_preds)
    print_metrics(oof_metrics)
    print_fold_summary(fold_metrics)

    # Average test predictions across folds
    test_preds_avg = test_preds_all.mean(axis=0)
    print(f"\n  Test preds: mean={test_preds_avg.mean():.3f}, std={test_preds_avg.std():.3f}")

    # Save submission
    submission = pd.DataFrame({
        "SMILES": test_df["smiles"],
        "Molecule Name": test_df["molecule_name"],
        "pEC50": test_preds_avg,
    })
    sub_path = SUBMISSION_DIR.joinpath("chemprop_mpnn.csv")
    submission.to_csv(sub_path, index=False)
    print(f"  Saved: {sub_path.name}")

    record_experiment(
        name="chemprop_mpnn",
        description="ChemProp MPNN (D-MPNN) with default bond message passing",
        model_type="chemprop",
        feature_set="molecular_graph",
        hyperparameters=CHEMPROP_PARAMS,
        fold_metrics=fold_metrics,
        submission_path="track1_activity/submissions/chemprop_mpnn.csv",
        notes=f"OOF RAE={oof_metrics['RAE']:.4f}, GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
    )


if __name__ == "__main__":
    main()
