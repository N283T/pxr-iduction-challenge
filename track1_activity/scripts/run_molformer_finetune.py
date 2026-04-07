"""MoLFormer-XL fine-tuning for pEC50 prediction.

Fine-tunes MoLFormer-XL-both-10pct with a regression head.
Uses UMAP split for more realistic CV estimates.
Saves OOF predictions to DB for ensemble.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))

import argparse

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from data import load_train_smiles_target, load_test_smiles
from evaluate import (
    compute_metrics,
    print_metrics,
    print_fold_summary,
    record_experiment,
    save_oof_predictions,
)
from splits import umap_split_indices, scaffold_split_indices

SUBMISSION_DIR = Path(__file__).resolve().parent.parent.joinpath("submissions")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"


class SmilesDataset(Dataset):
    """Dataset for SMILES strings with optional targets."""

    def __init__(self, smiles_list, tokenizer, targets=None, max_length=202):
        self.smiles = smiles_list
        self.tokenizer = tokenizer
        self.targets = targets
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.smiles[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        if self.targets is not None:
            item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return item


class MolFormerRegressor(nn.Module):
    """MoLFormer with regression head."""

    def __init__(self, hidden_dim, dropout, freeze_layers=0):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)

        # Freeze early layers if requested
        if freeze_layers > 0:
            for param in self.backbone.embeddings.parameters():
                param.requires_grad = False
            for i, layer in enumerate(self.backbone.encoder.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        backbone_dim = self.backbone.config.hidden_size  # 768
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token (first token) representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_output).squeeze(-1)


def train_and_predict(
    params,
    tokenizer,
    train_smiles,
    train_targets,
    val_smiles,
    val_targets,
    test_smiles=None,
):
    """Train MoLFormer regressor, return val and optionally test predictions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = SmilesDataset(train_smiles, tokenizer, train_targets)
    val_dataset = SmilesDataset(val_smiles, tokenizer, val_targets)

    train_loader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False)

    model = MolFormerRegressor(
        hidden_dim=params["head_hidden_dim"],
        dropout=params["dropout"],
        freeze_layers=params["freeze_layers"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": params["backbone_lr"]},
            {"params": model.head.parameters(), "lr": params["head_lr"]},
        ],
        weight_decay=params["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params["max_epochs"], eta_min=1e-7
    )

    criterion = nn.MSELoss()

    best_val_mae = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(params["max_epochs"]):
        # Train
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            preds = model(input_ids, attention_mask)
            loss = criterion(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # Validate
        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                preds = model(input_ids, attention_mask)
                val_preds_list.append(preds.cpu().numpy())

        val_preds = np.concatenate(val_preds_list)
        val_mae = np.mean(np.abs(val_targets - val_preds))

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= params["patience"]:
                break

    # Load best model and predict
    if best_state is None:
        raise RuntimeError(
            "Training failed: model never improved. Check learning rate."
        )
    model.load_state_dict(best_state)
    model.eval()

    val_preds_list = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            preds = model(input_ids, attention_mask)
            val_preds_list.append(preds.cpu().numpy())
    val_preds = np.concatenate(val_preds_list)

    test_preds = None
    if test_smiles is not None:
        test_dataset = SmilesDataset(test_smiles, tokenizer)
        test_loader = DataLoader(
            test_dataset, batch_size=params["batch_size"], shuffle=False
        )
        test_preds_list = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                preds = model(input_ids, attention_mask)
                test_preds_list.append(preds.cpu().numpy())
        test_preds = np.concatenate(test_preds_list)

    # Cleanup GPU memory
    del model
    torch.cuda.empty_cache()

    return val_preds, test_preds


def objective(trial, tokenizer, train_smiles, y_train, inner_splits):
    """Optuna objective: average RAE across inner CV folds."""
    params = {
        "head_hidden_dim": trial.suggest_categorical(
            "head_hidden_dim", [128, 256, 512]
        ),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4, step=0.05),
        "freeze_layers": trial.suggest_int("freeze_layers", 0, 8),
        "backbone_lr": trial.suggest_float("backbone_lr", 1e-6, 5e-5, log=True),
        "head_lr": trial.suggest_float("head_lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "max_epochs": 50,
        "patience": 8,
    }

    fold_raes = []
    for fold, (train_idx, val_idx) in enumerate(inner_splits):
        tr_smi = [train_smiles[i] for i in train_idx]
        va_smi = [train_smiles[i] for i in val_idx]
        tr_y = y_train[train_idx]
        va_y = y_train[val_idx]

        val_pred, _ = train_and_predict(params, tokenizer, tr_smi, tr_y, va_smi, va_y)
        metrics = compute_metrics(va_y, val_pred)
        fold_raes.append(metrics["RAE"])

        trial.report(np.mean(fold_raes), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(fold_raes)


def run_final_cv(
    best_params,
    tokenizer,
    train_smiles,
    y_train,
    test_smiles,
    test_df,
    outer_splits,
    split_name,
):
    """Run full outer CV with best params and save results."""
    final_params = {**best_params, "max_epochs": 80, "patience": 12}

    name = f"molformer_finetune_{split_name}"
    print(f"\n{'=' * 60}")
    print(f"  Final CV: {name}")
    print(f"  Params: {final_params}")
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
            final_params, tokenizer, tr_smi, tr_y, va_smi, va_y, test_smiles
        )

        if np.any(np.isnan(val_pred)):
            raise ValueError(f"Fold {fold}: model produced NaN predictions")

        oof_preds[val_idx] = val_pred
        test_preds_all[fold] = test_pred

        metrics = compute_metrics(va_y, val_pred)
        fold_metrics.append(metrics)
        print_metrics(metrics, label=f"Fold {fold}")

    oof_metrics = compute_metrics(y_train, oof_preds)
    print("\n  Overall OOF:")
    print_metrics(oof_metrics)
    print_fold_summary(fold_metrics)

    test_preds_avg = test_preds_all.mean(axis=0)

    submission = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": test_preds_avg,
        }
    )
    sub_path = SUBMISSION_DIR.joinpath(f"{name}.csv")
    submission.to_csv(sub_path, index=False)

    exp_id = record_experiment(
        name=name,
        description=f"MoLFormer-XL fine-tuned ({split_name} split)",
        model_type="molformer_finetune",
        feature_set="smiles_transformer",
        hyperparameters=final_params,
        fold_metrics=fold_metrics,
        submission_path=f"track1_activity/submissions/{name}.csv",
        notes=f"OOF RAE={oof_metrics['RAE']:.4f}, {split_name}_split, optuna_tuned",
    )
    save_oof_predictions(exp_id, oof_preds)

    return oof_metrics


def main():
    parser = argparse.ArgumentParser(description="MoLFormer fine-tuning with Optuna")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--split", choices=["umap", "scaffold"], default="umap")
    parser.add_argument("--n-clusters", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    print("Loading data...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    train_smiles = train_df["smiles"].tolist()
    y_train = train_df["pec50"].values
    test_smiles_list = test_df["smiles"].tolist()

    print(f"Train: {len(train_smiles)}, Test: {len(test_smiles_list)}")
    print(
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
    print(f"Split: {args.split}")

    print("Loading MoLFormer tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if args.split == "umap":
        split_fn = lambda n, s: umap_split_indices(
            train_smiles, n_splits=n, n_clusters=args.n_clusters, seed=s
        )
    else:
        split_fn = lambda n, s: scaffold_split_indices(train_smiles, n_splits=n, seed=s)

    outer_splits = split_fn(args.outer_folds, 42)
    inner_splits = split_fn(args.inner_folds, 123)

    print(f"\n{'=' * 60}")
    print(f"  Optuna tuning: {args.n_trials} trials, {args.inner_folds}-fold inner CV")
    print(f"{'=' * 60}")

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(
        lambda trial: objective(trial, tokenizer, train_smiles, y_train, inner_splits),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    n_complete = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    n_pruned = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    )
    n_failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
    print(f"  Trials: {n_complete} complete, {n_pruned} pruned, {n_failed} FAILED")
    if n_complete == 0:
        raise RuntimeError("All Optuna trials failed. Cannot proceed.")

    print(f"\n  Best trial: RAE={study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    # Final outer CV with best params
    oof_metrics = run_final_cv(
        study.best_params,
        tokenizer,
        train_smiles,
        y_train,
        test_smiles_list,
        test_df,
        outer_splits,
        args.split,
    )

    print(f"\n  Final OOF RAE: {oof_metrics['RAE']:.4f}")


if __name__ == "__main__":
    main()
