#!/usr/bin/env -S pixi run python
"""ChemProp Siamese pretraining on ChEMBL same-assay pChEMBL differences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from chemprop import data as chemprop_data
from chemprop import models, nn
from chemprop.data import BatchMolGraph
from lightning import pytorch as pl
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT.joinpath("data", "chembl", "pairwise_deep")
DEFAULT_CKPT_DIR = REPO_ROOT.joinpath(
    "track1_activity", "checkpoints", "chemprop_pairwise_chembl"
)

torch.set_float32_matmul_precision("medium")

DEFAULT_PARAMS = {
    "message_hidden_dim": 256,
    "depth": 4,
    "mp_dropout": 0.2,
    "activation": "relu",
    "aggregation": "norm",
    "ffn_hidden_dim": 256,
    "ffn_num_layers": 1,
    "ffn_dropout": 0.1,
    "batch_size": 128,
    "max_epochs": 30,
    "patience": 5,
    "learning_rate": 2.0e-4,
    "weight_decay": 1.0e-5,
}

AGG_REGISTRY = {
    "mean": nn.MeanAggregation,
    "sum": nn.SumAggregation,
    "norm": nn.NormAggregation,
}


class PairBatch(NamedTuple):
    bmg_a: BatchMolGraph
    bmg_b: BatchMolGraph
    delta: torch.Tensor
    value_a: torch.Tensor
    value_b: torch.Tensor
    weight: torch.Tensor


class PairDataset(Dataset):
    """Pair rows backed by precomputed ChemProp MolGraphs."""

    def __init__(
        self,
        pairs: pd.DataFrame,
        mol_graphs: dict[int, object],
        delta_mean: float,
        delta_std: float,
        value_mean: float,
        value_std: float,
    ):
        self.mol_a = pairs["mol_id_a"].to_numpy(dtype=np.int64)
        self.mol_b = pairs["mol_id_b"].to_numpy(dtype=np.int64)
        self.delta = (
            (pairs["delta"].to_numpy(dtype=np.float32) - delta_mean) / delta_std
        ).astype(np.float32)
        self.value_a = (
            (pairs["value_a"].to_numpy(dtype=np.float32) - value_mean) / value_std
        ).astype(np.float32)
        self.value_b = (
            (pairs["value_b"].to_numpy(dtype=np.float32) - value_mean) / value_std
        ).astype(np.float32)
        self.weight = pairs["weight"].to_numpy(dtype=np.float32)
        self.mol_graphs = mol_graphs

    def __len__(self) -> int:
        return len(self.delta)

    def __getitem__(self, idx: int):
        return (
            self.mol_graphs[int(self.mol_a[idx])],
            self.mol_graphs[int(self.mol_b[idx])],
            self.delta[idx],
            self.value_a[idx],
            self.value_b[idx],
            self.weight[idx],
        )


def collate_pairs(items) -> PairBatch:
    mg_a, mg_b, delta, value_a, value_b, weight = zip(*items, strict=False)
    return PairBatch(
        bmg_a=BatchMolGraph(mg_a),
        bmg_b=BatchMolGraph(mg_b),
        delta=torch.tensor(delta, dtype=torch.float32).unsqueeze(1),
        value_a=torch.tensor(value_a, dtype=torch.float32).unsqueeze(1),
        value_b=torch.tensor(value_b, dtype=torch.float32).unsqueeze(1),
        weight=torch.tensor(weight, dtype=torch.float32).unsqueeze(1),
    )


def build_base_mpnn(params: dict) -> models.MPNN:
    mp = nn.BondMessagePassing(
        d_h=params["message_hidden_dim"],
        depth=params["depth"],
        dropout=params["mp_dropout"],
        activation=params["activation"],
    )
    agg = AGG_REGISTRY[params["aggregation"]]()
    criterion = nn.metrics.MSE()
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
        warmup_epochs=1,
        init_lr=params["learning_rate"],
        max_lr=params["learning_rate"],
        final_lr=params["learning_rate"] * 0.1,
    )


class PairwiseChemProp(pl.LightningModule):
    def __init__(
        self,
        params: dict,
        delta_mean: float,
        delta_std: float,
        value_mean: float = 0.0,
        value_std: float = 1.0,
        diff_weight: float = 1.0,
        abs_weight: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(
            {
                "params": params,
                "delta_mean": delta_mean,
                "delta_std": delta_std,
                "value_mean": value_mean,
                "value_std": value_std,
                "diff_weight": diff_weight,
                "abs_weight": abs_weight,
            }
        )
        self.params = params
        self.delta_mean = float(delta_mean)
        self.delta_std = float(delta_std)
        self.value_mean = float(value_mean)
        self.value_std = float(value_std)
        self.diff_weight = float(diff_weight)
        self.abs_weight = float(abs_weight)
        self.base = build_base_mpnn(params)
        self.loss_fn = torch.nn.SmoothL1Loss(reduction="none", beta=0.5)

    def forward(self, bmg: BatchMolGraph) -> torch.Tensor:
        bmg.to(self.device)
        return self.base(bmg)

    def _step(self, batch: PairBatch, stage: str) -> torch.Tensor:
        batch.bmg_a.to(self.device)
        batch.bmg_b.to(self.device)
        delta = batch.delta.to(self.device)
        value_a = batch.value_a.to(self.device)
        value_b = batch.value_b.to(self.device)
        weight = batch.weight.to(self.device)
        pred_a = self.base(batch.bmg_a)
        pred_b = self.base(batch.bmg_b)
        pred_delta = pred_a - pred_b
        diff_loss_raw = self.loss_fn(pred_delta, delta)
        diff_loss = (diff_loss_raw * weight).sum() / weight.sum().clamp_min(1.0)
        abs_loss_raw = 0.5 * (
            self.loss_fn(pred_a, value_a) + self.loss_fn(pred_b, value_b)
        )
        abs_loss = (abs_loss_raw * weight).sum() / weight.sum().clamp_min(1.0)
        loss = self.diff_weight * diff_loss + self.abs_weight * abs_loss
        mae = torch.mean(torch.abs(pred_delta - delta))
        abs_mae = 0.5 * (
            torch.mean(torch.abs(pred_a - value_a))
            + torch.mean(torch.abs(pred_b - value_b))
        )
        sign_acc = torch.mean((torch.sign(pred_delta) == torch.sign(delta)).float())
        self.log(
            f"{stage}_loss", loss, prog_bar=False, on_epoch=True, batch_size=len(delta)
        )
        self.log(
            f"{stage}_diff_loss",
            diff_loss,
            prog_bar=False,
            on_epoch=True,
            batch_size=len(delta),
        )
        self.log(
            f"{stage}_abs_loss",
            abs_loss,
            prog_bar=False,
            on_epoch=True,
            batch_size=len(delta),
        )
        self.log(
            f"{stage}_mae_z", mae, prog_bar=False, on_epoch=True, batch_size=len(delta)
        )
        self.log(
            f"{stage}_abs_mae_z",
            abs_mae,
            prog_bar=False,
            on_epoch=True,
            batch_size=len(delta),
        )
        self.log(
            f"{stage}_sign_acc",
            sign_acc,
            prog_bar=False,
            on_epoch=True,
            batch_size=len(delta),
        )
        return loss

    def training_step(self, batch: PairBatch, _batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: PairBatch, _batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.params["learning_rate"],
            weight_decay=self.params["weight_decay"],
        )


def _sample_pairs_by_assay_iqr(
    pairs: pd.DataFrame,
    activities: pd.DataFrame,
    max_pairs: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    train = pairs[pairs["split"].eq("train")]
    val = pairs[pairs["split"].eq("val")]
    n_val = min(len(val), max(1, int(max_pairs * 0.1)))
    n_train = max_pairs - n_val

    assay_iqr = (
        activities.groupby("assay_id")["value"]
        .agg(lambda x: float(np.quantile(x, 0.75) - np.quantile(x, 0.25)))
        .rename("iqr")
        .reset_index()
    )
    assay_iqr["iqr"] = assay_iqr["iqr"].clip(lower=1e-3)
    train = train.assign(_pair_index=train.index.to_numpy(dtype=np.int64)).merge(
        assay_iqr, on="assay_id", how="left"
    )
    train["iqr"] = train["iqr"].fillna(train["iqr"].median()).clip(lower=1e-3)
    assay_weights = train.groupby("assay_id", sort=False)["iqr"].first()
    assay_probs = assay_weights.to_numpy(dtype=np.float64)
    assay_probs = assay_probs / assay_probs.sum()
    assay_ids = assay_weights.index.to_numpy()
    by_assay = {
        assay_id: sub["_pair_index"].to_numpy(dtype=np.int64)
        for assay_id, sub in train.groupby("assay_id")
    }

    chosen = []
    chosen_assays = rng.choice(
        assay_ids, size=min(n_train, len(train)), replace=True, p=assay_probs
    )
    for assay_id in chosen_assays:
        chosen.append(int(rng.choice(by_assay[assay_id])))
    if len(chosen) > 0:
        train_idx = np.asarray(chosen, dtype=np.int64)
    else:
        train_idx = np.empty(0, dtype=np.int64)
    val_idx = rng.choice(val.index.to_numpy(), size=min(n_val, len(val)), replace=False)
    out = pairs.loc[np.concatenate([train_idx, val_idx])].sample(
        frac=1.0, random_state=seed
    )
    return out


def load_data(
    data_dir: Path, max_pairs: int, seed: int, sampler: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    molecules = pd.read_parquet(data_dir.joinpath("molecules.parquet"))
    pairs = pd.read_parquet(data_dir.joinpath("pairs.parquet"))
    if max_pairs > 0 and len(pairs) > max_pairs:
        if sampler == "assay_iqr":
            activities = pd.read_parquet(data_dir.joinpath("activities.parquet"))
            pairs = _sample_pairs_by_assay_iqr(pairs, activities, max_pairs, seed)
        else:
            rng = np.random.default_rng(seed)
            train = pairs[pairs["split"].eq("train")]
            val = pairs[pairs["split"].eq("val")]
            n_val = min(len(val), max(1, int(max_pairs * 0.1)))
            n_train = max_pairs - n_val
            train_idx = rng.choice(
                train.index.to_numpy(), size=min(n_train, len(train)), replace=False
            )
            val_idx = rng.choice(
                val.index.to_numpy(), size=min(n_val, len(val)), replace=False
            )
            pairs = pairs.loc[np.concatenate([train_idx, val_idx])].sample(
                frac=1.0, random_state=seed
            )
    return molecules, pairs.reset_index(drop=True)


def build_mol_graphs(molecules: pd.DataFrame) -> dict[int, object]:
    datapoints = [
        chemprop_data.MoleculeDatapoint.from_smi(
            smi, np.asarray([0.0], dtype=np.float32)
        )
        for smi in molecules["std_smiles"].astype(str)
    ]
    dataset = chemprop_data.MoleculeDataset(datapoints)
    mol_ids = molecules["mol_id"].to_numpy(dtype=np.int64)
    return {int(mol_id): dataset[i].mg for i, mol_id in enumerate(mol_ids)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--max-pairs", type=int, default=200000)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--message-hidden-dim", type=int, default=None)
    parser.add_argument("--diff-weight", type=float, default=1.0)
    parser.add_argument("--abs-weight", type=float, default=0.0)
    parser.add_argument(
        "--sampler", choices=["uniform", "assay_iqr"], default="uniform"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = DEFAULT_PARAMS.copy()
    if args.max_epochs is not None:
        params["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        params["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        params["learning_rate"] = args.learning_rate
    if args.message_hidden_dim is not None:
        params["message_hidden_dim"] = args.message_hidden_dim

    pl.seed_everything(args.seed, workers=True)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading pairwise data from {args.data_dir}")
    molecules, pairs = load_data(args.data_dir, args.max_pairs, args.seed, args.sampler)
    print(f"  molecules: {len(molecules):,}")
    print(f"  pairs: {len(pairs):,} ({pairs['split'].value_counts().to_dict()})")

    train_pairs = pairs[pairs["split"].eq("train")].reset_index(drop=True)
    val_pairs = pairs[pairs["split"].eq("val")].reset_index(drop=True)
    if train_pairs.empty or val_pairs.empty:
        raise RuntimeError("Both train and val pair splits are required.")
    delta_mean = float(train_pairs["delta"].mean())
    delta_std = float(train_pairs["delta"].std(ddof=0))
    if delta_std < 1e-6:
        delta_std = 1.0
    value_mean = float(
        np.concatenate(
            [
                train_pairs["value_a"].to_numpy(dtype=np.float32),
                train_pairs["value_b"].to_numpy(dtype=np.float32),
            ]
        ).mean()
    )
    value_std = float(
        np.concatenate(
            [
                train_pairs["value_a"].to_numpy(dtype=np.float32),
                train_pairs["value_b"].to_numpy(dtype=np.float32),
            ]
        ).std()
    )
    if value_std < 1e-6:
        value_std = 1.0
    print(f"  train delta mean/std: {delta_mean:+.4f} / {delta_std:.4f}")
    print(f"  train value mean/std: {value_mean:+.4f} / {value_std:.4f}")
    print(
        f"  loss weights: diff={args.diff_weight:.3f} abs={args.abs_weight:.3f}; "
        f"sampler={args.sampler}"
    )

    print("Building ChemProp molecule graphs")
    mol_graphs = build_mol_graphs(molecules)

    train_ds = PairDataset(
        train_pairs, mol_graphs, delta_mean, delta_std, value_mean, value_std
    )
    val_ds = PairDataset(
        val_pairs, mol_graphs, delta_mean, delta_std, value_mean, value_std
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_pairs,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pairs,
        pin_memory=torch.cuda.is_available(),
    )

    model = PairwiseChemProp(
        params,
        delta_mean=delta_mean,
        delta_std=delta_std,
        value_mean=value_mean,
        value_std=value_std,
        diff_weight=args.diff_weight,
        abs_weight=args.abs_weight,
    )
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=params["patience"]
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath=str(args.ckpt_dir),
            filename="pairwise_best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
    ]
    trainer = pl.Trainer(
        max_epochs=params["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=False,
        callbacks=callbacks,
    )
    trainer.fit(model, train_loader, val_loader)

    state_path = args.ckpt_dir.joinpath("pairwise_pretrain.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "params": params,
            "delta_mean": delta_mean,
            "delta_std": delta_std,
            "value_mean": value_mean,
            "value_std": value_std,
            "diff_weight": args.diff_weight,
            "abs_weight": args.abs_weight,
            "sampler": args.sampler,
            "n_train_pairs": len(train_pairs),
            "n_val_pairs": len(val_pairs),
            "final_metrics": {
                k: float(v.detach().cpu()) for k, v in trainer.callback_metrics.items()
            },
        },
        state_path,
    )
    meta = {
        "data_dir": str(args.data_dir),
        "ckpt_dir": str(args.ckpt_dir),
        "params": params,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "value_mean": value_mean,
        "value_std": value_std,
        "diff_weight": args.diff_weight,
        "abs_weight": args.abs_weight,
        "sampler": args.sampler,
        "n_train_pairs": len(train_pairs),
        "n_val_pairs": len(val_pairs),
        "best_ckpt_path": callbacks[1].best_model_path,
        "final_metrics": {
            k: float(v.detach().cpu()) for k, v in trainer.callback_metrics.items()
        },
    }
    args.ckpt_dir.joinpath("pairwise_pretrain_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"Saved state_dict: {state_path}")
    print(f"Best Lightning checkpoint: {callbacks[1].best_model_path}")


if __name__ == "__main__":
    main()
