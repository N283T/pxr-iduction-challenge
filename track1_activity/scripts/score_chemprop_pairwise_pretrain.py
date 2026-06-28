#!/usr/bin/env -S pixi run python
"""Score PXR challenge compounds with a ChemProp pairwise-pretrained model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chemprop import data as chemprop_data
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from data import get_engine  # noqa: E402
from run_chemprop_pairwise_pretrain import PairwiseChemProp  # noqa: E402

DEFAULT_CKPT = REPO_ROOT.joinpath(
    "track1_activity", "checkpoints", "chemprop_pairwise_chembl", "pairwise_pretrain.pt"
)
DEFAULT_OUT = REPO_ROOT.joinpath(
    "data", "chembl", "pairwise_deep", "pxr_pairwise_chemprop_scores.csv"
)


def load_pxr_compounds() -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT 'train' AS split,
               t.id AS row_id,
               c.id AS compound_id,
               c.molecule_name,
               c.std_smiles AS smiles,
               t.pec50 AS pec50
        FROM train_activity t
        JOIN compounds c ON c.id = t.compound_id
        WHERE c.std_smiles IS NOT NULL
        UNION ALL
        SELECT CASE WHEN l.pec50 IS NULL THEN 'AS2' ELSE 'AS1' END AS split,
               t.id AS row_id,
               c.id AS compound_id,
               c.molecule_name,
               c.std_smiles AS smiles,
               l.pec50 AS pec50
        FROM test_activity t
        JOIN compounds c ON c.id = t.compound_id
        LEFT JOIN test_activity_phase1_labels l ON l.compound_id = t.compound_id
        WHERE c.std_smiles IS NOT NULL
        ORDER BY split, row_id
        """,
        get_engine(),
    )


def load_pairwise_model(ckpt_path: Path) -> PairwiseChemProp:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "params" in ckpt:
        model = PairwiseChemProp(
            ckpt["params"],
            delta_mean=ckpt["delta_mean"],
            delta_std=ckpt["delta_std"],
            value_mean=ckpt.get("value_mean", 0.0),
            value_std=ckpt.get("value_std", 1.0),
            diff_weight=ckpt.get("diff_weight", 1.0),
            abs_weight=ckpt.get("abs_weight", 0.0),
        )
        model.load_state_dict(ckpt["state_dict"])
        return model

    hyper = ckpt.get("hyper_parameters", {})
    params = hyper.get("params")
    if params is None:
        raise KeyError(f"Cannot find pairwise model params in {ckpt_path}")
    model = PairwiseChemProp(
        params,
        delta_mean=hyper.get("delta_mean", 0.0),
        delta_std=hyper.get("delta_std", 1.0),
        value_mean=hyper.get("value_mean", 0.0),
        value_std=hyper.get("value_std", 1.0),
        diff_weight=hyper.get("diff_weight", 1.0),
        abs_weight=hyper.get("abs_weight", 0.0),
    )
    model.load_state_dict(ckpt["state_dict"])
    return model


def score_smiles(
    model: PairwiseChemProp, smiles: list[str], batch_size: int
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    pts = [
        chemprop_data.MoleculeDatapoint.from_smi(
            smi, np.asarray([0.0], dtype=np.float32)
        )
        for smi in smiles
    ]
    loader = chemprop_data.build_dataloader(
        chemprop_data.MoleculeDataset(pts),
        batch_size=batch_size,
        shuffle=False,
    )
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            bmg = batch.bmg
            bmg.to(device)
            pred = model.base(bmg)
            scores.append(pred.detach().cpu().numpy().reshape(-1))
    return np.concatenate(scores)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def metric_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, sub in df[df["pec50"].notna()].groupby("split"):
        y = sub["pec50"].to_numpy(dtype=float)
        score = sub["pairwise_score"].to_numpy(dtype=float)
        row = {
            "split": split,
            "n": int(len(sub)),
            "spearman": safe_spearman(score, y),
            "pearson": float(np.corrcoef(score, y)[0, 1])
            if len(sub) > 2
            else float("nan"),
            "score_mean": float(np.mean(score)),
            "score_std": float(np.std(score)),
        }
        high = (y >= 6.0).astype(int)
        low = (y < 3.0).astype(int)
        if high.min() != high.max():
            row["gte6_auc"] = float(roc_auc_score(high, score))
            row["gte6_ap"] = float(average_precision_score(high, score))
        if low.min() != low.max():
            row["lt3_auc"] = float(roc_auc_score(low, -score))
            row["lt3_ap"] = float(average_precision_score(low, -score))
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_pairwise_model(args.ckpt)

    df = load_pxr_compounds()
    print(f"Scoring PXR compounds: {len(df):,}")
    df["pairwise_score"] = score_smiles(
        model, df["smiles"].astype(str).tolist(), args.batch_size
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    metrics = metric_rows(df)
    metric_path = args.out.with_name(args.out.stem + "_metrics.csv")
    metrics.to_csv(metric_path, index=False)
    print(f"Saved scores: {args.out}")
    print(f"Saved metrics: {metric_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
