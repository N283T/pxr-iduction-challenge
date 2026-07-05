"""Compute ADMET-AI (Stanford Swanson, 41 TDC endpoints + ~10 physchem)
predictions for all train + test compounds, cached to parquet.

After PR #151 #152 #153 (analog-prior + retrieval pivot 3-strike null),
this introduces externally-trained ADMET predictions as an orthogonal
information source to the existing pool (which is all SMILES embedding
based).

Output: data/admet_ai_predictions.parquet with columns:
  smiles_idx (int, 0..N-1, train order then test order),
  is_train (bool),
  smiles (string),
  <ADMET-AI feature columns>...

Legacy experiment script; internal design note was removed from the public repository.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
from admet_ai import ADMETModel

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
from data import load_test_smiles, load_train_smiles_target  # noqa: E402

OUT_PATH = REPO_ROOT.joinpath("data", "admet_ai_predictions.parquet")


def main() -> None:
    print("Loading SMILES (train + test) ...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    print(f"  train={len(train_df)}, test={len(test_df)}")

    all_smiles = train_df["smiles"].tolist() + test_df["smiles"].tolist()
    n_train = len(train_df)
    n_test = len(test_df)
    print(f"  combined SMILES list: {len(all_smiles)}")

    print("\nLoading ADMET-AI model ...")
    torch.set_float32_matmul_precision("high")  # mute the Ampere/Lovelace warning
    model = ADMETModel()

    print("Predicting (batched, all 4653 compounds at once) ...")
    preds = model.predict(smiles=all_smiles)
    if not isinstance(preds, pd.DataFrame):
        raise RuntimeError(
            f"ADMETModel.predict returned {type(preds).__name__}; "
            "expected DataFrame for batch input"
        )
    print(f"  predictions shape: {preds.shape}")
    print(f"  columns ({len(preds.columns)}): first 10 = {list(preds.columns)[:10]}")

    # ADMET-AI uses SMILES as index — make explicit columns
    preds = preds.reset_index().rename(columns={"index": "smiles", "SMILES": "smiles"})
    if "smiles" not in preds.columns:
        # Some versions of admet-ai use "SMILES" or use the raw index
        preds = preds.copy()
        preds.insert(0, "smiles", all_smiles)

    # If duplicate SMILES are in our input, ADMET-AI may dedupe; align by position
    if len(preds) != len(all_smiles):
        print(
            f"  WARNING: {len(preds)} predicted rows vs {len(all_smiles)} input. "
            "Re-aligning by SMILES match (assumes no NaN drops)."
        )
        preds = preds.set_index("smiles").reindex(all_smiles).reset_index()

    preds.insert(0, "is_train", [True] * n_train + [False] * n_test)
    preds.insert(0, "smiles_idx", list(range(len(all_smiles))))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote: {OUT_PATH}")
    print(f"  rows={len(preds)}, cols={len(preds.columns)}")

    # Sanity: NaN check, basic stats
    feat_cols = [
        c for c in preds.columns if c not in ("smiles", "smiles_idx", "is_train")
    ]
    print(f"  feature cols: {len(feat_cols)}")
    nan_per_col = preds[feat_cols].isna().sum()
    nan_cols = nan_per_col[nan_per_col > 0]
    if len(nan_cols) > 0:
        print(f"  WARN: {len(nan_cols)} columns have NaN values:")
        for c, n in nan_cols.items():
            print(f"    {c}: {n} NaN")
    else:
        print("  no NaN in any feature column")
    print("\nFirst 5 features stats (train portion):")
    for c in feat_cols[:5]:
        v = preds.loc[preds["is_train"], c].astype(float)
        print(
            f"    {c:>40}: mean={v.mean():.4f} std={v.std():.4f} "
            f"min={v.min():.4f} max={v.max():.4f}"
        )


if __name__ == "__main__":
    main()
