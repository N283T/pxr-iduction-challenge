"""Generate emax_pred side features for downstream pec50 stacking.

Hypothesis (issue #100 emax-MTL track, 2026-04-25):
  ``emax_estimate`` / ``emax_vs_pos_ctrl`` are 100% labelled in
  train_activity (4140/4140) but currently unused as targets. They are
  near-orthogonal to pec50 (Pearson r ~ -0.13) and capture *efficacy*
  whereas pec50 captures *potency* — different pharmacology axes.

  log2fc_pred (chemprop pretrain) is the pool's strongest decorrelating
  side feature today. The same Buterez 2024 strategy-2 pattern should
  work for emax: predict the missing labels, concatenate to the main
  feature set, let the downstream regressor lean on the new axis.

Phase 1A (LGBM + rdkit_desc_full): null result, +0.0106 LB MAE regress.
Phase 1B (TabPFN + cheme_2d_full_boltz_log2fc_pred 2103d): retest with
  the strongest M_emax we can quickly fit; if still null, conclude that
  emax-as-side-feature is bottlenecked by a deeper issue (e.g., the
  pool members already implicitly capture efficacy via their feature
  interactions) and move to chemprop-pretrain MTL (Phase 2).

Pipeline:
  1. Load feature matrix for all 4140 train + 513 test compounds.
     ``--feature`` switches between rdkit_desc_full (217d) and
     cheme_2d_full_boltz_log2fc_pred (2103d).
  2. 5-fold outer UMAP CV (canonical seed=42, k=50, Morgan+Jaccard) over
     the 4140 train rows. For each fold, train M_emax (LGBM or TabPFN)
     on emax_estimate and emax_vs_pos_ctrl separately (single-target).
     Predict out-of-fold values for the held-out 20%.
     -> 4140 OOF predictions per target, leak-free.
  3. Full-fit M_emax on all 4140 train -> predict emax for all 513 test.
  4. Save data/emax_predictions.parquet indexed by compound_id with
     columns ``emax_estimate_pred`` and ``emax_vs_pos_ctrl_pred``.

Usage:
    pixi run python track1_activity/scripts/run_emax_predict.py
    pixi run python track1_activity/scripts/run_emax_predict.py \\
        --model tabpfn --feature cheme_2d_full_boltz_log2fc_pred
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS, load_test_smiles, load_train_smiles_target  # noqa: E402
from splits import umap_split_indices  # noqa: E402

OUT_PATH = REPO_ROOT.joinpath("data", "emax_predictions.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

EMAX_COLS = ("emax_estimate", "emax_vs_pos_ctrl")
PRED_COLS = tuple(f"{c}_pred" for c in EMAX_COLS)


def load_train_emax_aligned(train_compound_ids: list[int]) -> np.ndarray:
    """emax_estimate + emax_vs_pos_ctrl aligned to train_compound_ids order."""
    conn = psycopg2.connect(**DB_PARAMS)
    df = pd.read_sql(
        """SELECT t.compound_id, t.emax_estimate, t.emax_vs_pos_ctrl
           FROM train_activity t
           ORDER BY t.id""",
        conn,
    )
    conn.close()
    df = df.set_index("compound_id").reindex(train_compound_ids)
    arr = df[list(EMAX_COLS)].to_numpy(dtype=np.float64)
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).any(axis=1).sum())
        raise RuntimeError(
            f"{n_bad} train rows have NaN/inf in emax columns; "
            "all train_activity rows should have emax labels."
        )
    return arr


def load_train_compound_ids() -> list[int]:
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute("SELECT compound_id FROM train_activity ORDER BY id")
    cids = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return cids


def load_test_compound_ids() -> list[int]:
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute("SELECT compound_id FROM test_activity ORDER BY id")
    cids = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return cids


def load_rdkit_full_for_ids(compound_ids: list[int]) -> np.ndarray:
    """Full RDKit descriptor matrix (217d) for arbitrary compound_ids."""
    from data import load_rdkit_full

    df = load_rdkit_full(compound_ids).reindex(compound_ids)
    arr = df.to_numpy(dtype=np.float64)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        mask = ~np.isfinite(col)
        if mask.any():
            col[mask] = float(np.nanmedian(col))
            arr[:, j] = col
    return arr


def load_cheme_features() -> tuple[np.ndarray, np.ndarray]:
    """Load cheme_2d_full_boltz_log2fc_pred (2103d) for train + test.

    Aligns to (train_activity.id, test_activity.id) order so the rows
    match load_train_compound_ids / load_test_compound_ids.
    """
    from run_train import load_features

    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    return load_features("cheme_2d_full_boltz_log2fc_pred", train_df, test_df)


def fit_lgbm(X: np.ndarray, y: np.ndarray, *, seed: int = 42, num_round: int = 500):
    import lightgbm as lgb

    params = {
        "objective": "regression",
        "metric": "mae",
        "verbose": -1,
        "seed": seed,
        "num_leaves": 63,
        "learning_rate": 0.02,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "lambda_l1": 0.01,
        "lambda_l2": 1.0,
    }
    return lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=num_round)


def fit_tabpfn(X: np.ndarray, y: np.ndarray, *, seed: int = 42):
    from tabpfn import TabPFNRegressor

    model = TabPFNRegressor(
        n_estimators=8,
        device="cuda",
        softmax_temperature=0.9,
        random_state=seed,
        ignore_pretraining_limits=True,
    )
    model.fit(X, y)
    return model


def fit_predict_M_emax(
    model_kind: str, X_tr, y_tr, X_pred, *, seed: int = 42
) -> np.ndarray:
    """Train M_emax with the chosen backbone and return predictions on X_pred."""
    if model_kind == "lgbm":
        m = fit_lgbm(X_tr, y_tr, seed=seed)
        return m.predict(X_pred)
    if model_kind == "tabpfn":
        m = fit_tabpfn(X_tr, y_tr, seed=seed)
        return m.predict(X_pred)
    raise ValueError(f"Unknown model: {model_kind}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["lgbm", "tabpfn"],
        default="lgbm",
        help="M_emax backbone. tabpfn is slower but typically more accurate "
        "on small (4140-row) tabular targets.",
    )
    parser.add_argument(
        "--feature",
        choices=["rdkit_desc_full", "cheme_2d_full_boltz_log2fc_pred"],
        default="rdkit_desc_full",
        help="Input features for M_emax. cheme_2d_full_boltz_log2fc_pred "
        "is the same 2103d set as the pool top member (overlap with "
        "downstream M_pec50 input but TabPFN handles redundant cols well).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_PATH,
        help="Parquet output path (default: data/emax_predictions.parquet)",
    )
    args = parser.parse_args()
    seed = 42
    n_splits = 5

    print(f"Config: model={args.model}, feature={args.feature}")
    print("Loading compound IDs and features...")
    train_ids = load_train_compound_ids()
    test_ids = load_test_compound_ids()
    print(f"  train={len(train_ids)}, test={len(test_ids)}")

    if args.feature == "rdkit_desc_full":
        X_train = load_rdkit_full_for_ids(train_ids)
        X_test = load_rdkit_full_for_ids(test_ids)
    else:
        X_train, X_test = load_cheme_features()
    print(f"  X_train={X_train.shape}, X_test={X_test.shape}")

    Y_train = load_train_emax_aligned(train_ids)
    print(f"  Y_train={Y_train.shape} (cols: {EMAX_COLS})")

    train_df = load_train_smiles_target()
    smiles = train_df["smiles"].tolist()
    splits = umap_split_indices(smiles, n_splits=n_splits, n_clusters=50, seed=seed)

    # ------------------------------------------------------------------
    # 5-fold UMAP cross-fit OOF predictions for train
    # ------------------------------------------------------------------
    oof_train = np.full_like(Y_train, np.nan, dtype=np.float64)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        for j, target_name in enumerate(EMAX_COLS):
            oof_train[va_idx, j] = fit_predict_M_emax(
                args.model,
                X_train[tr_idx],
                Y_train[tr_idx, j],
                X_train[va_idx],
                seed=seed,
            )
        # quick fold MAE
        fold_mae = np.mean(np.abs(oof_train[va_idx] - Y_train[va_idx]), axis=0)
        print(
            f"  [Fold {fold}] emax_estimate MAE={fold_mae[0]:.4f}, "
            f"emax_vs_pos_ctrl MAE={fold_mae[1]:.4f}"
        )

    overall_mae = np.mean(np.abs(oof_train - Y_train), axis=0)
    print(
        f"\nOOF MAE: emax_estimate={overall_mae[0]:.4f}, "
        f"emax_vs_pos_ctrl={overall_mae[1]:.4f}"
    )
    print(
        f"Target ranges: emax_estimate in "
        f"[{Y_train[:, 0].min():.2f}, {Y_train[:, 0].max():.2f}], "
        f"emax_vs_pos_ctrl in "
        f"[{Y_train[:, 1].min():.2f}, {Y_train[:, 1].max():.2f}]"
    )

    # ------------------------------------------------------------------
    # Full-fit on 4140 train -> predict test (513)
    # ------------------------------------------------------------------
    print("\nFull-fitting on all train -> predicting test...")
    pred_test = np.zeros((len(test_ids), 2), dtype=np.float64)
    for j, target_name in enumerate(EMAX_COLS):
        pred_test[:, j] = fit_predict_M_emax(
            args.model, X_train, Y_train[:, j], X_test, seed=seed
        )
        print(
            f"  {target_name}: test mean={pred_test[:, j].mean():.3f} "
            f"(train mean={Y_train[:, j].mean():.3f})"
        )

    # ------------------------------------------------------------------
    # Save parquet indexed by compound_id
    # ------------------------------------------------------------------
    df_train = pd.DataFrame(oof_train, index=train_ids, columns=list(PRED_COLS))
    df_test = pd.DataFrame(pred_test, index=test_ids, columns=list(PRED_COLS))
    out = pd.concat([df_train, df_test])
    out.index.name = "compound_id"
    out.to_parquet(args.out)
    print(f"\nWrote {args.out} ({len(out)} rows)")
    print(out.describe())


if __name__ == "__main__":
    main()
