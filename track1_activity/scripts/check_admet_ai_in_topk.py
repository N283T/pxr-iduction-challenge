"""Feature-importance check: do any ADMET-AI features rank in top-K
when concatenated to the cheme_2d_full_boltz_log2fc_pred_seed10ens
feature stack (2103d -> 2207d)?

Approach:
  1. Build cheme_2d_full_boltz_log2fc_pred_seed10ens (2103d) via the
     existing load_features pipeline.
  2. Concat ADMET-AI 104 features.
  3. Train per-fold LGBM (5-fold UMAP CV), record gain importances.
  4. Aggregate (mean) gain across folds, rank features.
  5. Report:
     - top-100 / top-200 / top-500 inclusion count for ADMET-AI cols
     - top ADMET-AI features by rank
     - sanity check on overall LGBM OOF MAE

Decision: if any ADMET-AI feature appears in top-100, augmented variant
worth implementing; if none in top-500, augmented null-likely.

No DB writes. Standalone diagnostic.

Legacy experiment script; internal design note was removed from the public repository.
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
from data import load_test_smiles, load_train_smiles_target  # noqa: E402
from splits import umap_split_indices  # noqa: E402

# Import the existing feature loader from run_train
import run_train  # noqa: E402

ADMET_PARQUET = REPO_ROOT.joinpath("data", "admet_ai_predictions.parquet")
FEATURE_NAME = "cheme_2d_full_boltz_log2fc_pred_seed10ens"
N_SPLITS = 5
N_CLUSTERS = 50
SEED = 42


def load_admet_features(
    n_train: int, n_test: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_parquet(ADMET_PARQUET)
    feat_cols = [c for c in df.columns if c not in ("smiles", "smiles_idx", "is_train")]
    train = df.loc[df["is_train"]].sort_values("smiles_idx")
    test = df.loc[~df["is_train"]].sort_values("smiles_idx")
    X_train = train[feat_cols].to_numpy(dtype=np.float64)
    X_test = test[feat_cols].to_numpy(dtype=np.float64)
    if len(X_train) != n_train:
        raise RuntimeError(f"ADMET train rows {len(X_train)} != {n_train}")
    if len(X_test) != n_test:
        raise RuntimeError(f"ADMET test rows {len(X_test)} != {n_test}")
    return X_train, X_test, feat_cols


def main() -> None:
    print("Loading data ...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y_train = train_df["pec50"].to_numpy(dtype=np.float64)

    print(
        f"\nBuilding feature: {FEATURE_NAME} (chemeleon 300d + 2d_full_boltz_log2fc_pred 1803d)"
    )
    X_base_train, X_base_test = run_train.load_features(FEATURE_NAME, train_df, test_df)
    print(f"  base shape: train {X_base_train.shape}, test {X_base_test.shape}")

    print("\nLoading ADMET-AI features ...")
    X_admet_train, X_admet_test, admet_cols = load_admet_features(
        n_train=len(train_df), n_test=len(test_df)
    )
    print(f"  ADMET shape: train {X_admet_train.shape}, test {X_admet_test.shape}")

    print("\nConcatenating base + ADMET ...")
    X_train = np.concatenate([X_base_train, X_admet_train], axis=1)
    X_test = np.concatenate([X_base_test, X_admet_test], axis=1)
    print(f"  combined: train {X_train.shape}, test {X_test.shape}")

    n_base = X_base_train.shape[1]
    n_admet = X_admet_train.shape[1]
    base_names = [f"base_{i}" for i in range(n_base)]
    feat_names = base_names + [f"admet:{c}" for c in admet_cols]
    assert len(feat_names) == X_train.shape[1]

    print(
        f"\nUMAP {N_SPLITS}-fold split (Morgan+Jaccard, k={N_CLUSTERS}, seed={SEED}) ..."
    )
    folds = umap_split_indices(
        train_df["smiles"].tolist(),
        n_splits=N_SPLITS,
        n_clusters=N_CLUSTERS,
        seed=SEED,
    )

    print("\nPer-fold LGBM (gain feature importance) ...")
    lgb_params = {
        "objective": "mae",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 30,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbose": -1,
        "seed": SEED,
    }
    importances = np.zeros((len(folds), X_train.shape[1]), dtype=np.float64)
    fold_maes = []
    oof_pred = np.zeros(len(X_train), dtype=np.float64)
    for fi, (tr_idx, va_idx) in enumerate(folds):
        dtr = lgb.Dataset(X_train[tr_idx], label=y_train[tr_idx])
        dva = lgb.Dataset(X_train[va_idx], label=y_train[va_idx], reference=dtr)
        booster = lgb.train(
            lgb_params,
            dtr,
            num_boost_round=2000,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        importances[fi] = booster.feature_importance(importance_type="gain")
        pred = booster.predict(X_train[va_idx])
        oof_pred[va_idx] = pred
        mae = float(np.mean(np.abs(pred - y_train[va_idx])))
        fold_maes.append(mae)
        print(f"  fold {fi}: MAE={mae:.4f} (n_iter={booster.best_iteration})")
    full_mae = float(np.mean(np.abs(oof_pred - y_train)))
    print(
        f"  full OOF MAE = {full_mae:.4f}  (per-fold mean = {np.mean(fold_maes):.4f})"
    )

    # Rank by mean importance
    mean_gain = importances.mean(axis=0)
    rank_order = np.argsort(-mean_gain)
    rank = np.zeros(len(mean_gain), dtype=np.int64)
    for r, idx in enumerate(rank_order):
        rank[idx] = r + 1  # 1-indexed

    # Inspect ADMET feature ranks
    admet_ranks = []
    for i, name in enumerate(feat_names):
        if name.startswith("admet:"):
            admet_ranks.append((rank[i], name, mean_gain[i]))
    admet_ranks.sort(key=lambda x: x[0])

    n_in_top100 = sum(1 for r, *_ in admet_ranks if r <= 100)
    n_in_top200 = sum(1 for r, *_ in admet_ranks if r <= 200)
    n_in_top500 = sum(1 for r, *_ in admet_ranks if r <= 500)

    print(
        f"\nADMET-AI features in top-K (out of {n_admet} ADMET cols, total {len(feat_names)} features):"
    )
    print(f"  top-100: {n_in_top100}")
    print(f"  top-200: {n_in_top200}")
    print(f"  top-500: {n_in_top500}")

    print("\nTop 30 ADMET-AI features by mean LGBM gain rank:")
    for r, name, g in admet_ranks[:30]:
        print(f"  rank {r:>4}  gain={g:.1f}  {name}")

    # Decision
    print("\n--- Decision ---")
    if n_in_top100 >= 1:
        print(
            f"PASS: {n_in_top100} ADMET-AI feature(s) in top-100 -> augmented variant "
            "worth implementing (TabPFN on combined 2207d, SWAP top500 candidate)."
        )
    elif n_in_top500 >= 5:
        print(
            f"MARGINAL: only {n_in_top100} in top-100 but {n_in_top500} in top-500. "
            "Augmented could still help via TabPFN (different model class than LGBM)."
        )
    else:
        print(
            f"NULL: only {n_in_top500} ADMET-AI features in top-500. "
            "Augmented variant unlikely to add signal — declare null."
        )


if __name__ == "__main__":
    main()
