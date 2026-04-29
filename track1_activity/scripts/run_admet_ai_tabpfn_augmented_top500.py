"""TabPFN on (cheme_2d_full_boltz_log2fc_pred_seed10ens + ADMET-AI) -> top-500 by LGBM gain.

Augmented + top-500 filter variant. Mirrors the existing
`tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap` pipeline
(per `boltz_affhead/13_tabpfn_top_k_importance.py`):

  1. Build cheme_2d_full_boltz_log2fc_pred_seed10ens (2103d).
  2. Concat ADMET-AI (104d) -> 2207d.
  3. Fit LGBM once on full train, get gain importance.
  4. Select top-500 features by gain.
  5. TabPFN 5-fold UMAP CV on the selected 500 features.

Sibling of run_admet_ai_tabpfn_augmented.py (which uses all 2207d). The
top-500 filter is the format that the existing pool's top500 member uses
(single OOF MAE ~0.397).

Pool member name: tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_admet_ai_top500_umap

If single OOF MAE < 0.397, this is a SWAP candidate for the existing
`tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap`.

Spec: docs/superpowers/specs/2026-04-29-admet-ai-features-design.md
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
from evaluate import (  # noqa: E402
    compute_metrics,
    record_experiment,
    save_oof_predictions,
)
from splits import umap_split_indices  # noqa: E402

import run_train  # noqa: E402

ADMET_PARQUET = REPO_ROOT.joinpath("data", "admet_ai_predictions.parquet")
SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
EXPERIMENT_NAME = (
    "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_admet_ai_top500_umap"
)
BASE_FEATURE = "cheme_2d_full_boltz_log2fc_pred_seed10ens"
TOP_K = 500

N_SPLITS = 5
N_CLUSTERS = 50
SEED = 42

TABPFN_PARAMS = {
    "n_estimators": 8,
    "device": "cuda",
    "softmax_temperature": 0.9,
    "random_state": 42,
    "ignore_pretraining_limits": True,
}


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

    print(f"\nBuilding base feature: {BASE_FEATURE}")
    X_base_train, X_base_test = run_train.load_features(BASE_FEATURE, train_df, test_df)
    print(f"  base shape: train {X_base_train.shape}, test {X_base_test.shape}")

    print("\nLoading ADMET-AI features ...")
    X_admet_train, X_admet_test, admet_cols = load_admet_features(
        n_train=len(train_df), n_test=len(test_df)
    )
    print(f"  ADMET shape: train {X_admet_train.shape}, test {X_admet_test.shape}")

    X_train = np.concatenate([X_base_train, X_admet_train], axis=1).astype(np.float64)
    X_test = np.concatenate([X_base_test, X_admet_test], axis=1).astype(np.float64)
    n_features_full = X_train.shape[1]
    n_base = X_base_train.shape[1]
    n_admet = X_admet_train.shape[1]
    print(f"  combined: train {X_train.shape}, test {X_test.shape}")
    print(f"  ({n_base} base + {n_admet} ADMET-AI = {n_features_full} total)")

    # Sanitize NaN/Inf (TabPFN rejects them)
    col_mean = np.nanmean(X_train, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    X_train = np.where(np.isfinite(X_train), X_train, col_mean)
    X_test = np.where(np.isfinite(X_test), X_test, col_mean)

    print("\nFitting LGBM for feature importance (full-train, single fit) ...")
    # Match the existing `13_tabpfn_top_k_importance.py` recipe exactly.
    lgbm = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=10,
        random_state=42,
        verbose=-1,
    )
    lgbm.fit(X_train, y_train)
    importance_gain = lgbm.booster_.feature_importance(importance_type="gain")
    order = np.argsort(-importance_gain)
    sel = order[:TOP_K]

    n_admet_in_top = int(np.sum(sel >= n_base))
    print(f"  top-{TOP_K} selected, {n_admet_in_top} ADMET-AI features included")

    feat_names_full = [f"base_{i}" for i in range(n_base)] + [
        f"admet:{c}" for c in admet_cols
    ]
    sel_admet = [
        feat_names_full[i] for i in sel if feat_names_full[i].startswith("admet:")
    ]
    print(f"  ADMET-AI features in top-{TOP_K}: {sel_admet[:15]}...")

    X_train_sel = X_train[:, sel]
    X_test_sel = X_test[:, sel]
    print(f"  selected: train {X_train_sel.shape}, test {X_test_sel.shape}")

    print(
        f"\nUMAP {N_SPLITS}-fold split (Morgan+Jaccard, k={N_CLUSTERS}, seed={SEED}) ..."
    )
    folds = umap_split_indices(
        train_df["smiles"].tolist(),
        n_splits=N_SPLITS,
        n_clusters=N_CLUSTERS,
        seed=SEED,
    )

    print("\nCross-validating TabPFN on top-500 features ...")
    from tabpfn import TabPFNRegressor

    oof_preds = np.zeros(len(X_train_sel), dtype=np.float64)
    fold_metrics = []
    for fi, (tr_idx, va_idx) in enumerate(folds):
        model = TabPFNRegressor(**TABPFN_PARAMS)
        model.fit(X_train_sel[tr_idx], y_train[tr_idx])
        oof_preds[va_idx] = model.predict(X_train_sel[va_idx])
        m = compute_metrics(y_train[va_idx], oof_preds[va_idx])
        fold_metrics.append(m)
        print(
            f"  fold {fi}: train={len(tr_idx)} val={len(va_idx)} "
            f"MAE={m['MAE']:.4f}  RAE={m['RAE']:.4f}  Sp={m['Spearman_R']:.4f}"
        )
    overall = compute_metrics(y_train, oof_preds)
    print(
        f"\n  full OOF: MAE={overall['MAE']:.4f}  RAE={overall['RAE']:.4f}  "
        f"Sp={overall['Spearman_R']:.4f}  R2={overall['R2']:.4f}"
    )

    print("\nFitting on ALL train (for test prediction) ...")
    full_model = TabPFNRegressor(**TABPFN_PARAMS)
    full_model.fit(X_train_sel, y_train)
    test_preds = full_model.predict(X_test_sel)
    print(
        f"  test preds: mean={test_preds.mean():.4f} std={test_preds.std():.4f} "
        f"min={test_preds.min():.4f} max={test_preds.max():.4f}"
    )

    sub = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": test_preds,
        }
    )
    sub_path = SUBMISSION_DIR.joinpath(f"{EXPERIMENT_NAME}.csv")
    sub.to_csv(sub_path, index=False)
    print(f"  wrote {sub_path}")

    exp_id = record_experiment(
        name=EXPERIMENT_NAME,
        description=(
            f"TabPFN on top-{TOP_K} of ({BASE_FEATURE} {n_base}d + ADMET-AI {n_admet}d), "
            "selected by LGBM gain (full-train fit)"
        ),
        model_type="tabpfn",
        feature_set=f"{BASE_FEATURE}_admet_ai_top{TOP_K}",
        hyperparameters={
            **TABPFN_PARAMS,
            "n_features_full": n_features_full,
            "n_base": n_base,
            "n_admet": n_admet,
            "top_k": TOP_K,
            "n_admet_in_top_k": n_admet_in_top,
            "n_splits": N_SPLITS,
            "n_clusters": N_CLUSTERS,
            "seed": SEED,
        },
        fold_metrics=fold_metrics,
        submission_path=f"track1_activity/submissions/{EXPERIMENT_NAME}.csv",
        notes=(
            f"OOF MAE={overall['MAE']:.4f}, augmented top-{TOP_K} "
            f"({n_admet_in_top} ADMET-AI features included), "
            f"target SWAP for top500 (existing single OOF ~0.397). "
            "Codex orthogonal-info pivot 2026-04-29"
        ),
        on_conflict_replace=True,
    )
    save_oof_predictions(exp_id, oof_preds)
    print(f"\nDone. Experiment id={exp_id}, name={EXPERIMENT_NAME}")
    print(f"  OOF: MAE={overall['MAE']:.4f}  Sp={overall['Spearman_R']:.4f}")


if __name__ == "__main__":
    main()
