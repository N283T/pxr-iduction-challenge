"""TabPFN on (cheme_2d_full_boltz_log2fc_pred_seed10ens + ADMET-AI) = 2207d.

Augmented variant of the existing top-weight pool member
`tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap` (single
OOF MAE ~0.397). Adds the 104 ADMET-AI features which the LGBM
top-K check (check_admet_ai_in_topk.py) ranked at positions 16, 23,
37 (3 features in top-100), confirming non-trivial biological signal
overlap (especially CYP3A4_Substrate_CarbonMangels at rank 23).

Pool member name: tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_admet_ai_umap

If this single OOF MAE beats the existing top500 (~0.397), it becomes a
SWAP candidate (replace top500 in ENSEMBLE_MODELS, no family share
concentration). Caruana bakeoff is the next step.

Legacy experiment script; internal design note was removed from the public repository.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
EXPERIMENT_NAME = "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_admet_ai_umap"
BASE_FEATURE = "cheme_2d_full_boltz_log2fc_pred_seed10ens"

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
    n_features = X_train.shape[1]
    n_base = X_base_train.shape[1]
    n_admet = X_admet_train.shape[1]
    print(f"  combined: train {X_train.shape}, test {X_test.shape}")
    print(f"  ({n_base} base + {n_admet} ADMET-AI = {n_features} total)")

    print(
        f"\nUMAP {N_SPLITS}-fold split (Morgan+Jaccard, k={N_CLUSTERS}, seed={SEED}) ..."
    )
    folds = umap_split_indices(
        train_df["smiles"].tolist(),
        n_splits=N_SPLITS,
        n_clusters=N_CLUSTERS,
        seed=SEED,
    )

    print("\nCross-validating TabPFN on combined features ...")
    from tabpfn import TabPFNRegressor

    oof_preds = np.zeros(len(X_train), dtype=np.float64)
    fold_metrics = []
    for fi, (tr_idx, va_idx) in enumerate(folds):
        model = TabPFNRegressor(**TABPFN_PARAMS)
        model.fit(X_train[tr_idx], y_train[tr_idx])
        oof_preds[va_idx] = model.predict(X_train[va_idx])
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
    full_model.fit(X_train, y_train)
    test_preds = full_model.predict(X_test)
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
            f"TabPFN on {BASE_FEATURE} ({n_base}d) + ADMET-AI ({n_admet}d) "
            f"= {n_features}d total"
        ),
        model_type="tabpfn",
        feature_set=f"{BASE_FEATURE}_admet_ai",
        hyperparameters={
            **TABPFN_PARAMS,
            "n_features": n_features,
            "n_base": n_base,
            "n_admet": n_admet,
            "n_splits": N_SPLITS,
            "n_clusters": N_CLUSTERS,
            "seed": SEED,
        },
        fold_metrics=fold_metrics,
        submission_path=f"track1_activity/submissions/{EXPERIMENT_NAME}.csv",
        notes=(
            f"OOF MAE={overall['MAE']:.4f}, augmented = {BASE_FEATURE} + ADMET-AI 104, "
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
