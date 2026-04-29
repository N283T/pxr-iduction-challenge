"""TabPFN on (chemeleon + 2d_full_boltz [no log2fc] + ADMET-AI) -> top-500 by LGBM gain.

Family-share-safe variant of the augmented ADMET-AI pool member. The original
augmented variant `tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_admet_ai_top500_umap`
counts as chemprop family because it includes log2fc_pred (chemprop pretrain
output). When ADDed to the 9-pool, family share rose to 0.887 (LB regress
zone per `project_family_share_lb_u_curve`).

This variant DROPS log2fc_pred — features are:
  chemeleon (300d) + 2d_full_boltz (1801d, no log2fc) + ADMET-AI (104d) = 2205d

Then LGBM-gain top-500 + TabPFN, mirroring the existing top500 pipeline.

Hypothesis: the chemprop family share will NOT include this member, so ADD
becomes safe. If it then earns caruana weight >= 0.01 AND Δ M2 (calibrated
OOF MAE) <= -0.003, it's a submit candidate.

Pool member name: tabpfn_cheme_2d_full_boltz_admet_ai_top500_umap

Spec: docs/superpowers/specs/2026-04-29-admet-ai-features-design.md (open question)
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
from data import (  # noqa: E402
    DB_PARAMS,
    load_test_smiles,
    load_train_smiles_target,
)
from evaluate import (  # noqa: E402
    compute_metrics,
    record_experiment,
    save_oof_predictions,
)
from splits import umap_split_indices  # noqa: E402

import run_train  # noqa: E402

ADMET_PARQUET = REPO_ROOT.joinpath("data", "admet_ai_predictions.parquet")
SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
EXPERIMENT_NAME = "tabpfn_cheme_2d_full_boltz_admet_ai_top500_umap"
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


def load_chemeleon(
    train_ids: list[int], test_ids: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Load chemeleon 300d embeddings from compound_chemeleon table."""
    with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT compound_id, embedding FROM compound_chemeleon ORDER BY compound_id"
        )
        rows = cur.fetchall()
    cheme_map = {int(r[0]): np.asarray(r[1], dtype=np.float32) for r in rows}
    cheme_tr = np.stack(
        [cheme_map.get(cid, np.zeros(300, dtype=np.float32)) for cid in train_ids]
    )
    cheme_te = np.stack(
        [cheme_map.get(cid, np.zeros(300, dtype=np.float32)) for cid in test_ids]
    )
    cheme_tr = np.nan_to_num(cheme_tr, nan=0.0, posinf=0.0, neginf=0.0)
    cheme_te = np.nan_to_num(cheme_te, nan=0.0, posinf=0.0, neginf=0.0)
    return cheme_tr.astype(np.float64), cheme_te.astype(np.float64)


def load_admet(n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
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

    # train_ids / test_ids for chemeleon load
    with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
        cur.execute("SELECT compound_id FROM train_activity ORDER BY id")
        train_ids = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT compound_id FROM test_activity ORDER BY id")
        test_ids = [r[0] for r in cur.fetchall()]
    assert len(train_ids) == len(train_df)
    assert len(test_ids) == len(test_df)

    print("\nBuilding feature: 2d_full_boltz (no log2fc, 1801d) ...")
    X_2df_train, X_2df_test = run_train.load_features(
        "2d_full_boltz", train_df, test_df
    )
    print(f"  2d_full_boltz: train {X_2df_train.shape}, test {X_2df_test.shape}")

    print("\nLoading chemeleon (300d) ...")
    X_cheme_train, X_cheme_test = load_chemeleon(train_ids, test_ids)
    print(f"  chemeleon: train {X_cheme_train.shape}, test {X_cheme_test.shape}")

    print("\nLoading ADMET-AI (104d) ...")
    X_admet_train, X_admet_test, admet_cols = load_admet(
        n_train=len(train_df), n_test=len(test_df)
    )
    print(f"  ADMET: train {X_admet_train.shape}, test {X_admet_test.shape}")

    X_train = np.concatenate(
        [X_cheme_train, X_2df_train.astype(np.float64), X_admet_train], axis=1
    )
    X_test = np.concatenate(
        [X_cheme_test, X_2df_test.astype(np.float64), X_admet_test], axis=1
    )
    n_features_full = X_train.shape[1]
    n_cheme = X_cheme_train.shape[1]
    n_2df = X_2df_train.shape[1]
    n_admet = X_admet_train.shape[1]
    print(f"\n  combined: train {X_train.shape}, test {X_test.shape}")
    print(
        f"  ({n_cheme} cheme + {n_2df} 2d_full_boltz + {n_admet} ADMET = {n_features_full} total)"
    )

    # Sanitize NaN/Inf
    col_mean = np.nanmean(X_train, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    X_train = np.where(np.isfinite(X_train), X_train, col_mean)
    X_test = np.where(np.isfinite(X_test), X_test, col_mean)

    print("\nFitting LGBM for feature importance (full-train, single fit) ...")
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

    n_admet_in_top = int(np.sum(sel >= n_cheme + n_2df))
    n_2df_in_top = int(np.sum((sel >= n_cheme) & (sel < n_cheme + n_2df)))
    n_cheme_in_top = int(np.sum(sel < n_cheme))
    print(
        f"  top-{TOP_K} selected: cheme={n_cheme_in_top}, 2d_full_boltz={n_2df_in_top}, ADMET={n_admet_in_top}"
    )

    feat_names_full = (
        [f"cheme_{i}" for i in range(n_cheme)]
        + [f"2df_{i}" for i in range(n_2df)]
        + [f"admet:{c}" for c in admet_cols]
    )
    sel_admet = [
        feat_names_full[i] for i in sel if feat_names_full[i].startswith("admet:")
    ]
    print(f"  ADMET features in top-{TOP_K}: {sel_admet[:15]}...")

    X_train_sel = X_train[:, sel]
    X_test_sel = X_test[:, sel]

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
            f"TabPFN on top-{TOP_K} of (chemeleon {n_cheme}d + 2d_full_boltz {n_2df}d "
            f"[no log2fc] + ADMET-AI {n_admet}d) = {n_features_full}d total. "
            "Family-share-safe: no log2fc_pred so not in chemprop family."
        ),
        model_type="tabpfn",
        feature_set="cheme_2d_full_boltz_admet_ai_top500_no_log2fc",
        hyperparameters={
            **TABPFN_PARAMS,
            "n_features_full": n_features_full,
            "n_cheme": n_cheme,
            "n_2df": n_2df,
            "n_admet": n_admet,
            "top_k": TOP_K,
            "n_admet_in_top_k": n_admet_in_top,
            "n_2df_in_top_k": n_2df_in_top,
            "n_cheme_in_top_k": n_cheme_in_top,
            "n_splits": N_SPLITS,
            "n_clusters": N_CLUSTERS,
            "seed": SEED,
        },
        fold_metrics=fold_metrics,
        submission_path=f"track1_activity/submissions/{EXPERIMENT_NAME}.csv",
        notes=(
            f"OOF MAE={overall['MAE']:.4f}, family-share-safe variant "
            f"(no log2fc_pred). {n_admet_in_top} ADMET in top-{TOP_K}. "
            "Codex orthogonal-info pivot 2026-04-29 (post-LB-proxy gate)"
        ),
        on_conflict_replace=True,
    )
    save_oof_predictions(exp_id, oof_preds)
    print(f"\nDone. Experiment id={exp_id}, name={EXPERIMENT_NAME}")
    print(f"  OOF: MAE={overall['MAE']:.4f}  Sp={overall['Spearman_R']:.4f}")


if __name__ == "__main__":
    main()
