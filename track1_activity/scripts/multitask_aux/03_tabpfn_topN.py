#!/usr/bin/env -S pixi run python
"""TabPFN on importance-ranked top-N features (issue #85).

Reuses the feature matrix + gain ranking produced by
``01_importance_lgbm.py``. Selects the top-N rows of
``reports/multitask_aux/importance_all_desc.csv`` by ``gain_mean``, slices
those columns out of the full feature matrix, and trains TabPFN with
5-fold UMAP CV (Morgan+Jaccard+k=50+seed=42, canonical split).

Default N=2000 fits comfortably inside TabPFN's soft limit. Smaller
values (e.g. 1000) give a cleaner-signal variant but drop coverage of
weaker 3D descriptors that are the main source of orthogonality to the
existing ``tabpfn_2d_full_boltz_umap`` member.

TabPFN cannot consume NaN, so we fill with 0 (matches the existing
``tabpfn_2d_full_boltz_umap`` convention — see
``run_train.py:2d_full_boltz`` branch). Auranofin (cid 1657) rows are
kept under this convention rather than dropped.

Usage:
    pixi run python track1_activity/scripts/multitask_aux/03_tabpfn_topN.py
    # Or a smaller / larger N for A/B:
    pixi run python track1_activity/scripts/multitask_aux/03_tabpfn_topN.py \
        --n-top 1500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(
    0, str(REPO_ROOT.joinpath("track1_activity", "scripts", "multitask_aux"))
)

from data import load_test_smiles, load_train_smiles_target  # noqa: E402
from evaluate import (  # noqa: E402
    compute_metrics,
    print_fold_summary,
    print_metrics,
    record_experiment,
    save_oof_predictions,
)
from splits import umap_split_indices  # noqa: E402

# Reuse the same build_matrix that produced the importance CSV
importance_mod = __import__("01_importance_lgbm")

SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = REPO_ROOT.joinpath("track1_activity", "reports", "multitask_aux")
IMPORTANCE_CSV = REPORT_DIR.joinpath("importance_all_desc.csv")

TABPFN_PARAMS = dict(
    n_estimators=8,
    device="cuda",
    softmax_temperature=0.9,
    random_state=42,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="TabPFN on top-N by importance")
    parser.add_argument(
        "--n-top",
        type=int,
        default=2000,
        help="Number of features to keep (by gain_mean). Default 2000.",
    )
    parser.add_argument(
        "--split", choices=["umap", "scaffold"], default="umap"
    )
    args = parser.parse_args()

    print(
        f"TabPFN top-{args.n_top} | split={args.split} | device={TABPFN_PARAMS['device']}"
    )

    # Load data once. build_matrix returns (X_train, X_test, feat_names).
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    train_ids = importance_mod.load_compound_ids("train")
    test_ids = importance_mod.load_compound_ids("test")
    assert len(train_ids) == len(train_df)

    print(f"  Building full feature matrix: train={len(train_ids)} test={len(test_ids)}")
    X_tr_full, X_te_full, feat_names = importance_mod.build_matrix(train_ids, test_ids)
    print(f"  Full shape: {X_tr_full.shape} (train) / {X_te_full.shape} (test)")

    # Rank by importance and pick top-N
    imp = pd.read_csv(IMPORTANCE_CSV)
    assert len(imp) == len(feat_names), (
        f"importance CSV rows ({len(imp)}) != matrix cols ({len(feat_names)}). "
        "Re-run 01_importance_lgbm.py to regenerate."
    )
    imp_sorted = imp.sort_values("gain_mean", ascending=False).head(args.n_top)
    selected_names = imp_sorted["feature"].tolist()

    name_to_idx = {n: i for i, n in enumerate(feat_names)}
    keep_cols = np.asarray([name_to_idx[n] for n in selected_names], dtype=np.int64)

    X_tr = X_tr_full[:, keep_cols].astype(np.float32)
    X_te = X_te_full[:, keep_cols].astype(np.float32)

    # TabPFN refuses NaN/Inf -- mirror the 2d_full_boltz zero-fill convention.
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

    y = train_df["pec50"].to_numpy(dtype=np.float64)

    print("  Family composition of selected features:")
    fam_counts = imp_sorted["family"].value_counts()
    for fam, count in fam_counts.items():
        print(f"    {fam:<25s}  {count:>4d}")

    # CV splits
    if args.split == "scaffold":
        from splits import scaffold_split_indices

        folds = scaffold_split_indices(train_df["smiles"].tolist(), n_splits=5, seed=42)
    else:
        folds = umap_split_indices(
            train_df["smiles"].tolist(), n_splits=5, n_clusters=50, seed=42
        )

    exp_name = f"tabpfn_desc_top{args.n_top}_{args.split}_default"
    print(f"  Experiment: {exp_name}")

    from tabpfn import TabPFNRegressor

    oof = np.zeros_like(y)
    fold_metrics = []
    test_pred_per_fold = []

    for k, (tr_idx, va_idx) in enumerate(folds):
        Xtr, Xva = X_tr[tr_idx], X_tr[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        print(f"\n[Fold {k}] train={len(tr_idx)}, val={len(va_idx)}")

        model = TabPFNRegressor(**TABPFN_PARAMS)
        model.fit(Xtr, ytr)
        val_preds = model.predict(Xva)

        if not np.isfinite(val_preds).all():
            raise RuntimeError(
                f"Fold {k}: val_preds contain "
                f"{int((~np.isfinite(val_preds)).sum())} NaN/Inf"
            )
        oof[va_idx] = val_preds

        metrics = compute_metrics(yva, val_preds)
        fold_metrics.append(metrics)
        print_metrics(metrics, label=f"Fold {k}")

        test_preds = model.predict(X_te)
        if not np.isfinite(test_preds).all():
            raise RuntimeError(f"Fold {k}: test_preds contain NaN/Inf")
        test_pred_per_fold.append(test_preds)

        del model

    oof_metrics = compute_metrics(y, oof)
    print("\n  Overall OOF:")
    print_metrics(oof_metrics)
    print_fold_summary(fold_metrics)

    test_preds_mean = np.mean(test_pred_per_fold, axis=0)
    print(
        f"\n  Test preds: mean={test_preds_mean.mean():.3f}, "
        f"std={test_preds_mean.std():.3f}"
    )

    sub = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": test_preds_mean,
        }
    )
    sub_path = SUBMISSION_DIR.joinpath(f"{exp_name}.csv")
    sub_tmp = sub_path.with_suffix(sub_path.suffix + ".tmp")
    sub.to_csv(sub_tmp, index=False)

    try:
        exp_id = record_experiment(
            name=exp_name,
            description=(
                f"TabPFN on top-{args.n_top} features by LGBM gain importance "
                f"(reports/multitask_aux/importance_all_desc.csv). "
                f"5-fold {args.split} CV."
            ),
            model_type="tabpfn",
            feature_set=f"desc_top{args.n_top}",
            hyperparameters={
                **TABPFN_PARAMS,
                "n_top": args.n_top,
                "source_importance": str(IMPORTANCE_CSV.name),
                "family_counts": fam_counts.to_dict(),
            },
            fold_metrics=fold_metrics,
            submission_path=f"track1_activity/submissions/{exp_name}.csv",
            num_boost_rounds=[0] * len(fold_metrics),
            notes=(
                f"OOF RAE={oof_metrics['RAE']:.4f}, n_top={args.n_top}, "
                f"source=01_importance_lgbm.py all-desc run"
            ),
        )
    except Exception:
        sub_tmp.unlink(missing_ok=True)
        raise

    sub_tmp.replace(sub_path)
    print(f"  Saved submission: {sub_path}")
    save_oof_predictions(exp_id, oof)
    print(f"\n  Done: {exp_name} -> RAE={oof_metrics['RAE']:.4f}")


if __name__ == "__main__":
    main()
