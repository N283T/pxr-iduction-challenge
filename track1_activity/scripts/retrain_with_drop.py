#!/usr/bin/env -S pixi run python
"""Retrain the current LightGBM ensemble members with Issue #52 big_tail drop.

For each LightGBM member of the ensemble, look up the original
hyperparameters from the `experiments` table, apply the 32-compound
big_tail drop to the training set, and rerun the same UMAP 5-fold CV
with the same hyperparameters. Register each new run as
`<original_name>_drop32`.

Pseudo-OOF convention: the ensemble code requires OOF arrays aligned
with the original 4,140-row train set. For the 32 dropped compounds
we save the final-model prediction (trained on the kept 4,108) in
place of a true OOF. The signal the ensemble optimizer sees is
therefore dominated by the real OOF rows (4,108 / 4,140 = 99.2%);
the comparison against the pre-drop baseline is interpretable.

Scope: gap-regularization variants (`_gap1.0`) are skipped on this
pass because the gap objective is not stored in the hyperparameters
JSON. They can be re-added in a follow-up if useful.

Usage:
    pixi run python track1_activity/scripts/retrain_with_drop.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))

from data import (
    DB_PARAMS,
    load_test_smiles,
    load_train_smiles_target,
)  # noqa: E402
from evaluate import (  # noqa: E402
    compute_metrics,
    load_oof_predictions,
    print_fold_summary,
    record_experiment,
    save_oof_predictions,
)
from splits import umap_split_indices  # noqa: E402

# Re-use feature loader + compound-id helper from run_train without invoking its argparse.
from run_train import load_compound_ids, load_features  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

DROP_PARQUET = REPO_ROOT.joinpath("data", "eda_redo", "07_drop_candidates.parquet")
DROP_SUFFIX = "_drop32"

# Target ensemble members: (ensemble_name, source_experiment_id, feature_set)
# IDs are from the 2026-04-09 snapshot of experiment_summary (LightGBM
# members that are currently in ENSEMBLE_MODELS in run_ensemble.py,
# excluding gap1.0 variants).
TARGETS: tuple[tuple[str, int, str], ...] = (
    ("lgbm_mordred_jazzy_umap", 240, "mordred_jazzy"),
    ("lgbm_chemeleon_umap", 111, "chemeleon"),
    ("lgbm_chemberta_5m_mtr_umap", 112, "chemberta_5m_mtr"),
    ("lgbm_molformer_xl_umap", 142, "molformer_xl"),
    ("lgbm_count_morgan_r2_2048_umap", 116, "count_morgan_r2_2048"),
    ("lgbm_count_atompair_2048_umap", 118, "count_atompair_2048"),
    ("lgbm_count_morgan_r3_2048_umap", 117, "count_morgan_r3_2048"),
    ("lgbm_avalon_2048_umap", 114, "avalon_2048"),
    ("lgbm_morgan_r2_2048_umap", 115, "morgan_r2_2048"),
    ("lgbm_atompair_2048_umap", 119, "atompair_2048"),
    ("lgbm_feat_morgan_r2_2048_umap", 120, "feat_morgan_r2_2048"),
    ("lgbm_rdkit_desc_umap", 113, "rdkit_desc"),
)

N_SPLITS = 5
SEED = 42


def load_hyperparameters(exp_id: int) -> dict:
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute("SELECT hyperparameters FROM experiments WHERE id = %s", (exp_id,))
    r = cur.fetchone()
    cur.close()
    conn.close()
    if r is None or r[0] is None:
        raise RuntimeError(f"experiment id {exp_id} not found or no hyperparameters")
    return dict(r[0]) if isinstance(r[0], dict) else json.loads(r[0])


def load_drop_ids() -> set[int]:
    df = pd.read_parquet(DROP_PARQUET)
    return set(
        int(c) for c in df[df["drop_reason"].isin(["big_tail", "both"])]["compound_id"]
    )


def cv_one_target(
    name: str,
    src_exp_id: int,
    feature_set: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: np.ndarray,
    train_cids: np.ndarray,
    drop_mask: np.ndarray,
    smiles_all: list[str],
) -> dict:
    print(f"\n{'=' * 70}\n[retrain] {name}  ->  {name}{DROP_SUFFIX}")
    print(f"  feature_set={feature_set}  src_experiment_id={src_exp_id}")
    hp = load_hyperparameters(src_exp_id)
    # LightGBM via scikit API: strip n_estimators if present (we control via
    # early_stopping + average best_iteration_).
    hp = dict(hp)
    hp.setdefault("objective", "regression")
    hp.setdefault("metric", "mae")
    hp.setdefault("verbose", -1)
    hp.setdefault("seed", SEED)

    X_train_all, X_test = load_features(feature_set, train_df, test_df)
    X_train = X_train_all[drop_mask]
    y_kept = y_train[drop_mask]
    smiles_kept = [s for s, m in zip(smiles_all, drop_mask) if m]
    print(
        f"  X_train {X_train_all.shape} -> {X_train.shape}  (dropped {(~drop_mask).sum()})"
    )

    splits = umap_split_indices(smiles_kept, n_splits=N_SPLITS, seed=SEED)
    oof_kept = np.full_like(y_kept, np.nan, dtype=float)
    fold_metrics: list[dict] = []
    num_boost_rounds: list[int] = []
    for fi, (tr_idx, va_idx) in enumerate(splits):
        model = lgb.LGBMRegressor(**hp, n_estimators=2000)
        model.fit(
            X_train[tr_idx],
            y_kept[tr_idx],
            eval_set=[(X_train[va_idx], y_kept[va_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        pred = model.predict(X_train[va_idx])
        oof_kept[va_idx] = pred
        m = compute_metrics(y_kept[va_idx], pred)
        fold_metrics.append(m)
        nbr = model.best_iteration_ or model.n_estimators
        num_boost_rounds.append(int(nbr))
        print(f"  fold {fi}: RAE {m['RAE']:.4f}  MAE {m['MAE']:.4f}  n_boost={nbr}")
    print_fold_summary(fold_metrics)

    # Final model trained on all kept rows, using mean(best_iteration) as n_estimators
    final_n = int(max(50, round(np.mean(num_boost_rounds))))
    final_model = lgb.LGBMRegressor(**hp, n_estimators=final_n)
    final_model.fit(X_train, y_kept)

    # Fill dropped rows with final-model predictions (pseudo-OOF)
    oof_full = np.full(len(y_train), np.nan, dtype=float)
    kept_positions = np.where(drop_mask)[0]
    oof_full[kept_positions] = oof_kept
    drop_positions = np.where(~drop_mask)[0]
    oof_full[drop_positions] = final_model.predict(X_train_all[drop_positions])
    assert np.all(np.isfinite(oof_full)), "OOF has non-finite after fill"

    # Test prediction
    test_pred = final_model.predict(X_test)
    sub_path_rel = Path("track1_activity", "submissions", f"{name}{DROP_SUFFIX}.csv")
    sub_path_abs = REPO_ROOT.joinpath(sub_path_rel)
    sub_df = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": test_pred,
        }
    )
    sub_df.to_csv(sub_path_abs, index=False)
    print(f"  wrote {sub_path_abs}")

    # Register experiment
    exp_id = record_experiment(
        name=f"{name}{DROP_SUFFIX}",
        description=(
            f"Issue #52 big_tail drop (32 cpds) retrain of '{name}' (src id {src_exp_id}). "
            f"Same hyperparameters; pseudo-OOF (final-model pred) for dropped rows."
        ),
        model_type="lgbm",
        feature_set=feature_set,
        hyperparameters=hp,
        fold_metrics=fold_metrics,
        submission_path=str(sub_path_rel),
        notes="issue_52_drop32",
        num_boost_rounds=num_boost_rounds,
    )
    save_oof_predictions(exp_id, oof_full)
    return {
        "name": name,
        "new_name": f"{name}{DROP_SUFFIX}",
        "src_exp_id": src_exp_id,
        "new_exp_id": exp_id,
        "feature_set": feature_set,
        "n_kept": int(drop_mask.sum()),
        "rae_mean": float(np.mean([m["RAE"] for m in fold_metrics])),
        "rae_std": float(np.std([m["RAE"] for m in fold_metrics])),
        "mae_mean": float(np.mean([m["MAE"] for m in fold_metrics])),
    }


def main() -> None:
    print(f"[retrain] drop list: {DROP_PARQUET}")
    drop_ids = load_drop_ids()
    print(f"[retrain] big_tail drop: {len(drop_ids)} compound_ids")

    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y_train = train_df["pec50"].values.astype(float)
    train_cids = np.array(load_compound_ids("train"))
    assert len(train_cids) == len(train_df), "train_cids / train_df length mismatch"
    drop_mask = ~np.isin(train_cids, list(drop_ids))
    smiles_all = train_df["smiles"].tolist()
    print(
        f"[retrain] train N={len(train_df)}  kept N={drop_mask.sum()}"
        f"  dropped N={(~drop_mask).sum()}"
    )

    results = []
    for name, src_exp_id, feature_set in TARGETS:
        try:
            r = cv_one_target(
                name,
                src_exp_id,
                feature_set,
                train_df,
                test_df,
                y_train,
                train_cids,
                drop_mask,
                smiles_all,
            )
            results.append(r)
        except Exception as e:
            print(f"  [!] {name} FAILED: {e}")
            raise

    # Summary vs baseline (both naive-from-DB and fair same-subset comparison)
    print("\n" + "=" * 80)
    print("[retrain] summary")
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    rows = []
    for r in results:
        cur.execute(
            "SELECT rae_mean FROM experiment_summary WHERE id = %s",
            (r["src_exp_id"],),
        )
        base = cur.fetchone()
        base_rae_full = float(base[0]) if base else float("nan")

        # Fair comparison: compute both OOF RAEs on the identical kept subset
        base_oof = load_oof_predictions(r["src_exp_id"])
        new_oof = load_oof_predictions(r["new_exp_id"])
        kept_y = y_train[drop_mask]
        base_rae_kept = compute_metrics(kept_y, base_oof[drop_mask])["RAE"]
        new_rae_kept = compute_metrics(kept_y, new_oof[drop_mask])["RAE"]
        rows.append(
            {
                **r,
                "baseline_rae_full": base_rae_full,
                "baseline_rae_kept": float(base_rae_kept),
                "drop_rae_kept": float(new_rae_kept),
                "delta_fair": float(new_rae_kept - base_rae_kept),
            }
        )
    conn.close()
    df = pd.DataFrame(rows)
    print(
        df[
            [
                "name",
                "baseline_rae_kept",
                "drop_rae_kept",
                "delta_fair",
                "baseline_rae_full",
                "new_exp_id",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    out_path = REPO_ROOT.joinpath(
        "data", "eda_redo", "retrain_with_drop_summary.parquet"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\n[retrain] wrote {out_path}")


if __name__ == "__main__":
    main()
