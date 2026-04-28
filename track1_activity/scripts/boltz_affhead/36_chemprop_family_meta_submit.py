"""Family-level meta-member ensemble (Codex 2026-04-28 advice).

Mechanism: chemprop family 3 members are collapsed into a single mean
meta-member BEFORE caruana selection, so the optimizer cannot
overweight the family. Caruana_bag20's max-weight ~0.30 then
structurally caps chemprop family share at ~0.30 (vs the 0.85+ that
caused id=38/40/41 LB regress).

Pipeline:
  1. Build chemprop_family_meta = mean(cheme_t10, cheme_top500,
     chemprop_pretrain_embed) over OOF + test predictions
  2. Register as new experiment row in DB (with OOF + submission CSV)
  3. Run run_ensemble.main() with new 7-pool:
       chemprop_family_meta + kermt + molformer_c3 +
       pooled_boltz + pooled_boltz_allpairs + att + gate
  4. Run importance calibrator -> submission CSV ready

Usage:
    pixi run python track1_activity/scripts/boltz_affhead/36_chemprop_family_meta_submit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS, load_test_smiles, load_train_smiles_target  # noqa: E402
from evaluate import (  # noqa: E402
    compute_metrics,
    load_oof_predictions,
    record_experiment,
    save_oof_predictions,
)

import run_ensemble  # noqa: E402
import run_ensemble_calibrate_importance  # noqa: E402

CHEMPROP_FAMILY = (
    "tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_umap_default",
    "tabpfn_chemprop_pretrain_embed_umap_default",
    "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap",
)

META_NAME = "tabpfn_chemprop_family_meta_umap"
SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")


def fetch_experiment_paths(names: tuple[str, ...]) -> dict[str, tuple[int, str]]:
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, submission_path FROM experiments WHERE name = ANY(%s)",
        (list(names),),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    out: dict[str, tuple[int, str]] = {}
    for exp_id, name, sub_path in rows:
        out[name] = (exp_id, sub_path)
    missing = [n for n in names if n not in out]
    if missing:
        raise RuntimeError(f"Missing experiment rows: {missing}")
    return out


def build_meta(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, dict]:
    info = fetch_experiment_paths(CHEMPROP_FAMILY)
    n_train = len(train_df)
    n_test = len(test_df)

    oofs: list[np.ndarray] = []
    tests: list[np.ndarray] = []
    for name in CHEMPROP_FAMILY:
        exp_id, sub_path = info[name]
        oof = load_oof_predictions(exp_id)
        if oof is None or len(oof) != n_train:
            raise RuntimeError(
                f"{name}: OOF missing or length mismatch ({None if oof is None else len(oof)} vs {n_train})"
            )
        csv_path = REPO_ROOT.joinpath(sub_path)
        df = pd.read_csv(csv_path)
        if "pEC50" not in df.columns or len(df) != n_test:
            raise RuntimeError(
                f"{name}: submission CSV invalid ({len(df)} rows, cols {list(df.columns)})"
            )
        oofs.append(oof)
        tests.append(df["pEC50"].to_numpy())
        print(
            f"  loaded {name}: OOF MAE={np.mean(np.abs(oof - train_df['pec50'].to_numpy())):.4f}"
        )

    oof_meta = np.mean(np.column_stack(oofs), axis=1)
    test_meta = np.mean(np.column_stack(tests), axis=1)

    metrics = compute_metrics(train_df["pec50"].to_numpy(), oof_meta)
    print(
        f"\n  chemprop_family_meta OOF: MAE={metrics['MAE']:.4f}  "
        f"RAE={metrics['RAE']:.4f}  Spearman={metrics['Spearman_R']:.4f}"
    )
    return oof_meta, test_meta, metrics


def register_meta(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    oof_meta: np.ndarray,
    test_meta: np.ndarray,
    metrics: dict,
) -> int:
    sub_filename = f"{META_NAME}.csv"
    sub_path_rel = f"track1_activity/submissions/{sub_filename}"
    sub_df = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": test_meta,
        }
    )
    sub_df.to_csv(SUBMISSION_DIR.joinpath(sub_filename), index=False)

    exp_id = record_experiment(
        name=META_NAME,
        description=(
            "Equal-weight mean of 3 chemprop family members "
            "(cheme_t10, cheme_top500, chemprop_pretrain_embed). "
            "Family-level collapse to prevent caruana family concentration "
            "(Codex 2026-04-28 advice)."
        ),
        model_type="ensemble_meta",
        feature_set="chemprop_family_meta",
        hyperparameters={
            "members": list(CHEMPROP_FAMILY),
            "weights": [1.0 / len(CHEMPROP_FAMILY)] * len(CHEMPROP_FAMILY),
        },
        fold_metrics=[metrics],
        submission_path=sub_path_rel,
        notes=f"OOF MAE={metrics['MAE']:.4f}, equal-weight mean of chemprop family",
        on_conflict_replace=True,
    )
    save_oof_predictions(exp_id, oof_meta)
    print(f"  registered {META_NAME} as experiment id={exp_id}")
    return exp_id


def build_new_pool() -> tuple[str, ...]:
    base = list(run_ensemble.ENSEMBLE_MODELS)
    pool = [META_NAME]
    for m in base:
        if m not in CHEMPROP_FAMILY:
            pool.append(m)
    return tuple(pool)


def main() -> None:
    print("===== Step 1: build chemprop_family_meta =====")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    oof_meta, test_meta, metrics = build_meta(train_df, test_df)

    print("\n===== Step 2: register meta in DB =====")
    register_meta(train_df, test_df, oof_meta, test_meta, metrics)

    new_pool = build_new_pool()
    print(f"\n===== Step 3: run_ensemble with new pool ({len(new_pool)} members) =====")
    for m in new_pool:
        marker = " (META)" if m == META_NAME else ""
        print(f"  {m}{marker}")

    orig = run_ensemble.ENSEMBLE_MODELS
    run_ensemble.ENSEMBLE_MODELS = new_pool
    try:
        run_ensemble.main()
    finally:
        run_ensemble.ENSEMBLE_MODELS = orig

    print("\n===== Step 4: importance calibrator =====")
    run_ensemble_calibrate_importance.main()

    out = SUBMISSION_DIR.joinpath("ens_caruana_bag20_calibrated_importance.csv")
    print(f"\nReady to submit: {out}")


if __name__ == "__main__":
    main()
