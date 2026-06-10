#!/usr/bin/env -S pixi run python
"""Train an AS1-augmented top500 TabPFN candidate for Phase 2.

This is an actual prediction candidate, not another diagnostic. It adds the
released Analog Set 1 labels to the training pool, fits the strong
CheMe/2D/Boltz/log2fc top500 TabPFN recipe once on train+AS1, and predicts the
still-blinded AS2 compounds.

Two CSVs are written:

- submission CSV: AS1 rows are filled with released labels, AS2 rows with model
  predictions. This is the practical Phase 2 submission-format artifact.
- model-only CSV: all 513 rows use model predictions, for diagnostics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from data import get_engine, load_test_smiles, load_train_smiles_target  # noqa: E402
import run_train  # noqa: E402

SUBMISSION_DIR = REPO_ROOT / "track1_activity" / "submissions"
OUT_DIR = REPO_ROOT / "track1_activity" / "analysis" / "phase2_as1_augmented"


def load_as1_labels() -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT
            t.id AS test_id,
            c.molecule_name,
            l.pec50 AS as1_pec50
        FROM test_activity_phase1_labels l
        JOIN test_activity t ON t.compound_id = l.compound_id
        JOIN compounds c ON c.id = l.compound_id
        ORDER BY t.id
        """,
        get_engine(),
    )


def fit_predict(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    version_enum = {
        "v3": ModelVersion.V3,
        "v2_6": ModelVersion.V2_6,
        "v2_5": ModelVersion.V2_5,
        "v2": ModelVersion.V2,
    }[args.tabpfn_version]
    model_path = TabPFNRegressor.create_default_for_version(version_enum).model_path

    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    as1 = load_as1_labels()
    as1_by_name = as1.set_index("molecule_name")["as1_pec50"]
    as1_mask = test_df["molecule_name"].isin(as1_by_name.index).to_numpy()
    as2_mask = ~as1_mask

    X_train, X_test = run_train.load_features(args.feature, train_df, test_df)
    y_train = train_df["pec50"].to_numpy(dtype=np.float32)
    y_as1 = test_df.loc[as1_mask, "molecule_name"].map(as1_by_name).to_numpy(
        dtype=np.float32
    )

    col_mean = np.nanmean(X_train, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    X_train = np.where(np.isfinite(X_train), X_train, col_mean).astype(np.float32)
    X_test = np.where(np.isfinite(X_test), X_test, col_mean).astype(np.float32)

    X_aug = np.concatenate([X_train, X_test[as1_mask]], axis=0)
    y_aug = np.concatenate([y_train, y_as1], axis=0)
    print(
        f"feature={args.feature} d={X_train.shape[1]} train={len(y_train)} "
        f"AS1={len(y_as1)} AS2={int(as2_mask.sum())} augmented={len(y_aug)}"
    )

    ranker = lgb.LGBMRegressor(
        n_estimators=args.lgbm_estimators,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=10,
        random_state=args.seed,
        verbose=-1,
    )
    ranker.fit(X_aug, y_aug)
    gain = ranker.booster_.feature_importance(importance_type="gain")
    selected = np.argsort(-gain)[: args.k]
    print(
        f"selected top K={args.k}; zero_gain_selected={int((gain[selected] == 0).sum())}"
    )

    model = TabPFNRegressor(
        device=args.device,
        n_estimators=args.n_estimators,
        softmax_temperature=args.softmax_temperature,
        average_before_softmax=args.average_before_softmax,
        random_state=args.seed,
        model_path=model_path,
        ignore_pretraining_limits=args.k > 500,
    )
    model.fit(X_aug[:, selected], y_aug)
    model_pred = model.predict(X_test[:, selected]).astype(np.float64)

    practical_pred = model_pred.copy()
    practical_pred[as1_mask] = test_df.loc[as1_mask, "molecule_name"].map(as1_by_name)

    base_name = (
        f"phase2_as1_aug_{args.feature}_top{args.k}_"
        f"tabpfn{args.tabpfn_version}_ne{args.n_estimators}_t{args.softmax_temperature:g}"
    ).replace(".", "p")
    practical_path = SUBMISSION_DIR / f"{base_name}_labels_as1.csv"
    model_only_path = SUBMISSION_DIR / f"{base_name}_model_only.csv"

    practical = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": practical_pred,
        }
    )
    model_only = practical.copy()
    model_only["pEC50"] = model_pred
    practical.to_csv(practical_path, index=False)
    model_only.to_csv(model_only_path, index=False)

    metadata = {
        "feature": args.feature,
        "k": args.k,
        "tabpfn_version": args.tabpfn_version,
        "n_estimators": args.n_estimators,
        "softmax_temperature": args.softmax_temperature,
        "average_before_softmax": args.average_before_softmax,
        "seed": args.seed,
        "n_train_original": int(len(y_train)),
        "n_as1_labels": int(len(y_as1)),
        "n_as2_blind": int(as2_mask.sum()),
        "practical_submission": str(practical_path.relative_to(REPO_ROOT)),
        "model_only_submission": str(model_only_path.relative_to(REPO_ROOT)),
        "selected_feature_indices": selected.tolist(),
    }
    return practical, model_only, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature",
        default="cheme_2d_full_boltz_log2fc_pred_seed10ens",
        help="run_train feature set to use before top-K selection.",
    )
    parser.add_argument("--k", type=int, default=500)
    parser.add_argument("--tabpfn-version", choices=["v3", "v2_6", "v2_5", "v2"], default="v3")
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--softmax-temperature", type=float, default=0.7)
    parser.add_argument("--average-before-softmax", action="store_true")
    parser.add_argument("--lgbm-estimators", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _practical, _model_only, metadata = fit_predict(args)
    pd.Series(metadata).to_json(
        OUT_DIR / "latest_as1_augmented_top500_metadata.json",
        indent=2,
        force_ascii=False,
    )
    print(f"wrote {metadata['practical_submission']}")
    print(f"wrote {metadata['model_only_submission']}")
    print(f"wrote {OUT_DIR / 'latest_as1_augmented_top500_metadata.json'}")


if __name__ == "__main__":
    main()
