#!/usr/bin/env -S pixi run python
"""Train AS1-augmented final TabPFN heads for existing production members.

This is a Phase 2 diagnostic/candidate builder. It keeps the existing feature
extractors and auxiliary pretraining artifacts fixed, adds released AS1 labels
to the pEC50 training pool, then fits final TabPFN regressors on train+AS1.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "track1_activity" / "src"))
sys.path.insert(0, str(REPO_ROOT / "track1_activity" / "scripts"))

from data import get_engine, load_test_smiles, load_train_smiles_target  # noqa: E402
import run_train  # noqa: E402

SUBMISSION_DIR = REPO_ROOT / "track1_activity" / "submissions"
OUT_DIR = REPO_ROOT / "track1_activity" / "analysis" / "phase2_as1_augmented_suite"
ID55_PATH = SUBMISSION_DIR / "ens_id51_top500_potent46_t40_soft_g35.csv"
PREV_FINAL_PATH = (
    SUBMISSION_DIR
    / "phase2_as1_aug_top500_id55blend_a0p4_pairrankchembl_q95_g0p15_labels_as1.csv"
)


@dataclass(frozen=True)
class SuiteConfig:
    name: str
    feature: str
    top_k: int | None = None
    tabpfn_version: str = "v3"
    n_estimators: int = 8
    softmax_temperature: float = 0.9
    seed: int = 42


CONFIGS: dict[str, SuiteConfig] = {
    "cheme_t10_full": SuiteConfig(
        name="cheme_t10_full",
        feature="cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens",
    ),
    "cheme_seed10_top500": SuiteConfig(
        name="cheme_seed10_top500",
        feature="cheme_2d_full_boltz_log2fc_pred_seed10ens",
        top_k=500,
        softmax_temperature=0.7,
    ),
    "cheme_t10_top500": SuiteConfig(
        name="cheme_t10_top500",
        feature="cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens",
        top_k=500,
        softmax_temperature=0.9,
    ),
    "chemprop_embed": SuiteConfig(
        name="chemprop_embed",
        feature="chemprop_pretrain_embed",
    ),
    "molformer_c3": SuiteConfig(
        name="molformer_c3",
        feature="molformer_c3_pretrain_embed",
    ),
    "kermt": SuiteConfig(
        name="kermt",
        feature="kermt_pretrain_embed",
    ),
    "attentivefp": SuiteConfig(
        name="attentivefp",
        feature="attentivefp_pretrain_embed",
    ),
    "gatedgcn": SuiteConfig(
        name="gatedgcn",
        feature="gatedgcn_pretrain_embed",
    ),
    "pooled_boltz": SuiteConfig(
        name="pooled_boltz",
        feature="pooled_boltz",
    ),
    "pooled_boltz_allpairs": SuiteConfig(
        name="pooled_boltz_allpairs",
        feature="pooled_boltz_allpairs",
    ),
}


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


def finite_impute_from_train(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    X_train = X_train.astype(np.float32, copy=True)
    X_test = X_test.astype(np.float32, copy=True)
    col_mean = np.nanmean(np.where(np.isfinite(X_train), X_train, np.nan), axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0).astype(np.float32)
    X_train[~np.isfinite(X_train)] = np.broadcast_to(col_mean, X_train.shape)[
        ~np.isfinite(X_train)
    ]
    X_test[~np.isfinite(X_test)] = np.broadcast_to(col_mean, X_test.shape)[
        ~np.isfinite(X_test)
    ]
    return X_train, X_test


def select_top_k(
    X_aug: np.ndarray,
    y_aug: np.ndarray,
    top_k: int,
    seed: int,
    n_estimators: int,
) -> np.ndarray:
    ranker = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=10,
        random_state=seed,
        verbose=-1,
    )
    ranker.fit(X_aug, y_aug)
    gain = ranker.booster_.feature_importance(importance_type="gain")
    return np.argsort(-gain)[:top_k]


def tabpfn_model(cfg: SuiteConfig, device: str):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    version_enum = {
        "v3": ModelVersion.V3,
        "v2_6": ModelVersion.V2_6,
        "v2_5": ModelVersion.V2_5,
        "v2": ModelVersion.V2,
    }[cfg.tabpfn_version]
    ref = TabPFNRegressor.create_default_for_version(version_enum)
    return TabPFNRegressor(
        device=device,
        n_estimators=cfg.n_estimators,
        softmax_temperature=cfg.softmax_temperature,
        random_state=cfg.seed,
        model_path=ref.model_path,
        ignore_pretraining_limits=True,
    )


def load_submission(path: Path, column: str) -> pd.DataFrame:
    return pd.read_csv(path).rename(
        columns={"Molecule Name": "molecule_name", "pEC50": column}
    )[["molecule_name", column]]


def train_one(cfg: SuiteConfig, args: argparse.Namespace) -> dict:
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    as1 = load_as1_labels()
    as1_by_name = as1.set_index("molecule_name")["as1_pec50"]
    as1_mask = test_df["molecule_name"].isin(as1_by_name.index).to_numpy()
    as2_mask = ~as1_mask

    X_train, X_test = run_train.load_features(cfg.feature, train_df, test_df)
    X_train, X_test = finite_impute_from_train(X_train, X_test)
    y_train = train_df["pec50"].to_numpy(dtype=np.float32)
    y_as1 = (
        test_df.loc[as1_mask, "molecule_name"]
        .map(as1_by_name)
        .to_numpy(dtype=np.float32)
    )
    X_aug = np.concatenate([X_train, X_test[as1_mask]], axis=0)
    y_aug = np.concatenate([y_train, y_as1], axis=0)

    selected: np.ndarray | None = None
    if cfg.top_k is not None:
        selected = select_top_k(
            X_aug=X_aug,
            y_aug=y_aug,
            top_k=cfg.top_k,
            seed=cfg.seed,
            n_estimators=args.lgbm_estimators,
        )
        X_aug_fit = X_aug[:, selected]
        X_test_fit = X_test[:, selected]
    else:
        X_aug_fit = X_aug
        X_test_fit = X_test

    model = tabpfn_model(cfg, device=args.device)
    model.fit(X_aug_fit, y_aug)
    model_pred = model.predict(X_test_fit).astype(np.float64)

    model_only = pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": model_pred,
        }
    )
    labels_as1 = model_only.copy()
    labels_as1.loc[as1_mask, "pEC50"] = (
        test_df.loc[as1_mask, "molecule_name"].map(as1_by_name).to_numpy(dtype=float)
    )

    base_name = (
        f"phase2_as1_aug_suite_{cfg.name}_"
        f"tabpfn{cfg.tabpfn_version}_ne{cfg.n_estimators}_"
        f"t{cfg.softmax_temperature:g}"
    ).replace(".", "p")
    model_path = SUBMISSION_DIR / f"{base_name}_model_only.csv"
    labels_path = SUBMISSION_DIR / f"{base_name}_labels_as1.csv"
    model_only.to_csv(model_path, index=False)
    labels_as1.to_csv(labels_path, index=False)

    id55 = load_submission(ID55_PATH, "id55_pred")
    prev = load_submission(PREV_FINAL_PATH, "prev_final_pred")
    audit = (
        model_only.rename(
            columns={"Molecule Name": "molecule_name", "pEC50": "model_pred"}
        )
        .merge(id55, on="molecule_name", validate="one_to_one")
        .merge(prev, on="molecule_name", validate="one_to_one")
    )
    audit["split"] = np.where(as1_mask, "AS1", "AS2")
    audit["as1_pec50"] = test_df["molecule_name"].map(as1_by_name)
    as1_rows = audit["split"].eq("AS1")
    as2_rows = audit["split"].eq("AS2")
    as1_y = audit.loc[as1_rows, "as1_pec50"].to_numpy(dtype=float)
    as1_pred = audit.loc[as1_rows, "model_pred"].to_numpy(dtype=float)
    as2_pred = audit.loc[as2_rows, "model_pred"].to_numpy(dtype=float)
    as2_id55 = audit.loc[as2_rows, "id55_pred"].to_numpy(dtype=float)
    as2_prev = audit.loc[as2_rows, "prev_final_pred"].to_numpy(dtype=float)

    summary = {
        "name": cfg.name,
        "feature": cfg.feature,
        "top_k": cfg.top_k,
        "tabpfn_version": cfg.tabpfn_version,
        "n_estimators": cfg.n_estimators,
        "softmax_temperature": cfg.softmax_temperature,
        "n_features_raw": int(X_train.shape[1]),
        "n_features_fit": int(X_aug_fit.shape[1]),
        "n_train_original": int(len(y_train)),
        "n_as1_labels": int(len(y_as1)),
        "n_as2": int(as2_mask.sum()),
        "model_only_path": str(model_path.relative_to(REPO_ROOT)),
        "labels_as1_path": str(labels_path.relative_to(REPO_ROOT)),
        "as1_mae_model_only": float(mean_absolute_error(as1_y, as1_pred)),
        "as1_bias_model_only": float(np.mean(as1_pred - as1_y)),
        "as1_spearman_model_only": float(stats.spearmanr(as1_y, as1_pred).statistic),
        "as2_mean_pred": float(np.mean(as2_pred)),
        "as2_std_pred": float(np.std(as2_pred)),
        "as2_mean_abs_shift_vs_id55": float(np.mean(np.abs(as2_pred - as2_id55))),
        "as2_p90_abs_shift_vs_id55": float(
            np.quantile(np.abs(as2_pred - as2_id55), 0.90)
        ),
        "as2_max_abs_shift_vs_id55": float(np.max(np.abs(as2_pred - as2_id55))),
        "as2_mean_abs_shift_vs_prev_final": float(np.mean(np.abs(as2_pred - as2_prev))),
        "as2_corr_vs_id55": float(np.corrcoef(as2_pred, as2_id55)[0, 1]),
    }
    if selected is not None:
        selected_path = OUT_DIR / f"{base_name}_selected_features.csv"
        pd.DataFrame({"feature_idx": selected}).to_csv(selected_path, index=False)
        summary["selected_features_path"] = str(selected_path.relative_to(REPO_ROOT))

    audit_path = OUT_DIR / f"{base_name}_audit.csv"
    audit.to_csv(audit_path, index=False)
    summary["audit_path"] = str(audit_path.relative_to(REPO_ROOT))
    return summary


def build_ensemble_summaries(rows: list[dict]) -> None:
    if not rows:
        return
    pred_frames = []
    for row in rows:
        p = REPO_ROOT / row["model_only_path"]
        pred = load_submission(p, row["name"])
        pred_frames.append(pred)
    merged = pred_frames[0]
    for frame in pred_frames[1:]:
        merged = merged.merge(frame, on="molecule_name", validate="one_to_one")
    pred_cols = [r["name"] for r in rows]
    corr = merged[pred_cols].corr(method="pearson")
    corr.to_csv(OUT_DIR / "model_only_test_prediction_correlation.csv")

    as1 = load_as1_labels().set_index("molecule_name")["as1_pec50"]
    is_as1 = merged["molecule_name"].isin(as1.index).to_numpy()
    mean_pred = merged[pred_cols].mean(axis=1).to_numpy(dtype=float)
    mean_df = pd.read_csv(REPO_ROOT / rows[0]["model_only_path"])
    mean_df["pEC50"] = mean_pred
    mean_path = SUBMISSION_DIR / "phase2_as1_aug_suite_mean_model_only.csv"
    mean_df.to_csv(mean_path, index=False)
    labels = mean_df.copy()
    labels.loc[is_as1, "pEC50"] = labels.loc[is_as1, "Molecule Name"].map(as1)
    labels_path = SUBMISSION_DIR / "phase2_as1_aug_suite_mean_labels_as1.csv"
    labels.to_csv(labels_path, index=False)

    id55 = load_submission(ID55_PATH, "id55_pred")
    joined = mean_df.rename(
        columns={"Molecule Name": "molecule_name", "pEC50": "mean_pred"}
    ).merge(id55, on="molecule_name", validate="one_to_one")
    as1_rows = joined["molecule_name"].isin(as1.index).to_numpy()
    y = joined.loc[as1_rows, "molecule_name"].map(as1).to_numpy(dtype=float)
    pred = joined.loc[as1_rows, "mean_pred"].to_numpy(dtype=float)
    as2 = ~as1_rows
    ens_row = {
        "name": "suite_mean",
        "model_only_path": str(mean_path.relative_to(REPO_ROOT)),
        "labels_as1_path": str(labels_path.relative_to(REPO_ROOT)),
        "as1_mae_model_only": float(mean_absolute_error(y, pred)),
        "as1_bias_model_only": float(np.mean(pred - y)),
        "as1_spearman_model_only": float(stats.spearmanr(y, pred).statistic),
        "as2_mean_abs_shift_vs_id55": float(
            np.mean(
                np.abs(
                    joined.loc[as2, "mean_pred"].to_numpy(dtype=float)
                    - joined.loc[as2, "id55_pred"].to_numpy(dtype=float)
                )
            )
        ),
        "as2_p90_abs_shift_vs_id55": float(
            np.quantile(
                np.abs(
                    joined.loc[as2, "mean_pred"].to_numpy(dtype=float)
                    - joined.loc[as2, "id55_pred"].to_numpy(dtype=float)
                ),
                0.90,
            )
        ),
        "as2_corr_vs_id55": float(
            np.corrcoef(
                joined.loc[as2, "mean_pred"].to_numpy(dtype=float),
                joined.loc[as2, "id55_pred"].to_numpy(dtype=float),
            )[0, 1]
        ),
    }
    pd.DataFrame([ens_row]).to_csv(OUT_DIR / "suite_mean_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "cheme_t10_full",
            "cheme_t10_top500",
            "cheme_seed10_top500",
            "chemprop_embed",
            "molformer_c3",
            "kermt",
            "attentivefp",
            "gatedgcn",
            "pooled_boltz",
            "pooled_boltz_allpairs",
        ],
        choices=sorted(CONFIGS),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lgbm-estimators", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in args.configs:
        cfg = CONFIGS[name]
        print(f"\n=== {name}: feature={cfg.feature}, top_k={cfg.top_k} ===")
        row = train_one(cfg, args)
        rows.append(row)
        print(json.dumps(row, indent=2, sort_keys=True))

    summary = pd.DataFrame(rows).sort_values("as1_mae_model_only")
    summary_path = OUT_DIR / "summary.csv"
    summary.to_csv(summary_path, index=False)
    build_ensemble_summaries(rows)
    print(f"\nWrote {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
