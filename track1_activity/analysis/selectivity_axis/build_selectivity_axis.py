#!/usr/bin/env python
"""Build an internal-only selectivity-axis Track 1 candidate.

This script uses counter-assay labels only to learn test-computable structural
signals. It then applies a small, clipped residual correction to the id48 anchor.
It does not modify the production ensemble.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
sys.path.insert(
    0, str(REPO_ROOT.joinpath("track1_activity", "analysis", "internal_decorrelation"))
)

from data import DESCRIPTOR_COLS, get_engine, load_train_smiles_target  # noqa: E402
from decorrelated_caruana_sweep import (  # noqa: E402
    build_id48_anchor_oof,
    fit_importance_weights,
    load_submission,
)
from features import count_morgan_fp, smiles_to_mols  # noqa: E402

try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


OUT_DIR = Path(__file__).resolve().parent.joinpath("outputs")
SUB_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
ID48_PATH = REPO_ROOT.joinpath(
    "track1_activity",
    "analysis",
    "compound_level_lb",
    "outputs",
    "meta_axis_candidates",
    "ens_meta_axis_a343.csv",
)
ID50_PATH = SUB_DIR.joinpath("ens_internal_decor_cap101_bf50_b40_i1_l20.csv")

RANDOM_STATE = 42
N_SPLITS = 5
SELECTIVITY_WINDOW = 0.30
SHRINKS = (0.10, 0.20, 0.30)
CLIPS = (0.03, 0.05)
MODES = ("raw", "centered", "anti_id50")


@dataclass(frozen=True)
class AuxPredictions:
    train: pd.DataFrame
    test: pd.DataFrame
    diagnostics: dict[str, float]


def load_activity_frame(split: str) -> pd.DataFrame:
    desc = ", ".join(f"d.{col}" for col in DESCRIPTOR_COLS)
    if split == "train":
        sql = f"""
            SELECT
                t.id AS row_id,
                t.compound_id,
                c.std_smiles AS smiles,
                c.molecule_name,
                t.pec50,
                t.emax_estimate,
                t.emax_vs_pos_ctrl,
                ca.pec50 AS counter_pec50,
                ca.emax_estimate AS counter_emax,
                ca.emax_vs_pos_ctrl AS counter_emax_vs_pos_ctrl,
                {desc}
            FROM train_activity t
            JOIN compounds c ON c.id = t.compound_id
            LEFT JOIN counter_assay ca ON ca.compound_id = t.compound_id
            LEFT JOIN compound_descriptors d ON d.compound_id = t.compound_id
            ORDER BY t.id
        """
    elif split == "test":
        sql = f"""
            SELECT
                t.id AS row_id,
                t.compound_id,
                c.std_smiles AS smiles,
                c.molecule_name,
                {desc}
            FROM test_activity t
            JOIN compounds c ON c.id = t.compound_id
            LEFT JOIN compound_descriptors d ON d.compound_id = t.compound_id
            ORDER BY t.id
        """
    else:
        raise ValueError(f"unknown split: {split}")
    return pd.read_sql(sql, get_engine())


def build_feature_matrix(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    desc_cols = [col for col in DESCRIPTOR_COLS if col in train.columns]
    all_desc = pd.concat([train[desc_cols], test[desc_cols]], axis=0, ignore_index=True)
    all_desc = all_desc.apply(pd.to_numeric, errors="coerce")
    all_desc = all_desc.fillna(all_desc.median(numeric_only=True)).fillna(0.0)
    scaler = StandardScaler()
    desc_scaled = scaler.fit_transform(all_desc.to_numpy(dtype=np.float32))

    all_smiles = pd.concat([train["smiles"], test["smiles"]], ignore_index=True)
    mols = smiles_to_mols(all_smiles)
    fp = count_morgan_fp(mols, radius=2, n_bits=2048).astype(np.float32)
    # Count fingerprints are very sparse but small enough here; log1p makes
    # repeated environments less dominant.
    fp = np.log1p(fp)
    x_all = np.hstack([desc_scaled.astype(np.float32), fp])
    return x_all[: len(train)], x_all[len(train) :]


def make_classifier() -> object:
    if HAS_LIGHTGBM:
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=25,
            reg_alpha=0.1,
            reg_lambda=2.0,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    return HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.04,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )


def make_regressor() -> object:
    if HAS_LIGHTGBM:
        return LGBMRegressor(
            n_estimators=400,
            learning_rate=0.025,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=3.0,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.035,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )


def predict_proba_positive(model: object, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    if proba.shape[1] == 1:
        return np.full(len(x), float(model.classes_[0] == 1), dtype=np.float64)
    return proba[:, list(model.classes_).index(1)].astype(np.float64)


def fit_auxiliary_predictions(
    x_train: np.ndarray,
    x_test: np.ndarray,
    train: pd.DataFrame,
) -> AuxPredictions:
    counter_active = train["counter_pec50"].notna().to_numpy()
    delta = train["pec50"].to_numpy(dtype=np.float64) - train["counter_pec50"].to_numpy(
        dtype=np.float64
    )
    nonselective = (counter_active & (np.abs(delta) <= SELECTIVITY_WINDOW)).astype(int)

    active_oof = np.zeros(len(train), dtype=np.float64)
    nonselective_oof = np.zeros(len(train), dtype=np.float64)
    delta_oof = np.zeros(len(train), dtype=np.float64)
    active_test_preds = []
    nonselective_test_preds = []
    delta_test_preds = []

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (fit_idx, pred_idx) in enumerate(kf.split(x_train), start=1):
        active_model = make_classifier()
        active_model.fit(x_train[fit_idx], counter_active[fit_idx].astype(int))
        active_oof[pred_idx] = predict_proba_positive(active_model, x_train[pred_idx])
        active_test_preds.append(predict_proba_positive(active_model, x_test))

        fit_active_idx = fit_idx[counter_active[fit_idx]]
        if nonselective[fit_active_idx].min() == nonselective[fit_active_idx].max():
            nonselective_oof[pred_idx] = float(nonselective[fit_active_idx].mean())
            nonselective_test_preds.append(
                np.full(len(x_test), float(nonselective[fit_active_idx].mean()))
            )
        else:
            nonselective_model = make_classifier()
            nonselective_model.fit(
                x_train[fit_active_idx], nonselective[fit_active_idx]
            )
            nonselective_oof[pred_idx] = predict_proba_positive(
                nonselective_model, x_train[pred_idx]
            )
            nonselective_test_preds.append(
                predict_proba_positive(nonselective_model, x_test)
            )

        delta_model = make_regressor()
        delta_model.fit(x_train[fit_active_idx], delta[fit_active_idx])
        delta_oof[pred_idx] = delta_model.predict(x_train[pred_idx])
        delta_test_preds.append(delta_model.predict(x_test))
        print(f"aux fold {fold}/{N_SPLITS} done")

    aux_train = pd.DataFrame(
        {
            "counter_active_prob": active_oof,
            "nonselective_prob": nonselective_oof,
            "selectivity_delta_pred": delta_oof,
        }
    )
    aux_test = pd.DataFrame(
        {
            "counter_active_prob": np.mean(active_test_preds, axis=0),
            "nonselective_prob": np.mean(nonselective_test_preds, axis=0),
            "selectivity_delta_pred": np.mean(delta_test_preds, axis=0),
        }
    )

    diagnostics = {
        "counter_active_auc": float(
            roc_auc_score(counter_active.astype(int), aux_train["counter_active_prob"])
        ),
        "nonselective_auc_active_rows": float(
            roc_auc_score(
                nonselective[counter_active],
                aux_train.loc[counter_active, "nonselective_prob"],
            )
        ),
        "selectivity_delta_mae_active_rows": float(
            mean_absolute_error(
                delta[counter_active],
                aux_train.loc[counter_active, "selectivity_delta_pred"],
            )
        ),
        "n_counter_active": float(counter_active.sum()),
        "n_nonselective": float(nonselective[counter_active].sum()),
    }
    return AuxPredictions(aux_train, aux_test, diagnostics)


def residual_design(aux: pd.DataFrame, anchor_pred: np.ndarray) -> np.ndarray:
    active = aux["counter_active_prob"].to_numpy(dtype=np.float64)
    nonselective = aux["nonselective_prob"].to_numpy(dtype=np.float64)
    delta = aux["selectivity_delta_pred"].to_numpy(dtype=np.float64)
    return np.column_stack(
        [
            active,
            nonselective,
            delta,
            anchor_pred,
            anchor_pred * nonselective,
            anchor_pred * delta,
        ]
    )


def crossfit_residual_correction(
    x_resid: np.ndarray,
    x_test_resid: np.ndarray,
    residual: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_resid)
    x_test_scaled = scaler.transform(x_test_resid)

    oof_corr = np.zeros(len(residual), dtype=np.float64)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + 17)
    for fit_idx, pred_idx in kf.split(x_scaled):
        model = Ridge(alpha=10.0)
        model.fit(x_scaled[fit_idx], residual[fit_idx])
        oof_corr[pred_idx] = model.predict(x_scaled[pred_idx])

    model = Ridge(alpha=10.0)
    model.fit(x_scaled, residual)
    test_corr = model.predict(x_test_scaled)
    diagnostics = {
        "raw_corr_std_train": float(oof_corr.std(ddof=0)),
        "raw_corr_std_test": float(test_corr.std(ddof=0)),
        "raw_corr_residual_r": float(np.corrcoef(oof_corr, residual)[0, 1]),
    }
    return oof_corr, test_corr, diagnostics


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def spearman(y: np.ndarray, pred: np.ndarray) -> float:
    return float(stats.spearmanr(y, pred).statistic)


def build_candidates(
    train: pd.DataFrame,
    test: pd.DataFrame,
    id48_oof: np.ndarray,
    id48_test_df: pd.DataFrame,
    raw_oof_corr: np.ndarray,
    raw_test_corr: np.ndarray,
) -> pd.DataFrame:
    y = train["pec50"].to_numpy(dtype=np.float64)
    base_mae = mae(y, id48_oof)
    base_sp = spearman(y, id48_oof)
    id48_test = id48_test_df["pEC50"].to_numpy(dtype=np.float64)
    id50_test = load_submission(ID50_PATH)["pEC50"].to_numpy(dtype=np.float64)
    id50_direction = id50_test - id48_test

    rows = []
    for mode in MODES:
        mode_oof_raw = raw_oof_corr.copy()
        mode_test_raw = raw_test_corr.copy()
        if mode in {"centered", "anti_id50"}:
            mode_oof_raw = mode_oof_raw - mode_oof_raw.mean()
            mode_test_raw = mode_test_raw - mode_test_raw.mean()
        if mode == "anti_id50":
            projection = float(
                np.dot(mode_test_raw, id50_direction)
                / np.dot(id50_direction, id50_direction)
            )
            mode_test_raw = mode_test_raw - projection * id50_direction

        for shrink in SHRINKS:
            for clip in CLIPS:
                oof_corr = np.clip(mode_oof_raw, -clip, clip) * shrink
                test_corr = np.clip(mode_test_raw, -clip, clip) * shrink
                pred_oof = id48_oof + oof_corr
                pred_test = id48_test + test_corr
                out = id48_test_df.copy()
                out["pEC50"] = pred_test
                name = (
                    f"ens_selectivity_axis_{mode}_"
                    f"s{int(shrink * 100):02d}_c{int(clip * 100):02d}"
                )
                path = SUB_DIR.joinpath(f"{name}.csv")
                out.to_csv(path, index=False)
                shift = pred_test - id48_test
                projection = float(
                    np.dot(shift, id50_direction)
                    / np.dot(id50_direction, id50_direction)
                )
                rows.append(
                    {
                        "name": name,
                        "path": str(path.relative_to(REPO_ROOT)),
                        "mode": mode,
                        "shrink": shrink,
                        "clip": clip,
                        "oof_mae": mae(y, pred_oof),
                        "oof_delta_mae": mae(y, pred_oof) - base_mae,
                        "oof_spearman": spearman(y, pred_oof),
                        "oof_delta_spearman": spearman(y, pred_oof) - base_sp,
                        "test_mean_shift": float(shift.mean()),
                        "test_mean_abs_shift": float(np.abs(shift).mean()),
                        "test_p90_abs_shift": float(np.quantile(np.abs(shift), 0.90)),
                        "test_max_abs_shift": float(np.abs(shift).max()),
                        "projection_on_id50_direction": projection,
                        "pearson_vs_id48": float(
                            np.corrcoef(pred_test, id48_test)[0, 1]
                        ),
                    }
                )
    return pd.DataFrame(rows).sort_values(["oof_delta_mae", "test_mean_abs_shift"])


def write_report(
    aux_diag: dict[str, float],
    corr_diag: dict[str, float],
    summary: pd.DataFrame,
) -> None:
    report = [
        "# Selectivity Axis Report",
        "",
        "Internal-only axis using counter-assay-derived selectivity labels.",
        "",
        "## Auxiliary Predictor Diagnostics",
        "",
        pd.DataFrame([aux_diag]).to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Residual Correction Diagnostics",
        "",
        pd.DataFrame([corr_diag]).to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Candidate Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Read",
        "",
        "Prefer candidates with negative OOF delta, non-negative Spearman delta,",
        "test mean_abs_shift <= 0.02, and low projection on the failed id50 direction.",
    ]
    OUT_DIR.joinpath("report.md").write_text("\n".join(report) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = load_activity_frame("train")
    test = load_activity_frame("test")
    x_train, x_test = build_feature_matrix(train, test)

    importance_weights = fit_importance_weights()
    y_ref = load_train_smiles_target()["pec50"].to_numpy(dtype=np.float64)
    if not np.allclose(y_ref, train["pec50"].to_numpy(dtype=np.float64)):
        raise RuntimeError("train target order mismatch")
    id48_oof = build_id48_anchor_oof(y_ref, importance_weights)
    id48_test_df = load_submission(ID48_PATH)
    if not (
        id48_test_df["Molecule Name"].to_numpy() == test["molecule_name"].to_numpy()
    ).all():
        raise RuntimeError("id48/test order mismatch")

    aux = fit_auxiliary_predictions(x_train, x_test, train)
    x_resid = residual_design(aux.train, id48_oof)
    x_test_resid = residual_design(
        aux.test, id48_test_df["pEC50"].to_numpy(dtype=np.float64)
    )
    residual = train["pec50"].to_numpy(dtype=np.float64) - id48_oof
    raw_oof_corr, raw_test_corr, corr_diag = crossfit_residual_correction(
        x_resid, x_test_resid, residual
    )
    summary = build_candidates(
        train, test, id48_oof, id48_test_df, raw_oof_corr, raw_test_corr
    )
    aux.train.to_csv(OUT_DIR.joinpath("aux_train_predictions.csv"), index=False)
    aux.test.to_csv(OUT_DIR.joinpath("aux_test_predictions.csv"), index=False)
    summary.to_csv(OUT_DIR.joinpath("summary.csv"), index=False)
    write_report(aux.diagnostics, corr_diag, summary)
    print(f"Wrote selectivity-axis outputs to {OUT_DIR}")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
