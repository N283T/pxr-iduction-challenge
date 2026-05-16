#!/usr/bin/env python
"""Retrain fast baselines on pseudo-public holdouts.

The split audit identifies train subsets that look more test-like than the
canonical UMAP folds. This script asks a narrower question: if we retrain simple
models on those pseudo-public holdouts, do model/feature rankings change enough
to give a better submission gate?
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "track1_activity" / "src"
SCRIPT_DIR = REPO_ROOT / "track1_activity" / "scripts"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_pseudo_public_splits import build_frame, split_registry  # noqa: E402
from data import (  # noqa: E402
    DESCRIPTOR_COLS,
    get_engine,
    load_jazzy,
    load_singleconc_features,
    load_test_smiles,
    load_train_descriptors,
    load_train_mordred,
    load_train_rdkit_full,
    load_train_smiles_target,
)
from run_ensemble import load_models, optimize_caruana  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "pseudo_public_retrain"
LF_PATH = (
    REPO_ROOT
    / "data"
    / "chemprop_pretrain_log2fc_predictions_optuna_trial10_seed5ens.parquet"
)

EVAL_SPLITS = (
    "umap_canonical",
    "public_adv_top513",
    "public_testnn_top513",
    "public_log2fc_top513",
    "public_hybrid_nolabel_top513",
    "public_hybrid_with_y_top513",
    "public_chembl_ext_nn_ge025",
)


@dataclass(frozen=True)
class FeatureSet:
    name: str
    frame: pd.DataFrame
    family: str
    leakage_free_retrain: bool


def train_ids() -> pd.Index:
    ids = pd.read_sql(
        "SELECT compound_id FROM train_activity ORDER BY id", get_engine()
    )["compound_id"].astype(int)
    return pd.Index(ids, name="compound_id")


def numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def morgan_bits(smiles: list[str], n_bits: int = 2048) -> pd.DataFrame:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    rows: list[np.ndarray] = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        arr = np.zeros(n_bits, dtype=np.uint8)
        if mol is not None:
            fp = gen.GetFingerprint(mol)
            for bit in fp.GetOnBits():
                arr[bit] = 1
        rows.append(arr)
    return pd.DataFrame(rows, columns=[f"morgan_{i}" for i in range(n_bits)])


def load_feature_sets() -> tuple[np.ndarray, list[FeatureSet]]:
    ids = train_ids()
    train = load_train_smiles_target()
    y = train["pec50"].to_numpy(dtype=np.float64)

    desc = load_train_descriptors()[list(DESCRIPTOR_COLS)].reset_index(drop=True)

    rdkit_full, _ = load_train_rdkit_full()
    rdkit_full = rdkit_full.reindex(ids).reset_index(drop=True)

    mordred, _ = load_train_mordred()
    mordred = mordred.reindex(ids).reset_index(drop=True)

    singleconc = load_singleconc_features(ids.tolist()).reset_index(drop=True)
    jazzy = load_jazzy(ids.tolist()).reindex(ids).reset_index(drop=True)
    lf = pd.read_parquet(LF_PATH).reindex(ids).reset_index(drop=True)
    lf = lf.assign(
        lf_mean=0.5 * (lf["log2fc_8p25_pred"] + lf["log2fc_33_pred"]),
        lf_delta=lf["log2fc_33_pred"] - lf["log2fc_8p25_pred"],
    )
    lf_cols = ["log2fc_8p25_pred", "log2fc_33_pred", "lf_mean", "lf_delta"]

    morgan = morgan_bits(train["smiles"].tolist())

    sets = [
        FeatureSet("rdkit41_lgbm", numeric_frame(desc), "2d", True),
        FeatureSet("rdkit_full_lgbm", numeric_frame(rdkit_full), "2d", True),
        FeatureSet("mordred_lgbm", numeric_frame(mordred), "2d", True),
        FeatureSet("morgan2048_lgbm", numeric_frame(morgan), "fingerprint", True),
        FeatureSet(
            "rdkit_full_lf_pred_lgbm",
            numeric_frame(pd.concat([rdkit_full, lf[lf_cols]], axis=1)),
            "2d_plus_log2fc_pred",
            True,
        ),
        FeatureSet(
            "mordred_lf_pred_lgbm",
            numeric_frame(pd.concat([mordred, lf[lf_cols]], axis=1)),
            "2d_plus_log2fc_pred",
            True,
        ),
        FeatureSet(
            "rdkit_full_singleconc_jazzy_lgbm",
            numeric_frame(pd.concat([rdkit_full, singleconc, jazzy], axis=1)),
            "2d_plus_aux_observed",
            True,
        ),
        FeatureSet(
            "mordred_singleconc_jazzy_lgbm",
            numeric_frame(pd.concat([mordred, singleconc, jazzy], axis=1)),
            "2d_plus_aux_observed",
            True,
        ),
    ]
    return y, sets


def lgbm_model(seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l1",
        metric="mae",
        n_estimators=450,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
        return float("nan")
    return float(stats.spearmanr(y_true, y_pred).correlation)


def metric_row(
    *,
    split: str,
    fold: int,
    model: str,
    family: str,
    leakage_free_retrain: bool,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | int | str | bool]:
    err = y_pred - y_true
    return {
        "split": split,
        "fold": fold,
        "model": model,
        "family": family,
        "leakage_free_retrain": leakage_free_retrain,
        "n_val": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": safe_spearman(y_true, y_pred),
        "bias": float(err.mean()),
        "pred_std": float(np.std(y_pred, ddof=1)),
    }


def run_lgbm_feature_sets(
    y: np.ndarray,
    feature_sets: list[FeatureSet],
    registry: dict[str, list[tuple[np.ndarray, np.ndarray]]],
) -> list[dict[str, float | int | str | bool]]:
    rows: list[dict[str, float | int | str | bool]] = []
    for split_name in EVAL_SPLITS:
        if split_name not in registry:
            continue
        for fold, (tr, va) in enumerate(registry[split_name]):
            for feature_set in feature_sets:
                X = feature_set.frame
                model = make_pipeline(
                    SimpleImputer(strategy="median"),
                    lgbm_model(seed=42 + fold),
                )
                model.fit(X.iloc[tr], y[tr])
                pred = model.predict(X.iloc[va])
                rows.append(
                    metric_row(
                        split=split_name,
                        fold=fold,
                        model=feature_set.name,
                        family=feature_set.family,
                        leakage_free_retrain=feature_set.leakage_free_retrain,
                        y_true=y[va],
                        y_pred=pred,
                    )
                )
    return rows


def run_oof_stack_diagnostics(
    y: np.ndarray,
    registry: dict[str, list[tuple[np.ndarray, np.ndarray]]],
) -> list[dict[str, float | int | str | bool]]:
    rows: list[dict[str, float | int | str | bool]] = []
    n_test = len(load_test_smiles())
    _names, oof, _test = load_models(y, n_test)
    for split_name in EVAL_SPLITS:
        if split_name not in registry:
            continue
        for fold, (tr, va) in enumerate(registry[split_name]):
            y_va = y[va]
            simple = oof[va].mean(axis=1)
            rows.append(
                metric_row(
                    split=split_name,
                    fold=fold,
                    model="current_pool_simple_mean_oof",
                    family="oof_stack_diagnostic",
                    leakage_free_retrain=False,
                    y_true=y_va,
                    y_pred=simple,
                )
            )

            ridge = make_pipeline(
                SimpleImputer(strategy="median"),
                RidgeCV(alphas=np.logspace(-3, 3, 13)),
            )
            ridge.fit(oof[tr], y[tr])
            pred = ridge.predict(oof[va])
            rows.append(
                metric_row(
                    split=split_name,
                    fold=fold,
                    model="current_pool_ridge_stack_oof",
                    family="oof_stack_diagnostic",
                    leakage_free_retrain=False,
                    y_true=y_va,
                    y_pred=pred,
                )
            )

            weights = optimize_caruana(
                oof[tr],
                y[tr],
                n_iter=80,
                init_top_n=3,
                bag_frac=0.65,
                n_bags=10,
                seed=42 + fold,
            )
            pred = oof[va] @ weights
            rows.append(
                metric_row(
                    split=split_name,
                    fold=fold,
                    model="current_pool_caruana_stack_oof",
                    family="oof_stack_diagnostic",
                    leakage_free_retrain=False,
                    y_true=y_va,
                    y_pred=pred,
                )
            )
    return rows


def aggregate(rows: list[dict[str, float | int | str | bool]]) -> pd.DataFrame:
    fold_metrics = pd.DataFrame(rows)
    summary = (
        fold_metrics.groupby(
            ["split", "model", "family", "leakage_free_retrain"], as_index=False
        )
        .agg(
            folds=("fold", "nunique"),
            n_val_mean=("n_val", "mean"),
            mae=("mae", "mean"),
            mae_std=("mae", "std"),
            r2=("r2", "mean"),
            spearman=("spearman", "mean"),
            bias=("bias", "mean"),
            pred_std=("pred_std", "mean"),
        )
        .fillna({"mae_std": 0.0})
    )
    summary["rank_mae"] = summary.groupby("split")["mae"].rank(
        method="min", ascending=True
    )
    summary["rank_spearman"] = summary.groupby("split")["spearman"].rank(
        method="min", ascending=False
    )
    summary["rank_mae_within_kind"] = summary.groupby(
        ["split", "leakage_free_retrain"]
    )["mae"].rank(method="min", ascending=True)
    summary["rank_spearman_within_kind"] = summary.groupby(
        ["split", "leakage_free_retrain"]
    )["spearman"].rank(method="min", ascending=False)
    return summary.sort_values(["split", "rank_mae", "model"])


def write_report(summary: pd.DataFrame, fold_metrics: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(OUT_DIR / "pseudo_public_retrain_fold_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / "pseudo_public_retrain_summary.csv", index=False)

    leakage_free = summary[summary["leakage_free_retrain"]].copy()
    top_by_split = (
        leakage_free.sort_values(["split", "mae"])
        .groupby("split", as_index=False)
        .head(5)
    )
    diagnostic = summary[~summary["leakage_free_retrain"]].copy()

    lines = [
        "# Pseudo-Public Holdout Retrain Battery",
        "",
        "Fast leakage-free LGBM baselines are retrained per split. OOF-stack rows",
        "are diagnostics only because the base predictions come from the existing",
        "canonical OOF pool rather than split-specific retraining.",
        "",
        "## Top leakage-free retrains by split",
        "",
        top_by_split[
            [
                "split",
                "rank_mae_within_kind",
                "model",
                "family",
                "folds",
                "n_val_mean",
                "mae",
                "mae_std",
                "spearman",
                "bias",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## OOF-stack diagnostics",
        "",
        diagnostic[
            [
                "split",
                "rank_mae_within_kind",
                "model",
                "folds",
                "n_val_mean",
                "mae",
                "mae_std",
                "spearman",
                "bias",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    (OUT_DIR / "pseudo_public_retrain_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    warnings.filterwarnings(
        "ignore", message="X does not have valid feature names.*", category=UserWarning
    )
    warnings.filterwarnings(
        "ignore",
        message="Skipping features without any observed values.*",
        category=UserWarning,
    )
    frame = build_frame()
    registry = split_registry(frame)
    y, feature_sets = load_feature_sets()
    rows = run_lgbm_feature_sets(y, feature_sets, registry)
    rows.extend(run_oof_stack_diagnostics(y, registry))
    fold_metrics = pd.DataFrame(rows)
    summary = aggregate(rows)
    write_report(summary, fold_metrics)

    top = (
        summary[summary["leakage_free_retrain"]]
        .sort_values(["split", "mae"])
        .groupby("split", as_index=False)
        .head(3)
    )
    print(
        top[
            [
                "split",
                "rank_mae_within_kind",
                "model",
                "mae",
                "spearman",
                "bias",
            ]
        ].to_markdown(index=False, floatfmt=".4f")
    )
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
