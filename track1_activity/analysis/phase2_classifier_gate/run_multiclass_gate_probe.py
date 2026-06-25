#!/usr/bin/env -S pixi run python
"""Probe Phase 2 multiclass activity classifiers as soft tail gates.

This is an experiment-only analysis. It trains bin classifiers on the
train+AS1 labeled pool using the existing Phase 2 folds, then tests whether
OOF class probabilities can improve tail-aware corrections around the current
id55 anchor and the Phase 2 OOF proxy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
sys.path.insert(
    0, str(REPO_ROOT.joinpath("track1_activity", "analysis", "phase2_htchem_pred_axis"))
)
sys.path.insert(
    0,
    str(REPO_ROOT.joinpath("track1_activity", "analysis", "phase2_validation_matrix")),
)

import run_train  # noqa: E402
from build_phase2_validation_matrix import build_phase2_feature_matrix  # noqa: E402
from data import get_engine, load_test_smiles, load_train_smiles_target  # noqa: E402
from external_rank_features import external_rank_feature_matrices  # noqa: E402
from run_top500_tabpfn_htchem import pred_htchem_for_challenge  # noqa: E402

OUT_ROOT = Path(__file__).resolve().parent.joinpath("outputs")
DOC_PATH = REPO_ROOT / "docs" / "track1_explain" / "phase2_classifier_gate.md"
POOL_FOLDS_PATH = (
    REPO_ROOT
    / "track1_activity"
    / "analysis"
    / "phase2_validation_matrix"
    / "outputs"
    / "phase2_labeled_pool_with_folds.csv"
)
PHASE2_OOF_PATH = (
    REPO_ROOT
    / "track1_activity"
    / "analysis"
    / "phase2_validation_matrix"
    / "outputs"
    / "phase2_lgbm_topk_oof_predictions.csv"
)
RISK_MAP_PATH = (
    REPO_ROOT
    / "track1_activity"
    / "analysis"
    / "phase2_as2_risk_map"
    / "outputs"
    / "all_test_risk_map.csv"
)

BIN_LABELS = ["lt3", "3to4", "4to5", "5to6", "gte6"]
LOW_LABEL = "lt3"
HIGH_LABEL = "gte6"


def load_pool_with_folds() -> pd.DataFrame:
    if not POOL_FOLDS_PATH.exists():
        raise FileNotFoundError(
            f"{POOL_FOLDS_PATH} is missing. Run build_phase2_validation_matrix.py first."
        )
    pool = pd.read_csv(POOL_FOLDS_PATH)
    pool["true_bin"] = pd.Categorical(
        pool["true_bin"], categories=BIN_LABELS, ordered=True
    ).astype("object")
    return pool


def load_test_frame() -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT
            t.id AS test_id,
            c.id AS compound_id,
            c.molecule_name,
            c.std_smiles AS smiles,
            l.pec50 AS as1_pec50
        FROM test_activity t
        JOIN compounds c ON c.id = t.compound_id
        LEFT JOIN test_activity_phase1_labels l ON l.compound_id = t.compound_id
        ORDER BY t.id
        """,
        get_engine(),
    )


def sanitize_matrix(x: np.ndarray) -> np.ndarray:
    col_mean = np.nanmean(x, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0).astype(np.float32)
    return np.where(np.isfinite(x), x, col_mean).astype(np.float32)


def append_pred_htchem(
    x_pool: np.ndarray,
    x_test: np.ndarray,
    pool: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    pred_htchem = pred_htchem_for_challenge()
    pool_col = pred_htchem.reindex(pool["compound_id"].astype(int)).to_numpy(
        dtype=np.float32
    )[:, None]
    test_col = pred_htchem.reindex(test["compound_id"].astype(int)).to_numpy(
        dtype=np.float32
    )[:, None]
    return np.concatenate([x_pool, pool_col], axis=1), np.concatenate(
        [x_test, test_col], axis=1
    )


def build_feature_matrices(
    feature: str,
    pool: pd.DataFrame,
    test: pd.DataFrame,
    use_htchem: bool,
    external_rank_features: str,
) -> tuple[np.ndarray, np.ndarray]:
    x_pool = build_phase2_feature_matrix(feature, pool)
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    _x_train, x_test = run_train.load_features(feature, train_df, test_df)
    if use_htchem:
        x_pool, x_test = append_pred_htchem(x_pool, x_test, pool, test)
    if external_rank_features != "none":
        extra_pool, extra_test, extra_cols = external_rank_feature_matrices(
            external_rank_features, pool, test
        )
        print(
            f"appending external_rank_features={external_rank_features} "
            f"n_cols={len(extra_cols)}"
        )
        x_pool = np.concatenate([x_pool, extra_pool], axis=1)
        x_test = np.concatenate([x_test, extra_test], axis=1)
    return sanitize_matrix(x_pool), sanitize_matrix(x_test)


def label_arrays(
    pool: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, int], dict[int, str]]:
    label_to_id = {label: idx for idx, label in enumerate(BIN_LABELS)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    y = pool["true_bin"].map(label_to_id).to_numpy(dtype=np.int64)
    return y, label_to_id, id_to_label


def select_features(
    x_train: np.ndarray,
    y_train: np.ndarray,
    top_k: int,
    seed: int,
    force_last_n: int = 0,
) -> np.ndarray:
    forced = (
        np.arange(x_train.shape[1] - force_last_n, x_train.shape[1], dtype=np.int64)
        if force_last_n > 0
        else np.empty(0, dtype=np.int64)
    )
    if top_k <= 0 or top_k >= x_train.shape[1]:
        selected = np.arange(x_train.shape[1], dtype=np.int64)
        return np.unique(np.concatenate([selected, forced])).astype(np.int64)
    ranker = lgb.LGBMClassifier(
        objective="multiclass",
        class_weight="balanced",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=10,
        random_state=seed,
        verbose=-1,
    )
    ranker.fit(x_train, y_train)
    gain = ranker.booster_.feature_importance(importance_type="gain")
    selected = np.argsort(-gain)[:top_k].astype(np.int64)
    return np.unique(np.concatenate([selected, forced])).astype(np.int64)


def make_classifier(args: argparse.Namespace, seed: int):
    if args.model == "lgbm":
        return lgb.LGBMClassifier(
            objective="multiclass",
            class_weight="balanced",
            n_estimators=args.lgbm_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            min_child_samples=args.min_child_samples,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_lambda=1.0,
            random_state=seed,
            verbose=-1,
        )

    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    version_enum = {
        "v3": ModelVersion.V3,
        "v2_6": ModelVersion.V2_6,
        "v2_5": ModelVersion.V2_5,
        "v2": ModelVersion.V2,
    }[args.tabpfn_version]
    model_path = TabPFNClassifier.create_default_for_version(version_enum).model_path
    return TabPFNClassifier(
        device=args.device,
        n_estimators=args.n_estimators,
        softmax_temperature=args.softmax_temperature,
        balance_probabilities=args.balance_probabilities,
        average_before_softmax=args.average_before_softmax,
        random_state=seed,
        model_path=model_path,
        ignore_pretraining_limits=True,
        show_progress_bar=True,
    )


def predict_proba_aligned(model, x: np.ndarray, n_classes: int) -> np.ndarray:
    raw = model.predict_proba(x)
    classes = getattr(model, "classes_", np.arange(raw.shape[1]))
    out = np.zeros((x.shape[0], n_classes), dtype=np.float64)
    for src_idx, cls in enumerate(classes):
        out[:, int(cls)] = raw[:, src_idx]
    row_sum = out.sum(axis=1, keepdims=True)
    return np.divide(
        out, row_sum, out=np.full_like(out, 1.0 / n_classes), where=row_sum > 0
    )


def run_oof(
    args: argparse.Namespace,
    pool: pd.DataFrame,
    x_pool: np.ndarray,
    x_test: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    n_classes = len(BIN_LABELS)
    oof = np.zeros((len(pool), n_classes), dtype=np.float64)
    test_fold_probas = []
    rows = []

    for fold in sorted(pool["fold"].unique()):
        val_idx = pool.index[pool["fold"].eq(fold)].to_numpy(dtype=np.int64)
        train_idx = pool.index[~pool["fold"].eq(fold)].to_numpy(dtype=np.int64)
        selected = select_features(
            x_pool[train_idx],
            y[train_idx],
            args.top_k,
            args.seed,
            args.force_last_n_features,
        )
        clf = make_classifier(args, args.seed + int(fold))
        clf.fit(x_pool[train_idx][:, selected], y[train_idx])
        oof[val_idx] = predict_proba_aligned(
            clf, x_pool[val_idx][:, selected], n_classes
        )
        test_fold_probas.append(
            predict_proba_aligned(clf, x_test[:, selected], n_classes)
        )
        rows.append(
            {
                "fold": int(fold),
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_features": int(len(selected)),
            }
        )
        print(
            f"fold={int(fold)} n_val={len(val_idx)} features={len(selected)} "
            f"val_acc={accuracy_score(y[val_idx], oof[val_idx].argmax(axis=1)):.4f}"
        )

    return oof, np.mean(np.stack(test_fold_probas), axis=0), pd.DataFrame(rows)


def train_final_test_proba(
    args: argparse.Namespace,
    x_pool: np.ndarray,
    x_test: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected = select_features(
        x_pool, y, args.top_k, args.seed, args.force_last_n_features
    )
    clf = make_classifier(args, args.seed)
    clf.fit(x_pool[:, selected], y)
    return predict_proba_aligned(clf, x_test[:, selected], len(BIN_LABELS)), selected


def binary_auc_rows(
    y: np.ndarray,
    proba: np.ndarray,
    label_to_id: dict[str, int],
) -> list[dict[str, float | str]]:
    rows = []
    for label in [LOW_LABEL, HIGH_LABEL]:
        cls = label_to_id[label]
        target = (y == cls).astype(int)
        score = proba[:, cls]
        row: dict[str, float | str] = {"class": label, "n_pos": int(target.sum())}
        if target.min() == target.max():
            row["roc_auc"] = np.nan
            row["average_precision"] = np.nan
        else:
            row["roc_auc"] = float(roc_auc_score(target, score))
            row["average_precision"] = float(average_precision_score(target, score))
        rows.append(row)
    return rows


def summarize_classifier(
    pool: pd.DataFrame,
    y: np.ndarray,
    proba: np.ndarray,
    label_to_id: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred = proba.argmax(axis=1)
    summary_rows = []
    for name, mask in {
        "all": np.ones(len(pool), dtype=bool),
        "source_train": pool["source"].eq("train").to_numpy(),
        "source_as1": pool["source"].eq("as1").to_numpy(),
    }.items():
        yy = y[mask]
        pp = pred[mask]
        summary_rows.append(
            {
                "slice": name,
                "n": int(mask.sum()),
                "accuracy": float(accuracy_score(yy, pp)),
                "balanced_accuracy": float(balanced_accuracy_score(yy, pp)),
                "macro_f1": float(f1_score(yy, pp, average="macro")),
                "weighted_f1": float(f1_score(yy, pp, average="weighted")),
                "log_loss": float(
                    log_loss(yy, proba[mask], labels=list(range(len(BIN_LABELS))))
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)

    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=list(range(len(BIN_LABELS))), zero_division=0
    )
    by_class = pd.DataFrame(
        {
            "class": BIN_LABELS,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "prob_mean": [float(proba[:, i].mean()) for i in range(len(BIN_LABELS))],
        }
    )

    auc = pd.DataFrame(binary_auc_rows(y, proba, label_to_id))
    cm = pd.DataFrame(
        confusion_matrix(y, pred, labels=list(range(len(BIN_LABELS)))),
        index=[f"true_{x}" for x in BIN_LABELS],
        columns=[f"pred_{x}" for x in BIN_LABELS],
    )
    return summary, by_class, auc, cm


def metric_row(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y_true
    spearman = stats.spearmanr(y_true, pred).statistic if len(y_true) >= 2 else np.nan
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(err))),
        "bias_pred_minus_true": float(np.mean(err)),
        "spearman": float(spearman),
        "pred_mean": float(np.mean(pred)),
        "true_mean": float(np.mean(y_true)),
    }


def load_anchor_as1(pool: pd.DataFrame) -> pd.DataFrame:
    risk = pd.read_csv(RISK_MAP_PATH)[
        ["compound_id", "molecule_name", "as1_pec50", "pred_id55"]
    ]
    as1 = pool[pool["source"].eq("as1")][
        ["pool_idx", "compound_id", "molecule_name", "pec50", "true_bin"]
    ].copy()
    merged = as1.merge(risk, on=["compound_id", "molecule_name"], how="left")
    if merged["pred_id55"].isna().any():
        raise ValueError("Missing id55 predictions for AS1 rows.")
    return merged


def scan_gates(
    pool: pd.DataFrame,
    y_class: np.ndarray,
    label_to_id: dict[str, int],
    oof_proba: np.ndarray,
    final_test_proba: np.ndarray,
    test: pd.DataFrame,
) -> pd.DataFrame:
    phase2_oof = pd.read_csv(PHASE2_OOF_PATH)[
        ["pool_idx", "phase2_oof_pred", "pec50", "source", "true_bin"]
    ].sort_values("pool_idx")
    phase2_y = phase2_oof["pec50"].to_numpy(dtype=np.float64)
    phase2_base = phase2_oof["phase2_oof_pred"].to_numpy(dtype=np.float64)
    as1_anchor = load_anchor_as1(pool)
    as1_idx = as1_anchor["pool_idx"].to_numpy(dtype=np.int64)
    as1_y = as1_anchor["pec50"].to_numpy(dtype=np.float64)
    as1_base = as1_anchor["pred_id55"].to_numpy(dtype=np.float64)

    low_cls = label_to_id[LOW_LABEL]
    high_cls = label_to_id[HIGH_LABEL]
    low_p = oof_proba[:, low_cls]
    high_p = oof_proba[:, high_cls]
    low_p_test = final_test_proba[:, low_cls]
    high_p_test = final_test_proba[:, high_cls]
    as2_mask_test = test["as1_pec50"].isna().to_numpy()

    rows = []
    thresholds = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    low_shifts = [0.0, -0.10, -0.20, -0.30, -0.40, -0.50, -0.70]
    high_shifts = [0.0, 0.10, 0.20, 0.30, 0.40, 0.55, 0.70]
    for low_t in thresholds:
        for high_t in thresholds:
            low_flag = low_p >= low_t
            high_flag = high_p >= high_t
            low_flag_as1 = low_flag[as1_idx]
            high_flag_as1 = high_flag[as1_idx]
            low_flag_test = low_p_test >= low_t
            high_flag_test = high_p_test >= high_t
            for low_shift in low_shifts:
                for high_shift in high_shifts:
                    if low_shift == 0.0 and high_shift == 0.0:
                        continue
                    phase2_pred = (
                        phase2_base
                        + low_flag.astype(float) * low_shift
                        + high_flag.astype(float) * high_shift
                    )
                    as1_pred = (
                        as1_base
                        + low_flag_as1.astype(float) * low_shift
                        + high_flag_as1.astype(float) * high_shift
                    )
                    all_row = metric_row(phase2_y, phase2_pred)
                    as1_row = metric_row(as1_y, as1_pred)
                    lt3_mask = y_class == low_cls
                    gte6_mask = y_class == high_cls
                    as1_lt3 = as1_anchor["true_bin"].eq(LOW_LABEL).to_numpy()
                    as1_gte6 = as1_anchor["true_bin"].eq(HIGH_LABEL).to_numpy()
                    rows.append(
                        {
                            "low_threshold": low_t,
                            "high_threshold": high_t,
                            "low_shift": low_shift,
                            "high_shift": high_shift,
                            "phase2_all_mae": all_row["mae"],
                            "phase2_all_bias": all_row["bias_pred_minus_true"],
                            "phase2_lt3_mae": metric_row(
                                phase2_y[lt3_mask], phase2_pred[lt3_mask]
                            )["mae"],
                            "phase2_gte6_mae": metric_row(
                                phase2_y[gte6_mask], phase2_pred[gte6_mask]
                            )["mae"],
                            "id55_as1_mae": as1_row["mae"],
                            "id55_as1_bias": as1_row["bias_pred_minus_true"],
                            "id55_as1_lt3_mae": metric_row(
                                as1_y[as1_lt3], as1_pred[as1_lt3]
                            )["mae"],
                            "id55_as1_gte6_mae": metric_row(
                                as1_y[as1_gte6], as1_pred[as1_gte6]
                            )["mae"],
                            "phase2_low_flags": int(low_flag.sum()),
                            "phase2_high_flags": int(high_flag.sum()),
                            "as1_low_flags": int(low_flag_as1.sum()),
                            "as1_high_flags": int(high_flag_as1.sum()),
                            "as2_low_flags": int((low_flag_test & as2_mask_test).sum()),
                            "as2_high_flags": int(
                                (high_flag_test & as2_mask_test).sum()
                            ),
                            "as2_mean_abs_shift": float(
                                np.mean(
                                    np.abs(
                                        low_flag_test[as2_mask_test].astype(float)
                                        * low_shift
                                        + high_flag_test[as2_mask_test].astype(float)
                                        * high_shift
                                    )
                                )
                            ),
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["id55_as1_mae", "phase2_all_mae", "as2_mean_abs_shift"]
    )


def scan_continuous_gates(
    pool: pd.DataFrame,
    y_class: np.ndarray,
    label_to_id: dict[str, int],
    oof_proba: np.ndarray,
) -> pd.DataFrame:
    phase2_oof = pd.read_csv(PHASE2_OOF_PATH)[
        ["pool_idx", "phase2_oof_pred", "pec50", "true_bin"]
    ].sort_values("pool_idx")
    phase2_y = phase2_oof["pec50"].to_numpy(dtype=np.float64)
    phase2_base = phase2_oof["phase2_oof_pred"].to_numpy(dtype=np.float64)
    as1_anchor = load_anchor_as1(pool)
    as1_idx = as1_anchor["pool_idx"].to_numpy(dtype=np.int64)
    as1_y = as1_anchor["pec50"].to_numpy(dtype=np.float64)
    as1_base = as1_anchor["pred_id55"].to_numpy(dtype=np.float64)

    low_cls = label_to_id[LOW_LABEL]
    high_cls = label_to_id[HIGH_LABEL]
    low_p = oof_proba[:, low_cls]
    high_p = oof_proba[:, high_cls]
    low_p_as1 = low_p[as1_idx]
    high_p_as1 = high_p[as1_idx]
    lt3_mask = y_class == low_cls
    gte6_mask = y_class == high_cls
    as1_lt3 = as1_anchor["true_bin"].eq(LOW_LABEL).to_numpy()
    as1_gte6 = as1_anchor["true_bin"].eq(HIGH_LABEL).to_numpy()

    rows = []
    for beta_low in np.round(np.linspace(-0.6, 0.1, 15), 4):
        for beta_high in np.round(np.linspace(-0.1, 0.7, 17), 4):
            phase2_pred = phase2_base + beta_low * low_p + beta_high * high_p
            as1_pred = as1_base + beta_low * low_p_as1 + beta_high * high_p_as1
            rows.append(
                {
                    "beta_low": beta_low,
                    "beta_high": beta_high,
                    "phase2_all_mae": metric_row(phase2_y, phase2_pred)["mae"],
                    "phase2_lt3_mae": metric_row(
                        phase2_y[lt3_mask], phase2_pred[lt3_mask]
                    )["mae"],
                    "phase2_gte6_mae": metric_row(
                        phase2_y[gte6_mask], phase2_pred[gte6_mask]
                    )["mae"],
                    "id55_as1_mae": metric_row(as1_y, as1_pred)["mae"],
                    "id55_as1_bias": metric_row(as1_y, as1_pred)[
                        "bias_pred_minus_true"
                    ],
                    "id55_as1_lt3_mae": metric_row(as1_y[as1_lt3], as1_pred[as1_lt3])[
                        "mae"
                    ],
                    "id55_as1_gte6_mae": metric_row(
                        as1_y[as1_gte6], as1_pred[as1_gte6]
                    )["mae"],
                }
            )
    return pd.DataFrame(rows).sort_values(["id55_as1_mae", "phase2_all_mae"])


def config_name(args: argparse.Namespace) -> str:
    parts = [args.model, args.feature]
    if args.use_htchem:
        parts.append("pred_htchem")
    if args.external_rank_features != "none":
        parts.append(f"external_{args.external_rank_features}")
    if args.force_last_n_features:
        parts.append(f"force_last{args.force_last_n_features}")
    if args.top_k > 0:
        parts.append(f"top{args.top_k}")
    if args.model == "tabpfn":
        parts.append(args.tabpfn_version)
        parts.append(f"ne{args.n_estimators}")
        parts.append(f"t{args.softmax_temperature:g}".replace(".", "p"))
        if args.balance_probabilities:
            parts.append("balanced")
    return "_".join(parts)


def write_report(
    out_dir: Path,
    name: str,
    args: argparse.Namespace,
    clf_summary: pd.DataFrame,
    by_class: pd.DataFrame,
    auc: pd.DataFrame,
    gate_scan: pd.DataFrame,
    continuous_gate_scan: pd.DataFrame,
    selected: np.ndarray,
) -> None:
    top_gate = gate_scan.head(12)
    top_continuous_as1 = continuous_gate_scan.head(8)
    top_continuous_phase2 = continuous_gate_scan.sort_values(
        ["phase2_all_mae", "id55_as1_mae"]
    ).head(8)
    lines = [
        "# Phase 2 classifier gate probe",
        "",
        "Experiment-only multiclass activity-bin classifier using the train+AS1",
        "Phase 2 folds. The objective is to test whether class probabilities can",
        "serve as soft low/high-tail gates, not to replace the submitted model.",
        "",
        "## Latest run",
        "",
        f"- Config: `{name}`",
        f"- Model: `{args.model}`",
        f"- Feature: `{args.feature}`",
        f"- HTChem scalar appended: `{args.use_htchem}`",
        f"- External rank features: `{args.external_rank_features}`",
        f"- Forced last-N features: `{args.force_last_n_features}`",
        f"- Top-K selection: `{args.top_k}`",
        f"- Final selected feature count: {len(selected)}",
        "",
        "## Classifier OOF summary",
        "",
        clf_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Class metrics",
        "",
        by_class.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Tail binary AUC",
        "",
        auc.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Best gate sweeps by id55 AS1 replay",
        "",
        top_gate.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Best continuous probability gates",
        "",
        "Sorted by id55 AS1 replay:",
        "",
        top_continuous_as1.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Sorted by Phase2 OOF:",
        "",
        top_continuous_phase2.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Generated files",
        "",
        f"- `{out_dir.relative_to(REPO_ROOT) / 'oof_class_probabilities.csv'}`",
        f"- `{out_dir.relative_to(REPO_ROOT) / 'test_class_probabilities.csv'}`",
        f"- `{out_dir.relative_to(REPO_ROOT) / 'gate_scan.csv'}`",
        f"- `{out_dir.relative_to(REPO_ROOT) / 'continuous_gate_scan.csv'}`",
        f"- `{out_dir.relative_to(REPO_ROOT) / 'classifier_summary.csv'}`",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature",
        default="cheme_2d_full_boltz_log2fc_pred_seed10ens",
    )
    parser.add_argument("--model", choices=["lgbm", "tabpfn"], default="lgbm")
    parser.add_argument("--use-htchem", action="store_true")
    parser.add_argument(
        "--external-rank-features",
        choices=["none", "chembl", "htchem", "chembl_htchem", "pairrank", "pairrank_all"],
        default="none",
    )
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--force-last-n-features", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lgbm-estimators", type=int, default=900)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument(
        "--tabpfn-version", choices=["v3", "v2_6", "v2_5", "v2"], default="v3"
    )
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--softmax-temperature", type=float, default=0.9)
    parser.add_argument("--average-before-softmax", action="store_true")
    parser.add_argument("--balance-probabilities", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    name = config_name(args)
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = load_pool_with_folds()
    test = load_test_frame()
    y, label_to_id, _id_to_label = label_arrays(pool)
    x_pool, x_test = build_feature_matrices(
        args.feature,
        pool,
        test,
        use_htchem=args.use_htchem,
        external_rank_features=args.external_rank_features,
    )
    print(
        f"config={name} pool={x_pool.shape} test={x_test.shape} "
        f"classes={dict(zip(*np.unique(y, return_counts=True)))}"
    )

    oof_proba, test_proba_cv, fold_df = run_oof(args, pool, x_pool, x_test, y)
    test_proba_final, selected = train_final_test_proba(args, x_pool, x_test, y)
    clf_summary, by_class, auc, cm = summarize_classifier(
        pool, y, oof_proba, label_to_id
    )
    gate_scan = scan_gates(pool, y, label_to_id, oof_proba, test_proba_final, test)
    continuous_gate_scan = scan_continuous_gates(pool, y, label_to_id, oof_proba)

    proba_cols = [f"p_{label}" for label in BIN_LABELS]
    oof_df = pool[
        [
            "pool_idx",
            "compound_id",
            "molecule_name",
            "pec50",
            "source",
            "true_bin",
            "fold",
        ]
    ].copy()
    for idx, col in enumerate(proba_cols):
        oof_df[col] = oof_proba[:, idx]
    oof_df["pred_bin"] = [BIN_LABELS[i] for i in oof_proba.argmax(axis=1)]
    test_df = test[["test_id", "compound_id", "molecule_name", "as1_pec50"]].copy()
    test_df["split"] = np.where(test_df["as1_pec50"].notna(), "AS1", "AS2")
    for idx, col in enumerate(proba_cols):
        test_df[f"cv_{col}"] = test_proba_cv[:, idx]
        test_df[f"final_{col}"] = test_proba_final[:, idx]
    test_df["final_pred_bin"] = [BIN_LABELS[i] for i in test_proba_final.argmax(axis=1)]

    oof_df.to_csv(out_dir / "oof_class_probabilities.csv", index=False)
    test_df.to_csv(out_dir / "test_class_probabilities.csv", index=False)
    fold_df.to_csv(out_dir / "fold_summary.csv", index=False)
    clf_summary.to_csv(out_dir / "classifier_summary.csv", index=False)
    by_class.to_csv(out_dir / "class_metrics.csv", index=False)
    auc.to_csv(out_dir / "tail_binary_auc.csv", index=False)
    cm.to_csv(out_dir / "confusion_matrix.csv")
    gate_scan.to_csv(out_dir / "gate_scan.csv", index=False)
    continuous_gate_scan.to_csv(out_dir / "continuous_gate_scan.csv", index=False)
    pd.Series(selected).to_csv(
        out_dir / "final_selected_feature_indices.csv", index=False
    )
    (out_dir / "metadata.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(
        out_dir,
        name,
        args,
        clf_summary,
        by_class,
        auc,
        gate_scan,
        continuous_gate_scan,
        selected,
    )

    print("\nClassifier summary")
    print(clf_summary.to_string(index=False))
    print("\nClass metrics")
    print(by_class.to_string(index=False))
    print("\nBest gate rows")
    print(gate_scan.head(10).to_string(index=False))
    print("\nBest continuous gate rows")
    print(continuous_gate_scan.head(10).to_string(index=False))
    print(f"\nWrote {out_dir}")
    print(f"Wrote {DOC_PATH}")


if __name__ == "__main__":
    main()
