#!/usr/bin/env -S pixi run python
"""Probe binary tail classifiers for Phase 2 gates.

The first use case is a high-activity classifier: ``gte6`` versus all other
activity bins. This keeps model capacity focused on the rare high tail instead
of asking one classifier to also separate the middle bins.
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
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_multiclass_gate_probe import (  # noqa: E402
    BIN_LABELS,
    OUT_ROOT,
    PHASE2_OOF_PATH,
    RISK_MAP_PATH,
    build_feature_matrices,
    load_pool_with_folds,
    load_test_frame,
)

DOC_PATH = REPO_ROOT / "docs" / "track1_explain" / "phase2_binary_tail_gate.md"


def target_array(pool: pd.DataFrame, positive_bin: str) -> np.ndarray:
    if positive_bin not in BIN_LABELS:
        raise ValueError(f"Unknown bin: {positive_bin}")
    return pool["true_bin"].eq(positive_bin).to_numpy(dtype=np.int64)


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
        objective="binary",
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
            objective="binary",
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


def positive_proba(model, x: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(x)
    classes = getattr(model, "classes_", np.arange(raw.shape[1]))
    out = np.zeros(x.shape[0], dtype=np.float64)
    for src_idx, cls in enumerate(classes):
        if int(cls) == 1:
            out = raw[:, src_idx].astype(np.float64)
            break
    return np.clip(out, 0.0, 1.0)


def run_oof(
    args: argparse.Namespace,
    pool: pd.DataFrame,
    x_pool: np.ndarray,
    x_test: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    oof = np.zeros(len(pool), dtype=np.float64)
    test_fold_scores = []
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
        oof[val_idx] = positive_proba(clf, x_pool[val_idx][:, selected])
        test_fold_scores.append(positive_proba(clf, x_test[:, selected]))
        pred = oof[val_idx] >= args.default_threshold
        rows.append(
            {
                "fold": int(fold),
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_pos_val": int(y[val_idx].sum()),
                "n_features": int(len(selected)),
                "val_ap": float(average_precision_score(y[val_idx], oof[val_idx])),
                "val_auc": float(roc_auc_score(y[val_idx], oof[val_idx])),
                "val_bal_acc": float(balanced_accuracy_score(y[val_idx], pred)),
            }
        )
        print(
            f"fold={int(fold)} n_pos={int(y[val_idx].sum())} "
            f"features={len(selected)} ap={rows[-1]['val_ap']:.4f} "
            f"auc={rows[-1]['val_auc']:.4f}"
        )

    return oof, np.mean(np.stack(test_fold_scores), axis=0), pd.DataFrame(rows)


def train_final_test_score(
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
    return positive_proba(clf, x_test[:, selected]), selected


def summarize_classifier(
    pool: pd.DataFrame,
    y: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    eps = 1e-7
    proba = np.column_stack([1.0 - score, score])
    proba = np.clip(proba, eps, 1.0 - eps)
    proba = proba / proba.sum(axis=1, keepdims=True)
    pred = (score >= threshold).astype(np.int64)

    for name, mask in {
        "all": np.ones(len(pool), dtype=bool),
        "source_train": pool["source"].eq("train").to_numpy(),
        "source_as1": pool["source"].eq("as1").to_numpy(),
    }.items():
        yy = y[mask]
        pp = pred[mask]
        ss = score[mask]
        rows.append(
            {
                "slice": name,
                "n": int(mask.sum()),
                "n_pos": int(yy.sum()),
                "accuracy": float(accuracy_score(yy, pp)),
                "balanced_accuracy": float(balanced_accuracy_score(yy, pp)),
                "f1": float(f1_score(yy, pp, zero_division=0)),
                "log_loss": float(log_loss(yy, proba[mask], labels=[0, 1])),
                "roc_auc": float(roc_auc_score(yy, ss)),
                "average_precision": float(average_precision_score(yy, ss)),
                "score_mean": float(ss.mean()),
            }
        )

    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=[0, 1], zero_division=0
    )
    by_class = pd.DataFrame(
        {
            "class": ["not_positive", "positive"],
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    return pd.DataFrame(rows), by_class


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


def scan_high_gate(
    pool: pd.DataFrame,
    y_bin: np.ndarray,
    positive_bin: str,
    oof_score: np.ndarray,
    final_test_score: np.ndarray,
    test: pd.DataFrame,
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
    pos_mask = y_bin.astype(bool)
    as1_pos = as1_anchor["true_bin"].eq(positive_bin).to_numpy()
    as2_mask_test = test["as1_pec50"].isna().to_numpy()

    thresholds = [
        0.02,
        0.05,
        0.08,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.45,
        0.55,
        0.65,
        0.75,
        0.85,
        0.95,
    ]
    shifts = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70]
    rows = []
    for threshold in thresholds:
        flag = oof_score >= threshold
        flag_as1 = flag[as1_idx]
        flag_test = final_test_score >= threshold
        for shift in shifts:
            phase2_pred = phase2_base + flag.astype(float) * shift
            as1_pred = as1_base + flag_as1.astype(float) * shift
            rows.append(
                {
                    "threshold": threshold,
                    "high_shift": shift,
                    "phase2_all_mae": metric_row(phase2_y, phase2_pred)["mae"],
                    "phase2_all_bias": metric_row(phase2_y, phase2_pred)[
                        "bias_pred_minus_true"
                    ],
                    "phase2_pos_mae": metric_row(
                        phase2_y[pos_mask], phase2_pred[pos_mask]
                    )["mae"],
                    "id55_as1_mae": metric_row(as1_y, as1_pred)["mae"],
                    "id55_as1_bias": metric_row(as1_y, as1_pred)[
                        "bias_pred_minus_true"
                    ],
                    "id55_as1_pos_mae": metric_row(as1_y[as1_pos], as1_pred[as1_pos])[
                        "mae"
                    ],
                    "phase2_flags": int(flag.sum()),
                    "phase2_pos_flags": int((flag & pos_mask).sum()),
                    "as1_flags": int(flag_as1.sum()),
                    "as1_pos_flags": int((flag_as1 & as1_pos).sum()),
                    "as2_flags": int((flag_test & as2_mask_test).sum()),
                    "as2_mean_abs_shift": float(
                        np.mean(np.abs(flag_test[as2_mask_test].astype(float) * shift))
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["id55_as1_mae", "phase2_all_mae", "as2_mean_abs_shift"]
    )


def config_name(args: argparse.Namespace) -> str:
    parts = ["binary", args.positive_bin, args.model, args.feature]
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
    summary: pd.DataFrame,
    by_class: pd.DataFrame,
    fold_df: pd.DataFrame,
    gate_scan: pd.DataFrame,
    selected: np.ndarray,
) -> None:
    lines = [
        "# Phase 2 binary tail gate probe",
        "",
        f"- Config: `{name}`",
        f"- Positive bin: `{args.positive_bin}`",
        f"- Model: `{args.model}`",
        f"- Feature: `{args.feature}`",
        f"- HTChem scalar appended: `{args.use_htchem}`",
        f"- External rank features: `{args.external_rank_features}`",
        f"- Top-K selection: `{args.top_k}`",
        f"- Final selected feature count: {len(selected)}",
        "",
        "## Fold summary",
        "",
        fold_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## OOF summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Threshold class metrics",
        "",
        by_class.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Best high-shift gates by id55 AS1 replay",
        "",
        gate_scan.head(16).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Generated files",
        "",
        f"- `{out_dir.relative_to(REPO_ROOT) / 'oof_binary_scores.csv'}`",
        f"- `{out_dir.relative_to(REPO_ROOT) / 'test_binary_scores.csv'}`",
        f"- `{out_dir.relative_to(REPO_ROOT) / 'gate_scan.csv'}`",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-bin", choices=BIN_LABELS, default="gte6")
    parser.add_argument(
        "--feature",
        default="cheme_2d_full_boltz_log2fc_pred_seed10ens",
    )
    parser.add_argument("--model", choices=["lgbm", "tabpfn"], default="tabpfn")
    parser.add_argument("--use-htchem", action="store_true")
    parser.add_argument(
        "--external-rank-features",
        choices=["none", "chembl", "htchem", "chembl_htchem", "pairrank", "pairrank_all"],
        default="none",
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--force-last-n-features", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--default-threshold", type=float, default=0.5)
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
    y = target_array(pool, args.positive_bin)
    x_pool, x_test = build_feature_matrices(
        args.feature,
        pool,
        test,
        use_htchem=args.use_htchem,
        external_rank_features=args.external_rank_features,
    )
    print(
        f"config={name} pool={x_pool.shape} test={x_test.shape} "
        f"positive={args.positive_bin} n_pos={int(y.sum())}"
    )

    oof_score, test_score_cv, fold_df = run_oof(args, pool, x_pool, x_test, y)
    test_score_final, selected = train_final_test_score(args, x_pool, x_test, y)
    summary, by_class = summarize_classifier(pool, y, oof_score, args.default_threshold)
    gate_scan = scan_high_gate(
        pool, y, args.positive_bin, oof_score, test_score_final, test
    )

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
    oof_df["target"] = y
    oof_df["score"] = oof_score
    oof_df["pred_default"] = oof_score >= args.default_threshold

    test_df = test[["test_id", "compound_id", "molecule_name", "as1_pec50"]].copy()
    test_df["split"] = np.where(test_df["as1_pec50"].notna(), "AS1", "AS2")
    test_df["cv_score"] = test_score_cv
    test_df["final_score"] = test_score_final

    oof_df.to_csv(out_dir / "oof_binary_scores.csv", index=False)
    test_df.to_csv(out_dir / "test_binary_scores.csv", index=False)
    fold_df.to_csv(out_dir / "fold_summary.csv", index=False)
    summary.to_csv(out_dir / "classifier_summary.csv", index=False)
    by_class.to_csv(out_dir / "threshold_class_metrics.csv", index=False)
    gate_scan.to_csv(out_dir / "gate_scan.csv", index=False)
    pd.Series(selected).to_csv(
        out_dir / "final_selected_feature_indices.csv", index=False
    )
    (out_dir / "metadata.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(out_dir, name, args, summary, by_class, fold_df, gate_scan, selected)

    print("\nClassifier summary")
    print(summary.to_string(index=False))
    print("\nThreshold class metrics")
    print(by_class.to_string(index=False))
    print("\nBest gate rows")
    print(gate_scan.head(12).to_string(index=False))
    print(f"\nWrote {out_dir}")
    print(f"Wrote {DOC_PATH}")


if __name__ == "__main__":
    main()
