#!/usr/bin/env -S pixi run python
"""Build high-tail classifiers from features correlated with ``gte6``.

This intentionally separates two questions:

1. Which broad feature families contain columns that rank high-activity
   compounds?
2. If we build a CV-safe panel from those columns, does a binary classifier
   improve high-tail gates?
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
from scipy.stats import rankdata
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_binary_tail_gate_probe import (  # noqa: E402
    make_classifier,
    positive_proba,
    scan_high_gate,
)
from run_multiclass_gate_probe import (  # noqa: E402
    BIN_LABELS,
    OUT_ROOT,
    append_pred_htchem,
    build_feature_matrices,
    load_pool_with_folds,
    load_test_frame,
)

DOC_PATH = REPO_ROOT / "docs" / "track1_explain" / "phase2_high_corr_panel.md"

DEFAULT_FAMILIES = [
    "cheme_2d_full_boltz_log2fc_pred_seed10ens",
    "chemprop_pretrain_embed",
    "chemprop_log2fc_htchem_pretrain_embed",
    "chemprop_assay_shape_drlatent_embed",
    "chemprop_counter_emax_embed",
    "chemprop_drlatent_embed",
    "chemprop_pretrain_optuna_trial10_embed",
    "molformer_c3_pretrain_embed",
    "chemberta_5m_mtr_pretrain_embed",
    "chemprop_mtr_embed",
    "molformer_c3_mtr_embed",
    "kermt_pretrain_embed",
    "attentivefp_pretrain_embed",
    "gatedgcn_pretrain_embed",
    "ka_gnn_pretrain_embed",
    "unimol_v2_pretrain_embed",
    "unimol_v2_log2fc_real_embed",
]


@dataclass(frozen=True)
class FamilyMatrix:
    name: str
    x_pool: np.ndarray
    x_test: np.ndarray
    col_names: list[str]


def sanitize_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    col_mean = np.nanmean(x, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0).astype(np.float32)
    return np.where(np.isfinite(x), x, col_mean).astype(np.float32)


def load_family(
    feature: str,
    pool: pd.DataFrame,
    test: pd.DataFrame,
    include_htchem_scalar: bool,
) -> FamilyMatrix:
    x_pool, x_test = build_feature_matrices(feature, pool, test, use_htchem=False)
    col_names = [f"{feature}__f{i:04d}" for i in range(x_pool.shape[1])]
    if include_htchem_scalar and feature == "cheme_2d_full_boltz_log2fc_pred_seed10ens":
        x_pool, x_test = append_pred_htchem(x_pool, x_test, pool, test)
        col_names.append("pred_htchem")
    return FamilyMatrix(
        name=feature,
        x_pool=sanitize_matrix(x_pool),
        x_test=sanitize_matrix(x_test),
        col_names=col_names,
    )


def auc_vector(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = y.astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return np.full(x.shape[1], np.nan, dtype=np.float64)
    ranks = rankdata(x, axis=0, method="average")
    pos_rank_sum = ranks[y].sum(axis=0)
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return np.asarray(auc, dtype=np.float64)


def top_precision(score: np.ndarray, y: np.ndarray, k: int) -> float:
    k = min(k, len(score))
    if k <= 0:
        return np.nan
    idx = np.argpartition(-score, k - 1)[:k]
    return float(y[idx].mean())


def screen_family(
    family: FamilyMatrix,
    y: np.ndarray,
    top_n: int,
    train_idx: np.ndarray | None = None,
) -> pd.DataFrame:
    rows = []
    x = family.x_pool if train_idx is None else family.x_pool[train_idx]
    yy = y if train_idx is None else y[train_idx]
    auc = auc_vector(x, yy)
    oriented_auc = np.maximum(auc, 1.0 - auc)
    signs = np.where(auc >= 0.5, 1.0, -1.0)
    top_candidates = np.argsort(-oriented_auc)[: min(max(top_n * 4, top_n), x.shape[1])]

    for col_idx in top_candidates:
        raw_score = x[:, col_idx].astype(np.float64)
        score = raw_score * signs[col_idx]
        if np.nanstd(score) <= 0:
            continue
        rows.append(
            {
                "family": family.name,
                "feature_idx": int(col_idx),
                "feature_name": family.col_names[col_idx],
                "direction": int(signs[col_idx]),
                "auc": float(auc[col_idx]),
                "oriented_auc": float(oriented_auc[col_idx]),
                "average_precision": float(average_precision_score(yy, score)),
                "top25_precision": top_precision(score, yy, 25),
                "top50_precision": top_precision(score, yy, 50),
                "top100_precision": top_precision(score, yy, 100),
                "mean_pos": float(raw_score[yy.astype(bool)].mean()),
                "mean_neg": float(raw_score[~yy.astype(bool)].mean()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["average_precision", "oriented_auc", "top50_precision"], ascending=False
    ).head(top_n)


def build_panel_from_screen(
    families: list[FamilyMatrix],
    y: np.ndarray,
    per_family: int,
    max_total: int,
    train_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    screen_rows = []
    pool_cols = []
    test_cols = []
    family_lookup = {family.name: family for family in families}

    for family in families:
        screen = screen_family(family, y, per_family, train_idx=train_idx)
        if screen.empty:
            continue
        screen_rows.append(screen)

    selected = pd.concat(screen_rows, ignore_index=True).sort_values(
        ["average_precision", "oriented_auc", "top50_precision"], ascending=False
    )
    selected = selected.head(max_total).reset_index(drop=True)

    for row in selected.itertuples(index=False):
        family = family_lookup[row.family]
        sign = float(row.direction)
        pool_cols.append(family.x_pool[:, int(row.feature_idx)] * sign)
        test_cols.append(family.x_test[:, int(row.feature_idx)] * sign)

    x_pool = np.column_stack(pool_cols).astype(np.float32)
    x_test = np.column_stack(test_cols).astype(np.float32)
    selected.insert(0, "panel_col", np.arange(len(selected), dtype=np.int64))
    return x_pool, x_test, selected


def select_base_features(
    x_pool: np.ndarray,
    pool: pd.DataFrame,
    y_binary: np.ndarray,
    train_idx: np.ndarray,
    top_k: int,
    objective: str,
    seed: int,
) -> np.ndarray:
    if top_k <= 0:
        return np.array([], dtype=np.int64)
    if objective == "binary":
        y_rank = y_binary
        clf = lgb.LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=10,
            random_state=seed,
            verbose=-1,
        )
    else:
        label_to_id = {label: idx for idx, label in enumerate(BIN_LABELS)}
        y_rank = pool["true_bin"].map(label_to_id).to_numpy(dtype=np.int64)
        clf = lgb.LGBMClassifier(
            objective="multiclass",
            class_weight="balanced",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=10,
            random_state=seed,
            verbose=-1,
        )
    clf.fit(x_pool[train_idx], y_rank[train_idx])
    gain = clf.booster_.feature_importance(importance_type="gain")
    return np.argsort(-gain)[: min(top_k, x_pool.shape[1])].astype(np.int64)


def summarize_scores(
    pool: pd.DataFrame,
    y: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = (score >= threshold).astype(np.int64)
    eps = 1e-7
    proba = np.column_stack([1.0 - score, score])
    proba = np.clip(proba, eps, 1.0 - eps)
    proba = proba / proba.sum(axis=1, keepdims=True)
    rows = []
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


def run_oof(
    args: argparse.Namespace,
    pool: pd.DataFrame,
    test: pd.DataFrame,
    families: list[FamilyMatrix],
    y: np.ndarray,
    base_pool: np.ndarray | None,
    base_test: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[pd.DataFrame]]:
    oof = np.zeros(len(pool), dtype=np.float64)
    test_scores = []
    fold_rows = []
    fold_selected = []

    for fold in sorted(pool["fold"].unique()):
        fold = int(fold)
        val_idx = pool.index[pool["fold"].eq(fold)].to_numpy(dtype=np.int64)
        train_idx = pool.index[~pool["fold"].eq(fold)].to_numpy(dtype=np.int64)
        x_pool_panel, x_test_panel, selected = build_panel_from_screen(
            families,
            y,
            per_family=args.per_family,
            max_total=args.panel_size,
            train_idx=train_idx,
        )
        if args.base_top_k > 0:
            if base_pool is None or base_test is None:
                raise RuntimeError("base_top_k requested without base matrices")
            base_selected = select_base_features(
                base_pool,
                pool,
                y,
                train_idx,
                args.base_top_k,
                args.base_ranker_objective,
                args.seed + fold,
            )
            x_pool_panel = np.concatenate(
                [base_pool[:, base_selected], x_pool_panel], axis=1
            ).astype(np.float32)
            x_test_panel = np.concatenate(
                [base_test[:, base_selected], x_test_panel], axis=1
            ).astype(np.float32)
            base_rows = pd.DataFrame(
                {
                    "panel_col": np.arange(len(base_selected), dtype=np.int64),
                    "family": args.base_feature,
                    "feature_idx": base_selected,
                    "feature_name": [
                        f"{args.base_feature}__topk_f{idx:04d}" for idx in base_selected
                    ],
                    "direction": 1,
                    "auc": np.nan,
                    "oriented_auc": np.nan,
                    "average_precision": np.nan,
                    "top25_precision": np.nan,
                    "top50_precision": np.nan,
                    "top100_precision": np.nan,
                    "mean_pos": np.nan,
                    "mean_neg": np.nan,
                }
            )
            selected = selected.copy()
            selected["panel_col"] += len(base_selected)
            selected = pd.concat([base_rows, selected], ignore_index=True)
        clf = make_classifier(args, args.seed + fold)
        clf.fit(x_pool_panel[train_idx], y[train_idx])
        oof[val_idx] = positive_proba(clf, x_pool_panel[val_idx])
        test_scores.append(positive_proba(clf, x_test_panel))
        fold_selected.append(selected.assign(fold=fold))
        fold_rows.append(
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_pos_val": int(y[val_idx].sum()),
                "panel_size": int(x_pool_panel.shape[1]),
                "val_auc": float(roc_auc_score(y[val_idx], oof[val_idx])),
                "val_ap": float(average_precision_score(y[val_idx], oof[val_idx])),
            }
        )
        print(
            f"fold={fold} panel={x_pool_panel.shape[1]} "
            f"ap={fold_rows[-1]['val_ap']:.4f} auc={fold_rows[-1]['val_auc']:.4f}"
        )

    return (
        oof,
        np.mean(np.stack(test_scores), axis=0),
        pd.DataFrame(fold_rows),
        fold_selected,
    )


def train_final_score(
    args: argparse.Namespace,
    families: list[FamilyMatrix],
    y: np.ndarray,
    pool: pd.DataFrame,
    base_pool: np.ndarray | None,
    base_test: np.ndarray | None,
) -> tuple[np.ndarray, pd.DataFrame]:
    x_pool_panel, x_test_panel, selected = build_panel_from_screen(
        families,
        y,
        per_family=args.per_family,
        max_total=args.panel_size,
        train_idx=None,
    )
    if args.base_top_k > 0:
        if base_pool is None or base_test is None:
            raise RuntimeError("base_top_k requested without base matrices")
        base_selected = select_base_features(
            base_pool,
            pool,
            y,
            np.arange(len(pool), dtype=np.int64),
            args.base_top_k,
            args.base_ranker_objective,
            args.seed,
        )
        x_pool_panel = np.concatenate(
            [base_pool[:, base_selected], x_pool_panel], axis=1
        ).astype(np.float32)
        x_test_panel = np.concatenate(
            [base_test[:, base_selected], x_test_panel], axis=1
        ).astype(np.float32)
        base_rows = pd.DataFrame(
            {
                "panel_col": np.arange(len(base_selected), dtype=np.int64),
                "family": args.base_feature,
                "feature_idx": base_selected,
                "feature_name": [
                    f"{args.base_feature}__topk_f{idx:04d}" for idx in base_selected
                ],
                "direction": 1,
                "auc": np.nan,
                "oriented_auc": np.nan,
                "average_precision": np.nan,
                "top25_precision": np.nan,
                "top50_precision": np.nan,
                "top100_precision": np.nan,
                "mean_pos": np.nan,
                "mean_neg": np.nan,
            }
        )
        selected = selected.copy()
        selected["panel_col"] += len(base_selected)
        selected = pd.concat([base_rows, selected], ignore_index=True)
    clf = make_classifier(args, args.seed)
    clf.fit(x_pool_panel, y)
    return positive_proba(clf, x_test_panel), selected


def config_name(args: argparse.Namespace) -> str:
    parts = [
        "high_corr_panel",
        args.model,
        f"pf{args.per_family}",
        f"p{args.panel_size}",
    ]
    if args.base_top_k > 0:
        parts.append(f"{args.base_ranker_objective}top{args.base_top_k}")
    if args.model == "tabpfn":
        parts.extend(
            [
                args.tabpfn_version,
                f"ne{args.n_estimators}",
                f"t{args.softmax_temperature:g}".replace(".", "p"),
            ]
        )
        if args.balance_probabilities:
            parts.append("balanced")
    return "_".join(parts)


def write_report(
    out_dir: Path,
    name: str,
    args: argparse.Namespace,
    global_screen: pd.DataFrame,
    family_summary: pd.DataFrame,
    clf_summary: pd.DataFrame,
    by_class: pd.DataFrame,
    fold_df: pd.DataFrame,
    gate_scan: pd.DataFrame,
) -> None:
    lines = [
        "# Phase 2 high-correlation feature panel",
        "",
        "Experiment-only probe: screen broad feature families for columns that",
        "rank `gte6` compounds, then train a CV-safe binary classifier on the",
        "selected high-correlation panel.",
        "",
        "## Latest Run",
        "",
        f"- Config: `{name}`",
        f"- Model: `{args.model}`",
        f"- Per-family screen count: {args.per_family}",
        f"- Max panel size: {args.panel_size}",
        f"- Base topK feature: `{args.base_feature}`",
        f"- Base topK count: {args.base_top_k}",
        f"- Base topK ranker objective: `{args.base_ranker_objective}`",
        "",
        "## Family Screen Summary",
        "",
        family_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Top Global Features",
        "",
        global_screen.head(30).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Fold Summary",
        "",
        fold_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Classifier Summary",
        "",
        clf_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Threshold Class Metrics",
        "",
        by_class.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Best High-Shift Gates",
        "",
        gate_scan.head(12).to_markdown(index=False, floatfmt=".4f"),
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="*", default=DEFAULT_FAMILIES)
    parser.add_argument("--per-family", type=int, default=20)
    parser.add_argument("--panel-size", type=int, default=250)
    parser.add_argument("--include-htchem-scalar", action="store_true")
    parser.add_argument(
        "--base-feature",
        default="cheme_2d_full_boltz_log2fc_pred_seed10ens",
    )
    parser.add_argument("--base-top-k", type=int, default=0)
    parser.add_argument(
        "--base-ranker-objective",
        choices=["binary", "multiclass"],
        default="binary",
    )
    parser.add_argument("--model", choices=["lgbm", "tabpfn"], default="tabpfn")
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
    y = pool["true_bin"].eq("gte6").to_numpy(dtype=np.int64)
    print(f"config={name} n_pool={len(pool)} n_pos={int(y.sum())}")

    families = []
    for feature in args.families:
        print(f"\nLoading family: {feature}")
        try:
            families.append(
                load_family(feature, pool, test, args.include_htchem_scalar)
            )
        except Exception as exc:
            print(f"  skip {feature}: {exc}")

    if not families:
        raise RuntimeError("No feature families loaded.")

    base_pool = base_test = None
    if args.base_top_k > 0:
        print(f"\nLoading base topK matrix: {args.base_feature}")
        base_pool, base_test = build_feature_matrices(
            args.base_feature,
            pool,
            test,
            use_htchem=args.include_htchem_scalar,
        )

    global_screens = []
    for family in families:
        print(f"Screening family: {family.name} ({family.x_pool.shape[1]} cols)")
        global_screens.append(
            screen_family(family, y, top_n=max(args.per_family * 3, 30))
        )
    global_screen = pd.concat(global_screens, ignore_index=True).sort_values(
        ["average_precision", "oriented_auc", "top50_precision"], ascending=False
    )
    family_summary = (
        global_screen.groupby("family", as_index=False)
        .agg(
            n_screened=("feature_idx", "count"),
            best_ap=("average_precision", "max"),
            best_auc=("oriented_auc", "max"),
            best_top50_precision=("top50_precision", "max"),
            median_top_feature_ap=("average_precision", "median"),
        )
        .sort_values(["best_ap", "best_auc"], ascending=False)
    )

    oof_score, test_score_cv, fold_df, fold_selected = run_oof(
        args, pool, test, families, y, base_pool, base_test
    )
    final_test_score, final_selected = train_final_score(
        args, families, y, pool, base_pool, base_test
    )
    clf_summary, by_class = summarize_scores(pool, y, oof_score, args.default_threshold)
    gate_scan = scan_high_gate(pool, y, "gte6", oof_score, final_test_score, test)

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
    oof_df["target_gte6"] = y
    oof_df["score"] = oof_score
    test_df = test[["test_id", "compound_id", "molecule_name", "as1_pec50"]].copy()
    test_df["split"] = np.where(test_df["as1_pec50"].notna(), "AS1", "AS2")
    test_df["cv_score"] = test_score_cv
    test_df["final_score"] = final_test_score

    global_screen.to_csv(out_dir / "global_feature_screen.csv", index=False)
    family_summary.to_csv(out_dir / "family_screen_summary.csv", index=False)
    pd.concat(fold_selected, ignore_index=True).to_csv(
        out_dir / "fold_selected_features.csv", index=False
    )
    final_selected.to_csv(out_dir / "final_selected_features.csv", index=False)
    fold_df.to_csv(out_dir / "fold_summary.csv", index=False)
    clf_summary.to_csv(out_dir / "classifier_summary.csv", index=False)
    by_class.to_csv(out_dir / "threshold_class_metrics.csv", index=False)
    gate_scan.to_csv(out_dir / "gate_scan.csv", index=False)
    oof_df.to_csv(out_dir / "oof_binary_scores.csv", index=False)
    test_df.to_csv(out_dir / "test_binary_scores.csv", index=False)
    (out_dir / "metadata.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(
        out_dir,
        name,
        args,
        global_screen,
        family_summary,
        clf_summary,
        by_class,
        fold_df,
        gate_scan,
    )

    print("\nFamily screen summary")
    print(family_summary.to_string(index=False))
    print("\nClassifier summary")
    print(clf_summary.to_string(index=False))
    print("\nBest gate rows")
    print(gate_scan.head(10).to_string(index=False))
    print(f"\nWrote {out_dir}")
    print(f"Wrote {DOC_PATH}")


if __name__ == "__main__":
    main()
