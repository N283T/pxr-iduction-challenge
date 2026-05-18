#!/usr/bin/env python
"""Full-train top-500 gain audit and TabPFN SHAP diagnostic.

This is an explanatory diagnostic, not the exact fold-local OOF procedure.
It fits one LGBM selector on all Track 1 train rows, selects top-500 features,
fits one TabPFN v2.6 model on those selected features, then computes first-order
imputation SHAP values for representative test compounds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load_test_smiles, load_train_smiles_target  # noqa: E402

import run_train  # noqa: E402
from top500_raw_feature_audit import build_feature_names, family_of  # noqa: E402


FEATURE = "cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens"
SUBMISSION = (
    "track1_activity/submissions/"
    "tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_top500_umap.csv"
)
OUT_DIR = REPO_ROOT.joinpath(
    "track1_activity",
    "analysis",
    "tabpfn_shape_diagnostic",
    "outputs",
    "optuna_trial10_top500_fulltrain_gain_tabpfn_shap",
)
DOC_ASSET_DIR = REPO_ROOT.joinpath(
    "docs",
    "track1_explain",
    "models",
    "assets",
    "optuna_trial10_top500",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", default=FEATURE)
    parser.add_argument("--K", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--softmax-temperature", type=float, default=0.9)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--n-explain", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--doc-asset-dir", type=Path, default=DOC_ASSET_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sanitize_train_test(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    col_mean = np.nanmean(x_train, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    return (
        np.where(np.isfinite(x_train), x_train, col_mean).astype(np.float32),
        np.where(np.isfinite(x_test), x_test, col_mean).astype(np.float32),
    )


def select_test_examples(n_test: int, n_explain: int) -> np.ndarray:
    sub = pd.read_csv(REPO_ROOT.joinpath(SUBMISSION))
    pred = sub["pEC50"].to_numpy(dtype=np.float64)
    if len(pred) != n_test:
        raise RuntimeError(f"submission length mismatch: {len(pred)} vs {n_test}")
    sorted_idx = np.argsort(pred)
    rank_positions = np.linspace(0, n_test - 1, n_explain, dtype=int)
    return sorted_idx[rank_positions]


def plot_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    path: Path,
    *,
    color: str,
) -> None:
    fig_h = max(4.0, 0.30 * len(df) + 1.2)
    fig, ax = plt.subplots(figsize=(8.0, fig_h))
    ax.barh(df[y_col], df[x_col], color=color)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_tabpfn(args: argparse.Namespace):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    model_path = TabPFNRegressor.create_default_for_version(ModelVersion.V2_6).model_path
    return TabPFNRegressor(
        device=args.device,
        n_estimators=args.n_estimators,
        softmax_temperature=args.softmax_temperature,
        average_before_softmax=False,
        random_state=args.seed,
        model_path=model_path,
        ignore_pretraining_limits=(args.K > 500),
        fit_mode="fit_preprocessors",
        show_progress_bar=False,
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.doc_asset_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    x_train, x_test = run_train.load_features(args.feature, train_df, test_df)
    names = build_feature_names()
    if len(names) != x_train.shape[1]:
        raise RuntimeError(f"feature name mismatch: {len(names)} vs {x_train.shape[1]}")
    x_train, x_test = sanitize_train_test(x_train, x_test)
    y = train_df["pec50"].to_numpy(dtype=np.float32)

    print("[full-train] fitting LGBM selector", flush=True)
    selector = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=10,
        random_state=args.seed,
        verbose=-1,
    )
    selector.fit(x_train, y)
    gain = selector.booster_.feature_importance(importance_type="gain")
    sel = np.argsort(-gain)[: args.K]
    selected = pd.DataFrame(
        {
            "rank": np.arange(1, len(sel) + 1),
            "global_feature_idx": sel,
            "feature": np.asarray(names)[sel],
            "family": [family_of(names[i]) for i in sel],
            "lgbm_gain": gain[sel],
            "gain_share_pct": gain[sel] / gain.sum() * 100.0,
        }
    )
    selected.to_csv(args.out_dir.joinpath("selected_top500_features.csv"), index=False)

    family_gain = (
        selected.groupby("family", as_index=False)
        .agg(n_selected=("feature", "size"), total_gain=("lgbm_gain", "sum"))
        .sort_values("total_gain", ascending=False)
    )
    family_gain["gain_share_pct"] = family_gain["total_gain"] / family_gain["total_gain"].sum()
    family_gain.to_csv(args.out_dir.joinpath("family_gain_summary.csv"), index=False)

    print("[full-train] fitting TabPFN v2.6", flush=True)
    fit_start = time.perf_counter()
    tabpfn = make_tabpfn(args)
    tabpfn.fit(x_train[:, sel], y)
    fit_s = time.perf_counter() - fit_start

    from tabpfn_extensions.interpretability.shapiq import (  # noqa: WPS433
        get_tabpfn_imputation_explainer,
    )

    print("[full-train] initializing imputation SHAP explainer", flush=True)
    explainer_start = time.perf_counter()
    explainer = get_tabpfn_imputation_explainer(
        tabpfn,
        x_train[:, sel],
        index="SV",
        max_order=1,
        imputer="baseline",
        random_state=args.seed,
    )
    explainer_s = time.perf_counter() - explainer_start

    explain_test_idx = select_test_examples(x_test.shape[0], args.n_explain)
    rows = []
    pred_rows = []
    wide_rows = []
    shap_start = time.perf_counter()
    for rank_group, test_idx in enumerate(explain_test_idx):
        cache_path = args.out_dir.joinpath(
            f"fulltrain_test{int(test_idx)}_budget{args.budget}_shap.npy"
        )
        x_one = x_test[test_idx, sel]
        pred = float(tabpfn.predict(x_one.reshape(1, -1))[0])
        if cache_path.exists() and not args.force:
            shap_values = np.load(cache_path)
        else:
            interaction_values = explainer.explain(x_one, budget=args.budget)
            shap_values = np.asarray(
                interaction_values.get_n_order_values(1), dtype=np.float64
            )
            np.save(cache_path, shap_values)
        if shap_values.shape != (args.K,):
            raise RuntimeError(f"unexpected SHAP shape: {shap_values.shape}")
        pred_rows.append(
            {
                "test_rank_group": rank_group,
                "test_index": int(test_idx),
                "molecule_name": test_df.iloc[int(test_idx)]["molecule_name"],
                "prediction": pred,
            }
        )
        wide = {
            "test_rank_group": rank_group,
            "test_index": int(test_idx),
            "molecule_name": test_df.iloc[int(test_idx)]["molecule_name"],
            "prediction": pred,
        }
        wide.update({f"f_{int(i)}": float(v) for i, v in zip(sel, shap_values)})
        wide_rows.append(wide)
        rows.append(
            pd.DataFrame(
                {
                    "test_rank_group": rank_group,
                    "test_index": int(test_idx),
                    "global_feature_idx": sel,
                    "feature": np.asarray(names)[sel],
                    "family": [family_of(names[i]) for i in sel],
                    "shap_value": shap_values,
                    "abs_shap": np.abs(shap_values),
                    "lgbm_gain": gain[sel],
                }
            )
        )
        print(
            f"[full-train] explained test_idx={int(test_idx)} "
            f"rank_group={rank_group}",
            flush=True,
        )

    shap_long = pd.concat(rows, ignore_index=True)
    pred_df = pd.DataFrame(pred_rows)
    wide_df = pd.DataFrame(wide_rows)
    feature_shap = (
        shap_long.groupby(["global_feature_idx", "feature", "family"], as_index=False)
        .agg(
            mean_abs_shap=("abs_shap", "mean"),
            mean_shap=("shap_value", "mean"),
            max_abs_shap=("abs_shap", "max"),
            n_explanations=("abs_shap", "size"),
            lgbm_gain=("lgbm_gain", "first"),
        )
        .sort_values("mean_abs_shap", ascending=False)
    )
    family_shap = (
        feature_shap.groupby("family", as_index=False)
        .agg(
            total_abs_shap=("mean_abs_shap", "sum"),
            mean_abs_shap_per_feature=("mean_abs_shap", "mean"),
            median_abs_shap_per_feature=("mean_abs_shap", "median"),
            mean_signed_shap=("mean_shap", "mean"),
            n_selected=("feature", "size"),
            total_gain=("lgbm_gain", "sum"),
        )
        .sort_values("total_abs_shap", ascending=False)
    )
    family_shap["share_abs_shap"] = (
        family_shap["total_abs_shap"] / family_shap["total_abs_shap"].sum()
    )
    if not np.isfinite(feature_shap["mean_abs_shap"]).all():
        raise RuntimeError("Non-finite SHAP values detected")
    if feature_shap["mean_abs_shap"].max() > 10.0:
        raise RuntimeError(
            "SHAP values look numerically unstable; inspect budget/approximator."
        )

    shap_long.to_csv(args.out_dir.joinpath("tabpfn_shap_long.csv"), index=False)
    pred_df.to_csv(args.out_dir.joinpath("explained_test_predictions.csv"), index=False)
    wide_df.to_csv(args.out_dir.joinpath("test_shap_values_wide.csv"), index=False)
    feature_shap.to_csv(args.out_dir.joinpath("tabpfn_shap_feature_summary.csv"), index=False)
    family_shap.to_csv(args.out_dir.joinpath("tabpfn_shap_family_summary.csv"), index=False)

    plot_bar(
        family_gain.sort_values("gain_share_pct", ascending=False),
        "gain_share_pct",
        "family",
        "Full-train LGBM Top500 Gain Share",
        "share of selected-feature gain",
        args.doc_asset_dir.joinpath("lgbm_family_gain_share.png"),
        color="#5b8e7d",
    )
    plot_bar(
        selected.head(25).sort_values("lgbm_gain", ascending=False),
        "lgbm_gain",
        "feature",
        "Top LGBM Gain Features",
        "gain",
        args.doc_asset_dir.joinpath("lgbm_top_gain_features.png"),
        color="#5b8e7d",
    )
    plot_bar(
        family_shap.sort_values("share_abs_shap", ascending=False),
        "share_abs_shap",
        "family",
        "Full-train TabPFN SHAP Family Share",
        "share of absolute SHAP",
        args.doc_asset_dir.joinpath("tabpfn_shap_family_share.png"),
        color="#b279a2",
    )
    plot_bar(
        feature_shap.head(25).sort_values("mean_abs_shap", ascending=False),
        "mean_abs_shap",
        "feature",
        "Top TabPFN SHAP Features",
        "mean absolute SHAP",
        args.doc_asset_dir.joinpath("tabpfn_top_shap_features.png"),
        color="#b279a2",
    )

    metadata = {
        "feature_set": args.feature,
        "mode": "full_train_explanatory_diagnostic",
        "K": args.K,
        "seed": args.seed,
        "tabpfn_version": "v2_6",
        "n_estimators": args.n_estimators,
        "softmax_temperature": args.softmax_temperature,
        "fit_mode": "fit_preprocessors",
        "shap_index": "SV",
        "shap_max_order": 1,
        "shap_imputer": "baseline",
        "shap_budget": args.budget,
        "n_explain_test_compounds": args.n_explain,
        "explained_test_indices": [int(i) for i in explain_test_idx],
        "fit_s": fit_s,
        "explainer_init_s": explainer_s,
        "explain_total_s": time.perf_counter() - shap_start,
        "note": (
            "This is not the exact fold-local OOF model explanation. It is a "
            "single full-train top500 diagnostic intended to summarize broad "
            "feature tendencies."
        ),
    }
    args.out_dir.joinpath("metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    report = [
        "# Optuna trial10 full-train top500 gain + TabPFN SHAP",
        "",
        f"Feature set: `{args.feature}`",
        f"SHAP: `SV`, baseline imputer, budget `{args.budget}`, "
        f"{args.n_explain} test compounds.",
        "",
        "## LGBM Family Gain",
        "",
        family_gain.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## TabPFN SHAP Family Summary",
        "",
        family_shap.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Top TabPFN SHAP Features",
        "",
        feature_shap.head(40).to_markdown(index=False, floatfmt=".5f"),
        "",
    ]
    args.out_dir.joinpath("report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"Wrote analysis outputs to {args.out_dir}")
    print(f"Wrote doc plots to {args.doc_asset_dir}")
    print("\nLGBM family gain:")
    print(family_gain.to_string(index=False))
    print("\nTabPFN SHAP family:")
    print(family_shap.to_string(index=False))


if __name__ == "__main__":
    main()
