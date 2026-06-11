#!/usr/bin/env -S pixi run python
"""Run proper top500 TabPFN with pred_htchem appended as one candidate feature."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

import run_train  # noqa: E402
from data import get_engine  # noqa: E402
from evaluate import record_experiment, save_oof_predictions  # noqa: E402
from splits import umap_split_indices  # noqa: E402

SUBMISSION_DIR = REPO_ROOT / "track1_activity" / "submissions"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "top500_tabpfn_htchem"
DOC_PATH = REPO_ROOT / "docs" / "track1_explain" / "phase2_htchem_top500_tabpfn.md"

TRAIN_TEST_EMBED = REPO_ROOT / "data" / "chemprop_pretrain_embed.parquet"
HTCHEM_EMBED = REPO_ROOT / "data" / "chemprop_pretrain_htchem_embed.parquet"
ALPHAS = np.logspace(-2, 4, 25)


def load_train() -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT
            t.id AS row_id,
            t.compound_id,
            t.pec50,
            c.std_smiles AS smiles,
            c.molecule_name
        FROM train_activity t
        JOIN compounds c ON c.id = t.compound_id
        ORDER BY t.id
        """,
        get_engine(),
    )


def load_test() -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT
            t.id AS row_id,
            t.compound_id,
            c.std_smiles AS smiles,
            c.molecule_name
        FROM test_activity t
        JOIN compounds c ON c.id = t.compound_id
        ORDER BY t.id
        """,
        get_engine(),
    )


def load_htchem_labels() -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT h.compound_id, h.corrected_pec50
        FROM htchem_activity h
        WHERE h.corrected_pec50 IS NOT NULL
        ORDER BY h.compound_id
        """,
        get_engine(),
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    num = np.sum(np.abs(y_true - y_pred))
    den = np.sum(np.abs(y_true - np.mean(y_true)))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RAE": float(num / den) if den > 0 else float("nan"),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman_R": float(spearmanr(y_pred, y_true).statistic),
        "Kendall_Tau": float(kendalltau(y_pred, y_true).statistic),
    }


def pred_htchem_for_challenge() -> pd.Series:
    htchem = load_htchem_labels()
    htchem_embed = pd.read_parquet(HTCHEM_EMBED)
    challenge_embed = pd.read_parquet(TRAIN_TEST_EMBED)
    x_ht = htchem_embed.loc[htchem["compound_id"].astype(int)].to_numpy(
        dtype=np.float32
    )
    y_ht = htchem["corrected_pec50"].to_numpy(dtype=np.float64)

    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=ALPHAS, scoring="neg_mean_absolute_error"),
    )
    model.fit(x_ht, y_ht)
    return pd.Series(
        model.predict(challenge_embed.to_numpy(dtype=np.float32)).astype(np.float32),
        index=challenge_embed.index.astype(int),
        name="pred_htchem",
    )


def sanitize_train_test(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    col_mean = np.nanmean(x_train, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, col_mean).astype(np.float32)
    x_test = np.where(np.isfinite(x_test), x_test, col_mean).astype(np.float32)
    return x_train, x_test


def write_doc(
    exp_name: str,
    overall: dict[str, float],
    fold_metrics: pd.DataFrame,
    selection_summary: pd.DataFrame,
    ref_row: tuple[float, float, float] | None,
) -> None:
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    ref_text = "Reference not found in experiment_summary."
    if ref_row is not None:
        ref_mae, ref_rae, ref_sp = ref_row
        ref_text = (
            f"Reference existing top500: MAE={ref_mae:.4f}, RAE={ref_rae:.4f}, "
            f"Spearman={ref_sp:.4f}. Delta MAE={overall['MAE'] - ref_mae:+.4f}."
        )
    text = f"""# Phase 2 HTChem top500 TabPFN

Experiment: `{exp_name}`

`pred_htchem` was appended as one scalar to `cheme_2d_full_boltz_log2fc_pred_seed10ens`, then top500 features were selected inside each outer fold using LGBM gain before fitting TabPFN.

## Overall OOF

| metric | value |
|:--|--:|
| MAE | {overall["MAE"]:.4f} |
| RAE | {overall["RAE"]:.4f} |
| Spearman_R | {overall["Spearman_R"]:.4f} |
| Kendall_Tau | {overall["Kendall_Tau"]:.4f} |

{ref_text}

## Fold Metrics

{fold_metrics.to_markdown(index=False, floatfmt=".4f")}

## pred_htchem Selection

{selection_summary.to_markdown(index=False, floatfmt=".4f")}

## Read

Use this as a SWAP diagnostic against the existing top500 member, not as an ADD by default. The gain probe said `pred_htchem` is consistently selected but low-share; the OOF result decides whether it is worth carrying into ensemble bakeoff.
"""
    DOC_PATH.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature", default="cheme_2d_full_boltz_log2fc_pred_seed10ens"
    )
    parser.add_argument("--K", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--softmax-temperature", type=float, default=0.9)
    parser.add_argument(
        "--tabpfn-version",
        choices=["v3", "v2_6", "v2_5", "v2"],
        default="v2_6",
    )
    args = parser.parse_args()

    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    version_enum = {
        "v3": ModelVersion.V3,
        "v2_6": ModelVersion.V2_6,
        "v2_5": ModelVersion.V2_5,
        "v2": ModelVersion.V2,
    }[args.tabpfn_version]
    model_path = TabPFNRegressor.create_default_for_version(version_enum).model_path

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    train = load_train()
    test = load_test()
    x_train_base, x_test_base = run_train.load_features(args.feature, train, test)
    pred_htchem = pred_htchem_for_challenge()
    train_ht = pred_htchem.reindex(train["compound_id"].astype(int)).to_numpy(
        dtype=np.float32
    )[:, None]
    test_ht = pred_htchem.reindex(test["compound_id"].astype(int)).to_numpy(
        dtype=np.float32
    )[:, None]
    x_train = np.concatenate([x_train_base, train_ht], axis=1)
    x_test = np.concatenate([x_test_base, test_ht], axis=1)
    x_train, x_test = sanitize_train_test(x_train, x_test)
    y = train["pec50"].to_numpy(dtype=np.float32)
    pred_htchem_idx = x_train.shape[1] - 1

    print(
        f"Feature: {args.feature} + pred_htchem  base_d={x_train_base.shape[1]} "
        f"model_d={x_train.shape[1]} K={args.K}"
    )
    print(
        f"TabPFN version={args.tabpfn_version}, n_estimators={args.n_estimators}, "
        f"temperature={args.softmax_temperature}"
    )

    splits = umap_split_indices(
        train["smiles"].tolist(), n_splits=5, n_clusters=50, seed=args.seed
    )
    oof = np.zeros(len(y), dtype=np.float32)
    test_preds_per_fold: list[np.ndarray] = []
    fold_metrics: list[dict[str, float]] = []
    selection_rows: list[dict[str, float | int | bool]] = []

    for fold, (tr_idx, va_idx) in enumerate(splits):
        ranker = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=10,
            random_state=args.seed,
            verbose=-1,
        )
        ranker.fit(x_train[tr_idx], y[tr_idx])
        gain = ranker.booster_.feature_importance(importance_type="gain")
        split = ranker.booster_.feature_importance(importance_type="split")
        order = np.argsort(-gain)
        selected = order[: args.K]
        rank = int(np.where(order == pred_htchem_idx)[0][0] + 1)
        selected_htchem = bool(pred_htchem_idx in set(selected.tolist()))
        total_gain = float(gain.sum())
        selection_rows.append(
            {
                "fold": fold,
                "pred_htchem_rank": rank,
                "pred_htchem_selected": selected_htchem,
                "pred_htchem_gain_share_pct": float(
                    gain[pred_htchem_idx] / total_gain * 100.0
                )
                if total_gain > 0
                else 0.0,
                "pred_htchem_split": int(split[pred_htchem_idx]),
                "zero_gain_selected": int((gain[selected] == 0).sum()),
            }
        )

        reg = TabPFNRegressor(
            device="cuda",
            n_estimators=args.n_estimators,
            softmax_temperature=args.softmax_temperature,
            random_state=args.seed,
            model_path=model_path,
            ignore_pretraining_limits=args.K > 500,
        )
        reg.fit(x_train[tr_idx][:, selected], y[tr_idx])
        oof[va_idx] = reg.predict(x_train[va_idx][:, selected])
        test_preds_per_fold.append(reg.predict(x_test[:, selected]))

        metrics = compute_metrics(y[va_idx], oof[va_idx])
        fold_metrics.append(metrics)
        print(
            f"  [Fold {fold}] MAE={metrics['MAE']:.4f} "
            f"Sp={metrics['Spearman_R']:.4f} "
            f"pred_htchem_rank={rank} selected={selected_htchem}"
        )

    overall = compute_metrics(y, oof)
    fold_df = pd.DataFrame(fold_metrics)
    selection_df = pd.DataFrame(selection_rows)
    fold_df.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    selection_df.to_csv(OUT_DIR / "pred_htchem_selection.csv", index=False)
    pd.DataFrame(
        {
            "compound_id": train["compound_id"],
            "molecule_name": train["molecule_name"],
            "true_pec50": y,
            "oof_pred": oof,
        }
    ).to_csv(OUT_DIR / "oof_predictions.csv", index=False)

    exp_name = (
        f"tabpfn_{args.feature}_pred_htchem_top{args.K}_umap_{args.tabpfn_version}"
    )
    if args.n_estimators != 8:
        exp_name += f"_n{args.n_estimators}"
    if args.softmax_temperature != 0.9:
        exp_name += f"_temp{str(args.softmax_temperature).replace('.', 'p')}"

    test_pred = np.mean(np.stack(test_preds_per_fold), axis=0)
    sub = pd.DataFrame(
        {
            "SMILES": test["smiles"],
            "Molecule Name": test["molecule_name"],
            "pEC50": test_pred,
        }
    )
    sub_path = SUBMISSION_DIR / f"{exp_name}.csv"
    sub.to_csv(sub_path, index=False)

    ref_row: tuple[float, float, float] | None = None
    with get_engine().connect() as conn:
        ref = pd.read_sql(
            """
            SELECT mae_mean, rae_mean, spearman_mean
            FROM experiment_summary
            WHERE name = %s
            """,
            conn,
            params=(f"tabpfn_{args.feature}_top{args.K}_umap",),
        )
    if len(ref) == 1:
        ref_row = tuple(
            float(ref.iloc[0][c]) for c in ["mae_mean", "rae_mean", "spearman_mean"]
        )

    exp_id = record_experiment(
        name=exp_name,
        description=(
            f"TabPFN on top-{args.K} per-fold LGBM-gain features from "
            f"{args.feature} plus pred_htchem scalar."
        ),
        model_type="tabpfn",
        feature_set=f"{args.feature}_pred_htchem",
        hyperparameters={
            "K": args.K,
            "seed": args.seed,
            "n_estimators": args.n_estimators,
            "softmax_temperature": args.softmax_temperature,
            "tabpfn_version": args.tabpfn_version,
            "lgbm_n_estimators": 500,
            "lgbm_learning_rate": 0.05,
            "lgbm_num_leaves": 63,
            "pred_htchem_source": "ChemProp LF embedding RidgeCV on HTChem corrected_pEC50",
            "pred_htchem_selected_folds": int(
                selection_df["pred_htchem_selected"].sum()
            ),
        },
        fold_metrics=fold_metrics,
        submission_path=f"track1_activity/submissions/{exp_name}.csv",
        notes=(
            f"Proper top500 CV with pred_htchem appended. OOF MAE={overall['MAE']:.4f}, "
            f"RAE={overall['RAE']:.4f}, Sp={overall['Spearman_R']:.4f}. "
            f"pred_htchem selected in {int(selection_df['pred_htchem_selected'].sum())}/5 folds."
        ),
    )
    save_oof_predictions(exp_id, oof)

    selection_summary = pd.DataFrame(
        [
            {
                "selected_folds": int(selection_df["pred_htchem_selected"].sum()),
                "mean_rank": float(selection_df["pred_htchem_rank"].mean()),
                "min_rank": int(selection_df["pred_htchem_rank"].min()),
                "max_rank": int(selection_df["pred_htchem_rank"].max()),
                "mean_gain_share_pct": float(
                    selection_df["pred_htchem_gain_share_pct"].mean()
                ),
                "mean_split": float(selection_df["pred_htchem_split"].mean()),
            }
        ]
    )
    selection_summary.to_csv(OUT_DIR / "pred_htchem_selection_summary.csv", index=False)
    pd.DataFrame([overall]).to_csv(OUT_DIR / "overall_metrics.csv", index=False)
    write_doc(exp_name, overall, fold_df, selection_summary, ref_row)

    print("\nOverall OOF:")
    for key, value in overall.items():
        print(f"  {key}={value:.4f}")
    if ref_row is not None:
        print(
            f"Reference tabpfn_{args.feature}_top{args.K}_umap: "
            f"MAE={ref_row[0]:.4f} RAE={ref_row[1]:.4f} Sp={ref_row[2]:.4f}"
        )
        print(f"Delta MAE vs reference: {overall['MAE'] - ref_row[0]:+.4f}")
    print(f"Saved submission: {sub_path}")
    print(f"Recorded experiment id={exp_id}: {exp_name}")
    print(f"Wrote doc: {DOC_PATH}")


if __name__ == "__main__":
    main()
