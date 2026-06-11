#!/usr/bin/env -S pixi run python
"""Analyze HTChem as an auxiliary predicted-activity axis.

This intentionally does not add HTChem rows to Track 1 training. It trains a
small model on HTChem corrected pEC50 using a frozen ChemProp low-fidelity
embedding, assigns pred_htchem to challenge compounds, and checks whether that
axis explains AS1 labels, id55 error shape, or AS2 regions not covered by the
existing predicted-log2fc signal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import get_engine  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs"
DOC_PATH = REPO_ROOT / "docs" / "track1_explain" / "phase2_htchem_pred_axis.md"
SUBMISSION_DIR = REPO_ROOT / "track1_activity" / "submissions"

TRAIN_TEST_EMBED = REPO_ROOT / "data" / "chemprop_pretrain_embed.parquet"
HTCHEM_EMBED = REPO_ROOT / "data" / "chemprop_pretrain_htchem_embed.parquet"
TRAIN_TEST_LOG2FC = (
    REPO_ROOT
    / "data"
    / "chemprop_pretrain_log2fc_predictions_optuna_trial10_seed5ens.parquet"
)
HTCHEM_LOG2FC = (
    REPO_ROOT / "data" / "chemprop_pretrain_htchem_log2fc_predictions.parquet"
)
ID55_PATH = SUBMISSION_DIR / "ens_id51_top500_potent46_t40_soft_g35.csv"

ALPHAS = np.logspace(-2, 4, 25)


def metric_row(
    y: pd.Series | np.ndarray, pred: pd.Series | np.ndarray
) -> dict[str, float]:
    y_arr = np.asarray(y, dtype=float)
    pred_arr = np.asarray(pred, dtype=float)
    err = pred_arr - y_arr
    return {
        "n": int(len(y_arr)),
        "mae": float(np.mean(np.abs(err))),
        "bias_pred_minus_true": float(np.mean(err)),
        "spearman": float(stats.spearmanr(y_arr, pred_arr).statistic),
        "pearson": float(stats.pearsonr(y_arr, pred_arr).statistic),
        "pred_mean": float(np.mean(pred_arr)),
        "true_mean": float(np.mean(y_arr)),
    }


def load_htchem_labels() -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT
            h.compound_id,
            c.molecule_name,
            COALESCE(c.std_smiles, c.smiles) AS smiles,
            h.source_type,
            h.corrected_pec50,
            h.corrected_pec50_se,
            h.product_yield_percent,
            h.emax_normalized
        FROM htchem_activity h
        JOIN compounds c ON c.id = h.compound_id
        WHERE h.corrected_pec50 IS NOT NULL
        ORDER BY h.compound_id
        """,
        get_engine(),
    )


def load_train_labels() -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT
            t.id AS row_id,
            c.id AS compound_id,
            c.molecule_name,
            c.std_smiles AS smiles,
            t.pec50 AS true_pec50
        FROM train_activity t
        JOIN compounds c ON c.id = t.compound_id
        ORDER BY t.id
        """,
        get_engine(),
    )


def load_test_labels() -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT
            t.id AS row_id,
            c.id AS compound_id,
            c.molecule_name,
            c.std_smiles AS smiles,
            l.pec50 AS true_pec50
        FROM test_activity t
        JOIN compounds c ON c.id = t.compound_id
        LEFT JOIN test_activity_phase1_labels l ON l.compound_id = t.compound_id
        ORDER BY t.id
        """,
        get_engine(),
    )


def load_embeddings() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_test = pd.read_parquet(TRAIN_TEST_EMBED)
    htchem = pd.read_parquet(HTCHEM_EMBED)
    return train_test, htchem


def fit_ridge_oof(
    x: np.ndarray, y: np.ndarray, strata: pd.Series
) -> tuple[np.ndarray, list[float]]:
    oof = np.zeros(len(y), dtype=np.float64)
    alphas: list[float] = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, valid_idx in splitter.split(x, strata):
        model = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=ALPHAS, scoring="neg_mean_absolute_error"),
        )
        model.fit(x[train_idx], y[train_idx])
        oof[valid_idx] = model.predict(x[valid_idx])
        alphas.append(float(model.named_steps["ridgecv"].alpha_))
    return oof, alphas


def assign_pred_htchem(
    htchem: pd.DataFrame,
    htchem_embed: pd.DataFrame,
    challenge_embed: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    x_ht = htchem_embed.loc[htchem["compound_id"].astype(int)].to_numpy(
        dtype=np.float32
    )
    y_ht = htchem["corrected_pec50"].to_numpy(dtype=np.float64)
    strata = pd.qcut(
        htchem["corrected_pec50"], 5, labels=False, duplicates="drop"
    ).astype(str)

    oof, alphas = fit_ridge_oof(x_ht, y_ht, strata)
    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=ALPHAS, scoring="neg_mean_absolute_error"),
    )
    model.fit(x_ht, y_ht)

    pred_challenge = pd.Series(
        model.predict(challenge_embed.to_numpy(dtype=np.float32)),
        index=challenge_embed.index.astype(int),
        name="pred_htchem",
    )
    htchem_oof = htchem.copy()
    htchem_oof["pred_htchem_oof"] = oof
    htchem_oof["abs_error"] = (
        htchem_oof["pred_htchem_oof"] - htchem_oof["corrected_pec50"]
    ).abs()
    htchem_oof.attrs["alphas"] = alphas
    htchem_oof.attrs["final_alpha"] = float(model.named_steps["ridgecv"].alpha_)
    return pred_challenge, htchem_oof


def add_log2fc(df: pd.DataFrame) -> pd.DataFrame:
    lf = pd.read_parquet(TRAIN_TEST_LOG2FC).copy()
    lf["lf_mean"] = 0.5 * (lf["log2fc_8p25_pred"] + lf["log2fc_33_pred"])
    return df.merge(lf.reset_index(), on="compound_id", how="left")


def add_id55(test: pd.DataFrame) -> pd.DataFrame:
    pred = pd.read_csv(ID55_PATH).rename(
        columns={"Molecule Name": "molecule_name", "pEC50": "pred_id55"}
    )[["molecule_name", "pred_id55"]]
    out = test.merge(pred, on="molecule_name", how="left")
    out["id55_error"] = out["pred_id55"] - out["true_pec50"]
    out["id55_abs_error"] = out["id55_error"].abs()
    return out


def correlation_rows(
    df: pd.DataFrame, columns: list[str], target: str, prefix: str
) -> pd.DataFrame:
    rows = []
    for col in columns:
        valid = df[[target, col]].dropna()
        if len(valid) < 3:
            continue
        rows.append(
            {
                "slice": prefix,
                "x": col,
                "target": target,
                "n": int(len(valid)),
                "spearman": float(stats.spearmanr(valid[col], valid[target]).statistic),
                "pearson": float(stats.pearsonr(valid[col], valid[target]).statistic),
            }
        )
    return pd.DataFrame(rows)


def write_doc(
    coverage: pd.DataFrame,
    htchem_summary: pd.DataFrame,
    challenge_summary: pd.DataFrame,
    corr: pd.DataFrame,
    as1_bins: pd.DataFrame,
) -> None:
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Phase 2 HTChem pred-axis probe

Purpose: use HTChem as an auxiliary external activity axis (`pred_htchem`), not as equivalent Track 1 training labels.

## Inputs

- HTChem corrected pEC50 rows: 441 unique compounds.
- Frozen representation: `data/chemprop_pretrain_embed.parquet` for challenge compounds plus `data/chemprop_pretrain_htchem_embed.parquet` extracted with the same checkpoint.
- Predictor: 5-fold activity-stratified RidgeCV on HTChem corrected pEC50.
- Existing challenge context: optuna trial10 predicted log2fc and id55 AS1 replay.

## Coverage

{coverage.to_markdown(index=False)}

## HTChem Model Check

{htchem_summary.to_markdown(index=False, floatfmt=".4f")}

## Challenge/AS1 Checks

{challenge_summary.to_markdown(index=False, floatfmt=".4f")}

## Correlations

{corr.to_markdown(index=False, floatfmt=".4f")}

## AS1 Error By pred_htchem Quantile

{as1_bins.to_markdown(index=False, floatfmt=".4f")}

## Read

This is the first usable `pred_htchem` axis. It is still a ChemProp-only frozen-embedding probe, so it should be treated like the early log2fc-axis checks: useful if it explains AS1 errors or AS2 regions differently from predicted log2fc, not automatically a member to add.
"""
    DOC_PATH.write_text(text)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    htchem = load_htchem_labels()
    train = load_train_labels()
    test = load_test_labels()
    challenge_embed, htchem_embed = load_embeddings()

    coverage = pd.DataFrame(
        [
            {
                "asset": "chemprop_pretrain_embed",
                "rows": len(challenge_embed),
                "train_cover": int(
                    train["compound_id"].isin(challenge_embed.index).sum()
                ),
                "test_cover": int(
                    test["compound_id"].isin(challenge_embed.index).sum()
                ),
                "htchem_cover": int(
                    htchem["compound_id"].isin(challenge_embed.index).sum()
                ),
            },
            {
                "asset": "chemprop_pretrain_htchem_embed",
                "rows": len(htchem_embed),
                "train_cover": int(train["compound_id"].isin(htchem_embed.index).sum()),
                "test_cover": int(test["compound_id"].isin(htchem_embed.index).sum()),
                "htchem_cover": int(
                    htchem["compound_id"].isin(htchem_embed.index).sum()
                ),
            },
        ]
    )
    coverage.to_csv(OUT_DIR / "pred_htchem_coverage.csv", index=False)

    pred_challenge, htchem_oof = assign_pred_htchem(
        htchem, htchem_embed, challenge_embed
    )
    htchem_oof.to_csv(OUT_DIR / "htchem_pred_oof.csv", index=False)

    htchem_lf = pd.read_parquet(HTCHEM_LOG2FC).reset_index()
    htchem_diag = htchem_oof.merge(htchem_lf, on="compound_id", how="left")
    htchem_diag["lf_mean"] = 0.5 * (
        htchem_diag["log2fc_8p25_pred"] + htchem_diag["log2fc_33_pred"]
    )
    htchem_diag.to_csv(OUT_DIR / "htchem_with_log2fc_and_oof.csv", index=False)

    train_pred = train.merge(pred_challenge.reset_index(), on="compound_id", how="left")
    test_pred = test.merge(pred_challenge.reset_index(), on="compound_id", how="left")
    train_pred = add_log2fc(train_pred)
    test_pred = add_log2fc(test_pred)
    test_pred = add_id55(test_pred)
    test_pred["phase2_slice"] = np.where(test_pred["true_pec50"].notna(), "as1", "as2")
    test_pred["pred_htchem_z"] = (
        test_pred["pred_htchem"] - test_pred["pred_htchem"].mean()
    ) / test_pred["pred_htchem"].std()
    test_pred["lf_mean_z"] = (
        test_pred["lf_mean"] - test_pred["lf_mean"].mean()
    ) / test_pred["lf_mean"].std()
    test_pred["htchem_minus_lf_z"] = test_pred["pred_htchem_z"] - test_pred["lf_mean_z"]
    test_pred.to_csv(OUT_DIR / "challenge_pred_htchem_context.csv", index=False)

    htchem_summary_rows = [
        {
            "slice": "htchem_all_oof",
            **metric_row(htchem_oof["corrected_pec50"], htchem_oof["pred_htchem_oof"]),
            "ridge_alpha_mean": float(np.mean(htchem_oof.attrs["alphas"])),
            "ridge_alpha_final": htchem_oof.attrs["final_alpha"],
        }
    ]
    for source, sub in htchem_oof.groupby("source_type"):
        htchem_summary_rows.append(
            {
                "slice": f"htchem_{source}_oof",
                **metric_row(sub["corrected_pec50"], sub["pred_htchem_oof"]),
                "ridge_alpha_mean": np.nan,
                "ridge_alpha_final": np.nan,
            }
        )
    htchem_summary = pd.DataFrame(htchem_summary_rows)
    htchem_summary.to_csv(OUT_DIR / "htchem_pred_oof_summary.csv", index=False)

    challenge_summary_rows = [
        {
            "slice": "train_true_vs_pred_htchem",
            **metric_row(train_pred["true_pec50"], train_pred["pred_htchem"]),
        },
        {
            "slice": "as1_true_vs_pred_htchem",
            **metric_row(
                test_pred.loc[test_pred["phase2_slice"] == "as1", "true_pec50"],
                test_pred.loc[test_pred["phase2_slice"] == "as1", "pred_htchem"],
            ),
        },
    ]
    as1 = test_pred[test_pred["phase2_slice"] == "as1"].copy()
    for label, sub in as1.groupby(
        pd.cut(
            as1["true_pec50"],
            [-np.inf, 3, 4, 5, 6, np.inf],
            labels=["lt3", "3to4", "4to5", "5to6", "gte6"],
        ),
        observed=True,
    ):
        challenge_summary_rows.append(
            {
                "slice": f"as1_true_bin_{label}_vs_pred_htchem",
                **metric_row(sub["true_pec50"], sub["pred_htchem"]),
            }
        )
    challenge_summary = pd.DataFrame(challenge_summary_rows)
    challenge_summary.to_csv(OUT_DIR / "challenge_pred_htchem_summary.csv", index=False)

    corr = pd.concat(
        [
            correlation_rows(
                htchem_diag,
                ["log2fc_8p25_pred", "log2fc_33_pred", "lf_mean", "pred_htchem_oof"],
                "corrected_pec50",
                "htchem",
            ),
            correlation_rows(
                train_pred,
                ["pred_htchem", "log2fc_8p25_pred", "log2fc_33_pred", "lf_mean"],
                "true_pec50",
                "train",
            ),
            correlation_rows(
                as1,
                [
                    "pred_htchem",
                    "log2fc_8p25_pred",
                    "log2fc_33_pred",
                    "lf_mean",
                    "pred_id55",
                ],
                "true_pec50",
                "as1",
            ),
            correlation_rows(
                as1,
                ["pred_htchem", "log2fc_8p25_pred", "log2fc_33_pred", "lf_mean"],
                "id55_error",
                "as1",
            ),
            correlation_rows(
                as1,
                ["pred_htchem", "log2fc_8p25_pred", "log2fc_33_pred", "lf_mean"],
                "id55_abs_error",
                "as1",
            ),
        ],
        ignore_index=True,
    )
    corr.to_csv(OUT_DIR / "pred_htchem_correlations.csv", index=False)

    as1["pred_htchem_quantile"] = pd.qcut(
        as1["pred_htchem"], 5, labels=["q1_low", "q2", "q3", "q4", "q5_high"]
    )
    as1_bins = (
        as1.groupby("pred_htchem_quantile", observed=True)
        .agg(
            n=("compound_id", "size"),
            true_mean=("true_pec50", "mean"),
            pred_htchem_mean=("pred_htchem", "mean"),
            lf_mean=("lf_mean", "mean"),
            id55_mae=("id55_abs_error", "mean"),
            id55_bias=("id55_error", "mean"),
        )
        .reset_index()
    )
    as1_bins.to_csv(OUT_DIR / "as1_id55_error_by_pred_htchem_quantile.csv", index=False)

    as2 = test_pred[test_pred["phase2_slice"] == "as2"].copy()
    as2.sort_values("htchem_minus_lf_z", ascending=False).head(40).to_csv(
        OUT_DIR / "as2_high_htchem_low_log2fc_cases.csv", index=False
    )
    as2.sort_values("htchem_minus_lf_z", ascending=True).head(40).to_csv(
        OUT_DIR / "as2_low_htchem_high_log2fc_cases.csv", index=False
    )

    write_doc(coverage, htchem_summary, challenge_summary, corr, as1_bins)

    print(f"Wrote outputs to {OUT_DIR}")
    print(f"Wrote doc to {DOC_PATH}")
    print(htchem_summary.to_string(index=False))
    print(challenge_summary.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
