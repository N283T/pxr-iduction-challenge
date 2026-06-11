#!/usr/bin/env -S pixi run python
"""Probe whether Phase 2 HTChem data is useful for Track 1 activity modeling.

This is intentionally a diagnostic layer before touching the production
ensemble. It asks:

1. How much HTChem label coverage and QC signal do we have?
2. Which train/AS1/AS2 regions are close to HTChem compounds?
3. Does a simple HTChem-only model transfer to train+AS1 labels?
4. Does adding HTChem labels improve a matched Morgan LightGBM Phase2 OOF?
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy import stats
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(
    0,
    str(REPO_ROOT.joinpath("track1_activity", "analysis", "phase2_validation_matrix")),
)

from build_phase2_validation_matrix import build_labeled_pool, build_phase2_splits  # noqa: E402
from data import get_engine  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs"
DOC_PATH = REPO_ROOT / "docs" / "track1_explain" / "phase2_htchem_probe.md"
AS1_CANDIDATES = {
    "id55_anchor": REPO_ROOT
    / "track1_activity"
    / "submissions"
    / "ens_id51_top500_potent46_t40_soft_g35.csv",
    "seed10_top500_temp0p7": REPO_ROOT
    / "track1_activity"
    / "submissions"
    / "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap_v3_temp0p7.csv",
    "optuna_t10_top500": REPO_ROOT
    / "track1_activity"
    / "submissions"
    / "tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_top500_umap.csv",
}


def metric_row(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    err = pred - y
    spearman = stats.spearmanr(y, pred).statistic if len(y) >= 2 else np.nan
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(err))),
        "bias_pred_minus_true": float(np.mean(err)),
        "spearman": float(spearman),
        "pred_mean": float(np.mean(pred)),
        "true_mean": float(np.mean(y)),
    }


def load_htchem_rows() -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT
            h.id AS htchem_id,
            h.compound_id,
            COALESCE(c.std_smiles, c.smiles) AS smiles,
            c.molecule_name,
            h.source_type,
            h.ocnt_id,
            h.batch_id,
            h.pec50,
            h.corrected_pec50,
            h.corrected_pec50_se,
            h.corrected_pec50_ci95,
            h.product_yield_percent,
            h.volatility,
            h.cad_yield_volatility_note,
            h.emax_normalized
        FROM htchem_activity h
        JOIN compounds c ON c.id = h.compound_id
        WHERE h.corrected_pec50 IS NOT NULL
        ORDER BY h.source_type, h.ocnt_id, h.batch_id
        """,
        get_engine(),
    )


def collapse_htchem(rows: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated HTChem rows to one label per compound.

    Semi-pure labels get priority over crude labels, then lower propagated SE.
    The collapsed row keeps a conservative sample weight for augmentation.
    """
    df = rows.copy()
    df["source_rank"] = df["source_type"].map({"semi_pure": 0, "crude": 1}).fillna(9)
    df["se_rank"] = df["corrected_pec50_se"].fillna(9.0)
    df = df.sort_values(["compound_id", "source_rank", "se_rank"])
    best = df.groupby("compound_id", as_index=False).head(1).copy()
    se = best["corrected_pec50_se"].fillna(best["corrected_pec50_se"].median())
    inv_var = 1.0 / np.square(np.maximum(se.to_numpy(dtype=np.float64), 0.12))
    inv_var = inv_var / np.nanmedian(inv_var)
    source_factor = np.where(best["source_type"].eq("semi_pure"), 1.0, 0.7)
    best["sample_weight"] = np.clip(inv_var * source_factor, 0.25, 2.0)
    best = best.rename(columns={"corrected_pec50": "htchem_pec50"})
    return best.reset_index(drop=True)


def morgan_count(smiles: pd.Series, n_bits: int = 2048) -> np.ndarray:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    invalid = [i for i, mol in enumerate(mols) if mol is None]
    if invalid:
        raise ValueError(f"invalid SMILES at rows {invalid[:10]}")
    return np.asarray(
        [gen.GetCountFingerprintAsNumPy(mol) for mol in mols], dtype=np.float32
    )


def morgan_bits(smiles: pd.Series, n_bits: int = 2048):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    invalid = [i for i, mol in enumerate(mols) if mol is None]
    if invalid:
        raise ValueError(f"invalid SMILES at rows {invalid[:10]}")
    return [gen.GetFingerprint(mol) for mol in mols]


def max_tanimoto(query_fps, ref_fps) -> tuple[np.ndarray, np.ndarray]:
    vals = []
    idxs = []
    for fp in query_fps:
        sims = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fp, ref_fps), dtype=np.float32
        )
        idx = int(np.argmax(sims))
        vals.append(float(sims[idx]))
        idxs.append(idx)
    return np.asarray(vals, dtype=np.float32), np.asarray(idxs, dtype=np.int64)


def load_test_pool() -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT
            t.id AS test_row_id,
            t.compound_id,
            COALESCE(c.std_smiles, c.smiles) AS smiles,
            c.molecule_name,
            l.pec50 AS as1_pec50,
            CASE WHEN l.pec50 IS NULL THEN 'as2' ELSE 'as1' END AS source
        FROM test_activity t
        JOIN compounds c ON c.id = t.compound_id
        LEFT JOIN test_activity_phase1_labels l ON l.compound_id = t.compound_id
        ORDER BY t.id
        """,
        get_engine(),
    )


def summarize_htchem(
    rows: pd.DataFrame, unique: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile_rows = []
    for label, df in {
        "all_rows": rows,
        "unique_compounds": unique,
        "crude_rows": rows[rows["source_type"].eq("crude")],
        "semi_pure_rows": rows[rows["source_type"].eq("semi_pure")],
    }.items():
        target_col = (
            "htchem_pec50" if "htchem_pec50" in df.columns else "corrected_pec50"
        )
        profile_rows.append(
            {
                "slice": label,
                "n": len(df),
                "n_unique_compound": df["compound_id"].nunique(),
                "mean_pec50": float(df[target_col].mean()),
                "median_pec50": float(df[target_col].median()),
                "min_pec50": float(df[target_col].min()),
                "max_pec50": float(df[target_col].max()),
                "mean_se": float(df["corrected_pec50_se"].mean()),
                "median_yield_percent": float(df["product_yield_percent"].median()),
            }
        )
    volatility = (
        rows.groupby(["source_type", "volatility"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["source_type", "n"], ascending=[True, False])
    )
    return pd.DataFrame(profile_rows), volatility


def coverage_tables(
    htchem: pd.DataFrame, labeled: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ht_fps = morgan_bits(htchem["smiles"])
    labeled_fps = morgan_bits(labeled["smiles"])
    test_fps = morgan_bits(test["smiles"])

    labeled_nn, labeled_idx = max_tanimoto(labeled_fps, ht_fps)
    test_nn, test_idx = max_tanimoto(test_fps, ht_fps)
    labeled_cov = labeled.copy()
    labeled_cov["htchem_nn_tanimoto"] = labeled_nn
    labeled_cov["htchem_nn_ocnt_id"] = htchem.iloc[labeled_idx]["ocnt_id"].to_numpy()
    test_cov = test.copy()
    test_cov["htchem_nn_tanimoto"] = test_nn
    test_cov["htchem_nn_ocnt_id"] = htchem.iloc[test_idx]["ocnt_id"].to_numpy()

    summary_rows = []
    for label, df in {
        "labeled_all": labeled_cov,
        "labeled_train": labeled_cov[labeled_cov["source"].eq("train")],
        "labeled_as1": labeled_cov[labeled_cov["source"].eq("as1")],
        "test_all": test_cov,
        "test_as1": test_cov[test_cov["source"].eq("as1")],
        "test_as2": test_cov[test_cov["source"].eq("as2")],
    }.items():
        row = {
            "slice": label,
            "n": len(df),
            "mean_nn": float(df["htchem_nn_tanimoto"].mean()),
            "p90_nn": float(df["htchem_nn_tanimoto"].quantile(0.90)),
            "max_nn": float(df["htchem_nn_tanimoto"].max()),
        }
        for threshold in [0.3, 0.4, 0.5, 0.6]:
            row[f"n_ge_{threshold:.1f}"] = int(
                (df["htchem_nn_tanimoto"] >= threshold).sum()
            )
        summary_rows.append(row)
    return pd.DataFrame(summary_rows), test_cov


def fit_lgbm(X_train, y_train, weights=None, seed: int = 42) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=15,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective="mae",
        random_state=seed,
        verbose=-1,
    )
    model.fit(X_train, y_train, sample_weight=weights)
    return model


def htchem_only_transfer(
    htchem: pd.DataFrame, labeled: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    Xh = morgan_count(htchem["smiles"])
    yh = htchem["htchem_pec50"].to_numpy(dtype=np.float64)
    wh = htchem["sample_weight"].to_numpy(dtype=np.float64)
    oof = np.full(len(htchem), np.nan, dtype=np.float64)
    for fold, (tr, va) in enumerate(
        KFold(n_splits=5, shuffle=True, random_state=42).split(Xh)
    ):
        model = fit_lgbm(Xh[tr], yh[tr], wh[tr], seed=42 + fold)
        oof[va] = model.predict(Xh[va])

    rows = [{"slice": "htchem_oof", **metric_row(yh, oof)}]
    Xl = morgan_count(labeled["smiles"])
    model = fit_lgbm(Xh, yh, wh, seed=42)
    pred = model.predict(Xl)
    pred_df = labeled[
        [
            "pool_idx",
            "compound_id",
            "molecule_name",
            "smiles",
            "pec50",
            "source",
            "true_bin",
        ]
    ].copy()
    pred_df["htchem_model_pred"] = pred
    pred_df["htchem_model_error"] = pred - pred_df["pec50"]
    masks = {
        "labeled_all": np.ones(len(pred_df), dtype=bool),
        "source_train": pred_df["source"].eq("train").to_numpy(),
        "source_as1": pred_df["source"].eq("as1").to_numpy(),
        "true_lt3": (pred_df["pec50"] < 3.0).to_numpy(),
        "true_gte6": (pred_df["pec50"] >= 6.0).to_numpy(),
    }
    for label, mask in masks.items():
        rows.append(
            {
                "slice": label,
                **metric_row(
                    pred_df.loc[mask, "pec50"].to_numpy(dtype=np.float64),
                    pred_df.loc[mask, "htchem_model_pred"].to_numpy(dtype=np.float64),
                ),
            }
        )
    return pd.DataFrame(rows), pred_df


def phase2_augmented_oof(
    htchem: pd.DataFrame, labeled: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splits, _ = build_phase2_splits(labeled, n_splits=5, n_clusters=100, seed=42)
    Xl = morgan_count(labeled["smiles"])
    yl = labeled["pec50"].to_numpy(dtype=np.float64)
    Xh = morgan_count(htchem["smiles"])
    yh = htchem["htchem_pec50"].to_numpy(dtype=np.float64)
    wh = htchem["sample_weight"].to_numpy(dtype=np.float64)

    settings = {
        "no_htchem": 0.0,
        "htchem_w0p1": 0.1,
        "htchem_w0p3": 0.3,
        "htchem_w0p5": 0.5,
        "htchem_w1p0": 1.0,
    }
    pred_cols = {}
    fold_rows = []
    for setting, ext_weight in settings.items():
        oof = np.full(len(labeled), np.nan, dtype=np.float64)
        for fold, (tr, va) in enumerate(splits):
            if ext_weight > 0:
                Xtr = np.vstack([Xl[tr], Xh])
                ytr = np.concatenate([yl[tr], yh])
                wtr = np.concatenate(
                    [np.ones(len(tr), dtype=np.float64), wh * ext_weight]
                )
            else:
                Xtr = Xl[tr]
                ytr = yl[tr]
                wtr = np.ones(len(tr), dtype=np.float64)
            model = fit_lgbm(Xtr, ytr, wtr, seed=100 + fold)
            pred = model.predict(Xl[va])
            oof[va] = pred
            fold_rows.append(
                {"setting": setting, "fold": fold, **metric_row(yl[va], pred)}
            )
        pred_cols[setting] = oof

    summary_rows = []
    masks = {
        "all": np.ones(len(labeled), dtype=bool),
        "source_train": labeled["source"].eq("train").to_numpy(),
        "source_as1": labeled["source"].eq("as1").to_numpy(),
        "true_lt3": (labeled["pec50"] < 3.0).to_numpy(),
        "true_gte6": (labeled["pec50"] >= 6.0).to_numpy(),
    }
    for setting, pred in pred_cols.items():
        for label, mask in masks.items():
            summary_rows.append(
                {"setting": setting, "slice": label, **metric_row(yl[mask], pred[mask])}
            )

    oof_df = labeled[
        [
            "pool_idx",
            "compound_id",
            "molecule_name",
            "smiles",
            "pec50",
            "source",
            "true_bin",
        ]
    ].copy()
    for setting, pred in pred_cols.items():
        oof_df[f"{setting}_pred"] = pred
    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows), oof_df


def as1_replay_by_htchem_coverage(test_cov: pd.DataFrame) -> pd.DataFrame:
    as1 = test_cov[test_cov["source"].eq("as1")].copy()
    rows = []
    slices = {
        "as1_all": np.ones(len(as1), dtype=bool),
        "htchem_nn_lt0p3": as1["htchem_nn_tanimoto"].lt(0.3).to_numpy(),
        "htchem_nn_ge0p3": as1["htchem_nn_tanimoto"].ge(0.3).to_numpy(),
        "htchem_nn_ge0p5": as1["htchem_nn_tanimoto"].ge(0.5).to_numpy(),
        "htchem_nn_ge0p6": as1["htchem_nn_tanimoto"].ge(0.6).to_numpy(),
    }
    for name, path in AS1_CANDIDATES.items():
        pred = pd.read_csv(path).rename(
            columns={"Molecule Name": "molecule_name", "pEC50": "pred"}
        )
        merged = as1.merge(
            pred[["molecule_name", "pred"]], on="molecule_name", how="inner"
        )
        if len(merged) != len(as1):
            raise RuntimeError(f"{name} aligned {len(merged)} of {len(as1)} AS1 rows")
        for slice_name, mask in slices.items():
            sub = merged.loc[mask]
            if sub.empty:
                continue
            rows.append(
                {
                    "candidate": name,
                    "slice": slice_name,
                    "mean_htchem_nn": float(sub["htchem_nn_tanimoto"].mean()),
                    **metric_row(
                        sub["as1_pec50"].to_numpy(dtype=np.float64),
                        sub["pred"].to_numpy(dtype=np.float64),
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    profile: pd.DataFrame,
    coverage: pd.DataFrame,
    transfer: pd.DataFrame,
    aug_summary: pd.DataFrame,
    as1_replay: pd.DataFrame,
) -> None:
    aug_all = aug_summary[aug_summary["slice"].eq("all")].sort_values("mae")
    aug_as1 = aug_summary[aug_summary["slice"].eq("source_as1")].sort_values("mae")
    as1_nn = as1_replay[
        as1_replay["slice"].isin(["as1_all", "htchem_nn_ge0p5"])
    ].sort_values(["slice", "mae"])
    lines = [
        "# Phase 2 HTChem probe",
        "",
        "Diagnostic analysis for the Phase 2 HTChem release. This does not change",
        "the production ensemble; it tests whether HTChem is a useful external SAR",
        "axis before heavier retraining.",
        "",
        "## HTChem label profile",
        "",
        profile.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Morgan nearest-neighbor coverage to HTChem",
        "",
        coverage.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## HTChem-only model transfer",
        "",
        transfer.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Phase2 Morgan LGBM augmentation, all rows",
        "",
        aug_all.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Phase2 Morgan LGBM augmentation, AS1 slice",
        "",
        aug_as1.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## AS1 replay by HTChem nearest-neighbor coverage",
        "",
        as1_nn.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Initial read",
        "",
        "- Treat corrected HTChem pEC50 as the primary label and keep QC columns",
        "  available for weighting and filtering.",
        "- The first useful question is not whether HTChem should replace the",
        "  current best model, but whether it provides a stable external SAR axis",
        "  around AS2-like chemistry.",
        "- If augmentation helps only in narrow Morgan-LGBM diagnostics, the next",
        "  step should be a low-weight feature/member, not a broad ensemble shift.",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = load_htchem_rows()
    htchem = collapse_htchem(rows)
    labeled = build_labeled_pool()
    test = load_test_pool()

    profile, volatility = summarize_htchem(rows, htchem)
    coverage, test_cov = coverage_tables(htchem, labeled, test)
    transfer, transfer_preds = htchem_only_transfer(htchem, labeled)
    aug_summary, aug_folds, aug_oof = phase2_augmented_oof(htchem, labeled)
    as1_replay = as1_replay_by_htchem_coverage(test_cov)

    rows.to_csv(OUT_DIR / "htchem_rows.csv", index=False)
    htchem.to_csv(OUT_DIR / "htchem_unique_labels.csv", index=False)
    profile.to_csv(OUT_DIR / "htchem_profile.csv", index=False)
    volatility.to_csv(OUT_DIR / "htchem_volatility_counts.csv", index=False)
    coverage.to_csv(OUT_DIR / "htchem_nn_coverage_summary.csv", index=False)
    test_cov.to_csv(OUT_DIR / "test_htchem_nn_coverage.csv", index=False)
    transfer.to_csv(OUT_DIR / "htchem_only_transfer_summary.csv", index=False)
    transfer_preds.to_csv(OUT_DIR / "htchem_only_labeled_predictions.csv", index=False)
    aug_summary.to_csv(
        OUT_DIR / "phase2_htchem_augmented_lgbm_summary.csv", index=False
    )
    aug_folds.to_csv(
        OUT_DIR / "phase2_htchem_augmented_lgbm_fold_metrics.csv", index=False
    )
    aug_oof.to_csv(OUT_DIR / "phase2_htchem_augmented_lgbm_oof.csv", index=False)
    as1_replay.to_csv(OUT_DIR / "as1_replay_by_htchem_nn.csv", index=False)
    write_report(profile, coverage, transfer, aug_summary, as1_replay)

    print(f"wrote {OUT_DIR}")
    print(f"wrote {DOC_PATH}")


if __name__ == "__main__":
    main()
