#!/usr/bin/env python
"""Audit alternative OOF metrics against LB-known submissions.

This is intentionally diagnostic: it does not train or submit models. It asks
whether any row-weighted or slice-specific OOF score orders historical
LB-known submissions better than plain global OOF MAE.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "track1_activity" / "src"
sys.path.insert(0, str(SRC_DIR))

from data import get_engine, load_test_smiles, load_train_smiles_with_counter  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "oof_metric_battery"
LF_PATH = (
    REPO_ROOT
    / "data"
    / "chemprop_pretrain_log2fc_predictions_optuna_trial10_seed5ens.parquet"
)


@dataclass(frozen=True)
class MetricSpec:
    name: str
    mask: np.ndarray
    weights: np.ndarray | None = None


def normalized(weights: np.ndarray) -> np.ndarray:
    out = np.asarray(weights, dtype=np.float64)
    out = np.clip(out, 1e-9, None)
    return out * (len(out) / out.sum())


def morgan_fps(smiles: list[str]) -> list:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"invalid SMILES: {smi}")
        fps.append(gen.GetFingerprint(mol))
    return fps


def fp_matrix(fps: list) -> np.ndarray:
    arr = np.zeros((len(fps), 2048), dtype=np.uint8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr


def max_bulk_tanimoto(query: list, refs: list) -> np.ndarray:
    out = np.zeros(len(query), dtype=np.float64)
    for i, fp in enumerate(query):
        sims = DataStructs.BulkTanimotoSimilarity(fp, refs)
        out[i] = max(sims) if sims else 0.0
    return out


def train_test_classifier_weights(
    train_fps: list, test_fps: list, *, kind: str, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, float]:
    X_train = fp_matrix(train_fps).astype(np.float32)
    X_test = fp_matrix(test_fps).astype(np.float32)
    X = np.vstack([X_train, X_test])
    y = np.concatenate(
        [np.zeros(len(X_train), dtype=np.int32), np.ones(len(X_test), dtype=np.int32)]
    )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    folds = np.array_split(perm, 5)
    p_oof = np.zeros(len(y), dtype=np.float64)
    if kind == "logreg":
        params = dict(max_iter=1000, solver="liblinear", C=1.0, random_state=seed)

        def model_factory() -> LogisticRegression:
            return LogisticRegression(**params)

    elif kind == "lgbm":
        params = dict(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            reg_lambda=1.0,
            random_state=seed,
            verbose=-1,
        )

        def model_factory() -> lgb.LGBMClassifier:
            return lgb.LGBMClassifier(**params)

    else:
        raise ValueError(kind)
    for k in range(5):
        va = folds[k]
        tr = np.concatenate([folds[j] for j in range(5) if j != k])
        model = model_factory()
        model.fit(X[tr], y[tr])
        p_oof[va] = model.predict_proba(X[va])[:, 1]
    auc = float(roc_auc_score(y, p_oof))
    full = model_factory()
    full.fit(X, y)
    p_train = full.predict_proba(X_train)[:, 1]
    raw_w = (p_train + 1e-6) / (1.0 - p_train + 1e-6)
    raw_w *= len(X_train) / len(X_test)
    clipped = normalized(np.clip(raw_w, 1.0 / 3.0, 3.0))
    return p_train, clipped, auc


def load_lb_oof_rows() -> pd.DataFrame:
    query = """
        SELECT
            lb.id AS lb_id,
            lb.submission_name,
            lb.file_path,
            lb.experiment_name,
            lb.lb_rank,
            lb.lb_mae,
            lb.lb_rae,
            lb.lb_spearman,
            lb.submitted_at,
            es.id AS experiment_id,
            es.mae_mean AS experiment_mae,
            es.rae_mean AS experiment_rae,
            es.spearman_mean AS experiment_spearman
        FROM lb_submissions lb
        JOIN experiment_summary es
          ON lb.experiment_name = es.name
        WHERE lb.track = 'activity'
          AND lb.lb_mae IS NOT NULL
        ORDER BY lb.id
    """
    rows = pd.read_sql(query, get_engine())
    if rows.empty:
        raise RuntimeError("no LB rows with matching experiment_summary entries")
    return rows


def load_oof(experiment_id: int, n_train: int) -> np.ndarray | None:
    query = """
        SELECT train_idx, oof_prediction
          FROM experiment_oof_predictions
         WHERE experiment_id = %(experiment_id)s
         ORDER BY train_idx
    """
    rows = pd.read_sql(
        query, get_engine(), params={"experiment_id": int(experiment_id)}
    )
    if len(rows) == 0:
        return None
    if len(rows) != n_train:
        return None
    if rows["train_idx"].to_numpy().tolist() != list(range(n_train)):
        return None
    return rows["oof_prediction"].to_numpy(dtype=np.float64)


def reconstruct_weighted_blend_oof(experiment_id: int, n_train: int) -> np.ndarray | None:
    query = """
        SELECT hyperparameters
          FROM experiments
         WHERE id = %(experiment_id)s
    """
    exp = pd.read_sql(query, get_engine(), params={"experiment_id": int(experiment_id)})
    if exp.empty:
        return None
    hp = exp.iloc[0]["hyperparameters"]
    if not isinstance(hp, dict) or "weights" not in hp:
        return None
    weights = hp["weights"]
    if not isinstance(weights, dict) or not weights:
        return None

    pred = np.zeros(n_train, dtype=np.float64)
    total = 0.0
    for member_name, weight in weights.items():
        member = pd.read_sql(
            """
            SELECT id
              FROM experiments
             WHERE name = %(name)s
             ORDER BY id DESC
             LIMIT 1
            """,
            get_engine(),
            params={"name": member_name},
        )
        if member.empty:
            return None
        member_oof = load_oof(int(member.iloc[0]["id"]), n_train)
        if member_oof is None:
            return None
        w = float(weight)
        pred += w * member_oof
        total += w
    if total <= 0.0:
        return None
    return pred / total


def load_oof_or_reconstruct(experiment_id: int, n_train: int) -> np.ndarray | None:
    direct = load_oof(experiment_id, n_train)
    if direct is not None:
        return direct
    return reconstruct_weighted_blend_oof(experiment_id, n_train)


def make_metric_specs(train_df: pd.DataFrame) -> tuple[list[MetricSpec], pd.DataFrame]:
    y = train_df["pec50"].to_numpy(dtype=np.float64)
    train_smiles = train_df["smiles"].tolist()
    test_smiles = load_test_smiles()["smiles"].tolist()
    train_fps = morgan_fps(train_smiles)
    test_fps = morgan_fps(test_smiles)

    nn_test = max_bulk_tanimoto(train_fps, test_fps)
    p_logreg, w_logreg, auc_logreg = train_test_classifier_weights(
        train_fps, test_fps, kind="logreg"
    )
    p_lgbm, w_lgbm, auc_lgbm = train_test_classifier_weights(
        train_fps, test_fps, kind="lgbm"
    )

    selectivity = y - train_df["counter_pec50"].to_numpy(dtype=np.float64)
    potent = (y >= 6.0) & (np.nan_to_num(selectivity, nan=-np.inf) >= 1.5)
    nn_potent = max_bulk_tanimoto(
        train_fps, [fp for fp, keep in zip(train_fps, potent) if keep]
    )
    potent_soft = np.clip((nn_potent - 0.25) / 0.20, 0.0, 1.0)

    ids = pd.read_sql(
        "SELECT compound_id FROM train_activity ORDER BY id", get_engine()
    )["compound_id"].astype(int).tolist()
    lf = pd.read_parquet(LF_PATH).loc[ids].copy()
    lf["lf_mean"] = 0.5 * (lf["log2fc_8p25_pred"] + lf["log2fc_33_pred"])

    specs: list[MetricSpec] = [
        MetricSpec("global_mae", np.ones(len(y), dtype=bool), None),
        MetricSpec(
            "weighted_logreg_testlikeness", np.ones(len(y), dtype=bool), w_logreg
        ),
        MetricSpec("weighted_lgbm_testlikeness", np.ones(len(y), dtype=bool), w_lgbm),
        MetricSpec(
            "weighted_lgbm_x_potent46",
            np.ones(len(y), dtype=bool),
            normalized(w_lgbm * (1.0 + 2.0 * potent_soft)),
        ),
    ]
    for label, values in (
        ("test_nn", nn_test),
        ("logreg_testlike", p_logreg),
        ("lgbm_testlike", p_lgbm),
        ("potent46_nn", nn_potent),
        ("log2fc33_pred", lf["log2fc_33_pred"].to_numpy(dtype=np.float64)),
        ("lf_mean", lf["lf_mean"].to_numpy(dtype=np.float64)),
        ("pec50", y),
    ):
        for q in (0.50, 0.70, 0.80, 0.90):
            specs.append(
                MetricSpec(
                    f"slice_{label}_top{int((1.0 - q) * 100)}",
                    values >= np.quantile(values, q),
                    None,
                )
            )
    diagnostics = pd.DataFrame(
        [
            {"name": "logreg_test_classifier_auc", "value": auc_logreg},
            {"name": "lgbm_test_classifier_auc", "value": auc_lgbm},
            {"name": "n_train", "value": float(len(y))},
            {"name": "n_potent46", "value": float(potent.sum())},
        ]
    )
    return specs, diagnostics


def metric_values(
    y: np.ndarray, pred: np.ndarray, spec: MetricSpec
) -> dict[str, float]:
    mask = spec.mask
    if mask.sum() < 5:
        return {}
    yy = y[mask]
    pp = pred[mask]
    weights = None if spec.weights is None else spec.weights[mask]
    abs_err = np.abs(yy - pp)
    mae = float(np.average(abs_err, weights=weights))
    bias = float(np.average(pp - yy, weights=weights))
    rmse = float(np.sqrt(np.average((yy - pp) ** 2, weights=weights)))
    spearman = float(stats.spearmanr(yy, pp).statistic)
    return {
        f"{spec.name}__mae": mae,
        f"{spec.name}__bias": bias,
        f"{spec.name}__rmse": rmse,
        f"{spec.name}__spearman": spearman,
        f"{spec.name}__n": float(mask.sum()),
    }


def correlation_rows(
    scored: pd.DataFrame, subset_name: str, subset: pd.DataFrame
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    if len(subset) < 5:
        return rows
    metric_cols = [
        col
        for col in scored.columns
        if "__" in col and not col.endswith("__n") and subset[col].notna().sum() >= 5
    ]
    for col in metric_cols:
        values = subset[col].to_numpy(dtype=np.float64)
        valid = np.isfinite(values) & np.isfinite(
            subset["lb_mae"].to_numpy(dtype=np.float64)
        )
        if valid.sum() < 5 or np.allclose(values[valid], values[valid][0]):
            continue
        lb_mae = subset["lb_mae"].to_numpy(dtype=np.float64)[valid]
        lb_rae = subset["lb_rae"].to_numpy(dtype=np.float64)[valid]
        lb_sp = subset["lb_spearman"].to_numpy(dtype=np.float64)[valid]
        rows.append(
            {
                "subset": subset_name,
                "metric": col,
                "n": int(valid.sum()),
                "spearman_vs_lb_mae": float(
                    stats.spearmanr(values[valid], lb_mae).statistic
                ),
                "pearson_vs_lb_mae": float(np.corrcoef(values[valid], lb_mae)[0, 1]),
                "spearman_vs_lb_rae": float(
                    stats.spearmanr(values[valid], lb_rae).statistic
                ),
                "spearman_vs_lb_spearman": float(
                    stats.spearmanr(values[valid], lb_sp).statistic
                ),
            }
        )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df = load_train_smiles_with_counter()
    y = train_df["pec50"].to_numpy(dtype=np.float64)
    specs, diagnostics = make_metric_specs(train_df)
    lb_rows = load_lb_oof_rows()

    scored_rows: list[dict[str, float | str]] = []
    skipped_rows: list[dict[str, float | str]] = []
    for row in lb_rows.itertuples(index=False):
        pred = load_oof_or_reconstruct(int(row.experiment_id), len(y))
        if pred is None:
            skipped_rows.append(
                {
                    "lb_id": int(row.lb_id),
                    "experiment_name": row.experiment_name,
                    "experiment_id": int(row.experiment_id),
                    "reason": "no experiment_oof_predictions rows",
                }
            )
            continue
        scored: dict[str, float | str] = {
            "lb_id": int(row.lb_id),
            "submission_name": row.submission_name,
            "experiment_name": row.experiment_name,
            "experiment_id": int(row.experiment_id),
            "lb_mae": float(row.lb_mae),
            "lb_rae": float(row.lb_rae),
            "lb_spearman": float(row.lb_spearman),
            "experiment_mae": float(row.experiment_mae),
            "experiment_rae": float(row.experiment_rae),
            "experiment_spearman": float(row.experiment_spearman),
            "submitted_at": row.submitted_at,
        }
        for spec in specs:
            scored.update(metric_values(y, pred, spec))
        scored_rows.append(scored)

    scored = pd.DataFrame(scored_rows)
    if scored.empty:
        raise RuntimeError("no LB rows had usable OOF predictions")
    scored.to_csv(OUT_DIR / "oof_metric_battery_scored_submissions.csv", index=False)
    pd.DataFrame(skipped_rows).to_csv(
        OUT_DIR / "oof_metric_battery_skipped.csv", index=False
    )
    diagnostics.to_csv(OUT_DIR / "oof_metric_battery_diagnostics.csv", index=False)

    dedup_latest = (
        scored.sort_values("lb_id")
        .drop_duplicates(["experiment_name"], keep="last")
        .reset_index(drop=True)
    )
    dedup_best = (
        scored.sort_values(["experiment_name", "lb_mae"])
        .drop_duplicates(["experiment_name"], keep="first")
        .reset_index(drop=True)
    )
    corr = pd.DataFrame(
        correlation_rows(scored, "all_matched_rows", scored)
        + correlation_rows(scored, "dedup_latest_experiment", dedup_latest)
        + correlation_rows(scored, "dedup_best_experiment", dedup_best)
    )
    corr["abs_spearman_vs_lb_mae"] = corr["spearman_vs_lb_mae"].abs()
    corr = corr.sort_values(
        ["subset", "spearman_vs_lb_mae", "pearson_vs_lb_mae"],
        ascending=[True, False, False],
    )
    corr.to_csv(OUT_DIR / "oof_metric_battery_correlations.csv", index=False)

    display = corr[
        [
            "subset",
            "metric",
            "n",
            "spearman_vs_lb_mae",
            "pearson_vs_lb_mae",
            "spearman_vs_lb_rae",
            "spearman_vs_lb_spearman",
        ]
    ]
    print("=== Diagnostics ===")
    print(diagnostics.to_markdown(index=False, floatfmt=".6f"))
    print(
        "\n=== Top correlations by subset (higher Spearman vs LB MAE is better for error metrics) ==="
    )
    for subset in display["subset"].unique():
        print(f"\n## {subset}")
        print(
            display[display["subset"] == subset]
            .head(15)
            .to_markdown(index=False, floatfmt=".4f")
        )
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
