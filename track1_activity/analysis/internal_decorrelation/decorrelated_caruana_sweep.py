#!/usr/bin/env python
"""No-external-data decorrelated Caruana sweep for Track 1.

This is an analysis-only candidate builder. It avoids the failed residual
correction pattern by building full internal-model ensembles, calibrating them
with the existing importance affine recipe, and only then making small blends
against the LB-proven id48 meta-axis anchor.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

from data import DB_PARAMS, load_test_smiles, load_train_smiles_target  # noqa: E402
from evaluate import load_oof_predictions  # noqa: E402
from run_ensemble import optimize_caruana  # noqa: E402
from run_ensemble_calibrate_importance import morgan_matrix  # noqa: E402

OUT_DIR = REPO_ROOT.joinpath(
    "track1_activity", "analysis", "internal_decorrelation"
).joinpath("outputs", "decorrelated_caruana_sweep")
SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")

ID48_ALPHA = 0.343
ID48_PATH = REPO_ROOT.joinpath(
    "track1_activity", "analysis", "compound_level_lb"
).joinpath("outputs", "meta_axis_candidates", "ens_meta_axis_a343.csv")
BASELINE9_PATH = SUBMISSION_DIR.joinpath(
    "ens_caruana_bag20_calibrated_importance_baseline_9pool.csv"
)
META42_PATH = SUBMISSION_DIR.joinpath(
    "ens_caruana_bag20_calibrated_importance_meta_id42.csv"
)

EXTERNAL_OR_FAILED_MARKERS = (
    "admet",
    "drugclip",
    "oe_",
    "openeye",
    "resid_",
    "potent_relu",
    "no_aux",
    "trial11",
)


@dataclass(frozen=True)
class Candidate:
    name: str
    exp_id: int
    mae: float
    spearman: float
    submission_path: Path
    oof: np.ndarray
    test: np.ndarray


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def spearman(y: np.ndarray, pred: np.ndarray) -> float:
    return float(stats.spearmanr(y, pred).statistic)


def residual_corr(y: np.ndarray, ref: np.ndarray, cand: np.ndarray) -> float:
    return float(np.corrcoef(y - ref, y - cand)[0, 1])


def load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Molecule Name" not in df.columns or "pEC50" not in df.columns:
        raise ValueError(f"not a Track 1 submission CSV: {path}")
    return df


def fit_importance_weights() -> np.ndarray:
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    x_train = morgan_matrix(train_df["smiles"].tolist())
    x_test = morgan_matrix(test_df["smiles"].tolist())
    x_all = np.vstack([x_train, x_test])
    y_all = np.concatenate(
        [np.zeros(len(x_train), dtype=np.int32), np.ones(len(x_test), dtype=np.int32)]
    )
    clf = LogisticRegression(max_iter=1000, solver="liblinear", C=1.0, random_state=42)
    clf.fit(x_all, y_all)
    p_test = clf.predict_proba(x_train)[:, 1]
    eps = 1e-6
    weights = (p_test + eps) / (1.0 - p_test + eps)
    weights *= len(x_train) / len(x_test)
    weights = np.clip(weights, 1.0 / 3.0, 3.0)
    return weights * (len(weights) / weights.sum())


def apply_importance_affine(
    y: np.ndarray,
    oof: np.ndarray,
    test: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    reg = LinearRegression()
    reg.fit(oof.reshape(-1, 1), y, sample_weight=weights)
    slope = float(reg.coef_[0])
    intercept = float(reg.intercept_)
    return slope * oof + intercept, slope * test + intercept, slope, intercept


def load_member_oof(name: str) -> np.ndarray:
    with psycopg2.connect(**DB_PARAMS) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM experiments WHERE name = %s ORDER BY id DESC LIMIT 1",
            (name,),
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError(f"missing experiment: {name}")
        pred = load_oof_predictions(int(row[0]))
    if pred is None:
        raise RuntimeError(f"missing OOF for {name}")
    return pred.astype(np.float64)


def reconstruct_latest_caruana_oof() -> np.ndarray:
    with psycopg2.connect(**DB_PARAMS) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT hyperparameters
            FROM experiments
            WHERE name = 'ens_caruana_bag20'
            ORDER BY id DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
    if row is None:
        raise RuntimeError("missing ens_caruana_bag20")
    weights = row[0]["weights"]
    out = None
    total = 0.0
    for name, weight in weights.items():
        pred = load_member_oof(name)
        if out is None:
            out = np.zeros_like(pred, dtype=np.float64)
        out += float(weight) * pred
        total += float(weight)
    if out is None or total <= 0:
        raise RuntimeError("invalid caruana weights")
    return out / total


def reconstruct_meta42_oof() -> np.ndarray:
    """Reconstruct the id42 family-meta 7-pool OOF from the archived weights."""
    weights = {
        "tabpfn_chemprop_family_meta_umap": 0.539,
        "tabpfn_kermt_pretrain_embed_umap_default": 0.148,
        "tabpfn_attentivefp_pretrain_embed_umap_default": 0.076,
        "tabpfn_gatedgcn_pretrain_embed_umap_default": 0.067,
        "tabpfn_pooled_boltz_allpairs_umap_default": 0.065,
        "tabpfn_pooled_boltz_umap_default": 0.056,
        "tabpfn_molformer_c3_pretrain_embed_umap": 0.049,
    }
    out = None
    total = 0.0
    for name, weight in weights.items():
        pred = load_member_oof(name)
        if out is None:
            out = np.zeros_like(pred, dtype=np.float64)
        out += float(weight) * pred
        total += float(weight)
    if out is None:
        raise RuntimeError("failed to reconstruct meta42 OOF")
    return out / total


def build_id48_anchor_oof(
    y: np.ndarray,
    importance_weights: np.ndarray,
) -> np.ndarray:
    base_raw = reconstruct_latest_caruana_oof()
    meta_raw = reconstruct_meta42_oof()
    base_test = load_submission(BASELINE9_PATH)["pEC50"].to_numpy(dtype=np.float64)
    meta_test = load_submission(META42_PATH)["pEC50"].to_numpy(dtype=np.float64)
    base_cal, _, _, _ = apply_importance_affine(
        y, base_raw, base_test, importance_weights
    )
    meta_cal, _, _, _ = apply_importance_affine(
        y, meta_raw, meta_test, importance_weights
    )
    return (1.0 - ID48_ALPHA) * base_cal + ID48_ALPHA * meta_cal


def query_candidate_rows(max_mae: float) -> list[tuple[int, str, float, float, str]]:
    with psycopg2.connect(**DB_PARAMS) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT e.id, e.name, es.mae_mean::float, es.spearman_mean::float,
                   e.submission_path
            FROM experiments e
            JOIN experiment_summary es ON es.id = e.id
            JOIN experiment_oof_predictions o ON o.experiment_id = e.id
            WHERE e.submission_path IS NOT NULL
              AND e.name NOT LIKE 'ens_%%'
              AND es.mae_mean <= %s
            GROUP BY e.id, e.name, es.mae_mean, es.spearman_mean, e.submission_path
            HAVING count(o.train_idx) = 4140
            ORDER BY es.mae_mean ASC
            """,
            (max_mae,),
        )
        return cur.fetchall()


def is_internal_candidate(name: str) -> bool:
    lower = name.lower()
    return not any(marker in lower for marker in EXTERNAL_OR_FAILED_MARKERS)


def load_candidates(y: np.ndarray, max_mae: float = 0.50) -> list[Candidate]:
    template = load_submission(ID48_PATH)
    candidates: list[Candidate] = []
    seen_names: set[str] = set()
    for exp_id, name, exp_mae, exp_sp, rel_path in query_candidate_rows(max_mae):
        if name in seen_names or not is_internal_candidate(name):
            continue
        path = REPO_ROOT.joinpath(rel_path)
        if not path.exists():
            continue
        sub = load_submission(path)
        if len(sub) != len(template):
            continue
        if not (
            sub["Molecule Name"].to_numpy() == template["Molecule Name"].to_numpy()
        ).all():
            continue
        oof = load_oof_predictions(int(exp_id))
        if oof is None or len(oof) != len(y):
            continue
        candidates.append(
            Candidate(
                name=name,
                exp_id=int(exp_id),
                mae=float(exp_mae),
                spearman=float(exp_sp),
                submission_path=path,
                oof=oof.astype(np.float64),
                test=sub["pEC50"].to_numpy(dtype=np.float64),
            )
        )
        seen_names.add(name)
    return candidates


def candidate_matrix(
    candidates: list[Candidate],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = [c.name for c in candidates]
    return (
        names,
        np.column_stack([c.oof for c in candidates]),
        np.column_stack([c.test for c in candidates]),
    )


def run_sweep() -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    y = load_train_smiles_target()["pec50"].to_numpy(dtype=np.float64)
    id48_sub = load_submission(ID48_PATH)
    id48_test = id48_sub["pEC50"].to_numpy(dtype=np.float64)

    iw = fit_importance_weights()
    id48_oof = build_id48_anchor_oof(y, iw)
    id48_mae = mae(y, id48_oof)
    id48_sp = spearman(y, id48_oof)

    all_candidates = load_candidates(y, max_mae=0.50)
    rows: list[dict[str, object]] = []
    materialized: list[dict[str, object]] = []
    for corr_cap in (0.78, 0.82, 0.86, 0.90, 0.94, 1.01):
        pool = [
            cand
            for cand in all_candidates
            if abs(residual_corr(y, id48_oof, cand.oof)) <= corr_cap
        ]
        if len(pool) < 4:
            continue
        names, oof_matrix, test_matrix = candidate_matrix(pool)
        for bag_frac in (0.35, 0.50, 0.70):
            for n_bags in (20, 40):
                for init_top_n in (1, 3):
                    weights = optimize_caruana(
                        oof_matrix,
                        y,
                        n_iter=120,
                        init_top_n=init_top_n,
                        bag_frac=bag_frac,
                        n_bags=n_bags,
                        seed=42,
                    )
                    raw_oof = oof_matrix @ weights
                    raw_test = test_matrix @ weights
                    cal_oof, cal_test, slope, intercept = apply_importance_affine(
                        y, raw_oof, raw_test, iw
                    )
                    nonzero = [(n, float(w)) for n, w in zip(names, weights) if w > 0]
                    top_weights = sorted(nonzero, key=lambda item: -item[1])[:8]
                    for blend_lambda in (0.10, 0.20, 0.30, 0.40):
                        blend_oof = (
                            1.0 - blend_lambda
                        ) * id48_oof + blend_lambda * cal_oof
                        blend_test = (
                            1.0 - blend_lambda
                        ) * id48_test + blend_lambda * cal_test
                        delta_test = blend_test - id48_test
                        rec = {
                            "corr_cap": corr_cap,
                            "bag_frac": bag_frac,
                            "n_bags": n_bags,
                            "init_top_n": init_top_n,
                            "blend_lambda": blend_lambda,
                            "pool_size": len(pool),
                            "nonzero_weights": len(nonzero),
                            "candidate_raw_mae": mae(y, raw_oof),
                            "candidate_cal_mae": mae(y, cal_oof),
                            "candidate_cal_sp": spearman(y, cal_oof),
                            "candidate_resid_r_vs_id48": residual_corr(
                                y, id48_oof, cal_oof
                            ),
                            "id48_proxy_mae": id48_mae,
                            "id48_proxy_sp": id48_sp,
                            "blend_proxy_mae": mae(y, blend_oof),
                            "blend_proxy_sp": spearman(y, blend_oof),
                            "delta_proxy_mae": mae(y, blend_oof) - id48_mae,
                            "delta_proxy_sp": spearman(y, blend_oof) - id48_sp,
                            "test_mean_shift_vs_id48": float(delta_test.mean()),
                            "test_mean_abs_shift_vs_id48": float(
                                np.abs(delta_test).mean()
                            ),
                            "test_p90_abs_shift_vs_id48": float(
                                np.quantile(np.abs(delta_test), 0.90)
                            ),
                            "test_max_abs_shift_vs_id48": float(
                                np.abs(delta_test).max()
                            ),
                            "test_pearson_vs_id48": float(
                                np.corrcoef(id48_test, blend_test)[0, 1]
                            ),
                            "top_weights_json": json.dumps(top_weights),
                        }
                        rows.append(rec)

                        conservative = (
                            rec["delta_proxy_mae"] <= -0.0002
                            and rec["test_mean_abs_shift_vs_id48"] <= 0.040
                            and rec["test_p90_abs_shift_vs_id48"] <= 0.085
                            and rec["test_pearson_vs_id48"] >= 0.997
                        )
                        if conservative:
                            stem = (
                                "ens_internal_decor"
                                f"_cap{int(round(corr_cap * 100)):03d}"
                                f"_bf{int(round(bag_frac * 100)):02d}"
                                f"_b{n_bags}"
                                f"_i{init_top_n}"
                                f"_l{int(round(blend_lambda * 100)):02d}"
                            )
                            out_path = SUBMISSION_DIR.joinpath(f"{stem}.csv")
                            out = id48_sub.copy()
                            out["pEC50"] = blend_test
                            out.to_csv(out_path, index=False)
                            rec_with_path = {
                                **rec,
                                "submission_path": str(out_path.relative_to(REPO_ROOT)),
                            }
                            materialized.append(rec_with_path)

    summary = pd.DataFrame(rows).sort_values(
        ["delta_proxy_mae", "test_mean_abs_shift_vs_id48"], ascending=[True, True]
    )
    summary.to_csv(OUT_DIR.joinpath("summary.csv"), index=False)
    mat = pd.DataFrame(materialized).sort_values(
        ["delta_proxy_mae", "test_mean_abs_shift_vs_id48"], ascending=[True, True]
    )
    mat.to_csv(OUT_DIR.joinpath("materialized_candidates.csv"), index=False)

    report_lines = [
        "# Internal Decorrelated Caruana Sweep",
        "",
        f"Loaded internal candidates: `{len(all_candidates)}`",
        f"id48 OOF proxy MAE: `{id48_mae:.5f}`",
        f"id48 OOF proxy Spearman: `{id48_sp:.5f}`",
        "",
        "## Top Proxy Rows",
        "",
        summary.head(25).to_markdown(index=False, floatfmt=".5f")
        if not summary.empty
        else "(none)",
        "",
        "## Materialized Conservative Candidates",
        "",
        mat.head(20).to_markdown(index=False, floatfmt=".5f")
        if not mat.empty
        else "(none)",
    ]
    OUT_DIR.joinpath("report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    result = run_sweep()
    print(result.head(25).to_string(index=False))
