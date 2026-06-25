#!/usr/bin/env -S pixi run python
"""Prototype Boltz-style intra-assay pairwise ranking for Phase 2 tail scores.

This is deliberately lightweight: it trains an antisymmetric pairwise classifier
from same-assay activity differences, then scores challenge compounds by
comparing them to assay reference compounds. It is a diagnostic, not a
submission generator.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score
from sqlalchemy import create_engine

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "track1_activity" / "src"))
sys.path.insert(0, str(REPO_ROOT / "track1_activity" / "analysis" / "chembl_pxr_probe"))

from chembl_pxr_activation_probe import filter_activation_ec50, tanimoto_matrix  # noqa: E402
from data import get_engine  # noqa: E402
from splits import _morgan_fp_matrix  # noqa: E402

OUT_ROOT = Path(__file__).resolve().parent / "outputs" / "pairwise_assay_rank"
DOC_PATH = REPO_ROOT / "docs" / "track1_explain" / "phase2_pairwise_assay_rank.md"
CHEMBL_URL = "postgresql+psycopg2:///chembl_36?host=/tmp&port=5433"
POOL_FOLDS_PATH = (
    REPO_ROOT
    / "track1_activity"
    / "analysis"
    / "phase2_validation_matrix"
    / "outputs"
    / "phase2_labeled_pool_with_folds.csv"
)
RISK_MAP_PATH = (
    REPO_ROOT
    / "track1_activity"
    / "analysis"
    / "phase2_as2_risk_map"
    / "outputs"
    / "all_test_risk_map.csv"
)


def load_pool() -> pd.DataFrame:
    return pd.read_csv(POOL_FOLDS_PATH)


def load_test() -> pd.DataFrame:
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


def load_challenge_inchikeys() -> set[str]:
    keys = pd.read_sql(
        """
        SELECT d.inchikey
        FROM compounds c
        LEFT JOIN compound_descriptors d ON d.compound_id = c.id
        WHERE c.id IN (
            SELECT compound_id FROM train_activity
            UNION
            SELECT compound_id FROM test_activity
        )
        """,
        get_engine(),
    )["inchikey"].dropna()
    return set(keys)


def quality_filter_assays(
    rows: pd.DataFrame,
    min_assay_n: int,
    min_std: float,
    min_unique: int,
) -> pd.DataFrame:
    summary = (
        rows.groupby("assay_id")
        .agg(
            n=("compound_key", "nunique"),
            std=("value", "std"),
            nunique=("value", "nunique"),
        )
        .reset_index()
    )
    keep = summary[
        summary["n"].ge(min_assay_n)
        & summary["std"].ge(min_std)
        & summary["nunique"].ge(min_unique)
    ]["assay_id"]
    return rows[rows["assay_id"].isin(set(keep))].reset_index(drop=True)


def load_chembl_rows(args: argparse.Namespace) -> pd.DataFrame:
    chembl = create_engine(CHEMBL_URL)
    raw = pd.read_sql(
        """
        SELECT
            s.standard_inchi_key AS inchikey,
            s.canonical_smiles AS smiles,
            m.chembl_id AS mol_chembl_id,
            a.chembl_id AS assay_chembl_id,
            a.assay_type,
            a.confidence_score,
            a.description,
            act.standard_type,
            act.pchembl_value
        FROM compound_structures s
        JOIN molecule_dictionary m ON m.molregno = s.molregno
        JOIN activities act ON act.molregno = m.molregno
        JOIN assays a ON a.assay_id = act.assay_id
        JOIN target_dictionary t ON t.tid = a.tid
        WHERE t.chembl_id = 'CHEMBL3401'
          AND s.canonical_smiles IS NOT NULL
          AND act.pchembl_value IS NOT NULL
        """,
        chembl,
    )
    if args.chembl_scope == "activation":
        raw = filter_activation_ec50(raw)
    elif args.chembl_scope == "all_pxr":
        raw = raw[raw["confidence_score"].ge(8)].copy()
    else:
        raise ValueError(args.chembl_scope)

    challenge_keys = load_challenge_inchikeys()
    raw = raw[~raw["inchikey"].isin(challenge_keys)].copy()
    rows = (
        raw.groupby(["assay_chembl_id", "inchikey", "smiles", "mol_chembl_id"], as_index=False)
        .agg(value=("pchembl_value", "median"))
        .rename(columns={"assay_chembl_id": "assay_id"})
    )
    rows["source"] = "chembl"
    rows["compound_id"] = -1
    rows["compound_key"] = "chembl:" + rows["inchikey"].astype(str)
    rows["molecule_name"] = rows["mol_chembl_id"]
    return quality_filter_assays(rows, args.min_assay_n, args.min_std, args.min_unique)


def load_htchem_rows(args: argparse.Namespace) -> pd.DataFrame:
    rows = pd.read_sql(
        """
        SELECT
            h.compound_id,
            c.molecule_name,
            COALESCE(c.std_smiles, c.smiles) AS smiles,
            h.source_type,
            h.corrected_pec50 AS value
        FROM htchem_activity h
        JOIN compounds c ON c.id = h.compound_id
        WHERE h.corrected_pec50 IS NOT NULL
        """,
        get_engine(),
    )
    rows["assay_id"] = "htchem_" + rows["source_type"].astype(str)
    rows["source"] = "htchem"
    rows["compound_key"] = "compound:" + rows["compound_id"].astype(str)
    rows["inchikey"] = ""
    return quality_filter_assays(rows, args.min_assay_n, args.min_std, args.min_unique)


def load_single_conc_rows(args: argparse.Namespace) -> pd.DataFrame:
    rows = pd.read_sql(
        """
        SELECT
            sc.compound_id,
            c.molecule_name,
            c.std_smiles AS smiles,
            sc.concentration_m,
            sc.log2_fc_estimate AS value
        FROM single_concentration sc
        JOIN compounds c ON c.id = sc.compound_id
        WHERE sc.log2_fc_estimate IS NOT NULL
        """,
        get_engine(),
    )
    rows = (
        rows.groupby(["compound_id", "molecule_name", "smiles", "concentration_m"], as_index=False)
        .agg(value=("value", "median"))
    )
    rows["assay_id"] = rows["concentration_m"].map(lambda x: f"single_conc_{x:.4g}")
    rows["source"] = "single_conc"
    rows["compound_key"] = "compound:" + rows["compound_id"].astype(str)
    rows["inchikey"] = ""
    return quality_filter_assays(rows, args.min_assay_n, args.min_std, args.min_unique)


def load_counter_rows(args: argparse.Namespace) -> pd.DataFrame:
    rows = pd.read_sql(
        """
        SELECT
            ca.compound_id,
            c.molecule_name,
            c.std_smiles AS smiles,
            ca.pec50 AS value
        FROM counter_assay ca
        JOIN compounds c ON c.id = ca.compound_id
        WHERE ca.pec50 IS NOT NULL
        """,
        get_engine(),
    )
    rows["assay_id"] = "counter_pec50"
    rows["source"] = "counter"
    rows["compound_key"] = "compound:" + rows["compound_id"].astype(str)
    rows["inchikey"] = ""
    return quality_filter_assays(rows, args.min_assay_n, args.min_std, args.min_unique)


def load_assay_rows(args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    sources = {x.strip() for x in args.sources.split(",") if x.strip()}
    if "chembl" in sources:
        frames.append(load_chembl_rows(args))
    if "htchem" in sources:
        frames.append(load_htchem_rows(args))
    if "single_conc" in sources:
        frames.append(load_single_conc_rows(args))
    if "counter" in sources:
        frames.append(load_counter_rows(args))
    if not frames:
        raise ValueError("No assay sources selected.")
    rows = pd.concat(frames, ignore_index=True)
    rows = rows.dropna(subset=["smiles", "value"]).reset_index(drop=True)
    rows["row_id"] = np.arange(len(rows), dtype=np.int64)
    return rows


def assay_summary(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["source", "assay_id"])
        .agg(
            n=("compound_key", "nunique"),
            mean=("value", "mean"),
            std=("value", "std"),
            min=("value", "min"),
            max=("value", "max"),
            nunique=("value", "nunique"),
        )
        .reset_index()
        .sort_values(["source", "n"], ascending=[True, False])
    )


def sample_pairs(
    rows: pd.DataFrame,
    fp: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(args.seed)
    pair_x: list[np.ndarray] = []
    pair_y: list[np.ndarray] = []
    pair_w: list[np.ndarray] = []
    source_rows = []

    for (source, assay_id), sub in rows.groupby(["source", "assay_id"], sort=False):
        idx = sub.index.to_numpy(dtype=np.int64)
        values = sub["value"].to_numpy(dtype=np.float32)
        n = len(idx)
        if n < 2:
            continue
        pairs: list[tuple[int, int, float]] = []
        all_possible = n * (n - 1) // 2
        if all_possible <= args.max_pairs_per_assay:
            for a in range(n):
                for b in range(a + 1, n):
                    delta = float(values[a] - values[b])
                    if abs(delta) >= args.min_pair_delta:
                        pairs.append((a, b, delta))
        else:
            seen: set[tuple[int, int]] = set()
            trials = 0
            max_trials = args.max_pairs_per_assay * 20
            while len(pairs) < args.max_pairs_per_assay and trials < max_trials:
                a, b = rng.choice(n, size=2, replace=False)
                if a > b:
                    a, b = b, a
                if (a, b) in seen:
                    trials += 1
                    continue
                seen.add((a, b))
                delta = float(values[a] - values[b])
                if abs(delta) >= args.min_pair_delta:
                    pairs.append((a, b, delta))
                trials += 1
        if not pairs:
            continue
        if len(pairs) > args.max_pairs_per_assay:
            pair_idx = rng.choice(len(pairs), size=args.max_pairs_per_assay, replace=False)
            pairs = [pairs[i] for i in pair_idx]

        x_parts = []
        y_parts = []
        w_parts = []
        for a, b, delta in pairs:
            ia = idx[a]
            ib = idx[b]
            x_ab = fp[ia].astype(np.float32) - fp[ib].astype(np.float32)
            y_ab = 1 if delta > 0 else 0
            weight = min(abs(delta), args.max_pair_weight)
            x_parts.append(x_ab)
            y_parts.append(y_ab)
            w_parts.append(weight)
            if args.add_reverse_pairs:
                x_parts.append(-x_ab)
                y_parts.append(1 - y_ab)
                w_parts.append(weight)
        pair_x.append(np.vstack(x_parts).astype(np.float32))
        pair_y.append(np.asarray(y_parts, dtype=np.int8))
        pair_w.append(np.asarray(w_parts, dtype=np.float32))
        source_rows.append(
            {
                "source": source,
                "assay_id": assay_id,
                "n_compounds": n,
                "n_unordered_pairs": len(pairs),
                "n_training_rows": len(y_parts),
                "value_std": float(np.std(values, ddof=1)),
            }
        )

    if not pair_x:
        raise RuntimeError("No pairwise training examples were generated.")
    return (
        np.vstack(pair_x),
        np.concatenate(pair_y),
        np.concatenate(pair_w),
        pd.DataFrame(source_rows),
    )


def make_pairwise_model(args: argparse.Namespace) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_lambda=1.0,
        random_state=args.seed,
        verbose=-1,
    )


def topk_indices(sim: np.ndarray, k: int) -> np.ndarray:
    k_eff = min(k, sim.shape[1])
    idx = np.argpartition(sim, kth=sim.shape[1] - k_eff, axis=1)[:, -k_eff:]
    row = np.arange(sim.shape[0])[:, None]
    order = np.argsort(-sim[row, idx], axis=1)
    return idx[row, order]


def score_against_refs(
    model: lgb.LGBMClassifier,
    query: pd.DataFrame,
    query_fp: np.ndarray,
    refs: pd.DataFrame,
    ref_fp: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    if refs.empty:
        return np.full(len(query), np.nan, dtype=np.float32)
    scores = np.zeros(len(query), dtype=np.float64)
    for start in range(0, len(query), args.score_chunk_size):
        end = min(start + args.score_chunk_size, len(query))
        sim = tanimoto_matrix(query_fp[start:end], ref_fp)
        idx = topk_indices(sim, args.score_top_k)
        q_compounds = query["compound_id"].iloc[start:end].to_numpy(dtype=np.int64)[:, None]
        r_compounds = refs["compound_id"].to_numpy(dtype=np.int64)[idx]
        valid = (r_compounds < 0) | (r_compounds != q_compounds)
        pair_features = []
        weights = []
        row_slices = []
        for local_i in range(end - start):
            ref_idx = idx[local_i][valid[local_i]]
            if len(ref_idx) == 0:
                ref_idx = idx[local_i]
            diff = query_fp[start + local_i].astype(np.float32) - ref_fp[ref_idx].astype(np.float32)
            pair_features.append(diff)
            sim_values = sim[local_i, ref_idx].astype(np.float64)
            weights.append(np.maximum(sim_values, 1e-3))
            row_slices.append(len(ref_idx))
        x_pair = np.vstack(pair_features)
        proba = model.predict_proba(x_pair)[:, 1]
        offset = 0
        for local_i, n_ref in enumerate(row_slices):
            p = proba[offset : offset + n_ref]
            w = weights[local_i]
            scores[start + local_i] = float(np.average(p, weights=w))
            offset += n_ref
    return scores.astype(np.float32)


def score_queries(
    model: lgb.LGBMClassifier,
    rows: pd.DataFrame,
    refs_fp: np.ndarray,
    pool: pd.DataFrame,
    test: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool_fp = _morgan_fp_matrix(pool["smiles"].tolist()).astype(np.int8)
    test_fp = _morgan_fp_matrix(test["smiles"].tolist()).astype(np.int8)

    pool_scores = pool[["pool_idx", "compound_id", "molecule_name", "pec50", "source", "true_bin"]].copy()
    test_scores = test[["test_id", "compound_id", "molecule_name", "as1_pec50"]].copy()
    test_scores["split"] = np.where(test_scores["as1_pec50"].notna(), "AS1", "AS2")

    for source in ["all", *sorted(rows["source"].unique())]:
        refs = rows if source == "all" else rows[rows["source"].eq(source)]
        ref_fp = refs_fp[refs.index.to_numpy(dtype=np.int64)]
        pool_scores[f"pairrank_{source}"] = score_against_refs(
            model, pool, pool_fp, refs.reset_index(drop=True), ref_fp, args
        )
        test_scores[f"pairrank_{source}"] = score_against_refs(
            model, test, test_fp, refs.reset_index(drop=True), ref_fp, args
        )
    return pool_scores, test_scores


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def evaluate_scores(pool_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    score_cols = [c for c in pool_scores.columns if c.startswith("pairrank_")]
    for col in score_cols:
        for slice_name, mask in {
            "all": np.ones(len(pool_scores), dtype=bool),
            "source_train": pool_scores["source"].eq("train").to_numpy(),
            "source_as1": pool_scores["source"].eq("as1").to_numpy(),
        }.items():
            sub = pool_scores.loc[mask, ["pec50", "true_bin", col]].dropna()
            if len(sub) < 3:
                continue
            high = sub["true_bin"].eq("gte6").to_numpy(dtype=int)
            low = sub["true_bin"].eq("lt3").to_numpy(dtype=int)
            score = sub[col].to_numpy(dtype=float)
            row = {
                "score": col,
                "slice": slice_name,
                "n": int(len(sub)),
                "spearman_pec50": safe_spearman(score, sub["pec50"].to_numpy(dtype=float)),
                "mean_score": float(np.mean(score)),
            }
            if high.min() != high.max():
                row["gte6_auc"] = float(roc_auc_score(high, score))
                row["gte6_ap"] = float(average_precision_score(high, score))
            if low.min() != low.max():
                row["lt3_auc"] = float(roc_auc_score(low, -score))
                row["lt3_ap"] = float(average_precision_score(low, -score))
            rows.append(row)
    return pd.DataFrame(rows)


def gate_scan(pool_scores: pd.DataFrame) -> pd.DataFrame:
    risk = pd.read_csv(RISK_MAP_PATH)[
        ["compound_id", "molecule_name", "as1_pec50", "pred_id55"]
    ]
    as1 = pool_scores[pool_scores["source"].eq("as1")].merge(
        risk, on=["compound_id", "molecule_name"], how="left"
    )
    rows = []
    score_cols = [c for c in pool_scores.columns if c.startswith("pairrank_")]
    for col in score_cols:
        score = as1[col].to_numpy(dtype=float)
        y = as1["pec50"].to_numpy(dtype=float)
        base = as1["pred_id55"].to_numpy(dtype=float)
        true_high = as1["true_bin"].eq("gte6").to_numpy()
        true_low = as1["true_bin"].eq("lt3").to_numpy()
        for q in [0.75, 0.80, 0.85, 0.90, 0.95]:
            threshold = float(np.nanquantile(score, q))
            flag = score >= threshold
            for shift in [0.05, 0.10, 0.15, 0.20, 0.30]:
                pred = base + flag.astype(float) * shift
                rows.append(
                    {
                        "score": col,
                        "mode": "high_lift",
                        "quantile": q,
                        "threshold": threshold,
                        "shift": shift,
                        "as1_mae": float(np.mean(np.abs(pred - y))),
                        "n_flags": int(flag.sum()),
                        "n_true_high_flags": int((flag & true_high).sum()),
                        "n_true_low_flags": int((flag & true_low).sum()),
                    }
                )
        for q in [0.05, 0.10, 0.15, 0.20, 0.25]:
            threshold = float(np.nanquantile(score, q))
            flag = score <= threshold
            for shift in [-0.05, -0.10, -0.15, -0.20, -0.30]:
                pred = base + flag.astype(float) * shift
                rows.append(
                    {
                        "score": col,
                        "mode": "low_drop",
                        "quantile": q,
                        "threshold": threshold,
                        "shift": shift,
                        "as1_mae": float(np.mean(np.abs(pred - y))),
                        "n_flags": int(flag.sum()),
                        "n_true_high_flags": int((flag & true_high).sum()),
                        "n_true_low_flags": int((flag & true_low).sum()),
                    }
                )
    return pd.DataFrame(rows).sort_values(["as1_mae", "score", "mode"])


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    assay_info: pd.DataFrame,
    pair_info: pd.DataFrame,
    summary: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    best_summary = summary.sort_values(["slice", "gte6_ap"], ascending=[True, False])
    lines = [
        "# Phase 2 pairwise assay-rank probe",
        "",
        "Prototype of Boltz-style same-assay pairwise learning. The model is trained",
        "to classify which compound is stronger within the same assay, then challenge",
        "compounds are scored by pairwise comparison to assay reference compounds.",
        "",
        "## Config",
        "",
        f"- Sources: `{args.sources}`",
        f"- ChEMBL scope: `{args.chembl_scope}`",
        f"- Min assay n/std/unique: `{args.min_assay_n}` / `{args.min_std}` / `{args.min_unique}`",
        f"- Max pairs per assay: `{args.max_pairs_per_assay}`",
        f"- Score top-k refs: `{args.score_top_k}`",
        "",
        "## Assay Sources",
        "",
        assay_info.groupby("source")
        .agg(assays=("assay_id", "nunique"), rows=("n", "sum"), mean_n=("n", "mean"))
        .reset_index()
        .to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Pair Training Rows",
        "",
        pair_info.groupby("source")
        .agg(
            assays=("assay_id", "nunique"),
            training_rows=("n_training_rows", "sum"),
            mean_compounds=("n_compounds", "mean"),
        )
        .reset_index()
        .to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Score Evaluation",
        "",
        best_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Best AS1 Gate Rows",
        "",
        gates.head(20).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Read",
        "",
        "This is a diagnostic scalar. A useful result would show stronger AS1 gte6 AP",
        "or a gate that improves id55 AS1 without broad shifts.",
        "",
        "## Generated Files",
        "",
        f"- `{out_dir.relative_to(REPO_ROOT) / 'pool_pairrank_scores.csv'}`",
        f"- `{out_dir.relative_to(REPO_ROOT) / 'test_pairrank_scores.csv'}`",
        f"- `{out_dir.relative_to(REPO_ROOT) / 'score_summary.csv'}`",
        f"- `{out_dir.relative_to(REPO_ROOT) / 'gate_scan.csv'}`",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        default="chembl,htchem,single_conc",
        help="Comma-separated sources: chembl,htchem,single_conc,counter",
    )
    parser.add_argument("--chembl-scope", choices=["activation", "all_pxr"], default="activation")
    parser.add_argument("--min-assay-n", type=int, default=5)
    parser.add_argument("--min-std", type=float, default=0.25)
    parser.add_argument("--min-unique", type=int, default=5)
    parser.add_argument("--min-pair-delta", type=float, default=0.25)
    parser.add_argument("--max-pair-weight", type=float, default=2.0)
    parser.add_argument("--max-pairs-per-assay", type=int, default=2500)
    parser.add_argument("--add-reverse-pairs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-estimators", type=int, default=450)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--min-child-samples", type=int, default=30)
    parser.add_argument("--score-top-k", type=int, default=64)
    parser.add_argument("--score-chunk-size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def config_name(args: argparse.Namespace) -> str:
    source_name = args.sources.replace(",", "_").replace(" ", "")
    name = f"{args.chembl_scope}_{source_name}_mpa{args.max_pairs_per_assay}_top{args.score_top_k}"
    if args.seed != 42:
        name += f"_seed{args.seed}"
    return name


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    )
    args = parse_args()
    out_dir = OUT_ROOT / config_name(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_assay_rows(args)
    rows.to_csv(out_dir / "assay_rows.csv", index=False)
    assay_info = assay_summary(rows)
    assay_info.to_csv(out_dir / "assay_summary.csv", index=False)
    print("Assay summary")
    print(assay_info.groupby("source").agg(assays=("assay_id", "nunique"), rows=("n", "sum")).to_string())

    ref_fp = _morgan_fp_matrix(rows["smiles"].tolist()).astype(np.int8)
    x_pair, y_pair, w_pair, pair_info = sample_pairs(rows, ref_fp, args)
    pair_info.to_csv(out_dir / "pair_training_summary.csv", index=False)
    print(f"Pair matrix: {x_pair.shape}, positives={int(y_pair.sum())}")

    model = make_pairwise_model(args)
    model.fit(x_pair, y_pair, sample_weight=w_pair)

    pool = load_pool()
    test = load_test()
    pool_scores, test_scores = score_queries(model, rows, ref_fp, pool, test, args)
    pool_scores.to_csv(out_dir / "pool_pairrank_scores.csv", index=False)
    test_scores.to_csv(out_dir / "test_pairrank_scores.csv", index=False)

    summary = evaluate_scores(pool_scores)
    gates = gate_scan(pool_scores)
    summary.to_csv(out_dir / "score_summary.csv", index=False)
    gates.to_csv(out_dir / "gate_scan.csv", index=False)
    (out_dir / "metadata.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.seed == 42:
        write_report(out_dir, args, assay_info, pair_info, summary, gates)

    print("\nScore summary")
    print(summary.to_string(index=False))
    print("\nBest gates")
    print(gates.head(12).to_string(index=False))
    print(f"\nWrote {out_dir}")
    if args.seed == 42:
        print(f"Wrote {DOC_PATH}")


if __name__ == "__main__":
    main()
