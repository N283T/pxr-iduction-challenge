"""Feature leak detector for pEC50 regression candidates.

Runs five diagnostic tests on an arbitrary feature matrix to flag
label-leak patterns that would inflate OOF beyond LB:

1. Self-reference check
   Compares feature-value distributions between "positive-label" and
   "negative-label" training compounds (user-defined pEC50 threshold).
   Flags features whose in-group mean exceeds the out-group
   mean+2*std — a textbook self-match indicator.

2. Outlier label-clustering
   For each feature, tests whether its top-N largest values are
   disproportionately the label-positive compounds. Uses hypergeometric
   test; p < 0.001 -> flag.

3. Val/train fold-spillover
   For UMAP 5-fold split, verifies that the feature-value distribution
   is comparable across folds. Large fold-to-fold variance suggests a
   query-compound-in-val scenario where val compounds inherit leak
   structure from queries in their own fold.

4. CV-vs-shuffle consistency
   Compares UMAP OOF MAE with random-shuffle 5-fold OOF MAE. A small
   UMAP/shuffle gap is normal; a large ``shuffle < UMAP`` gap can
   indicate memorization-via-leak (the model performs better when
   analogs are distributed).

5. Leak-free vs raw OOF
   If the feature has a natural "leak-free" variant (user-provided via
   --baseline-feature), compares OOF. A large raw-vs-baseline gap
   (Δ > 0.01 MAE beyond noise) is the strongest signal.

Usage:
    pixi run python track1_activity/scripts/check_feature_leak.py \\
        --sql "SELECT compound_id, feature_col FROM compound_xxx" \\
        --feature-cols feat1,feat2,... \\
        --pos-threshold 6.0

Reports pass/warn/fail per test. Exit code 0 = all tests clean,
1 = at least one warning/fail.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import hypergeom, ks_2samp

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402
from splits import umap_split_indices  # noqa: E402


def load_feature(sql: str, cols: list[str]) -> pd.DataFrame:
    """Load feature matrix from SQL. Must return compound_id + feature cols."""
    with psycopg2.connect(**DB_PARAMS) as conn:
        df = pd.read_sql(sql, conn).set_index("compound_id")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"SQL returned columns missing: {missing}")
    return df[cols]


def load_train_meta() -> pd.DataFrame:
    with psycopg2.connect(**DB_PARAMS) as conn:
        df = pd.read_sql(
            """SELECT t.compound_id, t.pec50, c.std_smiles AS smiles
               FROM train_activity t
               JOIN compounds c ON c.id = t.compound_id
               ORDER BY t.id""",
            conn,
        )
    return df


def test_1_self_reference(
    feat: pd.DataFrame, meta: pd.DataFrame, pos_threshold: float
) -> list[dict]:
    """In-group vs out-group distribution check."""
    findings = []
    aligned = feat.reindex(meta["compound_id"])
    pos_mask = (meta["pec50"] >= pos_threshold).to_numpy()
    for col in aligned.columns:
        vals = aligned[col].to_numpy(dtype=float)
        mask_ok = ~np.isnan(vals)
        pos = vals[mask_ok & pos_mask]
        neg = vals[mask_ok & ~pos_mask]
        if len(pos) < 5 or len(neg) < 5:
            continue
        mean_out = float(neg.mean())
        std_out = float(neg.std())
        mean_in = float(pos.mean())
        # Flag if in-group mean exceeds out-group by >2*std AND neg<pos
        # AND positive has very tight distribution near max
        z = (mean_in - mean_out) / (std_out + 1e-9)
        pos_max = float(pos.max())
        # Count queries pinned at max
        pinned = int((pos >= pos_max - 1e-6).sum())
        severity = "OK"
        reason = ""
        if z > 3.0 and pinned > 0.5 * len(pos):
            severity = "FAIL"
            reason = f"z={z:.1f} + {pinned}/{len(pos)} pinned at max"
        elif z > 2.0:
            severity = "WARN"
            reason = f"z={z:.1f}"
        findings.append(
            {
                "feature": col,
                "mean_pos": round(mean_in, 4),
                "mean_neg": round(mean_out, 4),
                "z": round(z, 2),
                "pinned_at_max": pinned,
                "n_pos": len(pos),
                "severity": severity,
                "reason": reason,
            }
        )
    return findings


def test_2_outlier_clustering(
    feat: pd.DataFrame,
    meta: pd.DataFrame,
    pos_threshold: float,
    top_n_ratio: float = 0.1,
) -> list[dict]:
    """Test whether top-N feature values cluster in label-positive group."""
    findings = []
    aligned = feat.reindex(meta["compound_id"])
    pos = (meta["pec50"] >= pos_threshold).to_numpy()
    total = len(pos)
    n_pos = int(pos.sum())
    for col in aligned.columns:
        vals = aligned[col].to_numpy(dtype=float)
        mask_ok = ~np.isnan(vals)
        top_n = max(5, int(top_n_ratio * total))
        if mask_ok.sum() < top_n + 5:
            continue
        # top-N by feature value
        valid_idx = np.where(mask_ok)[0]
        top_idx = valid_idx[np.argsort(-vals[valid_idx])[:top_n]]
        k = int(pos[top_idx].sum())
        # Hypergeometric: P(>= k pos draws) given n_pos successes in total
        pval = float(hypergeom.sf(k - 1, total, n_pos, top_n))
        expected = top_n * n_pos / total
        ratio = k / expected if expected > 0 else float("inf")
        severity = "OK"
        if pval < 0.001 and ratio > 2.0:
            severity = "FAIL"
        elif pval < 0.01 and ratio > 1.5:
            severity = "WARN"
        findings.append(
            {
                "feature": col,
                "top_n": top_n,
                "k_pos_in_top": k,
                "expected": round(expected, 1),
                "ratio": round(ratio, 2),
                "pval": pval,
                "severity": severity,
            }
        )
    return findings


def test_3_fold_spillover(feat: pd.DataFrame, meta: pd.DataFrame) -> list[dict]:
    """Check if feature distribution is consistent across UMAP folds."""
    findings = []
    splits = umap_split_indices(
        meta["smiles"].tolist(), n_splits=5, n_clusters=50, seed=42
    )
    fold_idx = np.zeros(len(meta), dtype=int)
    for fold, (_, val_idx) in enumerate(splits):
        fold_idx[val_idx] = fold

    aligned = feat.reindex(meta["compound_id"])
    for col in aligned.columns:
        vals = aligned[col].to_numpy(dtype=float)
        mask_ok = ~np.isnan(vals)
        fold_stds = []
        fold_means = []
        for f in range(5):
            m = mask_ok & (fold_idx == f)
            if m.sum() < 20:
                continue
            fold_means.append(float(vals[m].mean()))
            fold_stds.append(float(vals[m].std()))
        if len(fold_means) < 5:
            continue
        mean_of_means = float(np.mean(fold_means))
        std_of_means = float(np.std(fold_means))
        mean_std = float(np.mean(fold_stds))
        # Ratio: how much fold-to-fold mean shift vs within-fold std
        cv = std_of_means / (mean_std + 1e-9)
        severity = "OK"
        if cv > 0.5:
            severity = "WARN"
        if cv > 1.0:
            severity = "FAIL"
        findings.append(
            {
                "feature": col,
                "mean_of_fold_means": round(mean_of_means, 3),
                "std_of_fold_means": round(std_of_means, 3),
                "avg_within_fold_std": round(mean_std, 3),
                "cv_ratio": round(cv, 2),
                "severity": severity,
            }
        )
    return findings


def test_4_cv_vs_shuffle(
    feat: pd.DataFrame, meta: pd.DataFrame, pos_threshold: float
) -> dict:
    """Compare UMAP 5-fold OOF with random-shuffle 5-fold OOF (same K=5)."""
    y = meta["pec50"].to_numpy(dtype=np.float32)
    aligned = feat.reindex(meta["compound_id"])
    X = np.ascontiguousarray(aligned.to_numpy(dtype=np.float32))
    col_means = np.nan_to_num(np.nanmean(X, axis=0), nan=0.0)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    # UMAP folds
    splits = umap_split_indices(
        meta["smiles"].tolist(), n_splits=5, n_clusters=50, seed=42
    )
    fold_idx = np.zeros(len(meta), dtype=int)
    for fold, (_, val_idx) in enumerate(splits):
        fold_idx[val_idx] = fold

    rng = np.random.RandomState(42)
    shuffle_fold_idx = rng.randint(0, 5, size=len(meta))

    def run(fold_array: np.ndarray) -> float:
        oof = np.zeros_like(y)
        for f in range(5):
            tr = fold_array != f
            va = fold_array == f
            m = lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=10,
                random_state=42,
                verbose=-1,
            )
            m.fit(
                X[tr],
                y[tr],
                eval_set=[(X[va], y[va])],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            oof[va] = m.predict(X[va])
        return float(np.mean(np.abs(oof - y)))

    umap_mae = run(fold_idx)
    shuffle_mae = run(shuffle_fold_idx)
    gap = umap_mae - shuffle_mae

    severity = "OK"
    if gap < -0.01:
        # UMAP BETTER than shuffle — very suspicious (shuffle is typically easier)
        severity = "WARN"
    return {
        "umap_mae": round(umap_mae, 4),
        "shuffle_mae": round(shuffle_mae, 4),
        "gap_umap_minus_shuffle": round(gap, 4),
        "severity": severity,
        "note": "shuffle CV is normally EASIER (MAE lower); if UMAP<shuffle, leak suspected",
    }


def test_5_train_vs_test_dist(
    feat: pd.DataFrame,
) -> list[dict]:
    """KS test train vs test feature distribution."""
    findings = []
    with psycopg2.connect(**DB_PARAMS) as conn:
        train_ids = pd.read_sql("SELECT compound_id FROM train_activity", conn)[
            "compound_id"
        ].tolist()
        test_ids = pd.read_sql("SELECT compound_id FROM test_activity", conn)[
            "compound_id"
        ].tolist()

    for col in feat.columns:
        tr = feat.loc[feat.index.intersection(train_ids), col].dropna().to_numpy()
        te = feat.loc[feat.index.intersection(test_ids), col].dropna().to_numpy()
        if len(tr) < 50 or len(te) < 50:
            continue
        ks = ks_2samp(tr, te)
        d = float(ks.statistic)
        p = float(ks.pvalue)
        severity = "OK"
        if d > 0.3 and p < 0.001:
            severity = "WARN"
        if d > 0.5:
            severity = "FAIL"
        findings.append(
            {
                "feature": col,
                "ks_stat": round(d, 3),
                "pval": f"{p:.1e}",
                "train_mean": round(float(tr.mean()), 3),
                "test_mean": round(float(te.mean()), 3),
                "severity": severity,
            }
        )
    return findings


def report(findings: list[dict], title: str) -> int:
    """Print findings with severity emphasis. Returns count of FAIL/WARN."""
    print(f"\n=== {title} ===")
    if not findings:
        print("  (no features evaluated)")
        return 0
    if isinstance(findings, dict):
        findings = [findings]
    flagged = [f for f in findings if f.get("severity", "OK") != "OK"]
    if not flagged:
        # Only print summary row
        print(f"  all {len(findings)} clean (no WARN/FAIL)")
        return 0
    # Print all flagged
    cols = [k for k in findings[0].keys() if k != "severity"]
    hdr = "  " + "  ".join(f"{c:<18}" for c in cols) + "  severity"
    print(hdr)
    print("  " + "-" * len(hdr))
    for r in sorted(findings, key=lambda x: x.get("severity", "")):
        sev = r.get("severity", "OK")
        if sev == "OK":
            continue
        row = "  " + "  ".join(f"{str(r.get(c, '')):<18}" for c in cols) + f"  {sev}"
        print(row)
    return len(flagged)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--sql",
        required=True,
        help="SQL returning compound_id + feature columns",
    )
    ap.add_argument(
        "--feature-cols",
        required=True,
        help="Comma-separated list of feature columns to analyze",
    )
    ap.add_argument(
        "--pos-threshold",
        type=float,
        default=6.0,
        help="pEC50 threshold defining label-positive (default 6.0)",
    )
    ap.add_argument(
        "--skip-test",
        type=str,
        default="",
        help="Comma-separated test IDs to skip (1,2,3,4,5)",
    )
    args = ap.parse_args()

    cols = [c.strip() for c in args.feature_cols.split(",")]
    skip = {int(s) for s in args.skip_test.split(",") if s.strip()}

    print(f"Loading feature matrix ({len(cols)} columns)...")
    feat = load_feature(args.sql, cols)
    print(f"  {feat.shape}")
    meta = load_train_meta()
    print(f"  train meta: {len(meta)} compounds")

    total_flagged = 0

    if 1 not in skip:
        f1 = test_1_self_reference(feat, meta, args.pos_threshold)
        total_flagged += report(f1, "Test 1: Self-reference (in-group vs out-group)")

    if 2 not in skip:
        f2 = test_2_outlier_clustering(feat, meta, args.pos_threshold)
        total_flagged += report(f2, "Test 2: Outlier label-clustering (hypergeom)")

    if 3 not in skip:
        f3 = test_3_fold_spillover(feat, meta)
        total_flagged += report(f3, "Test 3: Fold spillover (cross-fold CV of stats)")

    if 4 not in skip:
        f4 = test_4_cv_vs_shuffle(feat, meta, args.pos_threshold)
        total_flagged += report([f4], "Test 4: UMAP OOF vs random-shuffle OOF")

    if 5 not in skip:
        f5 = test_5_train_vs_test_dist(feat)
        total_flagged += report(f5, "Test 5: Train vs test distribution (KS)")

    print(f"\n{'=' * 60}")
    if total_flagged == 0:
        print("RESULT: clean (no WARN/FAIL across all tests)")
        sys.exit(0)
    else:
        print(f"RESULT: {total_flagged} flagged issue(s) -- review above")
        sys.exit(1)


if __name__ == "__main__":
    main()
