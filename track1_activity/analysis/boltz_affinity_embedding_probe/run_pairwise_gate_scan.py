#!/usr/bin/env -S pixi run python
"""Scan pairwise Boltz-derived scores as sparse id55 gates.

The score is transductive over the 513 Track 1 test rows: train a pairwise
LightGBM model on original train_activity pairs, predict all test-test pairs,
then score each compound by its mean probability of beating the other test
compounds. AS1 labels are used only for replaying gate choices.

Experiment only; no submission files are written.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "outputs_pairwise_gate"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

import run_pairwise_probe  # noqa: E402
import run_probe  # noqa: E402
from data import DB_PARAMS, load_test_smiles, load_train_smiles_target  # noqa: E402


SHIFTS = [-0.3, -0.2, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.2, 0.3]
HIGH_QS = [0.75, 0.8, 0.85, 0.9, 0.95]
LOW_QS = [0.05, 0.1, 0.15, 0.2, 0.25]
ANCHOR = (
    REPO_ROOT / "track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv"
)


def load_test_metadata() -> pd.DataFrame:
    sql = """
    SELECT
        (t.id - 1)::int AS test_idx,
        t.compound_id,
        c.smiles,
        c.molecule_name,
        l.pec50 AS as1_pec50
    FROM test_activity t
    JOIN compounds c ON c.id = t.compound_id
    LEFT JOIN test_activity_phase1_labels l ON l.compound_id = t.compound_id
    ORDER BY t.id
    """
    with psycopg2.connect(**DB_PARAMS) as conn:
        return pd.read_sql(sql, conn)


def load_anchor(test: pd.DataFrame, anchor_path: Path) -> np.ndarray:
    sub = pd.read_csv(anchor_path)
    if len(sub) != len(test):
        raise ValueError(f"Anchor length {len(sub)} != test length {len(test)}")
    return sub["pEC50"].to_numpy(dtype=np.float64)


def load_feature_for_test(
    feature: run_probe.FeatureConfig, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    train_ids = run_probe.load_ids("train_activity")
    test_ids = test["compound_id"].astype(int).tolist()
    if feature.kind == "affinity":
        return run_probe.load_affinity_feature(feature.name, train_ids, test_ids)

    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    X_train, X_test = run_probe.run_train.load_features(feature.name, train_df, test_df)
    return X_train.astype(np.float32), X_test.astype(np.float32), train_ids


def pairwise_test_score(clf, X_test: np.ndarray) -> np.ndarray:
    n = len(X_test)
    i, j = np.triu_indices(n, k=1)
    X_pair = run_pairwise_probe.pair_matrix(X_test, i, j)
    prob_left = clf.predict_proba(X_pair)[:, 1]
    wins = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    np.add.at(wins, i, prob_left)
    np.add.at(wins, j, 1.0 - prob_left)
    np.add.at(counts, i, 1.0)
    np.add.at(counts, j, 1.0)
    return wins / counts


def score_feature(
    feature: run_probe.FeatureConfig,
    args: argparse.Namespace,
    test: pd.DataFrame,
    y_by_compound: dict[int, float],
    out_dir: Path,
) -> pd.DataFrame:
    print(f"\n=== {feature.name} ===")
    X_train, X_test, train_ids = load_feature_for_test(feature, test)
    X_train, X_test = run_probe.finite_impute(X_train, X_test)
    y_train = np.asarray(
        [y_by_compound[int(cid)] for cid in train_ids], dtype=np.float32
    )
    pair_i, pair_j = run_pairwise_probe.sample_train_pairs(
        y_train,
        n_pairs=args.n_pairs,
        seed=args.seed,
        min_abs_delta=args.min_abs_delta,
    )
    y_delta_train = (y_train[pair_i] - y_train[pair_j]).astype(np.float32)
    X_pair_train = run_pairwise_probe.pair_matrix(X_train, pair_i, pair_j)
    clf, reg, val_metrics = run_pairwise_probe.train_pair_models(
        X_pair_train, y_delta_train, args.seed
    )
    del reg, X_pair_train
    score = pairwise_test_score(clf, X_test)
    out = test.copy()
    out["feature"] = feature.name
    out["pairwise_win_score"] = score
    for key, value in val_metrics.items():
        out[key] = value
    out.to_csv(out_dir / f"{feature.name}__test_scores.csv", index=False)
    return out


def metric(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y
    return {
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "spearman": float(stats.spearmanr(y, pred).statistic),
    }


def scan_gates(scores: pd.DataFrame, anchor: np.ndarray) -> pd.DataFrame:
    df = scores.copy()
    df["anchor"] = anchor
    as1 = df[df["as1_pec50"].notna()].copy()
    y = as1["as1_pec50"].to_numpy(dtype=np.float64)
    base = as1["anchor"].to_numpy(dtype=np.float64)
    base_metrics = metric(y, base)
    rows = []
    for mode, quantiles in [("high_lift", HIGH_QS), ("low_drop", LOW_QS)]:
        for q in quantiles:
            threshold = float(df["pairwise_win_score"].quantile(q))
            if mode == "high_lift":
                flag_all = df["pairwise_win_score"] >= threshold
                allowed_shifts = [s for s in SHIFTS if s > 0]
            else:
                flag_all = df["pairwise_win_score"] <= threshold
                allowed_shifts = [s for s in SHIFTS if s < 0]
            flag_as1 = flag_all[df["as1_pec50"].notna()].to_numpy()
            for shift in allowed_shifts:
                pred = base.copy()
                pred[flag_as1] += shift
                m = metric(y, pred)
                flagged_as1 = as1.loc[flag_as1]
                rows.append(
                    {
                        "feature": str(df["feature"].iloc[0]),
                        "mode": mode,
                        "quantile": q,
                        "threshold": threshold,
                        "shift": shift,
                        "as1_mae": m["mae"],
                        "as1_bias": m["bias"],
                        "as1_spearman": m["spearman"],
                        "delta_mae_vs_anchor": m["mae"] - base_metrics["mae"],
                        "n_flags_all": int(flag_all.sum()),
                        "n_flags_as1": int(flag_as1.sum()),
                        "n_flags_as2": int(flag_all.sum() - flag_as1.sum()),
                        "n_true_high_flags_as1": int(
                            (flagged_as1["as1_pec50"] >= 6.0).sum()
                        ),
                        "n_true_low_flags_as1": int(
                            (flagged_as1["as1_pec50"] < 3.0).sum()
                        ),
                        "anchor_as1_mae": base_metrics["mae"],
                        "anchor_as1_bias": base_metrics["bias"],
                        "anchor_as1_spearman": base_metrics["spearman"],
                    }
                )
    return pd.DataFrame(rows).sort_values(["as1_mae", "n_flags_as2"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "feature", nargs="*", default=["pooled_boltz_allpairs", "boltz_affinity_g1g2"]
    )
    parser.add_argument("--n-pairs", type=int, default=150_000)
    parser.add_argument("--min-abs-delta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = OUT_DIR / (
        f"pairs{args.n_pairs}_mindelta{str(args.min_abs_delta).replace('.', 'p')}_seed{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    test = load_test_metadata()
    anchor = load_anchor(test, ANCHOR)
    y_by_compound = run_pairwise_probe.build_y_by_compound()
    selected = run_probe.choose_features(args.feature)
    scans = []
    for feature in selected:
        score_path = out_dir / f"{feature.name}__test_scores.csv"
        if score_path.exists() and not args.force:
            scores = pd.read_csv(score_path)
        else:
            scores = score_feature(feature, args, test, y_by_compound, out_dir)
        scan = scan_gates(scores, anchor)
        scan.to_csv(out_dir / f"{feature.name}__gate_scan.csv", index=False)
        scans.append(scan)
        print(f"\nBest {feature.name}")
        print(
            scan.head(12)[
                [
                    "feature",
                    "mode",
                    "quantile",
                    "shift",
                    "as1_mae",
                    "delta_mae_vs_anchor",
                    "n_flags_as1",
                    "n_flags_as2",
                    "n_true_high_flags_as1",
                    "n_true_low_flags_as1",
                ]
            ].to_string(index=False)
        )
    combined = pd.concat(scans, ignore_index=True).sort_values(
        ["as1_mae", "n_flags_as2"]
    )
    combined.to_csv(out_dir / "gate_scan.csv", index=False)
    metadata = {
        "n_pairs": args.n_pairs,
        "min_abs_delta": args.min_abs_delta,
        "seed": args.seed,
        "anchor": str(ANCHOR.relative_to(REPO_ROOT)),
        "features": [asdict(f) for f in selected],
        "note": "Experiment only; no submission files or experiment DB rows written.",
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Pairwise Score Gate Scan",
        "",
        "Pairwise win-score gates replayed against id55 AS1. AS1 labels are used only for gate scan readback.",
        "",
        combined.head(20)[
            [
                "feature",
                "mode",
                "quantile",
                "threshold",
                "shift",
                "as1_mae",
                "delta_mae_vs_anchor",
                "n_flags_as1",
                "n_flags_as2",
                "n_true_high_flags_as1",
                "n_true_low_flags_as1",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
