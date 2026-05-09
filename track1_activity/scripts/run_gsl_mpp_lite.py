#!/usr/bin/env python
"""GSL-MPP-lite transductive residual smoothing probe for Track 1.

This is a compact PXR adaptation of the GSL-MPP idea. It builds a
train+test molecule similarity graph, propagates anchor residuals over that
inter-molecule graph in a cross-fit way, and writes small clipped corrections
for leaderboard-safe inspection.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS, load_test_smiles, load_train_smiles_target  # noqa: E402
from evaluate import (  # noqa: E402
    compute_metrics,
    record_experiment,
    save_oof_predictions,
)
from gsl_mpp import (  # noqa: E402
    apply_residual_correction,
    morgan_bit_matrix,
    propagate_residuals,
    tanimoto_similarity,
    topk_row_normalized_adjacency,
)
from splits import umap_split_indices  # noqa: E402

SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
OUTPUT_ROOT = REPO_ROOT.joinpath(
    "track1_activity", "analysis", "gsl_mpp_lite", "outputs"
)
DEFAULT_ID55 = SUBMISSION_DIR.joinpath("ens_id51_top500_potent46_t40_soft_g35.csv")
DEFAULT_CARUANA = SUBMISSION_DIR.joinpath("ens_caruana_bag20.csv")


def load_caruana_oof() -> tuple[np.ndarray, int, dict[str, float]]:
    """Reconstruct latest ``ens_caruana_bag20`` OOF from stored weights."""
    with psycopg2.connect(**DB_PARAMS) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, hyperparameters FROM experiments
             WHERE name = 'ens_caruana_bag20'
             ORDER BY id DESC LIMIT 1
            """
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("No ens_caruana_bag20 experiment found in DB")
        exp_id, hp = row
        if isinstance(hp, str):
            hp = json.loads(hp)
        weights_map = {k: float(v) for k, v in hp["weights"].items()}

        oof = None
        for name, weight in weights_map.items():
            cur.execute(
                "SELECT id FROM experiments WHERE name = %s ORDER BY id DESC LIMIT 1",
                (name,),
            )
            member_row = cur.fetchone()
            if member_row is None:
                raise RuntimeError(f"Missing member experiment: {name}")
            member_id = member_row[0]
            cur.execute(
                """
                SELECT train_idx, oof_prediction
                  FROM experiment_oof_predictions
                 WHERE experiment_id = %s
                 ORDER BY train_idx
                """,
                (member_id,),
            )
            rows = cur.fetchall()
            if not rows:
                raise RuntimeError(f"Missing OOF predictions for {name} id={member_id}")
            member = np.asarray([r[1] for r in rows], dtype=np.float64)
            if oof is None:
                oof = np.zeros_like(member)
            oof += weight * member

    if oof is None:
        raise RuntimeError("Failed to reconstruct ens_caruana_bag20 OOF")
    return oof / sum(weights_map.values()), int(exp_id), weights_map


def load_anchor_test_csv(path: Path, expected_rows: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df) != expected_rows:
        raise RuntimeError(f"{path} has {len(df)} rows; expected {expected_rows}")
    if "pEC50" not in df.columns:
        raise RuntimeError(f"{path} has no pEC50 column")
    return df


def parse_grid(value: str, cast):
    return [cast(item) for item in value.split(",") if item != ""]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="initial")
    parser.add_argument(
        "--anchor-test-csv",
        type=Path,
        default=DEFAULT_ID55 if DEFAULT_ID55.exists() else DEFAULT_CARUANA,
        help="CSV whose pEC50 column receives the final test correction.",
    )
    parser.add_argument("--k-grid", default="8,16,32")
    parser.add_argument("--alpha-grid", default="0.5,0.85")
    parser.add_argument("--gamma-grid", default="-0.5,-0.25,0.25,0.5")
    parser.add_argument("--clip-grid", default="0.03,0.06")
    parser.add_argument("--n-iter", type=int, default=50)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--record-best", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    return parser


def summarize_shift(candidate: np.ndarray, anchor: np.ndarray) -> dict[str, float]:
    shift = candidate - anchor
    abs_shift = np.abs(shift)
    return {
        "test_mean_shift": float(shift.mean()),
        "test_mean_abs_shift": float(abs_shift.mean()),
        "test_p90_abs_shift": float(np.quantile(abs_shift, 0.9)),
        "test_max_abs_shift": float(abs_shift.max()),
        "test_gt_005": int((abs_shift > 0.05).sum()),
        "test_gt_010": int((abs_shift > 0.10).sum()),
        "test_gt_020": int((abs_shift > 0.20).sum()),
    }


def run_preflight(candidate_path: Path, run_dir: Path, name: str) -> str:
    if not DEFAULT_ID55.exists():
        return "SKIP: id55 anchor CSV missing"
    cmd = [
        "pixi",
        "run",
        "python",
        "track1_activity/scripts/submission_preflight.py",
        "--candidate",
        str(candidate_path),
        "--anchor",
        str(DEFAULT_ID55),
        "--name",
        f"gsl_mpp_lite_{name}_vs_id55",
    ]
    proc = subprocess.run(  # noqa: S603
        cmd,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    log_path = run_dir.joinpath(f"preflight_{name}.log")
    log_path.write_text(proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0:
        return f"ERROR: see {log_path}"
    return f"OK: see {log_path}"


def main() -> None:
    args = build_parser().parse_args()
    run_dir = OUTPUT_ROOT.joinpath(args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    k_grid = parse_grid(args.k_grid, int)
    alpha_grid = parse_grid(args.alpha_grid, float)
    gamma_grid = parse_grid(args.gamma_grid, float)
    clip_grid = parse_grid(args.clip_grid, float)

    print("Loading train/test and anchor predictions ...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y = train_df["pec50"].to_numpy(dtype=np.float64)
    n_train = len(train_df)
    n_test = len(test_df)
    anchor_oof, caruana_id, weights_map = load_caruana_oof()
    anchor_test_df = load_anchor_test_csv(args.anchor_test_csv, n_test)
    anchor_test = anchor_test_df["pEC50"].to_numpy(dtype=np.float64)

    anchor_metrics = compute_metrics(y, anchor_oof)
    print(
        f"Anchor OOF from ens_caruana_bag20 id={caruana_id}: "
        f"MAE={anchor_metrics['MAE']:.6f} Sp={anchor_metrics['Spearman_R']:.6f}"
    )
    print(f"Anchor test CSV: {args.anchor_test_csv}")

    smiles_all = train_df["smiles"].tolist() + test_df["smiles"].tolist()
    print("Building Morgan fingerprints and full molecule similarity graph ...")
    fps = morgan_bit_matrix(smiles_all)
    sim = tanimoto_similarity(fps, fps)

    print("Building canonical UMAP folds ...")
    folds = umap_split_indices(train_df["smiles"].tolist())
    base_residual = y - anchor_oof

    rows: list[dict[str, object]] = []
    candidate_payloads: dict[str, tuple[np.ndarray, np.ndarray, dict[str, object]]] = {}

    for k in k_grid:
        print(f"Propagating residuals for k={k} ...")
        adj = topk_row_normalized_adjacency(sim, k=k, include_self=False)
        final_seed = np.zeros(n_train + n_test, dtype=np.float64)
        final_seed[:n_train] = base_residual
        final_labeled = np.zeros(n_train + n_test, dtype=bool)
        final_labeled[:n_train] = True

        for alpha in alpha_grid:
            oof_resid = np.zeros(n_train, dtype=np.float64)
            for train_idx, val_idx in folds:
                seed = np.zeros(n_train + n_test, dtype=np.float64)
                labeled = np.zeros(n_train + n_test, dtype=bool)
                seed[train_idx] = base_residual[train_idx]
                labeled[train_idx] = True
                propagated = propagate_residuals(
                    adj,
                    seed,
                    labeled,
                    alpha=alpha,
                    n_iter=args.n_iter,
                    clamp_labeled=True,
                )
                oof_resid[val_idx] = propagated[val_idx]

            test_resid = propagate_residuals(
                adj,
                final_seed,
                final_labeled,
                alpha=alpha,
                n_iter=args.n_iter,
                clamp_labeled=True,
            )[n_train:]

            for gamma in gamma_grid:
                for clip in clip_grid:
                    oof_pred = apply_residual_correction(
                        anchor_oof, oof_resid, gamma=gamma, clip=clip
                    )
                    test_pred = apply_residual_correction(
                        anchor_test, test_resid, gamma=gamma, clip=clip
                    )
                    metrics = compute_metrics(y, oof_pred)
                    label = f"k{k}_a{alpha:g}_g{gamma:g}_c{clip:g}".replace(
                        ".", "p"
                    ).replace("-", "m")
                    row: dict[str, object] = {
                        "label": label,
                        "k": k,
                        "alpha": alpha,
                        "gamma": gamma,
                        "clip": clip,
                        "oof_mae": metrics["MAE"],
                        "oof_delta_mae": metrics["MAE"] - anchor_metrics["MAE"],
                        "oof_spearman": metrics["Spearman_R"],
                        "oof_delta_spearman": metrics["Spearman_R"]
                        - anchor_metrics["Spearman_R"],
                        "oof_resid_std": float(oof_resid.std()),
                        "test_resid_std": float(test_resid.std()),
                    }
                    row.update(summarize_shift(test_pred, anchor_test))
                    rows.append(row)
                    candidate_payloads[label] = (oof_pred, test_pred, row)

    summary = pd.DataFrame(rows).sort_values(
        ["oof_delta_mae", "oof_delta_spearman"], ascending=[True, False]
    )
    summary_path = run_dir.joinpath("summary.csv")
    summary.to_csv(summary_path, index=False)

    selected = summary.head(args.max_candidates)
    preflight_results: list[str] = []
    for _, row in selected.iterrows():
        label = str(row["label"])
        _oof_pred, test_pred, _payload = candidate_payloads[label]
        out_df = anchor_test_df.copy()
        out_df["pEC50"] = test_pred
        out_name = f"ens_gsl_mpp_lite_{args.run_name}_{label}.csv"
        out_path = SUBMISSION_DIR.joinpath(out_name)
        out_df.to_csv(out_path, index=False)
        if args.skip_preflight:
            preflight_results.append(f"{label}: SKIP")
        else:
            preflight_results.append(
                f"{label}: {run_preflight(out_path, run_dir, label)}"
            )

    best_label = str(selected.iloc[0]["label"])
    best_oof, _best_test, best_row = candidate_payloads[best_label]
    if args.record_best:
        exp_id = record_experiment(
            name=f"gsl_mpp_lite_{args.run_name}_{best_label}",
            description="GSL-MPP-lite residual smoothing around ens_caruana_bag20 OOF",
            model_type="gsl_mpp_lite",
            feature_set="morgan_r2_tanimoto_inter_molecule_graph",
            hyperparameters={
                "source_ensemble": "ens_caruana_bag20",
                "source_ensemble_id": caruana_id,
                "source_weights": weights_map,
                "anchor_test_csv": str(args.anchor_test_csv),
                "best": best_row,
            },
            fold_metrics=[compute_metrics(y, best_oof)],
            notes="Diagnostic GSL-MPP-lite residual smoother; do not submit without preflight review.",
            on_conflict_replace=True,
        )
        save_oof_predictions(exp_id, best_oof)

    report_path = run_dir.joinpath("report.md")
    report_path.write_text(
        "\n".join(
            [
                "# GSL-MPP-Lite Report",
                "",
                f"Run name: `{args.run_name}`",
                f"Source OOF ensemble: `ens_caruana_bag20` id `{caruana_id}`",
                f"Anchor test CSV: `{args.anchor_test_csv}`",
                "",
                "## Anchor OOF",
                "",
                f"MAE: `{anchor_metrics['MAE']:.6f}`",
                f"Spearman: `{anchor_metrics['Spearman_R']:.6f}`",
                "",
                "## Best candidates by OOF MAE delta",
                "",
                selected.to_markdown(index=False),
                "",
                "## Preflight",
                "",
                *[f"- {item}" for item in preflight_results],
                "",
                "## Read",
                "",
                "Treat this as a transductive molecule-graph diagnostic, not an automatic submission decision. ",
                "The OOF anchor is reconstructed from `ens_caruana_bag20`; if the test anchor is id55, ",
                "OOF and test anchors are intentionally not identical because id55 was a CSV-only perturbation.",
            ]
        )
        + "\n"
    )
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")
    print(
        selected[
            ["label", "oof_delta_mae", "oof_delta_spearman", "test_mean_abs_shift"]
        ]
    )


if __name__ == "__main__":
    main()
