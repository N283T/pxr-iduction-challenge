#!/usr/bin/env python
"""Learned GSL-MPP-style molecule graph residual probe for Track 1."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS, load_test_smiles, load_train_smiles_target  # noqa: E402
from evaluate import compute_metrics, record_experiment, save_oof_predictions  # noqa: E402
from gsl_mpp import (  # noqa: E402
    apply_residual_correction,
    morgan_bit_matrix,
    tanimoto_similarity,
)
from gsl_mpp_torch import build_topk_adjacency_torch, fit_dense_gsl_mpp  # noqa: E402
from splits import umap_split_indices  # noqa: E402

SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
OUTPUT_ROOT = REPO_ROOT.joinpath(
    "track1_activity", "analysis", "gsl_mpp_learned", "outputs"
)
DEFAULT_ID55 = SUBMISSION_DIR.joinpath("ens_id51_top500_potent46_t40_soft_g35.csv")
DEFAULT_CARUANA = SUBMISSION_DIR.joinpath("ens_caruana_bag20.csv")


def parse_grid(value: str, cast):
    return [cast(item) for item in value.split(",") if item != ""]


def load_caruana_oof() -> tuple[np.ndarray, int, dict[str, float]]:
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
            cur.execute(
                """
                SELECT oof_prediction
                  FROM experiment_oof_predictions
                 WHERE experiment_id = %s
                 ORDER BY train_idx
                """,
                (member_row[0],),
            )
            rows = cur.fetchall()
            if not rows:
                raise RuntimeError(f"Missing OOF predictions for {name}")
            member = np.asarray([r[0] for r in rows], dtype=np.float64)
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


def build_node_features(
    fps: np.ndarray,
    anchor_oof: np.ndarray,
    anchor_test: np.ndarray,
    svd_components: int,
    seed: int,
) -> np.ndarray:
    n_components = min(svd_components, fps.shape[0] - 1, fps.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    fp_features = svd.fit_transform(fps.astype(np.float32))
    anchor = np.concatenate([anchor_oof, anchor_test]).reshape(-1, 1)
    features = np.column_stack([fp_features, anchor])
    return StandardScaler().fit_transform(features).astype(np.float32)


def summarize_shift(
    candidate: np.ndarray, anchor: np.ndarray
) -> dict[str, float | int]:
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
        f"gsl_mpp_learned_{name}_vs_id55",
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    log_path = run_dir.joinpath(f"preflight_{name}.log")
    log_path.write_text(proc.stdout.rstrip() + "\n" + proc.stderr.rstrip() + "\n")
    if proc.returncode != 0:
        return f"ERROR: see {log_path}"
    return f"OK: see {log_path}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="initial")
    parser.add_argument(
        "--anchor-test-csv",
        type=Path,
        default=DEFAULT_ID55 if DEFAULT_ID55.exists() else DEFAULT_CARUANA,
    )
    parser.add_argument("--svd-components", type=int, default=256)
    parser.add_argument("--init-k", type=int, default=32)
    parser.add_argument("--learned-k", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-perspectives", type=int, default=8)
    parser.add_argument("--graph-skip", type=float, default=0.8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gamma-grid", default="-0.5,-0.25,0.25,0.5")
    parser.add_argument("--clip-grid", default="0.03,0.06")
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--record-best", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    return parser


def fit_one(
    features: np.ndarray,
    init_adj: np.ndarray,
    target_all: np.ndarray,
    train_mask: np.ndarray,
    args: argparse.Namespace,
    seed_offset: int,
) -> tuple[np.ndarray, list[float]]:
    return fit_dense_gsl_mpp(
        features,
        init_adj,
        target_all,
        train_mask,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        learned_k=args.learned_k,
        num_perspectives=args.num_perspectives,
        graph_skip=args.graph_skip,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed + seed_offset,
        device=args.device,
    )


def main() -> None:
    args = build_parser().parse_args()
    run_dir = OUTPUT_ROOT.joinpath(args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    gamma_grid = parse_grid(args.gamma_grid, float)
    clip_grid = parse_grid(args.clip_grid, float)

    print("Loading train/test and anchors ...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y = train_df["pec50"].to_numpy(dtype=np.float64)
    n_train = len(train_df)
    n_test = len(test_df)
    anchor_oof, caruana_id, weights_map = load_caruana_oof()
    anchor_test_df = load_anchor_test_csv(args.anchor_test_csv, n_test)
    anchor_test = anchor_test_df["pEC50"].to_numpy(dtype=np.float64)
    anchor_metrics = compute_metrics(y, anchor_oof)
    residual_target = y - anchor_oof
    target_all = np.zeros(n_train + n_test, dtype=np.float32)
    target_all[:n_train] = residual_target.astype(np.float32)

    print(
        f"Anchor ens_caruana_bag20 id={caruana_id}: "
        f"MAE={anchor_metrics['MAE']:.6f} Sp={anchor_metrics['Spearman_R']:.6f}"
    )

    print("Building graph and compact features ...")
    smiles_all = train_df["smiles"].tolist() + test_df["smiles"].tolist()
    fps = morgan_bit_matrix(smiles_all)
    sim = tanimoto_similarity(fps, fps)
    init_adj = (
        build_topk_adjacency_torch(
            similarity=torch.as_tensor(sim, dtype=torch.float32),
            k=args.init_k,
            include_self=False,
        )
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    features = build_node_features(
        fps,
        anchor_oof=anchor_oof,
        anchor_test=anchor_test,
        svd_components=args.svd_components,
        seed=args.seed,
    )

    print("Cross-fit learned graph residuals ...")
    folds = umap_split_indices(train_df["smiles"].tolist())
    oof_resid = np.zeros(n_train, dtype=np.float64)
    fold_rows: list[dict[str, float | int]] = []
    for fold_id, (train_idx, val_idx) in enumerate(folds):
        mask = np.zeros(n_train + n_test, dtype=bool)
        mask[train_idx] = True
        pred_all, history = fit_one(features, init_adj, target_all, mask, args, fold_id)
        oof_resid[val_idx] = pred_all[val_idx]
        fold_rows.append(
            {
                "fold": fold_id,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "loss_start": float(history[0]),
                "loss_end": float(history[-1]),
                "val_resid_std": float(pred_all[val_idx].std()),
            }
        )
        print(
            f"  fold {fold_id}: loss {history[0]:.4f}->{history[-1]:.4f}, "
            f"val_resid_std={pred_all[val_idx].std():.4f}"
        )

    print("Fit final all-train model for test residuals ...")
    final_mask = np.zeros(n_train + n_test, dtype=bool)
    final_mask[:n_train] = True
    final_pred, final_history = fit_one(
        features, init_adj, target_all, final_mask, args, seed_offset=100
    )
    test_resid = final_pred[n_train:]

    rows: list[dict[str, object]] = []
    payloads: dict[str, tuple[np.ndarray, np.ndarray, dict[str, object]]] = {}
    for gamma in gamma_grid:
        for clip in clip_grid:
            oof_pred = apply_residual_correction(anchor_oof, oof_resid, gamma, clip)
            test_pred = apply_residual_correction(anchor_test, test_resid, gamma, clip)
            metrics = compute_metrics(y, oof_pred)
            label = f"g{gamma:g}_c{clip:g}".replace(".", "p").replace("-", "m")
            row: dict[str, object] = {
                "label": label,
                "gamma": gamma,
                "clip": clip,
                "oof_mae": metrics["MAE"],
                "oof_delta_mae": metrics["MAE"] - anchor_metrics["MAE"],
                "oof_spearman": metrics["Spearman_R"],
                "oof_delta_spearman": metrics["Spearman_R"]
                - anchor_metrics["Spearman_R"],
                "oof_resid_std": float(oof_resid.std()),
                "test_resid_std": float(test_resid.std()),
                "final_loss_start": float(final_history[0]),
                "final_loss_end": float(final_history[-1]),
            }
            row.update(summarize_shift(test_pred, anchor_test))
            rows.append(row)
            payloads[label] = (oof_pred, test_pred, row)

    summary = pd.DataFrame(rows).sort_values(
        ["oof_delta_mae", "oof_delta_spearman"], ascending=[True, False]
    )
    summary_path = run_dir.joinpath("summary.csv")
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir.joinpath("fold_losses.csv"), index=False)

    selected = summary.head(args.max_candidates)
    preflight_results: list[str] = []
    for _, row in selected.iterrows():
        label = str(row["label"])
        _oof_pred, test_pred, _payload = payloads[label]
        out_df = anchor_test_df.copy()
        out_df["pEC50"] = test_pred
        out_name = f"ens_gsl_mpp_learned_{args.run_name}_{label}.csv"
        out_path = SUBMISSION_DIR.joinpath(out_name)
        out_df.to_csv(out_path, index=False)
        if args.skip_preflight:
            preflight_results.append(f"{label}: SKIP")
        else:
            preflight_results.append(
                f"{label}: {run_preflight(out_path, run_dir, label)}"
            )

    best_label = str(selected.iloc[0]["label"])
    best_oof, _best_test, best_row = payloads[best_label]
    if args.record_best:
        exp_id = record_experiment(
            name=f"gsl_mpp_learned_{args.run_name}_{best_label}",
            description="Learned GSL-MPP-style molecule graph residual model",
            model_type="gsl_mpp_learned",
            feature_set="morgan_svd_anchor_learned_inter_molecule_graph",
            hyperparameters={
                "source_ensemble": "ens_caruana_bag20",
                "source_ensemble_id": caruana_id,
                "source_weights": weights_map,
                "anchor_test_csv": str(args.anchor_test_csv),
                "args": vars(args),
                "best": best_row,
            },
            fold_metrics=[compute_metrics(y, best_oof)],
            notes="Diagnostic learned GSL-MPP residual model; do not submit without preflight review.",
            on_conflict_replace=True,
        )
        save_oof_predictions(exp_id, best_oof)

    report_path = run_dir.joinpath("report.md")
    report_path.write_text(
        "\n".join(
            [
                "# Learned GSL-MPP Report",
                "",
                f"Run name: `{args.run_name}`",
                f"Source OOF ensemble: `ens_caruana_bag20` id `{caruana_id}`",
                f"Anchor test CSV: `{args.anchor_test_csv}`",
                "",
                "## Hyperparameters",
                "",
                f"- svd_components: `{args.svd_components}`",
                f"- init_k / learned_k: `{args.init_k}` / `{args.learned_k}`",
                f"- epochs: `{args.epochs}`",
                f"- hidden_dim: `{args.hidden_dim}`",
                f"- graph_skip: `{args.graph_skip}`",
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
                "## Fold losses",
                "",
                pd.DataFrame(fold_rows).to_markdown(index=False),
                "",
                "## Preflight",
                "",
                *[f"- {item}" for item in preflight_results],
                "",
                "## Decision",
                "",
                "Do not submit automatically. The gate requires OOF MAE delta <= -0.0015 or Spearman delta >= +0.0010 with safe MAE, plus mean abs test shift <= 0.02 and non-HOLD preflight.",
                "",
                "## Read",
                "",
                "This is a learned molecule-graph residual model. Treat OOF gain, test shift, and preflight together before any cooldown spend.",
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
