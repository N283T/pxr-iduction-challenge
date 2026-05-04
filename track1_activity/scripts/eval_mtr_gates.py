#!/usr/bin/env -S pixi run python
"""MTR gate evaluation: G1 (single OOF MAE) and G2 (residual r vs pool members).

Reads OOF predictions from the DB, computes:
  - G1: candidate single-member OOF MAE <= G1_THRESHOLD (0.485)
  - G2: min Pearson r of candidate residuals vs each non-swap-target pool
        member residuals <= G2_THRESHOLD (0.85)

Output is written to both stdout and
  track1_activity/reports/mtr_gates_<candidate>.txt

Usage:
    pixi run python track1_activity/scripts/eval_mtr_gates.py \\
        --candidate tabpfn_chemprop_mtr_embed_umap \\
        --swap-target tabpfn_chemprop_pretrain_embed_umap_default
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS  # noqa: E402

# ---------------------------------------------------------------------------
# Current 9-pool snapshot (must match ENSEMBLE_MODELS in run_ensemble.py)
# Update this list if pool composition changes.
# ---------------------------------------------------------------------------
POOL_MEMBERS = [
    "tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_umap_default",
    "tabpfn_chemprop_pretrain_embed_umap_default",
    "tabpfn_pooled_boltz_umap_default",
    "tabpfn_pooled_boltz_allpairs_umap_default",
    "tabpfn_molformer_c3_pretrain_embed_umap",
    "tabpfn_kermt_pretrain_embed_umap_default",
    "tabpfn_attentivefp_pretrain_embed_umap_default",
    "tabpfn_gatedgcn_pretrain_embed_umap_default",
    "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap",
]

G1_THRESHOLD = 0.485  # single-member OOF MAE (max allowed)
G2_THRESHOLD = 0.85  # residual Pearson r (max allowed, lower = more diverse)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def get_experiment_id(experiment_name: str) -> int:
    """Return the most-recent experiment id for the given name."""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM experiments WHERE name = %s ORDER BY id DESC LIMIT 1",
        (experiment_name,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row is None:
        raise SystemExit(
            f"No experiment row found for '{experiment_name}'. "
            "Run the TabPFN UMAP CV first."
        )
    return row[0]


def get_oof_mae(experiment_id: int) -> float:
    """Return the mean per-fold MAE for this experiment from experiment_cv_results."""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute(
        "SELECT AVG(mae) FROM experiment_cv_results WHERE experiment_id = %s",
        (experiment_id,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row is None or row[0] is None:
        raise SystemExit(
            f"No experiment_cv_results rows for experiment_id={experiment_id}."
        )
    return float(row[0])


def get_oof_predictions(experiment_id: int) -> np.ndarray:
    """Return OOF predictions as a dense array ordered by train_idx (0..N-1)."""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute(
        "SELECT oof_prediction FROM experiment_oof_predictions "
        "WHERE experiment_id = %s ORDER BY train_idx",
        (experiment_id,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    if not rows:
        raise SystemExit(f"No OOF predictions in DB for experiment_id={experiment_id}.")
    return np.array([r[0] for r in rows], dtype=np.float64)


def get_train_pec50_ordered() -> np.ndarray:
    """Return pEC50 values for training set ordered by compound_id (= train_idx order)."""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute("SELECT pec50 FROM train_activity ORDER BY compound_id")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return np.array([r[0] for r in rows], dtype=np.float64)


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------


def evaluate_gates(
    candidate: str,
    swap_target: str,
    seed_tabpfn: int | None,
) -> tuple[bool, bool, dict]:
    """Run G1 and G2 gates for *candidate* vs *pool*.

    Returns (g1_pass, g2_pass, report_dict).
    """
    # Fetch candidate
    cand_id = get_experiment_id(candidate)
    cand_mae = get_oof_mae(cand_id)
    cand_oof = get_oof_predictions(cand_id)

    # Train ground truth (aligned by train_idx = position in compound_id ORDER)
    y_true = get_train_pec50_ordered()

    if len(cand_oof) != len(y_true):
        raise SystemExit(
            f"Candidate OOF length {len(cand_oof)} != train length {len(y_true)}. "
            "Possible partial CV run — check experiment_oof_predictions."
        )

    cand_resid = cand_oof - y_true

    # --- G1 ---
    g1_pass = cand_mae <= G1_THRESHOLD

    # --- G2: per pool member (excluding swap_target) ---
    g2_rows = []
    for member in POOL_MEMBERS:
        if member == swap_target:
            continue
        mem_id = get_experiment_id(member)
        mem_oof = get_oof_predictions(mem_id)
        if len(mem_oof) != len(y_true):
            raise SystemExit(
                f"Pool member '{member}' OOF length {len(mem_oof)} "
                f"!= train length {len(y_true)}."
            )
        mem_resid = mem_oof - y_true
        r = float(np.corrcoef(cand_resid, mem_resid)[0, 1])
        g2_rows.append({"member": member, "residual_r": r})

    # Sort ascending so the highest-correlation (hardest) member is last
    g2_rows_sorted = sorted(g2_rows, key=lambda x: x["residual_r"])
    g2_max_r = max(row["residual_r"] for row in g2_rows)
    g2_pass = g2_max_r <= G2_THRESHOLD

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "candidate": candidate,
        "experiment_id": cand_id,
        "swap_target": swap_target,
        "seed_tabpfn": seed_tabpfn,
        "g1": {
            "threshold": G1_THRESHOLD,
            "oof_mae": round(cand_mae, 4),
            "pass": g1_pass,
        },
        "g2": {
            "threshold": G2_THRESHOLD,
            "max_residual_r": round(g2_max_r, 4),
            "pass": g2_pass,
            "members": [
                {"member": r["member"], "residual_r": round(r["residual_r"], 4)}
                for r in g2_rows_sorted
            ],
        },
        "overall_pass": g1_pass and g2_pass,
    }
    return g1_pass, g2_pass, report


def format_report(report: dict) -> str:
    """Format the gate evaluation report as a human-readable string."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("MTR Gate Evaluation Report")
    lines.append(f"  Timestamp : {report['timestamp']}")
    lines.append(f"  Candidate : {report['candidate']} (id={report['experiment_id']})")
    lines.append(f"  Swap tgt  : {report['swap_target']}")
    if report["seed_tabpfn"] is not None:
        lines.append(f"  TabPFN seed: {report['seed_tabpfn']}")
    lines.append("")

    g1 = report["g1"]
    lines.append("--- Gate G1: single-member OOF MAE ---")
    lines.append(f"  OOF MAE   : {g1['oof_mae']:.4f}")
    lines.append(f"  Threshold : <= {g1['threshold']}")
    lines.append(f"  G1        : {'PASS' if g1['pass'] else 'FAIL'}")
    lines.append("")

    g2 = report["g2"]
    lines.append("--- Gate G2: residual Pearson r vs non-swap-target pool ---")
    col_w = max(len(r["member"]) for r in g2["members"]) + 2
    lines.append(f"  {'Member':<{col_w}}  residual_r")
    lines.append(f"  {'-' * col_w}  ----------")
    for row in g2["members"]:
        lines.append(f"  {row['member']:<{col_w}}  {row['residual_r']:.4f}")
    lines.append("")
    lines.append(f"  max residual r : {g2['max_residual_r']:.4f}")
    lines.append(f"  Threshold      : <= {g2['threshold']}")
    lines.append(f"  G2             : {'PASS' if g2['pass'] else 'FAIL'}")
    lines.append("")

    lines.append("--- Summary ---")
    lines.append(f"  G1 : {'PASS' if g1['pass'] else 'FAIL'}")
    lines.append(f"  G2 : {'PASS' if g2['pass'] else 'FAIL'}")
    lines.append(
        f"  Overall : {'PASS — proceed to G3 (caruana SWAP)' if report['overall_pass'] else 'FAIL — stop, do not submit'}"
    )
    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--candidate",
        required=True,
        help="experiment_name of the new variant to evaluate (e.g. tabpfn_chemprop_mtr_embed_umap)",
    )
    p.add_argument(
        "--swap-target",
        required=True,
        dest="swap_target",
        help="experiment_name being replaced — excluded from G2 comparison "
        "(e.g. tabpfn_chemprop_pretrain_embed_umap_default)",
    )
    p.add_argument(
        "--seed-tabpfn",
        type=int,
        default=None,
        dest="seed_tabpfn",
        help="TabPFN random seed used for the candidate run (informational only, "
        "not used in DB queries). Default: None (TabPFN's internal default).",
    )
    args = p.parse_args()

    g1_pass, g2_pass, report = evaluate_gates(
        candidate=args.candidate,
        swap_target=args.swap_target,
        seed_tabpfn=args.seed_tabpfn,
    )

    report_str = format_report(report)

    # Write to stdout
    print(report_str)

    # Write to file
    report_dir = REPO_ROOT.joinpath("track1_activity", "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    # Sanitize candidate name for use in filename
    safe_name = args.candidate.replace("/", "_").replace(" ", "_")
    out_path = report_dir.joinpath(f"mtr_gates_{safe_name}.txt")
    out_path.write_text(report_str + "\n")
    print(f"\nReport written to {out_path}")

    return 0 if (g1_pass and g2_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
