#!/usr/bin/env -S pixi run python
"""Caruana bakeoff for replacing the existing top500 member with HTChem top500."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "track1_activity" / "src"))
sys.path.insert(0, str(REPO_ROOT / "track1_activity" / "scripts"))

from data import DB_PARAMS, load_train_smiles_target  # noqa: E402
from run_ensemble import ENSEMBLE_MODELS, optimize_caruana  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "top500_htchem_swap"
DOC_PATH = REPO_ROOT / "docs" / "track1_explain" / "phase2_htchem_top500_swap.md"

OLD_TOP500 = "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap"
NEW_TOP500 = (
    "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_pred_htchem_top500_umap_v2_6"
)
CHEMPROP_FAMILY_BASE = {
    "tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_umap_default",
    "tabpfn_chemprop_pretrain_embed_umap_default",
}


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
        exp_id = row[0]
        cur.execute(
            """
            SELECT train_idx, oof_prediction
            FROM experiment_oof_predictions
            WHERE experiment_id = %s
            ORDER BY train_idx
            """,
            (exp_id,),
        )
        rows = cur.fetchall()
    if not rows:
        raise RuntimeError(f"no OOF predictions for {name}")
    return np.asarray([r[1] for r in rows], dtype=np.float64)


def caruana_bagged(oof_matrix: np.ndarray, y: np.ndarray, names: list[str]) -> dict:
    weight_runs = []
    for seed in [42, 43, 44, 45, 46]:
        weight_runs.append(
            optimize_caruana(
                oof_matrix, y, n_iter=100, init_top_n=3, n_bags=20, seed=seed
            )
        )
    weights = np.mean(weight_runs, axis=0)
    weights = weights / weights.sum()
    pred = oof_matrix @ weights
    return {
        "mae": float(np.mean(np.abs(pred - y))),
        "rae": float(np.sum(np.abs(pred - y)) / np.sum(np.abs(y - y.mean()))),
        "spearman": float(spearmanr(pred, y).statistic),
        "weights": dict(zip(names, [float(w) for w in weights])),
        "pred": pred,
    }


def summarize_variant(
    name: str,
    member_names: list[str],
    matrix: np.ndarray,
    y: np.ndarray,
    baseline: dict | None,
) -> tuple[dict, pd.DataFrame]:
    result = caruana_bagged(matrix, y, member_names)
    family = CHEMPROP_FAMILY_BASE.copy()
    if OLD_TOP500 in member_names:
        family.add(OLD_TOP500)
    if NEW_TOP500 in member_names:
        family.add(NEW_TOP500)
    weights = result["weights"]
    row = {
        "variant": name,
        "n_members": len(member_names),
        "mae": result["mae"],
        "rae": result["rae"],
        "spearman": result["spearman"],
        "delta_mae": 0.0 if baseline is None else result["mae"] - baseline["mae"],
        "delta_spearman": 0.0
        if baseline is None
        else result["spearman"] - baseline["spearman"],
        "old_top500_weight": weights.get(OLD_TOP500, 0.0),
        "new_top500_weight": weights.get(NEW_TOP500, 0.0),
        "chemprop_family_share": sum(weights.get(m, 0.0) for m in family),
    }
    weight_df = pd.DataFrame(
        [
            {
                "variant": name,
                "member": member,
                "weight": weight,
                "is_old_top500": member == OLD_TOP500,
                "is_new_top500": member == NEW_TOP500,
            }
            for member, weight in sorted(weights.items(), key=lambda item: -item[1])
        ]
    )
    return row, weight_df


def write_doc(
    summary: pd.DataFrame, weights: pd.DataFrame, correlations: pd.DataFrame
) -> None:
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Phase 2 HTChem top500 SWAP bakeoff

Current top500 member: `{OLD_TOP500}`

HTChem top500 member: `{NEW_TOP500}`

## Caruana Bag20 Summary

{summary.to_markdown(index=False, floatfmt=".5f")}

## Residual Correlation

{correlations.to_markdown(index=False, floatfmt=".5f")}

## Weights

{weights.to_markdown(index=False, floatfmt=".5f")}

## Read

SWAP is preferred over ADD if it improves OOF without increasing correlated family share. ADD is diagnostic only because two highly related top500 members can concentrate the same family axis.
"""
    DOC_PATH.write_text(text)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    y = load_train_smiles_target()["pec50"].to_numpy(dtype=np.float64)
    base_names = list(ENSEMBLE_MODELS)
    if OLD_TOP500 not in base_names:
        raise RuntimeError(f"{OLD_TOP500} not in ENSEMBLE_MODELS")

    base_oofs = [load_member_oof(name) for name in base_names]
    new_oof = load_member_oof(NEW_TOP500)

    corr_rows = []
    for name, oof in zip(base_names, base_oofs):
        corr_rows.append(
            {
                "member": name,
                "pearson_vs_new_oof": float(np.corrcoef(oof, new_oof)[0, 1]),
            }
        )
    corr_df = pd.DataFrame(corr_rows).sort_values("pearson_vs_new_oof", ascending=False)
    corr_df.to_csv(OUT_DIR / "new_member_correlations.csv", index=False)

    old_idx = base_names.index(OLD_TOP500)
    variants = {
        "baseline": (base_names, np.column_stack(base_oofs)),
        "swap_top500_to_htchem": (
            base_names[:old_idx] + [NEW_TOP500] + base_names[old_idx + 1 :],
            np.column_stack(base_oofs[:old_idx] + [new_oof] + base_oofs[old_idx + 1 :]),
        ),
        "add_htchem_top500": (
            base_names + [NEW_TOP500],
            np.column_stack(base_oofs + [new_oof]),
        ),
    }

    rows = []
    weight_frames = []
    baseline = None
    for variant, (names, matrix) in variants.items():
        row, weight_df = summarize_variant(variant, names, matrix, y, baseline)
        if variant == "baseline":
            baseline = row
        else:
            row["delta_mae"] = row["mae"] - baseline["mae"]
            row["delta_spearman"] = row["spearman"] - baseline["spearman"]
        rows.append(row)
        weight_frames.append(weight_df)

    summary = pd.DataFrame(rows)
    weights = pd.concat(weight_frames, ignore_index=True)
    summary.to_csv(OUT_DIR / "swap_summary.csv", index=False)
    weights.to_csv(OUT_DIR / "swap_weights.csv", index=False)
    write_doc(summary, weights, corr_df)

    print(summary.to_string(index=False))
    print("\nTop weights:")
    print(weights.groupby("variant").head(8).to_string(index=False))
    print(f"\nWrote outputs to {OUT_DIR}")
    print(f"Wrote doc to {DOC_PATH}")


if __name__ == "__main__":
    main()
