"""Caruana_bag20 bakeoff: 9-pool baseline vs +knn_alltrain vs +knn_potent46 vs +both.

Decides whether either of the new kNN pool members (from
run_knn_pool_member.py) carries non-zero caruana weight and improves OOF
MAE above the bag noise floor (-0.003).

No DB writes. Standalone diagnostic.

Legacy experiment script; internal design note was removed from the public repository.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import psycopg2
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
from data import DB_PARAMS, load_train_smiles_target  # noqa: E402
from run_ensemble import ENSEMBLE_MODELS, optimize_caruana  # noqa: E402

NEW_MEMBERS = ("knn_alltrain_umap", "knn_potent46_umap")


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
        raise RuntimeError(f"no OOF predictions for {name} (id={exp_id})")
    return np.asarray([r[1] for r in rows], dtype=np.float64)


def caruana_bagged(oof_matrix: np.ndarray, y: np.ndarray, names: list[str]) -> dict:
    """Run optimize_caruana 5 times, average weights for stability."""
    weights_runs = []
    for seed in [42, 43, 44, 45, 46]:
        w = optimize_caruana(
            oof_matrix, y, n_iter=100, init_top_n=3, n_bags=20, seed=seed
        )
        weights_runs.append(w)
    weights_mean = np.mean(weights_runs, axis=0)
    weights_mean = weights_mean / weights_mean.sum()
    blend = oof_matrix @ weights_mean
    mae = float(np.mean(np.abs(blend - y)))
    sp = float(spearmanr(blend, y).statistic)
    return {
        "weights": dict(zip(names, [round(float(w), 4) for w in weights_mean])),
        "mae": mae,
        "sp": sp,
    }


def main() -> None:
    print("Loading train labels ...")
    train_df = load_train_smiles_target()
    y = train_df["pec50"].to_numpy(dtype=np.float64)
    print(f"  train n={len(y)}")

    print(f"\nLoading {len(ENSEMBLE_MODELS)}-pool OOF (current production) ...")
    base_names = list(ENSEMBLE_MODELS)
    base_oofs = []
    for name in base_names:
        oof = load_member_oof(name)
        if len(oof) != len(y):
            raise RuntimeError(f"{name}: OOF length {len(oof)} != y length {len(y)}")
        base_oofs.append(oof)
        print(f"  {name}: mean={oof.mean():.3f} std={oof.std():.3f}")
    base_matrix = np.column_stack(base_oofs)

    print(f"\nLoading {len(NEW_MEMBERS)} new kNN members ...")
    new_oofs = {}
    for name in NEW_MEMBERS:
        oof = load_member_oof(name)
        if len(oof) != len(y):
            raise RuntimeError(f"{name}: OOF length {len(oof)} != y length {len(y)}")
        new_oofs[name] = oof
        single_mae = float(np.mean(np.abs(oof - y)))
        single_sp = float(spearmanr(oof, y).statistic)
        print(f"  {name}: single OOF MAE={single_mae:.4f} Sp={single_sp:.4f}")

    print("\nResidual correlation of new members vs each base member:")
    for new_name, new_oof in new_oofs.items():
        print(f"  {new_name}:")
        for base_name, base_oof in zip(base_names, base_oofs):
            r = float(np.corrcoef(new_oof, base_oof)[0, 1])
            print(f"    r vs {base_name:>60} = {r:+.4f}")

    print("\n=== Bakeoff: 4 caruana_bag20 variants (5-seed averaged weights) ===")
    variants = {
        "baseline_9pool": (base_names, base_matrix),
        "+knn_alltrain": (
            base_names + ["knn_alltrain_umap"],
            np.column_stack([base_matrix, new_oofs["knn_alltrain_umap"]]),
        ),
        "+knn_potent46": (
            base_names + ["knn_potent46_umap"],
            np.column_stack([base_matrix, new_oofs["knn_potent46_umap"]]),
        ),
        "+both": (
            base_names + list(NEW_MEMBERS),
            np.column_stack([base_matrix] + [new_oofs[n] for n in NEW_MEMBERS]),
        ),
    }

    baseline_mae = None
    for variant_name, (names, matrix) in variants.items():
        print(f"\n--- {variant_name} ({len(names)} members) ---")
        result = caruana_bagged(matrix, y, names)
        if variant_name == "baseline_9pool":
            baseline_mae = result["mae"]
        delta = result["mae"] - baseline_mae if baseline_mae is not None else 0.0
        print(f"  caruana OOF MAE = {result['mae']:.4f}  Δ vs baseline = {delta:+.4f}")
        print(f"  caruana OOF Sp  = {result['sp']:.4f}")
        print("  weights (top-12 by magnitude):")
        sorted_w = sorted(result["weights"].items(), key=lambda x: -x[1])
        for n, w in sorted_w[:12]:
            highlight = " <-- NEW" if n in NEW_MEMBERS else ""
            print(f"    {w:.4f}  {n}{highlight}")
        for n in NEW_MEMBERS:
            if n in result["weights"]:
                w = result["weights"][n]
                if w < 0.01:
                    print(f"  WARN: {n} weight {w:.4f} < 1% threshold")


if __name__ == "__main__":
    main()
