"""Caruana_bag20 SWAP bakeoff: replace existing top500 with admet_ai_top500.

Tests the hypothesis that
`tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_admet_ai_top500_umap`
(single OOF 0.3964, Sp 0.8490) can replace the existing
`tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap`
(single OOF ~0.397, Sp 0.846) without family share concentration risk.

Variants:
  baseline_9pool: current ENSEMBLE_MODELS (id=43 production)
  swap_admet_top500: replace top500 with admet_ai_top500
  add_admet_top500: keep top500 AND add admet_ai_top500 (10-pool, family share risk)

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

OLD_TOP500 = "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap"
NEW_TOP500 = "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_admet_ai_top500_umap"

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
        raise RuntimeError(f"no OOF predictions for {name} (id={exp_id})")
    return np.asarray([r[1] for r in rows], dtype=np.float64)


def caruana_bagged(oof_matrix: np.ndarray, y: np.ndarray, names: list[str]) -> dict:
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


def compute_family_share(weights: dict, names_in_family: set) -> float:
    return sum(weights.get(n, 0.0) for n in names_in_family)


def main() -> None:
    print("Loading train labels ...")
    train_df = load_train_smiles_target()
    y = train_df["pec50"].to_numpy(dtype=np.float64)
    print(f"  train n={len(y)}")

    base_names = list(ENSEMBLE_MODELS)
    print(f"\nBaseline pool: {len(base_names)} members")
    base_oofs = []
    for name in base_names:
        oof = load_member_oof(name)
        base_oofs.append(oof)
    base_matrix = np.column_stack(base_oofs)

    print(f"\nLoading new member OOF: {NEW_TOP500}")
    new_oof = load_member_oof(NEW_TOP500)
    single_mae = float(np.mean(np.abs(new_oof - y)))
    single_sp = float(spearmanr(new_oof, y).statistic)
    print(f"  single OOF MAE={single_mae:.4f} Sp={single_sp:.4f}")

    # Residual correlation vs existing pool
    print(f"\nResidual correlation of {NEW_TOP500} vs each base member:")
    for base_name, base_oof in zip(base_names, base_oofs):
        r = float(np.corrcoef(new_oof, base_oof)[0, 1])
        print(f"    r vs {base_name:>60} = {r:+.4f}")

    # Variants
    if OLD_TOP500 not in base_names:
        raise RuntimeError(f"{OLD_TOP500} not in ENSEMBLE_MODELS — design mismatch")
    swap_idx = base_names.index(OLD_TOP500)

    swap_names = base_names.copy()
    swap_names[swap_idx] = NEW_TOP500
    swap_oofs = base_oofs.copy()
    swap_oofs[swap_idx] = new_oof
    swap_matrix = np.column_stack(swap_oofs)

    add_names = base_names + [NEW_TOP500]
    add_matrix = np.column_stack([base_matrix, new_oof])

    variants = {
        "baseline_9pool": (base_names, base_matrix),
        "swap_admet_top500": (swap_names, swap_matrix),
        "add_admet_top500": (add_names, add_matrix),
    }

    chemprop_family_for_old = CHEMPROP_FAMILY_BASE | {OLD_TOP500}
    chemprop_family_for_new = CHEMPROP_FAMILY_BASE | {NEW_TOP500}

    print("\n=== Bakeoff (5-seed averaged caruana_bag20 weights) ===")
    baseline_mae = None
    baseline_sp = None
    for variant_name, (names, matrix) in variants.items():
        print(f"\n--- {variant_name} ({len(names)} members) ---")
        result = caruana_bagged(matrix, y, names)
        if variant_name == "baseline_9pool":
            baseline_mae = result["mae"]
            baseline_sp = result["sp"]
        delta_mae = result["mae"] - baseline_mae if baseline_mae is not None else 0.0
        delta_sp = result["sp"] - baseline_sp if baseline_sp is not None else 0.0
        print(
            f"  caruana OOF MAE = {result['mae']:.4f}  Δ vs baseline = {delta_mae:+.4f}"
        )
        print(
            f"  caruana OOF Sp  = {result['sp']:.4f}  Δ vs baseline = {delta_sp:+.4f}"
        )
        # family share
        if variant_name == "baseline_9pool":
            family = chemprop_family_for_old
        elif variant_name == "swap_admet_top500":
            family = chemprop_family_for_new
        else:  # add_admet_top500
            family = chemprop_family_for_old | {NEW_TOP500}
        share = compute_family_share(result["weights"], family)
        zone = "in 0.65-0.80" if 0.65 <= share <= 0.80 else "OUT"
        print(f"  chemprop family share: {share:.3f}  ({zone})")
        print("  weights:")
        sorted_w = sorted(result["weights"].items(), key=lambda x: -x[1])
        for n, w in sorted_w:
            highlight = ""
            if n == NEW_TOP500:
                highlight = " <-- NEW"
            elif n == OLD_TOP500:
                highlight = " <-- OLD"
            print(f"    {w:.4f}  {n}{highlight}")


if __name__ == "__main__":
    main()
