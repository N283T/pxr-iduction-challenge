"""Blend ratio bakeoff: search for OOF Sp+MAE optimal in
baseline-vs-meta linear combination.

Motivation: id=43 hybrid 50/50 LB result (MAE 0.4075 / Sp 0.8470, rank 2)
beat baseline on MAE (-0.0003) AND Sp (+0.0016) but Sp still rank 3
behind sia (0.8514) and Yan (0.8543). meta-only (id=42) had Sp 0.8476
top, suggesting meta-leaning ratios may push Sp up while staying near
baseline MAE.

Bakeoff sweeps α from 0.0 to 1.0 in 7 steps (α = baseline weight,
(1-α) = meta weight), reports OOF MAE / Sp / effective family share.

Caveats:
  - Meta OOF is the family_meta single-member proxy (same approximation
    used in 38_hybrid_meta_baseline_5050.py). The full 7-pool meta
    caruana blend would be slightly different, but trends are reliable.
  - Test predictions use the actual saved CSVs (no proxy).

Usage:
    pixi run python track1_activity/scripts/boltz_affhead/39_blend_ratio_bakeoff.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import DB_PARAMS, load_train_smiles_target  # noqa: E402
from evaluate import load_oof_predictions  # noqa: E402

SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
META_BACKUP = SUBMISSION_DIR.joinpath(
    "ens_caruana_bag20_calibrated_importance_meta_id42.csv"
)
BASELINE_BACKUP = SUBMISSION_DIR.joinpath(
    "ens_caruana_bag20_calibrated_importance_baseline_9pool.csv"
)

# Approximate effective chemprop family share for the two endpoints.
# baseline: caruana on 9-pool gives chemprop_family share 0.7573 (37 bakeoff).
# meta:     caruana on 7-pool gives chemprop_family_meta wt 0.539 (= effective
#           family share, since the meta IS the family).
FAMILY_SHARE_BASELINE = 0.7573
FAMILY_SHARE_META = 0.539


def reconstruct_baseline_oof() -> np.ndarray:
    """Pull current ens_caruana_bag20 (= baseline 9-pool after script #38)."""
    with psycopg2.connect(**DB_PARAMS) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, hyperparameters FROM experiments "
            "WHERE name = 'ens_caruana_bag20' ORDER BY id DESC LIMIT 1"
        )
        _, hp = cur.fetchone()
        weights = hp["weights"]

        oof_blend = None
        for name, w in weights.items():
            cur.execute(
                "SELECT id FROM experiments WHERE name = %s ORDER BY id DESC LIMIT 1",
                (name,),
            )
            mid = cur.fetchone()[0]
            oof = load_oof_predictions(mid)
            if oof_blend is None:
                oof_blend = np.zeros_like(oof, dtype=np.float64)
            oof_blend = oof_blend + w * oof
    return oof_blend / sum(weights.values())


def load_meta_oof_proxy() -> np.ndarray:
    """Use family_meta single-member OOF as proxy for the 7-pool meta blend."""
    with psycopg2.connect(**DB_PARAMS) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM experiments WHERE name = "
            "'tabpfn_chemprop_family_meta_umap' ORDER BY id DESC LIMIT 1"
        )
        meta_id = cur.fetchone()[0]
    return load_oof_predictions(meta_id)


def main() -> None:
    train_df = load_train_smiles_target()
    y = train_df["pec50"].to_numpy()

    print("Loading OOFs ...")
    base_oof = reconstruct_baseline_oof()
    meta_oof = load_meta_oof_proxy()
    print(
        f"  base OOF : MAE={np.mean(np.abs(base_oof - y)):.4f}, "
        f"Sp={stats.spearmanr(y, base_oof).statistic:.4f}"
    )
    print(
        f"  meta OOF : MAE={np.mean(np.abs(meta_oof - y)):.4f}, "
        f"Sp={stats.spearmanr(y, meta_oof).statistic:.4f}  (single-member proxy)"
    )

    base_df = pd.read_csv(BASELINE_BACKUP)
    meta_df = pd.read_csv(META_BACKUP)
    assert (base_df["SMILES"] == meta_df["SMILES"]).all()
    base_test = base_df["pEC50"].to_numpy()
    meta_test = meta_df["pEC50"].to_numpy()

    print("\n===== Blend ratio sweep =====")
    print(
        f"  {'α (base)':>10} {'OOF MAE':>10} {'OOF Sp':>10} "
        f"{'fam share':>10} {'test mean':>10} {'test std':>10}"
    )
    print(f"  {'-' * 70}")

    rows = []
    for alpha in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        blend_oof = alpha * base_oof + (1 - alpha) * meta_oof
        blend_test = alpha * base_test + (1 - alpha) * meta_test
        mae = float(np.mean(np.abs(blend_oof - y)))
        sp = float(stats.spearmanr(y, blend_oof).statistic)
        fam = alpha * FAMILY_SHARE_BASELINE + (1 - alpha) * FAMILY_SHARE_META
        rows.append(
            {
                "alpha": alpha,
                "MAE": mae,
                "Sp": sp,
                "family_share": fam,
                "test_mean": float(blend_test.mean()),
                "test_std": float(blend_test.std()),
            }
        )
        print(
            f"  {alpha:>10.2f} {mae:>10.4f} {sp:>10.4f} "
            f"{fam:>10.4f} {blend_test.mean():>10.4f} {blend_test.std():>10.4f}"
        )

    df = pd.DataFrame(rows)
    print("\n===== Findings =====")
    best_mae = df.loc[df["MAE"].idxmin()]
    best_sp = df.loc[df["Sp"].idxmax()]
    print(
        f"  Best OOF MAE: α={best_mae['alpha']:.2f} -> "
        f"MAE={best_mae['MAE']:.4f}, Sp={best_mae['Sp']:.4f}"
    )
    print(
        f"  Best OOF Sp : α={best_sp['alpha']:.2f} -> "
        f"MAE={best_sp['MAE']:.4f}, Sp={best_sp['Sp']:.4f}"
    )

    print("\n===== LB-known anchor points (for context) =====")
    print("  α=1.0 (baseline)  : LB MAE 0.4078 / Sp 0.8454 (id=32)")
    print("  α=0.5 (50/50)     : LB MAE 0.4075 / Sp 0.8470 (id=43)")
    print("  α=0.0 (meta)      : LB MAE 0.4091 / Sp 0.8476 (id=42)")


if __name__ == "__main__":
    main()
