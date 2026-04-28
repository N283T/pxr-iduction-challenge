"""Hybrid blend: id=32 baseline (9-pool) + id=42 family-meta (7-pool) at 50/50.

Motivation: id=42 family-meta LB result (MAE 0.4091 / Sp 0.8476) showed:
  - MAE +0.0013 worse vs baseline (id=32 LB 0.4078)
  - Sp +0.0022 BETTER vs baseline (first time Sp beat baseline)

Hypothesis: blend the two captures Sp gain while pulling MAE back to
baseline territory. family share of the blend lies between 0.54 (meta)
and 0.76 (baseline) ~ 0.65 — a region not yet observed on LB.

Pipeline:
  1. Regenerate 9-pool baseline caruana + importance cal (rebuilds DB row
     ens_caruana_bag20, overwrites the 7-pool meta version that was last
     written there). Save baseline test predictions to a stable filename.
  2. Load id=42 meta test predictions from the backup CSV.
  3. Blend 50/50 -> ens_hybrid_meta_baseline_5050.csv.
  4. Evaluate OOF-side blend (baseline-OOF + meta-OOF mean) and report
     MAE / Sp / family share.

Usage:
    pixi run python track1_activity/scripts/boltz_affhead/38_hybrid_meta_baseline_5050.py
"""

from __future__ import annotations

import shutil
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

import run_ensemble  # noqa: E402
import run_ensemble_calibrate_importance  # noqa: E402

SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
META_BACKUP = SUBMISSION_DIR.joinpath(
    "ens_caruana_bag20_calibrated_importance_meta_id42.csv"
)
BASELINE_BACKUP = SUBMISSION_DIR.joinpath(
    "ens_caruana_bag20_calibrated_importance_baseline_9pool.csv"
)
HYBRID_OUT = SUBMISSION_DIR.joinpath("ens_hybrid_meta_baseline_5050.csv")


def reconstruct_caruana_oof() -> tuple[np.ndarray, dict]:
    """Pull the latest ens_caruana_bag20 row from DB and reconstruct
    its OOF blend from member OOFs + stored weights."""
    with psycopg2.connect(**DB_PARAMS) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, hyperparameters FROM experiments "
            "WHERE name = 'ens_caruana_bag20' ORDER BY id DESC LIMIT 1"
        )
        exp_id, hp = cur.fetchone()
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
    return oof_blend / sum(weights.values()), weights


def main() -> None:
    if not META_BACKUP.exists():
        raise RuntimeError(f"meta backup missing: {META_BACKUP}")

    print("===== Step 1: regenerate 9-pool baseline =====")
    print("  (uses default ENSEMBLE_MODELS — 9-pool with att/gate)")
    run_ensemble.main()
    run_ensemble_calibrate_importance.main()

    cal_path = SUBMISSION_DIR.joinpath("ens_caruana_bag20_calibrated_importance.csv")
    shutil.copy(cal_path, BASELINE_BACKUP)
    print(f"  baseline backup: {BASELINE_BACKUP}")

    print("\n===== Step 2: load both submissions =====")
    base_df = pd.read_csv(BASELINE_BACKUP)
    meta_df = pd.read_csv(META_BACKUP)
    assert (base_df["SMILES"] == meta_df["SMILES"]).all(), "SMILES mismatch"
    base_pred = base_df["pEC50"].to_numpy()
    meta_pred = meta_df["pEC50"].to_numpy()
    print(
        f"  baseline test pred: mean={base_pred.mean():.4f} std={base_pred.std():.4f}"
    )
    print(
        f"  meta     test pred: mean={meta_pred.mean():.4f} std={meta_pred.std():.4f}"
    )
    print(f"  Pearson r: {np.corrcoef(base_pred, meta_pred)[0, 1]:.4f}")

    blend_pred = 0.5 * base_pred + 0.5 * meta_pred
    out = pd.DataFrame(
        {
            "SMILES": base_df["SMILES"],
            "Molecule Name": base_df["Molecule Name"],
            "pEC50": blend_pred,
        }
    )
    out.to_csv(HYBRID_OUT, index=False)
    print(f"\n  wrote: {HYBRID_OUT}")

    print("\n===== Step 3: OOF-side blend evaluation =====")
    train_df = load_train_smiles_target()
    y = train_df["pec50"].to_numpy()

    # Reconstruct baseline OOF (currently in DB after regeneration)
    base_oof, base_weights = reconstruct_caruana_oof()

    # Meta OOF must be reconstructed from id=42 era — we know the meta
    # variant pool from script #36. Pull tabpfn_chemprop_family_meta_umap
    # OOF directly to avoid stale DB state confusion.
    print("  reconstructing meta OOF blend from id=42 weight snapshot ...")
    # The meta caruana row was overwritten, so we re-fit by mean of
    # blend's OOF (members are stable). Approximation: load
    # tabpfn_chemprop_family_meta_umap OOF as proxy for meta blend OOF.
    with psycopg2.connect(**DB_PARAMS) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM experiments WHERE name = "
            "'tabpfn_chemprop_family_meta_umap' ORDER BY id DESC LIMIT 1"
        )
        meta_id = cur.fetchone()[0]
    meta_oof_proxy = load_oof_predictions(meta_id)

    blend_oof = 0.5 * base_oof + 0.5 * meta_oof_proxy

    def metrics(name: str, pred: np.ndarray) -> None:
        mae = float(np.mean(np.abs(pred - y)))
        sp = float(stats.spearmanr(y, pred).statistic)
        print(f"    {name:<35} MAE={mae:.4f}  Sp={sp:.4f}")

    metrics("baseline OOF (9-pool blend)", base_oof)
    metrics("meta OOF (proxy: family_meta single)", meta_oof_proxy)
    metrics("hybrid 50/50 OOF", blend_oof)

    print("\n  NOTE: meta OOF is the family_meta single-member proxy, NOT")
    print("  the full 7-pool meta caruana blend. The 7-pool blend would")
    print("  be slightly better (kermt/molformer weight). Treat hybrid")
    print("  OOF as a lower bound; LB will likely be a bit better.")

    print("\nReady to submit (after cooldown):")
    print(f"  {HYBRID_OUT}")


if __name__ == "__main__":
    main()
