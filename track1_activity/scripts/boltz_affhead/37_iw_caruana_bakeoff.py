"""IW caruana bakeoff (Codex 2026-04-28 advice).

Compare 4 combinations on the baseline 9-pool:

  vanilla caruana × no calibrator      [oracle baseline]
  vanilla caruana × importance cal     [current production = id=32]
  IW caruana × no calibrator           [pure IW signal]
  IW caruana × importance cal          [DOUBLE — two shift assumptions stacked]

Codex's warning: applying the same domain-classifier ratio in both
caruana selection AND the post-hoc affine = same shift hypothesis
trusted twice. So the meaningful comparison is between
``vanilla caruana × importance cal`` (current) and
``IW caruana × no calibrator`` (alternative single-stage shift correction).

Per-variant we report:
  - OOF MAE (raw, on plain MAE)
  - IW-OOF MAE (sample-weighted, the test-distribution proxy)
  - Spearman
  - chemprop family share (cheme_t10 + cheme_top500 + chemprop_pretrain_embed)
  - 5 top weights

Usage:
    pixi run python track1_activity/scripts/boltz_affhead/37_iw_caruana_bakeoff.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

from data import load_test_smiles, load_train_smiles_target  # noqa: E402
from importance_weights import compute_importance_weights  # noqa: E402
from scipy import stats  # noqa: E402

import run_ensemble  # noqa: E402

CHEMPROP_FAMILY = (
    "tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_umap_default",
    "tabpfn_chemprop_pretrain_embed_umap_default",
    "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap",
)


def family_share(names: list[str], weights: np.ndarray) -> float:
    return float(sum(w for n, w in zip(names, weights) if n in CHEMPROP_FAMILY))


def fit_importance_affine(
    oof: np.ndarray, y: np.ndarray, w: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply weighted affine regression on OOF (same recipe as
    run_ensemble_calibrate_importance)."""
    X = oof.reshape(-1, 1)
    # weighted least squares
    w_norm = w * (len(w) / w.sum())
    sqrt_w = np.sqrt(w_norm)
    Xw = X * sqrt_w[:, None]
    Xw = np.hstack([np.ones_like(Xw), Xw])
    yw = y * sqrt_w
    coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    intercept, slope = float(coef[0]), float(coef[1])
    oof_cal = oof * slope + intercept
    test_cal = test * slope + intercept
    return oof_cal, test_cal, {"slope": slope, "intercept": intercept}


def report(label: str, oof_blend: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    err = np.abs(oof_blend - y)
    mae = float(err.mean())
    iw_mae = float(np.sum(w * err) / w.sum())
    sp = float(stats.spearmanr(y, oof_blend).statistic)
    return {"label": label, "MAE": mae, "IW_MAE": iw_mae, "Sp": sp}


def main() -> None:
    print("Loading data + 9-pool OOF/test ...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y = train_df["pec50"].to_numpy(dtype=np.float64)

    names, oof_matrix, test_matrix = run_ensemble.load_models(y, n_test=len(test_df))
    print(f"  loaded {len(names)} members ({oof_matrix.shape}) ")

    print("Computing importance weights (Morgan FP + LogReg domain classifier) ...")
    iw = compute_importance_weights(
        train_df["smiles"].tolist(),
        test_df["smiles"].tolist(),
    )
    print(
        f"  iw: mean={iw.mean():.4f}, q25={np.quantile(iw, 0.25):.4f}, "
        f"q75={np.quantile(iw, 0.75):.4f}, max={iw.max():.4f}"
    )

    print("\n===== Caruana selection (vanilla vs IW) =====")
    w_vanilla = run_ensemble.optimize_caruana(oof_matrix, y, n_bags=20, seed=42)
    w_iw = run_ensemble.optimize_caruana(
        oof_matrix, y, n_bags=20, seed=42, sample_weight=iw
    )

    fs_vanilla = family_share(names, w_vanilla)
    fs_iw = family_share(names, w_iw)
    print(f"  vanilla caruana: chemprop family share = {fs_vanilla:.4f}")
    print(f"  IW      caruana: chemprop family share = {fs_iw:.4f}")

    print("\nTop weights:")
    for label, ww in [("vanilla", w_vanilla), ("IW", w_iw)]:
        print(f"  [{label}]")
        for n, x in sorted(zip(names, ww), key=lambda t: -t[1])[:7]:
            in_fam = " (FAM)" if n in CHEMPROP_FAMILY else ""
            print(f"    {n:<55} {x:.4f}{in_fam}")

    print("\n===== 4-way bakeoff =====")
    blends = {
        "vanilla_caruana": oof_matrix @ w_vanilla,
        "iw_caruana": oof_matrix @ w_iw,
    }
    test_blends = {
        "vanilla_caruana": test_matrix @ w_vanilla,
        "iw_caruana": test_matrix @ w_iw,
    }

    rows = []
    for car_name, blend in blends.items():
        # without calibrator
        rows.append({"caruana": car_name, "cal": "none", **report("", blend, y, iw)})
        # with importance calibrator
        oof_cal, _, params = fit_importance_affine(blend, y, iw, test_blends[car_name])
        rows.append(
            {
                "caruana": car_name,
                "cal": f"importance(slope={params['slope']:.3f},int={params['intercept']:.3f})",
                **report("", oof_cal, y, iw),
            }
        )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n===== Family share summary =====")
    print(f"  vanilla: {fs_vanilla:.4f}  (current production)")
    print(f"  IW     : {fs_iw:.4f}  (test-distribution corrected)")
    print(f"  Δ family share: {fs_iw - fs_vanilla:+.4f}")


if __name__ == "__main__":
    main()
