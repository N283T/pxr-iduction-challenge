"""Per-bin affine calibrator (MLPlatt-lite, no MLP).

After MLPlatt MLP (run_mlplatt_calibrator.py) gave +0.0095 to +0.0473 OOF MAE
across 5 context variants -- failing because 4-layer MLP overfits 4140
samples -- this script tests the simplest possible context-conditioned
calibrator:

  Per-bin affine: for each context bin b, fit y = a_b * pred + b_b independently.

Variants:
  V1 by_potency: 4 bins of base prediction quartile (4 x 2 params = 8 total)
  V2 by_nn_potent46: 4 bins of NN-Tanimoto to potent-46
  V3 by_potency_x_nn: 4 x 4 = 16 bins (potency x nn quartile)

This is much simpler than MLPlatt (8-16 params vs ~500 params) and should
not overfit. If even this can't beat the global importance affine, then
context-conditioning calibration is structurally dead on PXR at this scale.

Includes monotonic post-processing: if any bin's slope < 0, it's clipped
to 0 (since negative slope inverts ranking within bin).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
from data import (  # noqa: E402
    DB_PARAMS,
    load_test_smiles,
    load_train_smiles_target,
    load_train_smiles_with_counter,
)
from splits import _morgan_fp_matrix, umap_split_indices  # noqa: E402

OUT_DIR = REPO_ROOT.joinpath("docs", "superpowers", "runs")
SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
SEED = 42
N_SPLITS = 5
N_CLUSTERS = 50
POTENT_PEC50 = 6.0
POTENT_SEL = 1.5


def load_pool() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    with psycopg2.connect(**DB_PARAMS) as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT id, hyperparameters FROM experiments WHERE name = 'ens_caruana_bag20'
               ORDER BY id DESC LIMIT 1"""
        )
        _, hp = cur.fetchone()
        weights = hp["weights"]
        oofs: dict[str, np.ndarray] = {}
        for name in weights:
            cur.execute(
                "SELECT id FROM experiments WHERE name = %s ORDER BY id DESC LIMIT 1",
                (name,),
            )
            mid = cur.fetchone()[0]
            cur.execute(
                """SELECT train_idx, oof_prediction FROM experiment_oof_predictions
                   WHERE experiment_id = %s ORDER BY train_idx""",
                (mid,),
            )
            rows = cur.fetchall()
            oofs[name] = np.asarray([r[1] for r in rows], dtype=np.float64)
    test_preds: dict[str, np.ndarray] = {}
    for name in weights:
        sub = pd.read_csv(SUBMISSION_DIR.joinpath(f"{name}.csv"))
        col = [c for c in sub.columns if c.lower() == "pec50"][0]
        test_preds[name] = sub[col].to_numpy(dtype=np.float64)
    return oofs, test_preds, weights


def potent46_idx() -> np.ndarray:
    df = load_train_smiles_with_counter()
    sel = df["pec50"] - df["counter_pec50"]
    mask = (df["pec50"] >= POTENT_PEC50) & (sel >= POTENT_SEL)
    return np.flatnonzero(mask.to_numpy())


def nn_tanimoto(fps: np.ndarray, anchor_fps: np.ndarray, exclude_self: bool) -> np.ndarray:
    q_pop = fps.sum(axis=1).astype(np.int32)
    a_pop = anchor_fps.sum(axis=1).astype(np.int32)
    inter = fps.astype(np.int32) @ anchor_fps.T.astype(np.int32)
    union = q_pop[:, None] + a_pop[None, :] - inter
    sim = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    if exclude_self:
        sim = np.where(sim >= 0.999, -np.inf, sim)
    return sim.max(axis=1).astype(np.float64)


def per_bin_affine_oof(
    oof: np.ndarray, y: np.ndarray, w: np.ndarray, bins: np.ndarray, folds: list
) -> np.ndarray:
    """Per-bin affine OOF prediction. For each fold, fit affine within each
    bin using train-fold rows, predict on val-fold rows.
    """
    out = np.zeros_like(y)
    n_bins = int(bins.max() + 1)
    for tr, va in folds:
        for b in range(n_bins):
            tr_mask = bins[tr] == b
            va_mask = bins[va] == b
            if tr_mask.sum() < 30 or va_mask.sum() == 0:
                # Fall back to global affine for this bin/fold combo
                reg = LinearRegression()
                reg.fit(oof[tr].reshape(-1, 1), y[tr], sample_weight=w[tr])
                if va_mask.sum() > 0:
                    out[va[va_mask]] = reg.predict(oof[va[va_mask]].reshape(-1, 1))
                continue
            reg = LinearRegression()
            reg.fit(
                oof[tr][tr_mask].reshape(-1, 1),
                y[tr][tr_mask],
                sample_weight=w[tr][tr_mask],
            )
            slope = float(reg.coef_[0])
            # Monotonicity guard: if slope < 0, fall back to slope=0 (constant)
            # which preserves inverted -> stable but discards score info.
            if slope < 0:
                # Use intercept-only (constant prediction = bin mean)
                bin_mean = float(np.average(y[tr][tr_mask], weights=w[tr][tr_mask]))
                out[va[va_mask]] = bin_mean
            else:
                out[va[va_mask]] = reg.predict(oof[va][va_mask].reshape(-1, 1))
    return out


def per_bin_affine_full(
    oof: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    bins_train: np.ndarray,
    test_pred: np.ndarray,
    bins_test: np.ndarray,
) -> np.ndarray:
    """Train affine per bin on FULL train, apply to test."""
    out = np.zeros_like(test_pred)
    n_bins = int(max(bins_train.max(), bins_test.max()) + 1)
    # Global fallback
    g_reg = LinearRegression()
    g_reg.fit(oof.reshape(-1, 1), y, sample_weight=w)
    for b in range(n_bins):
        tr_mask = bins_train == b
        te_mask = bins_test == b
        if te_mask.sum() == 0:
            continue
        if tr_mask.sum() < 30:
            out[te_mask] = g_reg.predict(test_pred[te_mask].reshape(-1, 1))
            continue
        reg = LinearRegression()
        reg.fit(
            oof[tr_mask].reshape(-1, 1),
            y[tr_mask],
            sample_weight=w[tr_mask],
        )
        slope = float(reg.coef_[0])
        if slope < 0:
            bin_mean = float(np.average(y[tr_mask], weights=w[tr_mask]))
            out[te_mask] = bin_mean
        else:
            out[te_mask] = reg.predict(test_pred[te_mask].reshape(-1, 1))
    return out


def main() -> None:
    print("Loading data ...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y_train = train_df["pec50"].to_numpy(dtype=np.float64)
    train_smiles = train_df["smiles"].tolist()
    test_smiles = test_df["smiles"].tolist()

    X_tr_fp = _morgan_fp_matrix(train_smiles)
    X_te_fp = _morgan_fp_matrix(test_smiles)

    print("Loading 9-pool ...")
    oofs, test_preds, global_w = load_pool()
    norm = sum(global_w.values())
    base_oof = np.zeros_like(y_train)
    base_test = np.zeros(len(test_df), dtype=np.float64)
    for name, w in global_w.items():
        base_oof += (w / norm) * oofs[name]
        base_test += (w / norm) * test_preds[name]

    from importance_weights import compute_importance_weights  # noqa: E402

    iw = compute_importance_weights(train_smiles, test_smiles)

    # Reference: global importance affine
    g_reg = LinearRegression()
    g_reg.fit(base_oof.reshape(-1, 1), y_train, sample_weight=iw)
    slope_g = float(g_reg.coef_[0])
    intercept_g = float(g_reg.intercept_)
    base_oof_cal = slope_g * base_oof + intercept_g
    base_test_cal = slope_g * base_test + intercept_g
    base_cal_mae = float(np.mean(np.abs(base_oof_cal - y_train)))
    base_sp = float(spearmanr(base_oof_cal, y_train).statistic)
    print(f"  global affine: slope={slope_g:.4f} intercept={intercept_g:.4f}")
    print(f"  base cal OOF MAE={base_cal_mae:.4f}  Sp={base_sp:.4f}")

    # === Bins ===
    # Potency bin (quartiles of base_oof_cal)
    pec_edges = np.quantile(base_oof_cal, [0.25, 0.5, 0.75])
    potency_train = np.digitize(base_oof_cal, pec_edges).astype(np.int64)
    potency_test = np.digitize(base_test_cal, pec_edges).astype(np.int64)
    # NN-potent46 quartiles
    pot_idx = potent46_idx()
    nn_train = nn_tanimoto(X_tr_fp, X_tr_fp[pot_idx], exclude_self=True)
    nn_test = nn_tanimoto(X_te_fp, X_tr_fp[pot_idx], exclude_self=False)
    nn_edges = np.quantile(nn_train, [0.25, 0.5, 0.75])
    nn_train_bin = np.digitize(nn_train, nn_edges).astype(np.int64)
    nn_test_bin = np.digitize(nn_test, nn_edges).astype(np.int64)

    # Cross product (4x4 = 16 bins)
    pn_train = potency_train * 4 + nn_train_bin
    pn_test = potency_test * 4 + nn_test_bin

    print("\n  bin sizes:")
    print(f"    potency train: {np.bincount(potency_train)}  test: {np.bincount(potency_test)}")
    print(f"    nn      train: {np.bincount(nn_train_bin)}  test: {np.bincount(nn_test_bin)}")
    print(
        f"    cross   train: {np.bincount(pn_train, minlength=16)}  "
        f"test: {np.bincount(pn_test, minlength=16)}"
    )

    folds = umap_split_indices(train_smiles, n_splits=N_SPLITS, n_clusters=N_CLUSTERS, seed=SEED)

    print("\n=== Per-bin affine OOF bake-off ===\n")
    rows = []
    test_predictions = {}
    for variant, bins_tr, bins_te in [
        ("by_potency_4bin", potency_train, potency_test),
        ("by_nn_potent46_4bin", nn_train_bin, nn_test_bin),
        ("by_potency_x_nn_16bin", pn_train, pn_test),
    ]:
        # OOF
        out = per_bin_affine_oof(base_oof, y_train, iw, bins_tr, folds)
        cal_mae = float(np.mean(np.abs(out - y_train)))
        sp = float(spearmanr(out, y_train).statistic)
        d_m2 = cal_mae - base_cal_mae
        d_sp = sp - base_sp
        m2_pass = d_m2 <= -0.003
        sp_pass = d_sp >= -0.002
        all_pass = m2_pass and sp_pass

        # Test prediction
        test_out = per_bin_affine_full(
            base_oof, y_train, iw, bins_tr, base_test, bins_te
        )

        print(
            f"  {variant:>26}  cal_MAE={cal_mae:.4f}  ΔM2={d_m2:+.4f}  "
            f"Sp={sp:.4f}  ΔSp={d_sp:+.4f}  gate={'PASS' if all_pass else 'fail'}"
        )
        rows.append({
            "variant": variant,
            "cal_mae": cal_mae,
            "d_m2": d_m2,
            "sp": sp,
            "d_sp": d_sp,
            "all_pass": all_pass,
        })
        test_predictions[variant] = test_out

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR.joinpath("2026-04-30-perbin-affine.csv"), index=False)

    # Save submission CSVs for any that pass
    print("\n=== Submission CSV generation ===\n")
    test_smiles_list = test_df["smiles"].tolist()
    test_names = test_df["molecule_name"].astype(str).tolist()
    for row in rows:
        variant = row["variant"]
        out_path = SUBMISSION_DIR.joinpath(
            f"ens_caruana_bag20_perbin_{variant}.csv"
        )
        out_df = pd.DataFrame({
            "SMILES": test_smiles_list,
            "Molecule Name": test_names,
            "pEC50": test_predictions[variant],
        })
        out_df.to_csv(out_path, index=False)
        print(
            f"  saved {out_path.name}  test mean={np.mean(test_predictions[variant]):.3f}  "
            f"std={np.std(test_predictions[variant]):.3f}  (gate={row['all_pass']})"
        )

    if any(r["all_pass"] for r in rows):
        best = min((r for r in rows if r["all_pass"]), key=lambda r: r["d_m2"])
        print(
            f"\n  RECOMMEND: {best['variant']} ΔM2={best['d_m2']:+.4f} ΔSp={best['d_sp']:+.4f}"
        )
    else:
        print("\n  All variants FAIL strict gate. Context-conditioned calibration is dead on PXR.")


if __name__ == "__main__":
    main()
