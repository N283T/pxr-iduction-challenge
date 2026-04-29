"""Proximity-gated affine calibrator for the caruana_bag20 ensemble.

Local version of run_ensemble_calibrate_importance.py:
  1. Compute NN-Tanimoto from each train/test compound to the potent-46 set
     (train pec50 >= 6 AND selectivity = pec50 - counter_pec50 >= 1.5).
  2. Split train and test into "near" (NN >= T) and "far" (NN < T) strata,
     where T = median(NN-Tanimoto over test).
  3. Per stratum, fit a LogReg train-vs-test domain classifier (Morgan FP)
     and a weighted LinearRegression on (oof_pred, y_train) using the
     density-ratio sample weights (clipped [1/3, 3]).
  4. Apply each stratum's affine to its test slice, write submission.

Output: track1_activity/submissions/ens_caruana_bag20_calibrated_proximity.csv

LB A/B mandatory regardless of OOF gate (calibrator family changes have a
documented OOF/LB sign-flip history).

Spec: docs/superpowers/specs/2026-04-29-potent46-proximity-gated-calibrator-design.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import spearmanr
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
from data import (  # noqa: E402
    DB_PARAMS,
    load_test_smiles,
    load_train_smiles_target,
    load_train_smiles_with_counter,
)

POTENT_PEC50_THRESHOLD = 6.0
POTENT_SEL_THRESHOLD = 1.5
WEIGHT_CLIP_LO = 1.0 / 3.0
WEIGHT_CLIP_HI = 3.0
MIN_STRATUM_TRAIN = 200


def morgan_matrix(
    smiles_list: list[str], radius: int = 2, n_bits: int = 2048
) -> np.ndarray:
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = gen.GetFingerprint(mol)
        out[i] = np.asarray(fp, dtype=np.uint8)
    return out


def load_potent46_indices() -> np.ndarray:
    """Indices into the train_df row order for potent-46 members.

    potent-46 = train pec50 >= 6 AND selectivity >= 1.5
    selectivity = train_pec50 - counter_pec50 (NaN where counter row missing,
    those rows are excluded).
    """
    df = load_train_smiles_with_counter()
    sel = df["pec50"] - df["counter_pec50"]
    mask = (df["pec50"] >= POTENT_PEC50_THRESHOLD) & (sel >= POTENT_SEL_THRESHOLD)
    indices = np.flatnonzero(mask.to_numpy())
    return indices


def load_caruana_oof_and_test() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    import json

    with psycopg2.connect(**DB_PARAMS) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, hyperparameters FROM experiments
             WHERE name = 'ens_caruana_bag20'
             ORDER BY id DESC LIMIT 1
            """
        )
        exp_id, hp = cur.fetchone()
        weights_map = hp["weights"]
        print(f"  using ens_caruana_bag20 experiment id={exp_id}")
        print(f"  {len(weights_map)} member weights in hyperparameters")

        oof_stack = None
        for name, weight in weights_map.items():
            cur.execute(
                "SELECT id FROM experiments WHERE name = %s ORDER BY id DESC LIMIT 1",
                (name,),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f"member experiment missing: {name}")
            mid = row[0]
            cur.execute(
                """
                SELECT train_idx, oof_prediction
                  FROM experiment_oof_predictions
                 WHERE experiment_id = %s
                 ORDER BY train_idx
                """,
                (mid,),
            )
            rows = cur.fetchall()
            if not rows:
                raise RuntimeError(f"OOF missing for member {name} (id={mid})")
            member = np.asarray([r[1] for r in rows], dtype=np.float64)
            if oof_stack is None:
                oof_stack = np.zeros_like(member)
            oof_stack = oof_stack + weight * member

    oof_preds = oof_stack / sum(weights_map.values())

    sub_path = REPO_ROOT.joinpath(
        "track1_activity", "submissions", "ens_caruana_bag20.csv"
    )
    test_sub = pd.read_csv(sub_path)
    test_col = [c for c in test_sub.columns if c.lower() == "pec50"][0]
    test_preds = test_sub[test_col].to_numpy(dtype=np.float64)
    print(
        f"  reconstructed OOF: n={len(oof_preds)}, "
        f"mean={oof_preds.mean():.4f}, std={oof_preds.std():.4f}"
    )
    _ = json
    return oof_preds, test_preds, test_sub


def compute_nn_tanimoto(
    query_fps: np.ndarray,
    anchor_fps: np.ndarray,
    query_global_idx: np.ndarray | None = None,
    anchor_global_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Max Tanimoto similarity from each query row to any anchor row.

    Both inputs are uint8 (N, n_bits) bit matrices (bits in 0/1).
    Tanimoto = popcount(A AND B) / popcount(A OR B).

    If `query_global_idx` and `anchor_global_idx` are provided, any anchor
    whose global index matches the query's global index is excluded from
    that query's max (self-exclude). Pass None on either side to disable.
    """
    if anchor_fps.shape[0] == 0:
        raise ValueError("anchor_fps is empty")

    # popcount per row, precomputed
    q_pop = query_fps.sum(axis=1).astype(np.int32)  # (Nq,)
    a_pop = anchor_fps.sum(axis=1).astype(np.int32)  # (Na,)

    # Intersection: matrix product of bit matrices (each cell = popcount(q AND a))
    inter = query_fps.astype(np.int32) @ anchor_fps.T.astype(np.int32)  # (Nq, Na)
    union = q_pop[:, None] + a_pop[None, :] - inter  # (Nq, Na)
    # Avoid div-by-zero (both all-zero is impossible for valid mols, but guard)
    tanimoto = np.where(union > 0, inter / np.maximum(union, 1), 0.0)

    # Self-exclude: set tanimoto[i, j] = -inf where query_global_idx[i] == anchor_global_idx[j]
    if query_global_idx is not None and anchor_global_idx is not None:
        # broadcast comparison (Nq, Na)
        mask = query_global_idx[:, None] == anchor_global_idx[None, :]
        tanimoto = np.where(mask, -np.inf, tanimoto)

    nn = tanimoto.max(axis=1)
    # If self-exclude removed every anchor for some query (shouldn't happen
    # unless anchors are a subset of size 1 == query), guard the -inf:
    if np.any(np.isneginf(nn)):
        raise RuntimeError(
            "compute_nn_tanimoto: query has no anchor after self-exclude"
        )
    return nn.astype(np.float64)


def fit_stratum_calibrator(
    train_fps_stratum: np.ndarray,
    test_fps_stratum: np.ndarray,
    oof_stratum: np.ndarray,
    y_stratum: np.ndarray,
    label: str,
) -> tuple[float, float, np.ndarray]:
    """Fit per-stratum domain classifier + weighted affine.

    Returns: (slope, intercept, sample_weights). Prints a one-line summary.
    """
    n_train = len(train_fps_stratum)
    n_test = len(test_fps_stratum)
    if n_train < 2:
        raise RuntimeError(f"stratum '{label}': only {n_train} train rows, cannot fit")
    if n_test < 1:
        raise RuntimeError(f"stratum '{label}': zero test rows")

    # Domain classifier (within stratum only)
    X_all = np.vstack([train_fps_stratum, test_fps_stratum])
    y_all = np.concatenate(
        [np.zeros(n_train, dtype=np.int32), np.ones(n_test, dtype=np.int32)]
    )
    clf = LogisticRegression(max_iter=1000, solver="liblinear", C=1.0, random_state=42)
    clf.fit(X_all, y_all)
    p_test = clf.predict_proba(train_fps_stratum)[:, 1]
    eps = 1e-6
    w = (p_test + eps) / (1.0 - p_test + eps)
    w = w * (n_train / n_test)
    w = np.clip(w, WEIGHT_CLIP_LO, WEIGHT_CLIP_HI)
    w = w * (n_train / w.sum())

    reg = LinearRegression()
    reg.fit(oof_stratum.reshape(-1, 1), y_stratum, sample_weight=w)
    slope = float(reg.coef_[0])
    intercept = float(reg.intercept_)
    print(
        f"  stratum '{label}': n_train={n_train} n_test={n_test} "
        f"w[mean={w.mean():.3f} q25={np.quantile(w, 0.25):.3f} "
        f"q75={np.quantile(w, 0.75):.3f} max={w.max():.3f}]"
        f" -> y = {slope:.4f} * pred + {intercept:.4f}"
    )
    return slope, intercept, w


def main() -> None:
    print("Loading data ...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    oof_preds, test_preds, test_sub = load_caruana_oof_and_test()

    y_train = train_df["pec50"].to_numpy(dtype=np.float64)
    assert len(y_train) == len(oof_preds), (
        f"y_train {len(y_train)} vs oof {len(oof_preds)}"
    )
    assert len(test_df) == len(test_preds), (
        f"test_df {len(test_df)} vs test_preds {len(test_preds)}"
    )
    print(f"  train={len(train_df)}, test={len(test_df)}")
    raw_train_mae = float(np.mean(np.abs(oof_preds - y_train)))
    raw_train_sp = float(spearmanr(oof_preds, y_train).statistic)
    print(f"  raw OOF MAE={raw_train_mae:.4f}  Spearman={raw_train_sp:.4f}")

    print("Computing Morgan fingerprints (train + test) ...")
    X_train_fp = morgan_matrix(train_df["smiles"].tolist())
    X_test_fp = morgan_matrix(test_df["smiles"].tolist())
    print(f"  X_train_fp {X_train_fp.shape}, X_test_fp {X_test_fp.shape}")

    print("Loading potent-46 indices (train pec50>=6, sel>=1.5) ...")
    potent_idx = load_potent46_indices()
    print(f"  potent-46 size: {len(potent_idx)}")
    if len(potent_idx) == 0:
        raise RuntimeError("potent-46 set is empty — check thresholds / counter join")

    potent_fps = X_train_fp[potent_idx]
    train_global_idx = np.arange(len(train_df))
    test_global_idx = np.full(
        len(test_df), -1, dtype=np.int64
    )  # never matches potent_idx (>=0)

    print("Computing NN-Tanimoto to potent-46 ...")
    prox_train = compute_nn_tanimoto(
        X_train_fp,
        potent_fps,
        query_global_idx=train_global_idx,
        anchor_global_idx=potent_idx,
    )
    prox_test = compute_nn_tanimoto(
        X_test_fp,
        potent_fps,
        query_global_idx=test_global_idx,
        anchor_global_idx=potent_idx,
    )
    print(
        f"  prox_train: mean={prox_train.mean():.4f} median={np.median(prox_train):.4f} "
        f"q25={np.quantile(prox_train, 0.25):.4f} q75={np.quantile(prox_train, 0.75):.4f}"
    )
    print(
        f"  prox_test:  mean={prox_test.mean():.4f} median={np.median(prox_test):.4f} "
        f"q25={np.quantile(prox_test, 0.25):.4f} q75={np.quantile(prox_test, 0.75):.4f}"
    )

    threshold = float(np.median(prox_test))
    print(f"  threshold T = test median = {threshold:.4f}")
    near_train = prox_train >= threshold
    near_test = prox_test >= threshold
    print(
        f"  train near={near_train.sum()} far={(~near_train).sum()} | "
        f"test near={near_test.sum()} far={(~near_test).sum()}"
    )

    if near_train.sum() < MIN_STRATUM_TRAIN or (~near_train).sum() < MIN_STRATUM_TRAIN:
        print(
            f"  GATE FAIL: train stratum size below MIN_STRATUM_TRAIN={MIN_STRATUM_TRAIN}"
        )
        raise SystemExit(1)

    print("Fitting per-stratum calibrators ...")
    slope_n, int_n, _ = fit_stratum_calibrator(
        X_train_fp[near_train],
        X_test_fp[near_test],
        oof_preds[near_train],
        y_train[near_train],
        label="near",
    )
    slope_f, int_f, _ = fit_stratum_calibrator(
        X_train_fp[~near_train],
        X_test_fp[~near_test],
        oof_preds[~near_train],
        y_train[~near_train],
        label="far",
    )

    # Apply per-stratum to OOF and test
    oof_cal = np.empty_like(oof_preds)
    oof_cal[near_train] = slope_n * oof_preds[near_train] + int_n
    oof_cal[~near_train] = slope_f * oof_preds[~near_train] + int_f
    test_cal = np.empty_like(test_preds)
    test_cal[near_test] = slope_n * test_preds[near_test] + int_n
    test_cal[~near_test] = slope_f * test_preds[~near_test] + int_f

    # Per-stratum + global metrics
    def _mae(a, b):
        return float(np.mean(np.abs(a - b)))

    def _sp(a, b):
        return float(spearmanr(a, b).statistic)

    near_mae_raw = _mae(oof_preds[near_train], y_train[near_train])
    near_mae_cal = _mae(oof_cal[near_train], y_train[near_train])
    far_mae_raw = _mae(oof_preds[~near_train], y_train[~near_train])
    far_mae_cal = _mae(oof_cal[~near_train], y_train[~near_train])
    global_mae_raw = _mae(oof_preds, y_train)
    global_mae_cal = _mae(oof_cal, y_train)
    global_sp_raw = _sp(oof_preds, y_train)
    global_sp_cal = _sp(oof_cal, y_train)
    print("Per-stratum OOF MAE:")
    print(
        f"  near: raw={near_mae_raw:.4f}  cal={near_mae_cal:.4f}  Δ={near_mae_cal - near_mae_raw:+.4f}"
    )
    print(
        f"  far : raw={far_mae_raw:.4f}  cal={far_mae_cal:.4f}  Δ={far_mae_cal - far_mae_raw:+.4f}"
    )
    print(
        f"  global: raw MAE={global_mae_raw:.4f}  cal MAE={global_mae_cal:.4f}  "
        f"Δ MAE={global_mae_cal - global_mae_raw:+.4f}\n"
        f"          raw Sp ={global_sp_raw:.4f}  cal Sp ={global_sp_cal:.4f}  "
        f"Δ Sp ={global_sp_cal - global_sp_raw:+.4f}"
    )

    # Comparison with global importance baseline (re-fit here, no shelling out)
    print("Re-fitting global importance baseline for direct comparison ...")
    X_all = np.vstack([X_train_fp, X_test_fp])
    y_all = np.concatenate(
        [
            np.zeros(len(X_train_fp), dtype=np.int32),
            np.ones(len(X_test_fp), dtype=np.int32),
        ]
    )
    clf_g = LogisticRegression(
        max_iter=1000, solver="liblinear", C=1.0, random_state=42
    )
    clf_g.fit(X_all, y_all)
    p_test_g = clf_g.predict_proba(X_train_fp)[:, 1]
    w_g = ((p_test_g + 1e-6) / (1.0 - p_test_g + 1e-6)) * (
        len(X_train_fp) / len(X_test_fp)
    )
    w_g = np.clip(w_g, WEIGHT_CLIP_LO, WEIGHT_CLIP_HI)
    w_g = w_g * (len(w_g) / w_g.sum())
    reg_g = LinearRegression()
    reg_g.fit(oof_preds.reshape(-1, 1), y_train, sample_weight=w_g)
    oof_g = float(reg_g.coef_[0]) * oof_preds + float(reg_g.intercept_)
    g_mae = _mae(oof_g, y_train)
    g_sp = _sp(oof_g, y_train)
    print(
        f"  global importance affine: y = {float(reg_g.coef_[0]):.4f} * pred + {float(reg_g.intercept_):.4f}\n"
        f"  global importance OOF: MAE={g_mae:.4f}  Sp={g_sp:.4f}"
    )

    # Gate
    mae_ok = global_mae_cal <= g_mae
    sp_ok = global_sp_cal >= g_sp - 0.005
    near_san = (near_mae_cal - near_mae_raw) <= 0.01
    far_san = (far_mae_cal - far_mae_raw) <= 0.01
    print("\nGATE CHECK:")
    print(
        f"  proximity OOF MAE <= global importance MAE: {mae_ok} ({global_mae_cal:.4f} vs {g_mae:.4f})"
    )
    print(
        f"  proximity OOF Sp >= global importance Sp - 0.005: {sp_ok} ({global_sp_cal:.4f} vs {g_sp - 0.005:.4f})"
    )
    print(
        f"  near stratum cal-vs-raw within +0.01: {near_san} (Δ={near_mae_cal - near_mae_raw:+.4f})"
    )
    print(
        f"  far stratum cal-vs-raw within +0.01: {far_san} (Δ={far_mae_cal - far_mae_raw:+.4f})"
    )
    print(f"  GATE PASS = {mae_ok and sp_ok and near_san and far_san}")

    # Always write the submission (gate informs LB decision, not file write)
    out_sub = test_sub.copy()
    test_col = [c for c in out_sub.columns if c.lower() == "pec50"][0]
    out_sub[test_col] = test_cal
    out_path = REPO_ROOT.joinpath(
        "track1_activity", "submissions", "ens_caruana_bag20_calibrated_proximity.csv"
    )
    out_sub.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path}")
    print(f"  test mean: {test_cal.mean():.4f}, std: {test_cal.std():.4f}")


if __name__ == "__main__":
    main()
