"""Anchor residual model on top of caruana_bag20 + importance calibrator.

Codex proposal #2 (2026-04-29): predict the per-compound residual of the
calibrated base model using anchor (potent-46) features only, then add a
damped correction `final = base + alpha * residual_hat`.

Pipeline:
  1. Reconstruct caruana_bag20 OOF + test preds, apply importance affine
     (re-fit inline, same recipe as run_ensemble_calibrate_importance.py).
  2. For each train + test compound, compute 4 anchor features:
       nn_tanimoto: NN Tanimoto to potent-46 (train self-excluded)
       anchor_pec50: pEC50 of the nearest potent anchor
       base_pred: calibrated base prediction
       pred_minus_anchor: base_pred - anchor_pec50
  3. UMAP 5-fold CV: per fold, fit small LGBM on (k-1) folds' residuals
     (target = y - base_pred), predict on held-out fold -> OOF residual_hat.
  4. corrected_oof = base_pred_oof + alpha * residual_hat (alpha=0.5).
  5. Fit residual model on ALL train, predict residual for test.
  6. Apply correction, write submission.

Output: track1_activity/submissions/ens_caruana_bag20_anchor_residual.csv

LB A/B mandatory regardless of OOF gate.

Legacy experiment script; internal design note was removed from the public repository.
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
from sklearn.linear_model import LinearRegression, LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
from data import (  # noqa: E402
    DB_PARAMS,
    load_test_smiles,
    load_train_smiles_target,
    load_train_smiles_with_counter,
)
from splits import umap_split_indices  # noqa: E402

POTENT_PEC50_THRESHOLD = 6.0
POTENT_SEL_THRESHOLD = 1.5
WEIGHT_CLIP_LO = 1.0 / 3.0
WEIGHT_CLIP_HI = 3.0
ALPHA = 0.4  # mid-plateau of OOF MAE sweep (LinearRegression residual)
N_SPLITS = 5
N_CLUSTERS = 50
SEED = 42
RESIDUAL_MAX_WARN = 0.5
RESIDUAL_MAX_ABORT = 1.0
RESIDUAL_Q99_WARN = 0.3


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


def load_potent46_indices_and_pec50() -> tuple[np.ndarray, np.ndarray]:
    """potent-46 row indices into train_df + their pEC50 values.

    potent-46 = pec50 >= 6 AND pec50 - counter_pec50 >= 1.5.
    """
    df = load_train_smiles_with_counter()
    sel = df["pec50"] - df["counter_pec50"]
    mask = (df["pec50"] >= POTENT_PEC50_THRESHOLD) & (sel >= POTENT_SEL_THRESHOLD)
    indices = np.flatnonzero(mask.to_numpy())
    pec50_values = df.loc[indices, "pec50"].to_numpy(dtype=np.float64)
    return indices, pec50_values


def compute_nn_with_anchor(
    query_fps: np.ndarray,
    anchor_fps: np.ndarray,
    anchor_pec50: np.ndarray,
    query_global_idx: np.ndarray | None = None,
    anchor_global_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """For each query, return (NN Tanimoto, NN's pEC50) against anchors.

    Tanimoto = popcount(A AND B) / popcount(A OR B). Self-exclude masks
    matching (query, anchor) pairs to -inf. Returns the argmax anchor's
    pec50 alongside the max similarity.
    """
    if anchor_fps.shape[0] == 0:
        raise ValueError("anchor_fps is empty")
    q_pop = query_fps.sum(axis=1).astype(np.int32)
    a_pop = anchor_fps.sum(axis=1).astype(np.int32)
    inter = query_fps.astype(np.int32) @ anchor_fps.T.astype(np.int32)
    union = q_pop[:, None] + a_pop[None, :] - inter
    tanimoto = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    if query_global_idx is not None and anchor_global_idx is not None:
        mask = query_global_idx[:, None] == anchor_global_idx[None, :]
        tanimoto = np.where(mask, -np.inf, tanimoto)
    nn_idx = tanimoto.argmax(axis=1)
    nn_tan = tanimoto[np.arange(len(tanimoto)), nn_idx]
    if np.any(np.isneginf(nn_tan)):
        raise RuntimeError(
            "compute_nn_with_anchor: query has no anchor after self-exclude"
        )
    nn_pec50 = anchor_pec50[nn_idx]
    return nn_tan.astype(np.float64), nn_pec50.astype(np.float64)


def load_caruana_oof_and_test() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Reconstruct ens_caruana_bag20 OOF and load test preds (raw, pre-calib)."""
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
        print(
            f"  using ens_caruana_bag20 experiment id={exp_id}, {len(weights_map)} members"
        )

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

    oof_preds_raw = oof_stack / sum(weights_map.values())

    sub_path = REPO_ROOT.joinpath(
        "track1_activity", "submissions", "ens_caruana_bag20.csv"
    )
    test_sub = pd.read_csv(sub_path)
    test_col = [c for c in test_sub.columns if c.lower() == "pec50"][0]
    test_preds_raw = test_sub[test_col].to_numpy(dtype=np.float64)
    print(
        f"  reconstructed OOF: n={len(oof_preds_raw)}, "
        f"mean={oof_preds_raw.mean():.4f}, std={oof_preds_raw.std():.4f}"
    )
    return oof_preds_raw, test_preds_raw, test_sub


def fit_global_importance_affine(
    X_train_fp: np.ndarray,
    X_test_fp: np.ndarray,
    oof_raw: np.ndarray,
    y_train: np.ndarray,
) -> tuple[float, float]:
    """Fit the global importance-weighted affine, return (slope, intercept)."""
    X_all = np.vstack([X_train_fp, X_test_fp])
    y_all = np.concatenate(
        [
            np.zeros(len(X_train_fp), dtype=np.int32),
            np.ones(len(X_test_fp), dtype=np.int32),
        ]
    )
    clf = LogisticRegression(max_iter=1000, solver="liblinear", C=1.0, random_state=42)
    clf.fit(X_all, y_all)
    p_test = clf.predict_proba(X_train_fp)[:, 1]
    eps = 1e-6
    w = (p_test + eps) / (1.0 - p_test + eps)
    w = w * (len(X_train_fp) / len(X_test_fp))
    w = np.clip(w, WEIGHT_CLIP_LO, WEIGHT_CLIP_HI)
    w = w * (len(w) / w.sum())
    reg = LinearRegression()
    reg.fit(oof_raw.reshape(-1, 1), y_train, sample_weight=w)
    return float(reg.coef_[0]), float(reg.intercept_)


def make_residual_model() -> LinearRegression:
    """Linear residual model (chosen after LGBM overfit observed in v1).

    LGBM with 4 anchor features overfits the per-fold noise: best alpha
    only at 0.05-0.15 with -0.0001 MAE, then degrades quickly. Linear
    finds a clean monotone alpha plateau (-0.0006 at 0.30-0.50) driven
    almost entirely by the nn_tanimoto coefficient (+0.35; other coefs
    < |0.04|), confirming the residual signal is "base under-predicts
    for potent-46 analogs". Linear's restricted hypothesis space is the
    right inductive bias here.
    """
    return LinearRegression()


def main() -> None:
    print("Loading data ...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    oof_raw, test_raw, test_sub = load_caruana_oof_and_test()

    y_train = train_df["pec50"].to_numpy(dtype=np.float64)
    assert len(y_train) == len(oof_raw)
    assert len(test_df) == len(test_raw)
    print(f"  train={len(train_df)}, test={len(test_df)}")

    print("Computing Morgan fingerprints (train + test) ...")
    X_train_fp = morgan_matrix(train_df["smiles"].tolist())
    X_test_fp = morgan_matrix(test_df["smiles"].tolist())
    print(f"  X_train_fp {X_train_fp.shape}, X_test_fp {X_test_fp.shape}")

    print("Fitting global importance affine (base calibrator) ...")
    slope_g, intercept_g = fit_global_importance_affine(
        X_train_fp, X_test_fp, oof_raw, y_train
    )
    print(f"  affine: y = {slope_g:.4f} * pred + {intercept_g:.4f}")
    base_oof = slope_g * oof_raw + intercept_g
    base_test = slope_g * test_raw + intercept_g

    base_oof_mae = float(np.mean(np.abs(base_oof - y_train)))
    base_oof_sp = float(spearmanr(base_oof, y_train).statistic)
    print(f"  base OOF: MAE={base_oof_mae:.4f}  Sp={base_oof_sp:.4f}")

    print("Loading potent-46 anchors ...")
    potent_idx, potent_pec50 = load_potent46_indices_and_pec50()
    print(
        f"  potent-46 size: {len(potent_idx)} "
        f"(pec50: mean={potent_pec50.mean():.2f}, min={potent_pec50.min():.2f}, max={potent_pec50.max():.2f})"
    )
    if len(potent_idx) == 0:
        raise RuntimeError("potent-46 set is empty")

    print("Computing anchor features ...")
    train_global_idx = np.arange(len(train_df))
    test_global_idx = np.full(len(test_df), -1, dtype=np.int64)
    nn_train, anchor_pec50_train = compute_nn_with_anchor(
        X_train_fp,
        X_train_fp[potent_idx],
        potent_pec50,
        query_global_idx=train_global_idx,
        anchor_global_idx=potent_idx,
    )
    nn_test, anchor_pec50_test = compute_nn_with_anchor(
        X_test_fp,
        X_train_fp[potent_idx],
        potent_pec50,
        query_global_idx=test_global_idx,
        anchor_global_idx=potent_idx,
    )
    pred_minus_anchor_train = base_oof - anchor_pec50_train
    pred_minus_anchor_test = base_test - anchor_pec50_test

    feat_train = np.column_stack(
        [nn_train, anchor_pec50_train, base_oof, pred_minus_anchor_train]
    )
    feat_test = np.column_stack(
        [nn_test, anchor_pec50_test, base_test, pred_minus_anchor_test]
    )
    feat_names = ["nn_tanimoto", "anchor_pec50", "base_pred", "pred_minus_anchor"]
    print(f"  feat_train {feat_train.shape}, feat_test {feat_test.shape}")
    for i, name in enumerate(feat_names):
        v = feat_train[:, i]
        print(
            f"    {name:>20}: mean={v.mean():.3f} q25={np.quantile(v, 0.25):.3f} "
            f"q75={np.quantile(v, 0.75):.3f} min={v.min():.3f} max={v.max():.3f}"
        )

    residual_target = y_train - base_oof
    print(
        f"  residual target stats: mean={residual_target.mean():+.4f} "
        f"std={residual_target.std():.4f} "
        f"q25={np.quantile(residual_target, 0.25):+.4f} "
        f"q75={np.quantile(residual_target, 0.75):+.4f}"
    )

    print(
        f"Building UMAP {N_SPLITS}-fold split (Morgan+Jaccard, k={N_CLUSTERS}, seed={SEED}) ..."
    )
    folds = umap_split_indices(
        train_df["smiles"].tolist(),
        n_splits=N_SPLITS,
        n_clusters=N_CLUSTERS,
        seed=SEED,
    )
    print(f"  built {len(folds)} folds")

    print("Cross-validating residual model ...")
    residual_oof = np.zeros_like(residual_target)
    fold_mae = []
    for fi, (tr_idx, va_idx) in enumerate(folds):
        model = make_residual_model()
        model.fit(feat_train[tr_idx], residual_target[tr_idx])
        residual_oof[va_idx] = model.predict(feat_train[va_idx])
        fold_residual_mae = float(
            np.mean(np.abs(residual_oof[va_idx] - residual_target[va_idx]))
        )
        fold_mae.append(fold_residual_mae)
        print(
            f"  fold {fi}: train={len(tr_idx)} val={len(va_idx)} "
            f"residual-MAE={fold_residual_mae:.4f}"
        )
    print(
        f"  per-fold residual MAE: mean={np.mean(fold_mae):.4f} std={np.std(fold_mae):.4f}"
    )

    abs_resid_oof = np.abs(residual_oof)
    print(
        f"  |residual_hat| OOF: mean={abs_resid_oof.mean():.4f} "
        f"q99={np.quantile(abs_resid_oof, 0.99):.4f} max={abs_resid_oof.max():.4f}"
    )

    corrected_oof = base_oof + ALPHA * residual_oof
    corrected_oof_mae = float(np.mean(np.abs(corrected_oof - y_train)))
    corrected_oof_sp = float(spearmanr(corrected_oof, y_train).statistic)
    print(
        f"  corrected OOF (alpha={ALPHA}): MAE={corrected_oof_mae:.4f}  "
        f"Sp={corrected_oof_sp:.4f}  "
        f"Δ MAE={corrected_oof_mae - base_oof_mae:+.4f}  "
        f"Δ Sp={corrected_oof_sp - base_oof_sp:+.4f}"
    )

    print("Fitting residual model on ALL train (for test prediction) ...")
    full_model = make_residual_model()
    full_model.fit(feat_train, residual_target)
    residual_test = full_model.predict(feat_test)
    abs_resid_test = np.abs(residual_test)
    print(
        f"  |residual_hat| test: mean={abs_resid_test.mean():.4f} "
        f"q99={np.quantile(abs_resid_test, 0.99):.4f} max={abs_resid_test.max():.4f}"
    )

    test_corrected = base_test + ALPHA * residual_test
    print(f"  test base mean={base_test.mean():.4f} std={base_test.std():.4f}")
    print(
        f"  test corrected mean={test_corrected.mean():.4f} std={test_corrected.std():.4f}"
    )

    # Linear coefficients (sanity)
    print("Residual linear coefficients (full-train fit):")
    for name, coef in zip(feat_names, full_model.coef_):
        print(f"    {name:>20}: {coef:+.4f}")
    print(f"    {'intercept':>20}: {full_model.intercept_:+.4f}")

    # Gate
    mae_ok = corrected_oof_mae <= base_oof_mae
    sp_ok = corrected_oof_sp >= base_oof_sp - 0.005
    max_ok = abs_resid_oof.max() <= RESIDUAL_MAX_ABORT
    max_warn = abs_resid_oof.max() <= RESIDUAL_MAX_WARN
    q99_warn = np.quantile(abs_resid_oof, 0.99) <= RESIDUAL_Q99_WARN
    print("\nGATE CHECK:")
    print(
        f"  corrected OOF MAE <= base OOF MAE: {mae_ok} ({corrected_oof_mae:.4f} vs {base_oof_mae:.4f})"
    )
    print(
        f"  corrected OOF Sp >= base OOF Sp - 0.005: {sp_ok} ({corrected_oof_sp:.4f} vs {base_oof_sp - 0.005:.4f})"
    )
    print(
        f"  max |residual_hat| <= {RESIDUAL_MAX_ABORT}: {max_ok} (max={abs_resid_oof.max():.4f})"
    )
    print(f"  max |residual_hat| <= {RESIDUAL_MAX_WARN} (warn): {max_warn}")
    print(f"  q99 |residual_hat| <= {RESIDUAL_Q99_WARN} (warn): {q99_warn}")
    print(f"  GATE PASS = {mae_ok and sp_ok and max_ok}")

    # Always write submission
    out_sub = test_sub.copy()
    test_col = [c for c in out_sub.columns if c.lower() == "pec50"][0]
    out_sub[test_col] = test_corrected
    out_path = REPO_ROOT.joinpath(
        "track1_activity", "submissions", "ens_caruana_bag20_anchor_residual.csv"
    )
    out_sub.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
