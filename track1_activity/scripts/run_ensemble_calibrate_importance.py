"""Importance-weighted linear calibrator for the caruana_bag20 ensemble.

Covariate-shift correction:
  1. Train an LGBM classifier that distinguishes train from test compounds
     using Morgan r=2 2048-bit fingerprints.
  2. Per-train-compound importance weight w(x) = P(test|x) / (1 - P(test|x))
     (Bayes density ratio).
  3. Weighted linear regression (slope, intercept) on OOF caruana pred vs
     y_train, then apply the affine to test caruana pred.

Output: track1_activity/submissions/ens_caruana_bag20_calibrated_importance.csv

This replicates the recipe that produced the 2026-04-21 evening LB MAE 0.4154
(rank 5) submission. Kept as a standalone script because it's outside the
nested-CV 4-way selection in run_ensemble_calibrate.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
from data import DB_PARAMS, load_test_smiles, load_train_smiles_target  # noqa: E402


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


def load_caruana_oof_and_test() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Reconstruct ens_caruana_bag20 OOF from member OOF + stored weights.

    Ensemble experiments don't write to experiment_oof_predictions (only
    single-model experiments do), so we pull the weights from the ensemble
    experiment's hyperparameters JSON and weighted-sum each member's OOF.
    """
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

        # Resolve each member experiment id by name (latest occurrence)
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

    oof_preds = oof_stack / sum(weights_map.values())  # normalise if not sum-to-1

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
    _ = json  # keep import used (linter)
    return oof_preds, test_preds, test_sub


def main() -> None:
    print("Loading data ...")
    train_df = load_train_smiles_target()  # columns: compound_id, smiles, pec50
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
    print(
        f"  raw OOF MAE vs truth: "
        f"{np.mean(np.abs(oof_preds - y_train)):.4f}, "
        f"mean test pred: {test_preds.mean():.4f}, std: {test_preds.std():.4f}"
    )

    print("Computing Morgan fingerprints ...")
    X_train_fp = morgan_matrix(train_df["smiles"].tolist())
    X_test_fp = morgan_matrix(test_df["smiles"].tolist())
    print(f"  X_train_fp {X_train_fp.shape}, X_test_fp {X_test_fp.shape}")

    print("Training train-vs-test classifier ...")
    X_all = np.vstack([X_train_fp, X_test_fp])
    y_all = np.concatenate(
        [
            np.zeros(len(X_train_fp), dtype=np.int32),
            np.ones(len(X_test_fp), dtype=np.int32),
        ]
    )
    # LogisticRegression domain classifier (matches 2026-04-21 production
    # recipe per lb_submissions.notes for submission id=19: "Morgan-FP+LogReg
    # domain classifier, density-ratio clipped [1/3, 3]").
    clf = LogisticRegression(max_iter=1000, solver="liblinear", C=1.0, random_state=42)
    clf.fit(X_all, y_all)
    p_test_given_x = clf.predict_proba(X_train_fp)[:, 1]
    eps = 1e-6
    w = (p_test_given_x + eps) / (1.0 - p_test_given_x + eps)
    w = w * (len(X_train_fp) / len(X_test_fp))
    # CRITICAL: clip to [1/3, 3] to prevent extreme weights from destabilising
    # the fit. Yesterday's production recipe used this clip.
    w = np.clip(w, 1.0 / 3.0, 3.0)
    # Normalise so weights sum to N (doesn't affect the fit point, just
    # readability of per-sample weight magnitudes).
    w = w * (len(w) / w.sum())
    print(
        f"  importance weights (clipped [1/3, 3]): mean={w.mean():.4f}, "
        f"q25={np.quantile(w, 0.25):.4f}, q75={np.quantile(w, 0.75):.4f}, max={w.max():.4f}"
    )

    print("Fitting weighted affine on OOF ...")
    reg = LinearRegression()
    reg.fit(oof_preds.reshape(-1, 1), y_train, sample_weight=w)
    slope = float(reg.coef_[0])
    intercept = float(reg.intercept_)
    print(f"  affine: y = {slope:.4f} * pred + {intercept:.4f}")

    # Apply calibration
    oof_cal = slope * oof_preds + intercept
    test_cal = slope * test_preds + intercept
    from scipy.stats import spearmanr

    raw_mae = float(np.mean(np.abs(oof_preds - y_train)))
    cal_mae = float(np.mean(np.abs(oof_cal - y_train)))
    raw_sp = float(spearmanr(oof_preds, y_train).statistic)
    cal_sp = float(spearmanr(oof_cal, y_train).statistic)
    print(
        f"  OOF raw   MAE={raw_mae:.4f}  Spearman={raw_sp:.4f}\n"
        f"  OOF cal   MAE={cal_mae:.4f}  Spearman={cal_sp:.4f}\n"
        f"  Δ MAE={cal_mae - raw_mae:+.4f}  Δ Spearman={cal_sp - raw_sp:+.4f}"
    )

    # Write submission
    out_sub = test_sub.copy()
    test_col = [c for c in out_sub.columns if c.lower() == "pec50"][0]
    out_sub[test_col] = test_cal
    out_path = REPO_ROOT.joinpath(
        "track1_activity", "submissions", "ens_caruana_bag20_calibrated_importance.csv"
    )
    out_sub.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path}")
    print(f"  test mean: {test_cal.mean():.4f}, std: {test_cal.std():.4f}")


if __name__ == "__main__":
    main()
