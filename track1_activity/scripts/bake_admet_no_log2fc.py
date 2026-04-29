"""Caruana_bag20 ADD bakeoff: 9-pool + tabpfn_cheme_2d_full_boltz_admet_ai_top500_umap.

Family-share-safe variant of the ADMET-AI augmented member (no log2fc_pred,
so not chemprop family). Tests:
  1. caruana weight on new member (>= 0.01 needed)
  2. M2 (calibrated OOF MAE) Δ vs base9pool (target <= -0.003)
  3. chemprop family share remains in 0.65-0.80 zone
  4. test prediction sanity

If all pass, this is a submit candidate.

Spec: docs/superpowers/specs/2026-04-29-lb-proxy-discovery-design.md
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
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
from data import (  # noqa: E402
    DB_PARAMS,
    load_test_smiles,
    load_train_smiles_target,
)
from run_ensemble import ENSEMBLE_MODELS, optimize_caruana  # noqa: E402

NEW_MEMBER = "tabpfn_cheme_2d_full_boltz_admet_ai_top500_umap"
SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")

CHEMPROP_FAMILY = {
    "tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_umap_default",
    "tabpfn_chemprop_pretrain_embed_umap_default",
    "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap",
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
            """SELECT train_idx, oof_prediction FROM experiment_oof_predictions
               WHERE experiment_id = %s ORDER BY train_idx""",
            (exp_id,),
        )
        rows = cur.fetchall()
    return np.asarray([r[1] for r in rows], dtype=np.float64)


def load_member_test(name: str) -> np.ndarray:
    df = pd.read_csv(SUBMISSION_DIR.joinpath(f"{name}.csv"))
    test_col = [c for c in df.columns if c.lower() == "pec50"][0]
    return df[test_col].to_numpy(dtype=np.float64)


def morgan_matrix(smiles_list: list[str]) -> np.ndarray:
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    out = np.zeros((len(smiles_list), 2048), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = gen.GetFingerprint(mol)
        out[i] = np.asarray(fp, dtype=np.uint8)
    return out


def fit_global_importance(
    Xt: np.ndarray, Xv: np.ndarray, oof: np.ndarray, y: np.ndarray
) -> tuple[float, float, np.ndarray]:
    Xall = np.vstack([Xt, Xv])
    yall = np.concatenate([np.zeros(len(Xt), dtype=int), np.ones(len(Xv), dtype=int)])
    clf = LogisticRegression(max_iter=1000, solver="liblinear", C=1.0, random_state=42)
    clf.fit(Xall, yall)
    p = clf.predict_proba(Xt)[:, 1]
    eps = 1e-6
    w = (p + eps) / (1 - p + eps) * (len(Xt) / len(Xv))
    w = np.clip(w, 1.0 / 3.0, 3.0)
    w = w * (len(w) / w.sum())
    reg = LinearRegression()
    reg.fit(oof.reshape(-1, 1), y, sample_weight=w)
    return float(reg.coef_[0]), float(reg.intercept_), w


def caruana_bagged(oof_matrix: np.ndarray, y: np.ndarray, names: list[str]) -> dict:
    weights_runs = []
    for seed in [42, 43, 44, 45, 46]:
        w = optimize_caruana(
            oof_matrix, y, n_iter=100, init_top_n=3, n_bags=20, seed=seed
        )
        weights_runs.append(w)
    weights_mean = np.mean(weights_runs, axis=0)
    weights_mean = weights_mean / weights_mean.sum()
    blend_oof = oof_matrix @ weights_mean
    return {
        "weights": dict(zip(names, [float(w) for w in weights_mean])),
        "blend_oof": blend_oof,
    }


def main() -> None:
    print("Loading data ...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y_train = train_df["pec50"].to_numpy(dtype=np.float64)
    print(f"  train n={len(y_train)}, test n={len(test_df)}")

    print("Computing Morgan FPs ...")
    Xtr = morgan_matrix(train_df["smiles"].tolist())
    Xte = morgan_matrix(test_df["smiles"].tolist())

    print(f"\nLoading {len(ENSEMBLE_MODELS)}-pool baseline OOF + test ...")
    base_names = list(ENSEMBLE_MODELS)
    base_oofs = [load_member_oof(n) for n in base_names]
    base_tests = [load_member_test(n) for n in base_names]
    base_oof_matrix = np.column_stack(base_oofs)
    base_test_matrix = np.column_stack(base_tests)

    print(f"\nLoading new member: {NEW_MEMBER}")
    new_oof = load_member_oof(NEW_MEMBER)
    new_test = load_member_test(NEW_MEMBER)
    single_mae = float(np.mean(np.abs(new_oof - y_train)))
    single_sp = float(spearmanr(new_oof, y_train).statistic)
    print(f"  single OOF MAE={single_mae:.4f}  Sp={single_sp:.4f}")

    print("\nResidual r vs each base member:")
    for bn, bo in zip(base_names, base_oofs):
        r = float(np.corrcoef(new_oof, bo)[0, 1])
        print(f"    r vs {bn:>60} = {r:+.4f}")

    print("\n=== Bakeoff: 2 variants ===")

    # variant A: baseline 9pool
    print(f"\n--- baseline_9pool ({len(base_names)} members) ---")
    base_res = caruana_bagged(base_oof_matrix, y_train, base_names)
    base_blend_oof = base_res["blend_oof"]
    base_test_blend = base_test_matrix @ np.array(
        [base_res["weights"][n] for n in base_names]
    )

    # variant B: ADD new
    print(f"\n--- add_new ({len(base_names) + 1} members) ---")
    add_names = base_names + [NEW_MEMBER]
    add_oof_matrix = np.column_stack([base_oof_matrix, new_oof])
    add_test_matrix = np.column_stack([base_test_matrix, new_test])
    add_res = caruana_bagged(add_oof_matrix, y_train, add_names)
    add_blend_oof = add_res["blend_oof"]
    add_test_blend = add_test_matrix @ np.array(
        [add_res["weights"][n] for n in add_names]
    )

    # Compute candidate metrics for each
    print("\n=== Computing M2 (calibrated OOF MAE) gate ===")

    # Fit ONE global importance affine on the BASELINE blend (the production calibrator)
    slope_g, intercept_g, w_imp = fit_global_importance(
        Xtr, Xte, base_blend_oof, y_train
    )
    print(f"  importance affine: y = {slope_g:.4f} * pred + {intercept_g:.4f}")

    def metrics_for(blend_oof: np.ndarray, blend_test: np.ndarray, label: str):
        m1 = float(np.mean(np.abs(blend_oof - y_train)))
        cal_oof = slope_g * blend_oof + intercept_g
        m2 = float(np.mean(np.abs(cal_oof - y_train)))
        m4 = float(np.average(np.abs(cal_oof - y_train), weights=w_imp))
        sp = float(spearmanr(cal_oof, y_train).statistic)
        cal_test = slope_g * blend_test + intercept_g
        return {
            "label": label,
            "M1_raw_oof_mae": m1,
            "M2_calibrated_oof_mae": m2,
            "M4_iw_calibrated_oof_mae": m4,
            "calibrated_oof_spearman": sp,
            "test_pred_mean": float(np.mean(cal_test)),
            "test_pred_std": float(np.std(cal_test)),
        }

    base_metrics = metrics_for(base_blend_oof, base_test_blend, "baseline_9pool")
    add_metrics = metrics_for(add_blend_oof, add_test_blend, "add_new")

    print()
    for m in [base_metrics, add_metrics]:
        print(f"  {m['label']}:")
        print(f"    M1 raw OOF MAE        = {m['M1_raw_oof_mae']:.4f}")
        print(f"    M2 calibrated OOF MAE = {m['M2_calibrated_oof_mae']:.4f}")
        print(f"    M4 iw-calibrated MAE  = {m['M4_iw_calibrated_oof_mae']:.4f}")
        print(f"    cal Sp                = {m['calibrated_oof_spearman']:.4f}")
        print(
            f"    test pred             = mean {m['test_pred_mean']:.4f}, std {m['test_pred_std']:.4f}"
        )

    print()
    print("=== Deltas (add_new vs baseline) ===")
    print(
        f"  ΔM1 = {add_metrics['M1_raw_oof_mae'] - base_metrics['M1_raw_oof_mae']:+.4f}"
    )
    print(
        f"  ΔM2 = {add_metrics['M2_calibrated_oof_mae'] - base_metrics['M2_calibrated_oof_mae']:+.4f}"
    )
    print(
        f"  ΔM4 = {add_metrics['M4_iw_calibrated_oof_mae'] - base_metrics['M4_iw_calibrated_oof_mae']:+.4f}"
    )
    print(
        f"  ΔSp = {add_metrics['calibrated_oof_spearman'] - base_metrics['calibrated_oof_spearman']:+.4f}"
    )

    print()
    print("=== Caruana weights for add_new variant ===")
    for n, w in sorted(add_res["weights"].items(), key=lambda x: -x[1]):
        highlight = " <-- NEW" if n == NEW_MEMBER else ""
        print(f"  {w:.4f}  {n}{highlight}")

    new_w = add_res["weights"].get(NEW_MEMBER, 0)
    family_share = sum(add_res["weights"].get(n, 0) for n in CHEMPROP_FAMILY)
    print()
    print("=== Gate decision ===")
    print(
        f"  new member weight: {new_w:.4f}  ({'PASS >= 0.01' if new_w >= 0.01 else 'FAIL < 0.01'})"
    )
    print(
        f"  ΔM2 (calibrated OOF MAE): "
        f"{add_metrics['M2_calibrated_oof_mae'] - base_metrics['M2_calibrated_oof_mae']:+.4f}  "
        f"({'PASS <= -0.003' if (add_metrics['M2_calibrated_oof_mae'] - base_metrics['M2_calibrated_oof_mae']) <= -0.003 else 'check tighter'})"
    )
    print(
        f"  chemprop family share: {family_share:.3f}  "
        f"({'PASS in 0.65-0.80' if 0.65 <= family_share <= 0.80 else 'CHECK out of zone'})"
    )

    pass_all = (
        new_w >= 0.01
        and (
            add_metrics["M2_calibrated_oof_mae"] - base_metrics["M2_calibrated_oof_mae"]
        )
        <= -0.003
        and 0.65 <= family_share <= 0.80
    )
    print(f"\n  GATE PASS = {pass_all}")

    # Save calibrated test prediction as a candidate submission CSV
    if pass_all or True:  # always write so user can decide; gate pass labelled
        sub_in = pd.read_csv(
            SUBMISSION_DIR.joinpath("ens_caruana_bag20_calibrated_importance.csv")
        )
        out = sub_in.copy()
        test_col = [c for c in out.columns if c.lower() == "pec50"][0]
        cal_test = slope_g * add_test_blend + intercept_g
        out[test_col] = cal_test
        out_name = "ens_caruana_bag20_admet_ai_no_log2fc_calibrated_importance.csv"
        out_path = SUBMISSION_DIR.joinpath(out_name)
        out.to_csv(out_path, index=False)
        print(f"\nWrote candidate submission CSV: {out_path}")


if __name__ == "__main__":
    main()
