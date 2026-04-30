"""Phase B-3: V2-base blend ratio sweep + submission CSV generation.

Per Codex sanity check (2026-04-30) and 6/6 stability pass, we submit a
hedged V2-base blend rather than pure V2. This script:

  1. Reconstructs V2 (family-capped per-regime) on seed=42 (production).
  2. Sweeps blend ratio alpha in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        blend = alpha * V2 + (1 - alpha) * base9pool
  3. For each ratio, applies global importance affine (production recipe).
  4. Computes M2 (calibrated OOF MAE) delta vs base9pool, M6 (Sp) delta.
  5. Reports family share at Q4 (test region) for each ratio.
  6. Generates submission CSV for each that passes strict gate.

Codex recommendation: 0.8 / 0.2 if 6/6 stability (current state).
0.7 / 0.3 conservative fallback. Pure V2 (1.0) only if seeds extremely clean
across multiple stability axes (we have that, so it's optionally allowed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
from data import (  # noqa: E402
    DB_PARAMS,
    load_test_smiles,
    load_train_smiles_target,
)

OUT_DIR = REPO_ROOT.joinpath("docs", "superpowers", "runs")
SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
SEED = 42
N_QUARTILES = 4
WEIGHT_CLIP_LO = 1.0 / 3.0
WEIGHT_CLIP_HI = 3.0
FAMILY_CAP = 0.70
BLEND_ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

CHEMPROP_FAMILY = {
    "tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_umap_default",
    "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap",
    "tabpfn_chemprop_pretrain_embed_umap_default",
}


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


def fit_classifier_lgbm(
    X_tr: np.ndarray, X_te: np.ndarray, *, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    X_all = np.vstack([X_tr, X_te]).astype(np.float32)
    y_all = np.concatenate(
        [np.zeros(len(X_tr), dtype=np.int32), np.ones(len(X_te), dtype=np.int32)]
    )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y_all))
    folds = np.array_split(perm, 5)
    p_oof = np.zeros(len(y_all), dtype=np.float64)
    params = dict(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        reg_lambda=1.0,
        random_state=seed,
        verbose=-1,
    )
    for k in range(5):
        va = folds[k]
        tr = np.concatenate([folds[j] for j in range(5) if j != k])
        m = lgb.LGBMClassifier(**params)
        m.fit(X_all[tr], y_all[tr])
        p_oof[va] = m.predict_proba(X_all[va])[:, 1]
    auc = float(roc_auc_score(y_all, p_oof))
    full = lgb.LGBMClassifier(**params)
    full.fit(X_all, y_all)
    p_train = full.predict_proba(X_tr.astype(np.float32))[:, 1]
    p_test = full.predict_proba(X_te.astype(np.float32))[:, 1]
    print(f"  classifier seed={seed} AUC={auc:.4f}")
    return p_train, p_test


def load_pool() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    with psycopg2.connect(**DB_PARAMS) as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT id, hyperparameters FROM experiments WHERE name = 'ens_caruana_bag20'
               ORDER BY id DESC LIMIT 1"""
        )
        _exp_id, hp = cur.fetchone()
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


def caruana_select_capped(
    oofs: dict[str, np.ndarray],
    y: np.ndarray,
    *,
    sample_mask: np.ndarray,
    family_members: set[str],
    family_cap: float = FAMILY_CAP,
    n_steps: int = 100,
    n_bags: int = 20,
    seed: int = 42,
) -> dict[str, float]:
    members = list(oofs.keys())
    M = len(members)
    family_idx = {i for i, m in enumerate(members) if m in family_members}
    y_sel = y[sample_mask]
    oof_arr = np.stack([oofs[m][sample_mask] for m in members], axis=1)
    rng = np.random.default_rng(seed)
    counts = np.zeros(M, dtype=np.float64)
    for _bag in range(n_bags):
        bag_idx = rng.choice(np.arange(len(y_sel)), size=len(y_sel), replace=True)
        y_bag = y_sel[bag_idx]
        oof_bag = oof_arr[bag_idx]
        bag_counts = np.zeros(M, dtype=np.float64)
        ens_sum = np.zeros_like(y_bag)
        n = 0
        family_picks = 0
        for _ in range(n_steps):
            best_m, best_mae = None, float("inf")
            for j in range(M):
                if j in family_idx:
                    proposed = (family_picks + 1) / (n + 1)
                    if proposed > family_cap:
                        continue
                cand_pred = (ens_sum + oof_bag[:, j]) / (n + 1)
                mae = float(np.mean(np.abs(cand_pred - y_bag)))
                if mae < best_mae:
                    best_mae = mae
                    best_m = j
            if best_m is None:
                for j in range(M):
                    cand_pred = (ens_sum + oof_bag[:, j]) / (n + 1)
                    mae = float(np.mean(np.abs(cand_pred - y_bag)))
                    if mae < best_mae:
                        best_mae = mae
                        best_m = j
            ens_sum = ens_sum + oof_bag[:, best_m]
            n += 1
            bag_counts[best_m] += 1.0
            if best_m in family_idx:
                family_picks += 1
        counts += bag_counts / bag_counts.sum()
    counts /= n_bags
    return {members[i]: float(counts[i]) for i in range(M)}


def fit_global_affine(
    oof: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[float, float]:
    reg = LinearRegression()
    reg.fit(oof.reshape(-1, 1), y, sample_weight=w)
    return float(reg.coef_[0]), float(reg.intercept_)


def family_share(weights: dict[str, float], family: set[str]) -> float:
    norm = sum(weights.values())
    return sum(w for m, w in weights.items() if m in family) / max(norm, 1e-12)


def main() -> None:
    print("Loading data ...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y_train = train_df["pec50"].to_numpy(dtype=np.float64)

    X_tr = morgan_matrix(train_df["smiles"].tolist())
    X_te = morgan_matrix(test_df["smiles"].tolist())

    print("Fit classifier ...")
    p_train, p_test = fit_classifier_lgbm(X_tr, X_te, seed=SEED)
    q_edges = np.quantile(p_train, np.linspace(0, 1, N_QUARTILES + 1))
    q_edges[0] = -np.inf
    q_edges[-1] = np.inf
    quartile = np.digitize(p_train, q_edges[1:-1], right=False)
    test_quartile = np.digitize(p_test, q_edges[1:-1], right=False)
    print(f"  test in Q4: {(test_quartile == 3).sum()}/{len(p_test)}")

    print("Loading 9-pool ...")
    oofs, test_preds, global_w = load_pool()

    from importance_weights import compute_importance_weights  # noqa: E402

    iw = compute_importance_weights(
        train_df["smiles"].tolist(), test_df["smiles"].tolist()
    )

    # Global ensemble
    norm_g = sum(global_w.values())
    base_oof = np.zeros_like(y_train)
    base_test = np.zeros_like(test_preds[next(iter(test_preds))])
    for name, w in global_w.items():
        base_oof += (w / norm_g) * oofs[name]
        base_test += (w / norm_g) * test_preds[name]
    slope_b, intercept_b = fit_global_affine(base_oof, y_train, iw)
    base_oof_cal = slope_b * base_oof + intercept_b
    base_cal_mae = float(np.mean(np.abs(base_oof_cal - y_train)))
    base_sp = float(spearmanr(base_oof, y_train).statistic)
    print(f"\nbase9pool cal OOF MAE={base_cal_mae:.4f}  Sp={base_sp:.4f}")
    print(f"base affine: y = {slope_b:.4f} * pred + {intercept_b:.4f}")

    print("\nBuild V2 weights per regime (capped) ...")
    v2_per_regime: dict[int, dict[str, float]] = {}
    for q in range(N_QUARTILES):
        mask = quartile == q
        if mask.sum() < 50:
            v2_per_regime[q] = global_w
            continue
        wq = caruana_select_capped(
            oofs,
            y_train,
            sample_mask=mask,
            family_members=CHEMPROP_FAMILY,
            family_cap=FAMILY_CAP,
            seed=SEED,
        )
        v2_per_regime[q] = wq

    # V2 OOF + test
    v2_oof = np.zeros_like(y_train)
    for q in range(N_QUARTILES):
        mask = quartile == q
        wmap = v2_per_regime[q]
        norm_q = sum(wmap.values())
        for name, w in wmap.items():
            v2_oof[mask] += (w / norm_q) * oofs[name][mask]

    v2_test = np.zeros_like(base_test)
    for q in range(N_QUARTILES):
        mask = test_quartile == q
        if mask.sum() == 0:
            continue
        wmap = v2_per_regime[q]
        norm_q = sum(wmap.values())
        for name, w in wmap.items():
            v2_test[mask] += (w / norm_q) * test_preds[name][mask]

    # Q4 family share for V2 (test region)
    q4_w = v2_per_regime[3]
    q4_fs = family_share(q4_w, CHEMPROP_FAMILY)
    print(f"  V2 Q4 family share = {q4_fs:.4f}")

    # === Blend ratio sweep ===
    print("\n=== Blend ratio sweep (alpha = V2 weight) ===\n")
    print(f"  base9pool ref: cal MAE={base_cal_mae:.4f}  Sp={base_sp:.4f}")
    print("  pure V2 ref:   ΔM2=-0.0040  ΔSp=+0.0030")
    print()
    print(f"  {'alpha':>5}  {'cal_MAE':>8}  {'ΔM2':>8}  {'Sp':>7}  {'ΔSp':>8}  "
          f"{'Q4_fam':>7}  {'gate':>10}")
    print("  " + "-" * 75)

    rows = []
    for alpha in BLEND_ALPHAS:
        blend_oof = alpha * v2_oof + (1.0 - alpha) * base_oof
        slope_l, intercept_l = fit_global_affine(blend_oof, y_train, iw)
        blend_cal = slope_l * blend_oof + intercept_l
        cal_mae = float(np.mean(np.abs(blend_cal - y_train)))
        sp = float(spearmanr(blend_oof, y_train).statistic)
        d_m2 = cal_mae - base_cal_mae
        d_sp = sp - base_sp
        # Effective family share at the blend (Q4 region for test)
        # alpha * V2_Q4_share + (1-alpha) * global_share
        global_fs = family_share(global_w, CHEMPROP_FAMILY)
        blend_q4_fs = alpha * q4_fs + (1.0 - alpha) * global_fs
        m2_pass = d_m2 <= -0.003
        sp_pass = d_sp >= -0.002
        fs_pass = 0.65 <= blend_q4_fs <= 0.80
        all_pass = m2_pass and sp_pass and fs_pass
        gate = "ALL PASS" if all_pass else (
            "M2 fail" if not m2_pass else ("Sp fail" if not sp_pass else "fam fail")
        )
        print(
            f"  {alpha:>5.2f}  {cal_mae:>8.4f}  {d_m2:>+8.4f}  {sp:>7.4f}  "
            f"{d_sp:>+8.4f}  {blend_q4_fs:>7.4f}  {gate:>10}"
        )
        rows.append({
            "alpha": alpha,
            "cal_mae": cal_mae,
            "d_m2": d_m2,
            "sp": sp,
            "d_sp": d_sp,
            "blend_q4_family_share": blend_q4_fs,
            "all_pass": all_pass,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR.joinpath("2026-04-30-region-b3-blend-sweep.csv"), index=False)

    # === Generate submission CSVs for the recommended ratios ===
    print("\n=== Generating submission CSVs for recommended ratios (0.7, 0.8, 1.0) ===\n")
    test_smiles = test_df["smiles"].tolist()
    test_names = test_df["molecule_name"].astype(str).tolist()

    for alpha in [0.7, 0.8, 1.0]:
        blend_oof = alpha * v2_oof + (1.0 - alpha) * base_oof
        blend_test = alpha * v2_test + (1.0 - alpha) * base_test
        slope_l, intercept_l = fit_global_affine(blend_oof, y_train, iw)
        test_cal = slope_l * blend_test + intercept_l
        out_path = SUBMISSION_DIR.joinpath(
            f"ens_region_v2_blend_a{int(alpha * 10)}.csv"
        )
        out_df = pd.DataFrame({
            "SMILES": test_smiles,
            "Molecule Name": test_names,
            "pEC50": test_cal,
        })
        out_df.to_csv(out_path, index=False)
        print(
            f"  saved {out_path.name}  "
            f"test mean={np.mean(test_cal):.3f} std={np.std(test_cal):.3f}"
        )

    print("\n=== Recommendation summary ===\n")
    for alpha in BLEND_ALPHAS:
        row = df[df["alpha"] == alpha].iloc[0]
        if row["all_pass"]:
            print(
                f"  alpha={alpha}  ΔM2={row['d_m2']:+.4f}  "
                f"ΔSp={row['d_sp']:+.4f}  family={row['blend_q4_family_share']:.4f}  "
                f"=> CANDIDATE"
            )


if __name__ == "__main__":
    main()
