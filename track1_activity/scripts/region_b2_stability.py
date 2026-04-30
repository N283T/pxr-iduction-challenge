"""Region-conditioned V2: 5-seed stability check (Phase B-2 follow-up).

Per Codex 2026-04-30 sanity check, before LB submitting V2 we need to verify
the per-regime caruana selection is stable across seeds. Three things to check:

  1. Sign + magnitude of ΔM2 across 5 seeds — signal not noise upper-bound.
     Target: median ΔM2 ≤ -0.0025, worst seed not big positive reversal.
  2. Q4 top-2 (top500 + pooled_boltz) consistency — same 2 members in top
     across seeds. If one disappears for some seed, structure unstable.
  3. Family share post-cap stability — should pin near 0.70, but the
     non-family 30% allocation should not vary wildly (risk: cap-driven
     instability picking up noise).

Decision rules per Codex:
  - If 4/5 seeds show improvement AND top-2 stable AND ΔSp not catastrophic:
    proceed to blended submit (0.7 * V2 + 0.3 * base9pool).
  - If half seeds reverse: pure V2 / V2-blend both blocked, defer.
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
N_QUARTILES = 4
WEIGHT_CLIP_LO = 1.0 / 3.0
WEIGHT_CLIP_HI = 3.0
FAMILY_CAP = 0.70
SEEDS = [42, 7, 100, 314, 2026]

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
    print(f"    classifier seed={seed}: AUC={auc:.4f}")
    return p_train, p_test


def load_pool() -> tuple[dict[str, np.ndarray], dict[str, float]]:
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
    return oofs, weights


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


def family_share(weights: dict[str, float], family: set[str]) -> float:
    norm = sum(weights.values())
    return sum(w for m, w in weights.items() if m in family) / max(norm, 1e-12)


def fit_global_affine(
    oof: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[float, float]:
    reg = LinearRegression()
    reg.fit(oof.reshape(-1, 1), y, sample_weight=w)
    return float(reg.coef_[0]), float(reg.intercept_)


def main() -> None:
    print("Loading data ...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y_train = train_df["pec50"].to_numpy(dtype=np.float64)

    X_tr = morgan_matrix(train_df["smiles"].tolist())
    X_te = morgan_matrix(test_df["smiles"].tolist())

    print("Loading 9-pool ...")
    oofs, global_w = load_pool()

    from importance_weights import compute_importance_weights  # noqa: E402

    iw = compute_importance_weights(
        train_df["smiles"].tolist(), test_df["smiles"].tolist()
    )

    # Global ensemble baseline
    global_oof = np.zeros_like(y_train)
    norm_g = sum(global_w.values())
    for name, w in global_w.items():
        global_oof += (w / norm_g) * oofs[name]
    slope_g, intercept_g = fit_global_affine(global_oof, y_train, iw)
    global_oof_cal = slope_g * global_oof + intercept_g
    global_cal_mae = float(np.mean(np.abs(global_oof_cal - y_train)))
    global_sp = float(spearmanr(global_oof, y_train).statistic)
    print(f"global cal OOF MAE = {global_cal_mae:.4f}  Sp = {global_sp:.4f}")

    print("\n=== Per-seed V2 stability ===\n")
    rows = []
    q4_top500_weights = []
    q4_pooled_boltz_weights = []
    q4_optuna_t10_weights = []
    family_share_q4 = []

    members = list(oofs.keys())
    print(f"  9 pool members: {members}")
    for seed in SEEDS:
        print(f"\n  --- seed {seed} ---")
        # Build classifier + quartile partition with this seed
        p_train, p_test = fit_classifier_lgbm(X_tr, X_te, seed=seed)
        q_edges = np.quantile(p_train, np.linspace(0, 1, N_QUARTILES + 1))
        q_edges[0] = -np.inf
        q_edges[-1] = np.inf
        quartile = np.digitize(p_train, q_edges[1:-1], right=False)
        test_quartile = np.digitize(p_test, q_edges[1:-1], right=False)
        n_test_q4 = (test_quartile == 3).sum()
        print(f"    test in Q4: {n_test_q4}/{len(p_test)} ({100 * n_test_q4 / len(p_test):.1f}%)")

        # Per-regime capped caruana
        per_regime: dict[int, dict[str, float]] = {}
        for q in range(N_QUARTILES):
            mask = quartile == q
            if mask.sum() < 50:
                per_regime[q] = global_w
                continue
            wq = caruana_select_capped(
                oofs,
                y_train,
                sample_mask=mask,
                family_members=CHEMPROP_FAMILY,
                family_cap=FAMILY_CAP,
                seed=seed,
            )
            per_regime[q] = wq

        # Reconstruct OOF
        v2_oof = np.zeros_like(y_train)
        for q in range(N_QUARTILES):
            mask = quartile == q
            wmap = per_regime[q]
            norm_q = sum(wmap.values())
            for name, w in wmap.items():
                v2_oof[mask] += (w / norm_q) * oofs[name][mask]
        slope_l, intercept_l = fit_global_affine(v2_oof, y_train, iw)
        v2_cal = slope_l * v2_oof + intercept_l
        v2_cal_mae = float(np.mean(np.abs(v2_cal - y_train)))
        v2_sp = float(spearmanr(v2_oof, y_train).statistic)

        d_m2 = v2_cal_mae - global_cal_mae
        d_sp = v2_sp - global_sp

        # Q4 weights
        q4_w = per_regime[3]
        q4_norm = sum(q4_w.values())
        q4_w_norm = {k: v / q4_norm for k, v in q4_w.items()}
        top500_w = q4_w_norm.get("tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap", 0)
        pb_w = q4_w_norm.get("tabpfn_pooled_boltz_umap_default", 0)
        opt_w = q4_w_norm.get(
            "tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_umap_default", 0
        )
        fs_q4 = family_share(q4_w, CHEMPROP_FAMILY)

        q4_top500_weights.append(top500_w)
        q4_pooled_boltz_weights.append(pb_w)
        q4_optuna_t10_weights.append(opt_w)
        family_share_q4.append(fs_q4)

        rows.append({
            "seed": seed,
            "v2_cal_mae": v2_cal_mae,
            "v2_sp": v2_sp,
            "d_m2": d_m2,
            "d_sp": d_sp,
            "q4_top500": top500_w,
            "q4_pooled_boltz": pb_w,
            "q4_optuna_t10": opt_w,
            "q4_family_share": fs_q4,
        })
        print(
            f"    ΔM2={d_m2:+.4f}  ΔSp={d_sp:+.4f}  "
            f"Q4: top500={top500_w:.3f} pb={pb_w:.3f} opt_t10={opt_w:.3f} "
            f"fam={fs_q4:.3f}"
        )

    df = pd.DataFrame(rows)
    print("\n=== Stability summary ===\n")
    print(df.to_string(index=False, float_format="%.4f"))

    print("\n  ΔM2 across seeds:")
    print(f"    median = {df['d_m2'].median():+.4f}")
    print(f"    mean   = {df['d_m2'].mean():+.4f}")
    print(f"    min    = {df['d_m2'].min():+.4f}")
    print(f"    max    = {df['d_m2'].max():+.4f}")
    n_improve = (df["d_m2"] <= -0.001).sum()
    print(f"    seeds with improvement (ΔM2 <= -0.001): {n_improve}/5")
    n_real_improve = (df["d_m2"] <= -0.003).sum()
    print(f"    seeds with strict improvement (ΔM2 <= -0.003): {n_real_improve}/5")

    print("\n  Q4 top500 weight across seeds:")
    print(f"    range [{min(q4_top500_weights):.3f}, {max(q4_top500_weights):.3f}], "
          f"std={np.std(q4_top500_weights):.3f}")
    print("  Q4 pooled_boltz weight across seeds:")
    print(f"    range [{min(q4_pooled_boltz_weights):.3f}, {max(q4_pooled_boltz_weights):.3f}], "
          f"std={np.std(q4_pooled_boltz_weights):.3f}")
    print("  Q4 family share across seeds:")
    print(f"    range [{min(family_share_q4):.3f}, {max(family_share_q4):.3f}], "
          f"std={np.std(family_share_q4):.3f}")

    # Codex decision criteria
    print("\n=== Codex decision criteria ===\n")
    median_d_m2 = df["d_m2"].median()
    worst_d_m2 = df["d_m2"].max()
    median_d_sp = df["d_sp"].median()
    n_seeds_improve = (df["d_m2"] <= -0.001).sum()

    crit1 = median_d_m2 <= -0.0025
    crit2 = worst_d_m2 < 0.001  # no big positive reversal
    crit3 = median_d_sp >= -0.001  # Sp not catastrophic
    crit4 = n_seeds_improve >= 4  # 4/5 seeds show improvement
    top500_stable = np.std(q4_top500_weights) <= 0.05
    pb_stable = np.std(q4_pooled_boltz_weights) <= 0.05

    print(f"  C1: median ΔM2 ≤ -0.0025  ({median_d_m2:+.4f})  {'PASS' if crit1 else 'FAIL'}")
    print(f"  C2: worst ΔM2 < +0.001    ({worst_d_m2:+.4f})  {'PASS' if crit2 else 'FAIL'}")
    print(f"  C3: median ΔSp ≥ -0.001   ({median_d_sp:+.4f})  {'PASS' if crit3 else 'FAIL'}")
    print(f"  C4: ≥4/5 seeds ΔM2≤-0.001 ({n_seeds_improve}/5)        {'PASS' if crit4 else 'FAIL'}")
    print(f"  C5: Q4 top500 std ≤ 0.05  ({np.std(q4_top500_weights):.4f})  {'PASS' if top500_stable else 'FAIL'}")
    print(f"  C6: Q4 pooled_boltz std ≤ 0.05 ({np.std(q4_pooled_boltz_weights):.4f})  {'PASS' if pb_stable else 'FAIL'}")
    n_pass = sum([crit1, crit2, crit3, crit4, top500_stable, pb_stable])
    print(f"\n  Total: {n_pass}/6 pass")
    if n_pass >= 5:
        print("  RECOMMEND: 0.7 V2 + 0.3 base hedged blend (or 0.8/0.2 if 6/6)")
    elif n_pass >= 4:
        print("  RECOMMEND: 0.5 V2 + 0.5 base safer blend, or defer")
    else:
        print("  RECOMMEND: defer; per-regime caruana too unstable")

    df.to_csv(OUT_DIR.joinpath("2026-04-30-region-b2-stability.csv"), index=False)


if __name__ == "__main__":
    main()
