"""Region-conditioned ensemble diagnostic (Phase B-1).

Per Codex consult 2026-04-30: rather than improve the LB-proxy, change the
prediction itself by routing per-test-compound through different ensemble
weights based on a regime indicator (e.g., test-likeness from adversarial
classifier).

This script answers the prerequisite question: do the 9 pool members
SPECIALIZE across regions, or are their MAEs flat? Without specialization,
region-conditioning offers no improvement.

Procedure:
  1. Fit LightGBM p(test|train) classifier (Morgan FP).
  2. Bin train compounds into 4 quartiles by test-likeness.
  3. For each pool member, compute MAE within each quartile.
  4. Compute regime-MAE variance per member; high variance = specialization.
  5. Compute spread per quartile (which member dominates where).
  6. Run global caruana on FULL train, then run separate caruanas on each
     quartile-restricted train. Compare per-regime weight maps.

Output:
  - Heatmap CSV: rows = members, cols = quartiles, values = MAE.
  - Per-quartile caruana weights table.
  - Decision: variance >= 0.005 across quartiles for >= 3 members => proceed
    to Phase B-2.
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
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
from data import (  # noqa: E402
    DB_PARAMS,
    load_test_smiles,
    load_train_smiles_target,
)

OUT_DIR = REPO_ROOT.joinpath("track1_activity", "reports")
SEED = 42
N_QUARTILES = 4


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
) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (p_train, p_test, AUC)."""
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
    return p_train, p_test, auc


def load_pool_oof_members() -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Returns (member_name -> OOF, member_name -> caruana global weight)."""
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


def caruana_select(
    oofs: dict[str, np.ndarray],
    y: np.ndarray,
    *,
    sample_mask: np.ndarray | None = None,
    n_steps: int = 100,
    n_bags: int = 20,
    seed: int = 42,
) -> dict[str, float]:
    """Caruana forward selection with replacement, bagged.

    sample_mask: boolean array selecting which train rows to use. If None,
    all rows are used.
    """
    members = list(oofs.keys())
    M = len(members)
    if sample_mask is None:
        sample_mask = np.ones(len(y), dtype=bool)
    y_sel = y[sample_mask]
    oof_arr = np.stack([oofs[m][sample_mask] for m in members], axis=1)  # (n, M)

    rng = np.random.default_rng(seed)
    counts = np.zeros(M, dtype=np.float64)
    for _bag in range(n_bags):
        bag_idx = rng.choice(np.arange(len(y_sel)), size=len(y_sel), replace=True)
        y_bag = y_sel[bag_idx]
        oof_bag = oof_arr[bag_idx]
        bag_counts = np.zeros(M, dtype=np.float64)
        ens_sum = np.zeros_like(y_bag)
        n = 0
        for _ in range(n_steps):
            best_m, best_mae = None, float("inf")
            for j in range(M):
                cand_sum = ens_sum + oof_bag[:, j]
                cand_pred = cand_sum / (n + 1)
                mae = float(np.mean(np.abs(cand_pred - y_bag)))
                if mae < best_mae:
                    best_mae = mae
                    best_m = j
            ens_sum = ens_sum + oof_bag[:, best_m]
            n += 1
            bag_counts[best_m] += 1.0
        counts += bag_counts / bag_counts.sum()
    counts /= n_bags
    return {members[i]: float(counts[i]) for i in range(M)}


def main() -> None:
    print("Loading data ...")
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y_train = train_df["pec50"].to_numpy(dtype=np.float64)

    print("  morgan FPs ...")
    X_tr = morgan_matrix(train_df["smiles"].tolist())
    X_te = morgan_matrix(test_df["smiles"].tolist())

    print("  LightGBM adversarial classifier ...")
    p_train, p_test, auc = fit_classifier_lgbm(X_tr, X_te)
    print(f"    AUC (5-fold) = {auc:.4f}")

    # Bin into quartiles by p_train
    q_edges = np.quantile(p_train, np.linspace(0, 1, N_QUARTILES + 1))
    q_edges[0] = -np.inf
    q_edges[-1] = np.inf
    quartile = np.digitize(p_train, q_edges[1:-1], right=False)  # 0..3
    print("\n  Quartile distribution (train):")
    for q in range(N_QUARTILES):
        mask = quartile == q
        print(
            f"    Q{q + 1}: n={mask.sum()}, "
            f"p_train range [{p_train[mask].min():.4f}, {p_train[mask].max():.4f}], "
            f"y mean={y_train[mask].mean():.3f}, std={y_train[mask].std():.3f}"
        )

    # Where does TEST land? (assign each test compound to a quartile by p_test)
    test_quartile = np.digitize(p_test, q_edges[1:-1], right=False)
    print("\n  Quartile distribution (test):")
    for q in range(N_QUARTILES):
        n = (test_quartile == q).sum()
        print(f"    Q{q + 1}: n={n} ({100 * n / len(p_test):.1f}%)")

    print("\nLoading 9-pool members ...")
    oofs, weights = load_pool_oof_members()
    print(f"  pool size = {len(oofs)}")
    members = list(oofs.keys())
    print("  global caruana weights:")
    for m, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        print(f"    {w:.3f}  {m}")

    # === Member × Quartile MAE heatmap ===
    print("\n=== Member x Quartile MAE ===")
    rows = []
    for name in members:
        oof = oofs[name]
        row = {"member": name, "global_w": weights[name]}
        global_mae = float(np.mean(np.abs(oof - y_train)))
        row["global_mae"] = global_mae
        for q in range(N_QUARTILES):
            mask = quartile == q
            mae_q = float(np.mean(np.abs(oof[mask] - y_train[mask])))
            row[f"Q{q + 1}_mae"] = mae_q
        # Variance across quartiles
        q_maes = np.array([row[f"Q{q + 1}_mae"] for q in range(N_QUARTILES)])
        row["q_std"] = float(q_maes.std())
        row["q_max_minus_min"] = float(q_maes.max() - q_maes.min())
        rows.append(row)
    df_heatmap = pd.DataFrame(rows).sort_values("global_w", ascending=False)
    print()
    cols = (
        ["member", "global_w", "global_mae"]
        + [f"Q{q + 1}_mae" for q in range(N_QUARTILES)]
        + ["q_std", "q_max_minus_min"]
    )
    print(df_heatmap[cols].to_string(index=False, float_format="%.4f"))

    # === Global ensemble MAE per quartile ===
    print("\n=== Global ensemble MAE by quartile ===")
    norm = sum(weights.values())
    ens_oof = np.zeros_like(y_train)
    for name, w in weights.items():
        ens_oof += w * oofs[name]
    ens_oof /= norm
    overall_mae = float(np.mean(np.abs(ens_oof - y_train)))
    print(f"  overall MAE = {overall_mae:.4f}")
    for q in range(N_QUARTILES):
        mask = quartile == q
        mae_q = float(np.mean(np.abs(ens_oof[mask] - y_train[mask])))
        print(f"    Q{q + 1}: ens MAE = {mae_q:.4f} (n={mask.sum()})")

    # === Per-regime caruana ===
    print("\n=== Per-regime caruana ===")
    print("  global caruana (sanity check) ...")
    global_w = caruana_select(oofs, y_train, sample_mask=None, seed=SEED)
    sum_w = sum(global_w.values())
    print(f"    sum of weights = {sum_w:.4f}")

    per_regime_w: dict[int, dict[str, float]] = {}
    for q in range(N_QUARTILES):
        mask = quartile == q
        if mask.sum() < 50:
            print(f"  Q{q + 1}: too few samples ({mask.sum()}), skip")
            continue
        w_q = caruana_select(oofs, y_train, sample_mask=mask, seed=SEED)
        per_regime_w[q] = w_q
        print(f"\n  Q{q + 1} (n={mask.sum()}) caruana weights (top 5):")
        for m, w in sorted(w_q.items(), key=lambda kv: -kv[1])[:5]:
            delta = w - global_w[m]
            print(f"    {w:.3f}  ({delta:+.3f} vs global)  {m}")

    # Reconstruct OOF using per-regime weights
    print("\n=== Region-conditioned OOF MAE ===")
    region_oof = np.zeros_like(y_train)
    for q in range(N_QUARTILES):
        mask = quartile == q
        if q not in per_regime_w:
            # fallback to global
            wmap = global_w
        else:
            wmap = per_regime_w[q]
        norm_q = sum(wmap.values())
        for name, w in wmap.items():
            region_oof[mask] += (w / norm_q) * oofs[name][mask]
    region_overall_mae = float(np.mean(np.abs(region_oof - y_train)))
    print(f"  region-conditioned overall MAE = {region_overall_mae:.4f}")
    print(f"  global ensemble overall MAE    = {overall_mae:.4f}")
    print(f"  delta                          = {region_overall_mae - overall_mae:+.4f}")
    for q in range(N_QUARTILES):
        mask = quartile == q
        rg = float(np.mean(np.abs(region_oof[mask] - y_train[mask])))
        gl = float(np.mean(np.abs(ens_oof[mask] - y_train[mask])))
        print(f"  Q{q + 1}: region={rg:.4f} vs global={gl:.4f}  Δ={rg - gl:+.4f}")

    out_csv = OUT_DIR.joinpath("2026-04-30-region-diagnostic-heatmap.csv")
    df_heatmap.to_csv(out_csv, index=False)
    print(f"\nHeatmap written to {out_csv}")

    # Decision summary
    print("\n=== Decision summary ===")
    n_specialised = (df_heatmap["q_max_minus_min"] >= 0.05).sum()
    print(
        f"  members with q_max_minus_min >= 0.05: {n_specialised} / {len(df_heatmap)}"
    )
    print(
        f"  region-conditioned OOF improvement: {overall_mae - region_overall_mae:+.4f}"
    )
    if region_overall_mae < overall_mae - 0.001:
        print("  [PROCEED] Phase B-2 candidate (Δ <= -0.001 at OOF)")
    elif region_overall_mae < overall_mae - 0.0005:
        print("  [MARGINAL] consider regime granularity tweaks")
    else:
        print("  [ABANDON] no OOF signal from region-conditioning at this granularity")


if __name__ == "__main__":
    main()
