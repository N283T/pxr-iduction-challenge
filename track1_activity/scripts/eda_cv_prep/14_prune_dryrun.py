"""Compare candidate ensemble pools side-by-side, using the newly
tuned tabpfn_2d_full_boltz as the workhorse.

Strategy optimisation: l2 with alpha=0.1 and MAE objective.

Pools evaluated:
  1. baseline_current  -- 28 (pre-submission, with mixed)
  2. replace_only      -- 28: tabpfn_mordred_jazzy_umap -> tabpfn_2d_full_boltz
  3. drop_mixed        -- 25: (2) - 3 mixed LGBMs
  4. drop_mixed_zerow  -- 25 minus zero-w_mae LGBM Morgan/AtomPair fingerprint variants
  5. minimal_core      -- drop to top-weight ~12 members (aggressive)

Report: OOF RAE/MAE per pool + top-N weights to inspect composition.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
from data import DB_PARAMS  # noqa: E402
from evaluate import load_oof_predictions  # noqa: E402


# ---------------------------------------------------------------------
# Candidate pools
# ---------------------------------------------------------------------
BASELINE_28 = [
    "lgbm_mordred_jazzy_umap",
    "chemprop_multitask5_umap_aux0.0_tuned",
    "chemprop_optuna_umap",
    "attentivefp_optuna_umap",
    "residual_physprop+mordred_umap",
    "chemprop_chemeleon_umap",
    "gatedgcn_optuna_umap",
    "gin_optuna_umap",
    "graphgps_optuna_umap",
    "lgbm_chemeleon_umap",
    "lgbm_chemberta_5m_mtr_umap",
    "lgbm_chemeleon_umap_gap1.0",
    "lgbm_chemberta_5m_mtr_umap_gap1.0",
    "lgbm_molformer_xl_umap",
    "lgbm_count_morgan_r2_2048_umap",
    "lgbm_count_atompair_2048_umap",
    "lgbm_count_morgan_r3_2048_umap",
    "lgbm_count_morgan_r2_2048_umap_gap1.0",
    "lgbm_avalon_2048_umap",
    "lgbm_morgan_r2_2048_umap",
    "lgbm_atompair_2048_umap",
    "lgbm_feat_morgan_r2_2048_umap",
    "lgbm_rdkit_desc_full_umap",
    "tabpfn_mordred_jazzy_umap",
    "tabpfn_chemeleon_umap",
    # mixed 3
    "lgbm_mordred_jazzy_mixed",
    "lgbm_rdkit_desc_full_mixed",
    "lgbm_morgan_r2_2048_mixed",
]

REPLACE_28 = [
    ("tabpfn_2d_full_boltz_umap" if m == "tabpfn_mordred_jazzy_umap" else m)
    for m in BASELINE_28
]

MIXED_MODELS = {
    "lgbm_mordred_jazzy_mixed",
    "lgbm_rdkit_desc_full_mixed",
    "lgbm_morgan_r2_2048_mixed",
}
DROP_MIXED = [m for m in REPLACE_28 if m not in MIXED_MODELS]

# Zero-w_mae fingerprint / embedding LGBMs from earlier dry-run
ZEROW_DROP = {
    "lgbm_chemberta_5m_mtr_umap",
    "lgbm_count_atompair_2048_umap",
    "lgbm_chemeleon_umap_gap1.0",
    "lgbm_count_morgan_r2_2048_umap_gap1.0",
    "lgbm_count_morgan_r2_2048_umap",
    "lgbm_atompair_2048_umap",
    "lgbm_count_morgan_r3_2048_umap",
    "lgbm_chemberta_5m_mtr_umap_gap1.0",
    "lgbm_molformer_xl_umap",
    "lgbm_morgan_r2_2048_umap",
    "lgbm_feat_morgan_r2_2048_umap",
    "lgbm_avalon_2048_umap",
    "lgbm_rdkit_desc_full_umap",
}
DROP_MIXED_ZEROW = [m for m in DROP_MIXED if m not in ZEROW_DROP]

# Minimal core: only top-weight members from prior runs
MINIMAL_CORE = [
    "chemprop_optuna_umap",
    "chemprop_chemeleon_umap",
    "chemprop_multitask5_umap_aux0.0_tuned",
    "attentivefp_optuna_umap",
    "gatedgcn_optuna_umap",
    "residual_physprop+mordred_umap",
    "lgbm_chemeleon_umap",
    "tabpfn_chemeleon_umap",
    "tabpfn_2d_full_boltz_umap",
    "lgbm_mordred_jazzy_umap",
]

POOLS = {
    "baseline_28 (current, submitted)": BASELINE_28,
    "replace_only (2d_full_boltz swap)": REPLACE_28,
    "drop_mixed": DROP_MIXED,
    "drop_mixed_zerow": DROP_MIXED_ZEROW,
    "minimal_core": MINIMAL_CORE,
}


# ---------------------------------------------------------------------
# Ensemble optimisation helpers
# ---------------------------------------------------------------------
def normalize(w: np.ndarray) -> np.ndarray:
    a = np.abs(w)
    s = a.sum()
    if s < 1e-12:
        raise RuntimeError("weights collapsed to zero")
    return a / s


def rae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sum(np.abs(y - p)) / np.sum(np.abs(y - y.mean())))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def optimise(oof: np.ndarray, y: np.ndarray, alpha: float, metric="mae") -> np.ndarray:
    n = oof.shape[1]
    equal = np.ones(n) / n
    met_fn = mae if metric == "mae" else rae

    def obj(w):
        wn = normalize(w)
        return met_fn(y, oof @ wn) + alpha * np.sum((wn - equal) ** 2)

    r = minimize(
        obj, equal.copy(), method="Nelder-Mead",
        options={"maxiter": 50000, "xatol": 1e-8, "fatol": 1e-8},
    )
    return normalize(r.x)


def evaluate(pool: list[str], y: np.ndarray, ids: dict[str, int]) -> dict:
    oofs = np.column_stack([load_oof_predictions(ids[n]) for n in pool])
    w = optimise(oofs, y, alpha=0.1, metric="mae")
    pred = oofs @ w
    return {
        "n": len(pool),
        "oof_rae": rae(y, pred),
        "oof_mae": mae(y, pred),
        "weights": dict(zip(pool, w)),
    }


def main() -> None:
    conn = psycopg2.connect(**DB_PARAMS)
    y = pd.read_sql(
        "SELECT pec50 FROM train_activity ORDER BY id", conn
    )["pec50"].to_numpy(dtype=np.float64)

    all_names = set()
    for p in POOLS.values():
        all_names |= set(p)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name FROM experiments WHERE name = ANY(%s)",
        (list(all_names),),
    )
    ids = dict((n, i) for i, n in cur.fetchall())
    conn.close()
    missing = all_names - set(ids)
    if missing:
        raise SystemExit(f"Missing experiments in DB: {missing}")

    print(
        f"{'pool':<38} {'n':>3} {'OOF RAE':>8} {'OOF MAE':>8} "
        f"{'dMAE vs baseline':>18}"
    )
    print("-" * 80)
    baseline_mae = None
    results = {}
    for name, pool in POOLS.items():
        r = evaluate(pool, y, ids)
        results[name] = r
        if baseline_mae is None:
            baseline_mae = r["oof_mae"]
        d = r["oof_mae"] - baseline_mae
        arrow = "[better]" if d < 0 else ("[worse]" if d > 0 else "")
        print(
            f"{name:<38} {r['n']:>3} {r['oof_rae']:>8.4f} "
            f"{r['oof_mae']:>8.4f} {d:>+14.4f}  {arrow}"
        )

    # Detailed weights for replace_only and drop_mixed_zerow
    print()
    for name in ("replace_only (2d_full_boltz swap)", "drop_mixed_zerow", "minimal_core"):
        print(f"\n=== {name} weight (top by weight) ===")
        r = results[name]
        for n, w in sorted(r["weights"].items(), key=lambda kv: -kv[1])[:12]:
            marker = " **" if "2d_full_boltz" in n else ""
            print(f"  {w:>6.3f}  {n}{marker}")


if __name__ == "__main__":
    main()
