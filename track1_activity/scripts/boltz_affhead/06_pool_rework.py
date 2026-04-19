"""Holistic ensemble rework: search over pool compositions.

Candidates (11 registered):
  # Existing 8-model pool
  chemprop_optuna_umap                              (OOF 0.521)
  chemprop_chemeleon_umap                           (OOF 0.549)
  chemprop_multitask5_umap_aux0.0_tuned             (OOF 0.524)
  attentivefp_optuna_umap                           (OOF 0.528)
  gatedgcn_optuna_umap                              (OOF 0.546)
  residual_physprop+mordred_umap                    (OOF 0.528)
  tabpfn_2d_full_boltz_umap                         (OOF 0.483)
  tabpfn_chemeleon_umap                             (OOF 0.512)
  # New Boltz-pooled variants
  lgbm_pooled_boltz_umap                            (OOF 0.512)
  tabpfn_pooled_boltz_umap_default                  (OOF 0.486)
  tabpfn_pooled_boltz_allpairs_umap_default         (OOF 0.486)

Outputs:
  - Full correlation matrix among all 11
  - Table of candidate pool compositions with OOF MAE + concentration
  - Recommendation based on "max OOF MAE with no single weight > 0.4"
    (robustness heuristic: prevents over-reliance on one member that
    OOF sees as stronger than LB rewards)
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))
from data import DB_PARAMS  # noqa: E402
from evaluate import load_oof_predictions  # noqa: E402


EXISTING_8 = (
    "chemprop_optuna_umap",
    "chemprop_chemeleon_umap",
    "chemprop_multitask5_umap_aux0.0_tuned",
    "attentivefp_optuna_umap",
    "gatedgcn_optuna_umap",
    "residual_physprop+mordred_umap",
    "tabpfn_2d_full_boltz_umap",
    "tabpfn_chemeleon_umap",
)

NEW_CANDIDATES = (
    "lgbm_pooled_boltz_umap",
    "tabpfn_pooled_boltz_umap_default",
    "tabpfn_pooled_boltz_allpairs_umap_default",
)

ALL_11 = EXISTING_8 + NEW_CANDIDATES


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rae(y, p):
    return float(np.sum(np.abs(y - p)) / np.sum(np.abs(y - y.mean())))


def norm(w):
    a = np.abs(w)
    return a / a.sum()


def opt(pool, oofs, y, alpha=0.0):
    X = np.column_stack([oofs[n] for n in pool])
    n = X.shape[1]
    eq = np.ones(n) / n

    def obj(w):
        wn = norm(w)
        return mae(y, X @ wn) + alpha * np.sum((wn - eq) ** 2)

    r = minimize(
        obj,
        eq.copy(),
        method="Nelder-Mead",
        options=dict(maxiter=50000, xatol=1e-8, fatol=1e-8),
    )
    w = norm(r.x)
    return w, mae(y, X @ w), rae(y, X @ w)


def main() -> None:
    conn = psycopg2.connect(**DB_PARAMS)
    y = pd.read_sql(
        "SELECT pec50 FROM train_activity ORDER BY id", conn
    )["pec50"].to_numpy(dtype=np.float64)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name FROM experiments WHERE name = ANY(%s)",
        (list(ALL_11),),
    )
    ids = dict((n, i) for i, n in cur.fetchall())
    conn.close()
    missing = set(ALL_11) - set(ids)
    if missing:
        raise SystemExit(f"Missing in DB: {missing}")

    oofs = {n: load_oof_predictions(ids[n]) for n in ALL_11}
    singles = {n: mae(y, o) for n, o in oofs.items()}

    print("\n=== Single-model OOF MAE ===")
    for n in sorted(ALL_11, key=lambda m: singles[m]):
        marker = "  *" if n in NEW_CANDIDATES else ""
        print(f"  {singles[n]:.4f}  {n}{marker}")

    print("\n=== Correlation matrix (top 11) ===")
    short = {n: n[:26] for n in ALL_11}
    header = "".join(f"{short[n]:>28}" for n in ALL_11)
    print(" " * 35 + header)
    for a in ALL_11:
        cells = []
        for b in ALL_11:
            if a == b:
                cells.append(f"{'1.000':>28}")
            else:
                r = stats.pearsonr(oofs[a], oofs[b]).statistic
                cells.append(f"{r:>28.3f}")
        print(f"{short[a]:<35}" + "".join(cells))

    # Pool candidates to try
    compositions = [
        ("8-base (current)", EXISTING_8),
        ("8 + lgbm_pooled", EXISTING_8 + ("lgbm_pooled_boltz_umap",)),
        ("8 + tabpfn_pooled_core", EXISTING_8 + ("tabpfn_pooled_boltz_umap_default",)),
        (
            "8 + tabpfn_pooled_allpairs",
            EXISTING_8 + ("tabpfn_pooled_boltz_allpairs_umap_default",),
        ),
        # Drop low-weight members
        (
            "Drop 3 low-weight (residual, gatedgcn, attentivefp)",
            tuple(
                m
                for m in EXISTING_8
                if m
                not in (
                    "residual_physprop+mordred_umap",
                    "gatedgcn_optuna_umap",
                    "attentivefp_optuna_umap",
                )
            ),
        ),
        # Swap-in candidates
        (
            "Drop 3 low + tabpfn_pooled_allpairs",
            tuple(
                m
                for m in EXISTING_8
                if m
                not in (
                    "residual_physprop+mordred_umap",
                    "gatedgcn_optuna_umap",
                    "attentivefp_optuna_umap",
                )
            )
            + ("tabpfn_pooled_boltz_allpairs_umap_default",),
        ),
        # Full 11
        ("All 11", ALL_11),
        # Minimal 5 aggressive
        (
            "Minimal 5 (tabpfn + cp_optuna + cp_mt5 + tabpfn_pooled)",
            (
                "tabpfn_2d_full_boltz_umap",
                "tabpfn_chemeleon_umap",
                "chemprop_optuna_umap",
                "chemprop_multitask5_umap_aux0.0_tuned",
                "tabpfn_pooled_boltz_allpairs_umap_default",
            ),
        ),
        # TabPFN heavy (both tabpfn variants + diverse graph members)
        (
            "TabPFN-heavy (2 tabpfn + 3 chemprop + 1 tabpfn_pooled)",
            (
                "tabpfn_2d_full_boltz_umap",
                "tabpfn_chemeleon_umap",
                "chemprop_optuna_umap",
                "chemprop_chemeleon_umap",
                "chemprop_multitask5_umap_aux0.0_tuned",
                "tabpfn_pooled_boltz_allpairs_umap_default",
            ),
        ),
    ]

    print("\n=== Pool composition A/B (vanilla weights) ===")
    print(
        f'{"pool":<60} {"n":>3} {"MAE":>7} {"RAE":>7} '
        f'{"max_w":>6} {"max_member":>28}'
    )
    print("-" * 130)
    results = []
    for tag, pool in compositions:
        w, m, r = opt(pool, oofs, y, alpha=0.0)
        wd = dict(zip(pool, w))
        mx_name = max(wd, key=wd.get)
        mx_w = wd[mx_name]
        results.append((tag, pool, m, r, mx_w, mx_name, wd))
        print(
            f"{tag:<60} {len(pool):>3} {m:>7.4f} {r:>7.4f} "
            f"{mx_w:>6.3f} {mx_name[:28]:>28}"
        )

    print("\n=== Same pools with L2 alpha=0.1 (robust weights) ===")
    print(
        f'{"pool":<60} {"n":>3} {"MAE":>7} {"RAE":>7} '
        f'{"max_w":>6} {"max_member":>28}'
    )
    print("-" * 130)
    for tag, pool in compositions:
        w, m, r = opt(pool, oofs, y, alpha=0.1)
        wd = dict(zip(pool, w))
        mx_name = max(wd, key=wd.get)
        mx_w = wd[mx_name]
        print(
            f"{tag:<60} {len(pool):>3} {m:>7.4f} {r:>7.4f} "
            f"{mx_w:>6.3f} {mx_name[:28]:>28}"
        )

    # Brute-force search small pools (n=5..7) for lowest OOF MAE under
    # robustness constraint (max weight <= 0.4 in vanilla).
    print(
        "\n=== Brute-force search: all n=5..7 subsets with vanilla max_w <= 0.4 ==="
    )
    candidate_subsets = []
    for sz in (5, 6, 7):
        for combo in combinations(ALL_11, sz):
            w, m, r = opt(list(combo), oofs, y, alpha=0.0)
            if w.max() <= 0.4:
                candidate_subsets.append(
                    (m, r, sz, w.max(), combo, dict(zip(combo, w)))
                )
    candidate_subsets.sort()
    print(f"Total qualifying subsets: {len(candidate_subsets)}")
    print("\nTop 10 by vanilla OOF MAE under max_w<=0.4:")
    for m, r, sz, mw, combo, wd in candidate_subsets[:10]:
        members = ", ".join(sorted(combo, key=lambda n: -wd[n])[:3])
        print(f"  MAE={m:.4f}  RAE={r:.4f}  n={sz}  max_w={mw:.3f}  top3: {members}...")


if __name__ == "__main__":
    main()
