"""OOF reliability diagnostics for Track 1 ensemble reweighting.

This script intentionally does not write submission CSVs. It stress-tests
current ensemble-pool reweighting under alternative validation proxies and
compares the induced test movement with submitted LB directions.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy import stats
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "track1_activity" / "src"
SCRIPTS_DIR = REPO_ROOT / "track1_activity" / "scripts"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from data import get_engine, load_test_smiles, load_train_smiles_with_counter  # noqa: E402
from evaluate import compute_metrics  # noqa: E402
from importance_weights import compute_importance_weights  # noqa: E402
from run_conservative_blend_probes import (  # noqa: E402
    load_latest_caruana_weight_map,
    load_pool_by_names,
    movement_summary,
    normalize_weight_map,
)
from splits import (  # noqa: E402
    mixed_analog_diversity_split_indices,
    test_nn_split_indices,
    umap_split_indices,
)
from submission_preflight import (  # noqa: E402
    DEFAULT_ANCHOR,
    align_submission,
    bad_axis_correlations,
    load_submission,
)

OUT_DIR = REPO_ROOT / "track1_activity" / "analysis" / "oof_proxy_diagnostics"


@dataclass(frozen=True)
class BlendSetting:
    split_name: str
    weight_name: str
    l2_anchor: float
    max_weight: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--bootstrap", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def mae(y: np.ndarray, pred: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    err = np.abs(y - pred)
    if sample_weight is None:
        return float(np.mean(err))
    return float(np.average(err, weights=sample_weight))


def fit_simplex_weighted_mae(
    X: np.ndarray,
    y: np.ndarray,
    anchor: np.ndarray,
    *,
    l2_anchor: float,
    sample_weight: np.ndarray | None = None,
    max_weight: float | None = None,
) -> np.ndarray:
    n = X.shape[1]
    bounds = [(0.0, 1.0 if max_weight is None else max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)

    def objective(w: np.ndarray) -> float:
        return mae(y, X @ w, sw) + l2_anchor * float(np.sum((w - anchor) ** 2))

    result = minimize(
        objective,
        anchor.copy(),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"SLSQP failed: {result.message}")
    w = np.clip(result.x, 0.0, None)
    return w / w.sum()


def normalized(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.clip(weights, 1e-6, None)
    return weights * (len(weights) / weights.sum())


def load_id55_anchor() -> np.ndarray:
    return load_submission(DEFAULT_ANCHOR)["pEC50"].to_numpy(dtype=np.float64)


def candidate_axis_rows(
    *,
    name: str,
    test: np.ndarray,
    raw_anchor_test: np.ndarray,
    id55_test: np.ndarray,
) -> dict[str, float | str]:
    row: dict[str, float | str] = {"name": name}
    row.update({f"raw_{k}": v for k, v in movement_summary(raw_anchor_test, test).items()})

    delta_vs_id55 = test - id55_test
    row.update(
        {
            "id55_abs_delta_mean": float(np.mean(np.abs(delta_vs_id55))),
            "id55_abs_delta_p90": float(np.quantile(np.abs(delta_vs_id55), 0.90)),
            "id55_abs_delta_max": float(np.max(np.abs(delta_vs_id55))),
            "id55_pearson": float(np.corrcoef(id55_test, test)[0, 1]),
            "id55_spearman": float(stats.spearmanr(id55_test, test).statistic),
        }
    )
    for axis in bad_axis_correlations(delta_vs_id55):
        row[f"{axis.label}_pearson"] = axis.pearson
        row[f"{axis.label}_spearman"] = axis.spearman
        row[f"{axis.label}_projection"] = axis.candidate_projection
    return row


def compute_nn_to_potent46(
    train_smiles: list[str], y: np.ndarray, selectivity: np.ndarray
) -> np.ndarray:
    potent_mask = (y >= 6.0) & (np.nan_to_num(selectivity, nan=-np.inf) >= 1.5)
    potent_idx = np.where(potent_mask)[0]
    if len(potent_idx) == 0:
        raise RuntimeError("No potent46 compounds found")

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mols = [Chem.MolFromSmiles(smi) for smi in train_smiles]
    invalid = [i for i, mol in enumerate(mols) if mol is None]
    if invalid:
        raise RuntimeError(f"Invalid train SMILES at indices {invalid[:10]}")
    fps = [gen.GetFingerprint(mol) for mol in mols if mol is not None]
    potent_fps = [fps[i] for i in potent_idx]
    out = np.zeros(len(train_smiles), dtype=np.float64)
    for i, fp in enumerate(fps):
        sims = DataStructs.BulkTanimotoSimilarity(fp, potent_fps)
        out[i] = max(sims) if sims else 0.0
    return out


def build_row_weights(
    train_smiles: list[str],
    test_smiles: list[str],
    y: np.ndarray,
    selectivity: np.ndarray,
    member_oof: np.ndarray,
) -> dict[str, np.ndarray]:
    iw = compute_importance_weights(train_smiles, test_smiles)
    nn_potent = compute_nn_to_potent46(train_smiles, y, selectivity)
    potent_soft = np.clip((nn_potent - 0.25) / 0.20, 0.0, 1.0)

    disagreement = member_oof.std(axis=1)
    disagree_rank = stats.rankdata(disagreement, method="average") / len(disagreement)
    low_disagreement = 1.5 - disagree_rank

    return {
        "uniform": np.ones(len(y), dtype=np.float64),
        "importance": normalized(iw),
        "potent46_soft": normalized(1.0 + 2.0 * potent_soft),
        "importance_x_potent46": normalized(iw * (1.0 + 2.0 * potent_soft)),
        "low_disagreement": normalized(low_disagreement),
    }


def build_splits(
    train_smiles: list[str],
    test_smiles: list[str],
    y: np.ndarray,
    selectivity: np.ndarray,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    return {
        "umap": umap_split_indices(train_smiles, n_splits=5, seed=42),
        "mixed_analog_t20": mixed_analog_diversity_split_indices(
            train_smiles, y, selectivity, n_splits=5, analog_tanimoto_threshold=0.20
        ),
        "mixed_analog_t25": mixed_analog_diversity_split_indices(
            train_smiles, y, selectivity, n_splits=5, analog_tanimoto_threshold=0.25
        ),
        "mixed_analog_t30": mixed_analog_diversity_split_indices(
            train_smiles, y, selectivity, n_splits=5, analog_tanimoto_threshold=0.30
        ),
        "test_nn_t25": test_nn_split_indices(
            train_smiles, test_smiles, n_splits=5, test_nn_threshold=0.25
        ),
        "test_nn_t30": test_nn_split_indices(
            train_smiles, test_smiles, n_splits=5, test_nn_threshold=0.30
        ),
    }


def run_setting(
    *,
    setting: BlendSetting,
    X: np.ndarray,
    X_test: np.ndarray,
    y: np.ndarray,
    split: list[tuple[np.ndarray, np.ndarray]],
    row_weight: np.ndarray,
    anchor_w: np.ndarray,
    anchor_oof: np.ndarray,
    anchor_test: np.ndarray,
    id55_test: np.ndarray,
) -> tuple[dict[str, float | str], np.ndarray]:
    oof = np.full(len(y), np.nan, dtype=np.float64)
    fold_weights = []
    for train_idx, val_idx in split:
        w = fit_simplex_weighted_mae(
            X[train_idx],
            y[train_idx],
            anchor_w,
            l2_anchor=setting.l2_anchor,
            sample_weight=row_weight[train_idx],
            max_weight=setting.max_weight,
        )
        oof[val_idx] = X[val_idx] @ w
        fold_weights.append(w)

    full_w = fit_simplex_weighted_mae(
        X,
        y,
        anchor_w,
        l2_anchor=setting.l2_anchor,
        sample_weight=row_weight,
        max_weight=setting.max_weight,
    )
    test = X_test @ full_w
    metrics = compute_metrics(y, oof)
    fold_weights_arr = np.vstack(fold_weights)
    row: dict[str, float | str] = {
        "name": (
            f"{setting.split_name}__{setting.weight_name}"
            f"__l2_{str(setting.l2_anchor).replace('.', 'p')}"
        ),
        "split": setting.split_name,
        "row_weight": setting.weight_name,
        "l2_anchor": setting.l2_anchor,
        "max_weight": "none" if setting.max_weight is None else setting.max_weight,
        "MAE": float(metrics["MAE"]),
        "Spearman_R": float(metrics["Spearman_R"]),
        "delta_mae_vs_raw_anchor": float(metrics["MAE"] - mae(y, anchor_oof)),
        "oof_abs_delta_p95": float(np.quantile(np.abs(oof - anchor_oof), 0.95)),
        "weight_l1_from_anchor": float(np.abs(full_w - anchor_w).sum()),
        "weight_max": float(full_w.max()),
        "fold_weight_l1_mean": float(np.abs(fold_weights_arr - anchor_w).sum(axis=1).mean()),
        "fold_weight_l1_std": float(np.abs(fold_weights_arr - anchor_w).sum(axis=1).std()),
    }
    row.update(
        candidate_axis_rows(
            name=str(row["name"]),
            test=test,
            raw_anchor_test=anchor_test,
            id55_test=id55_test,
        )
    )
    return row, full_w


def bootstrap_setting(
    *,
    name: str,
    X: np.ndarray,
    X_test: np.ndarray,
    y: np.ndarray,
    row_weight: np.ndarray,
    anchor_w: np.ndarray,
    anchor_test: np.ndarray,
    l2_anchor: float,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float | str]:
    rng = np.random.default_rng(seed)
    weight_rows = []
    p95_shifts = []
    l1_shifts = []
    n = len(y)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        w = fit_simplex_weighted_mae(
            X[idx],
            y[idx],
            anchor_w,
            l2_anchor=l2_anchor,
            sample_weight=row_weight[idx],
        )
        weight_rows.append(w)
        test = X_test @ w
        p95_shifts.append(float(np.quantile(np.abs(test - anchor_test), 0.95)))
        l1_shifts.append(float(np.abs(w - anchor_w).sum()))
    weights = np.vstack(weight_rows)
    return {
        "name": name,
        "l2_anchor": l2_anchor,
        "n_bootstrap": n_bootstrap,
        "weight_l1_mean": float(np.mean(l1_shifts)),
        "weight_l1_std": float(np.std(l1_shifts)),
        "test_p95_shift_mean": float(np.mean(p95_shifts)),
        "test_p95_shift_std": float(np.std(p95_shifts)),
        "top_member_weight_mean": float(np.mean(weights.max(axis=1))),
        "top_member_weight_p90": float(np.quantile(weights.max(axis=1), 0.90)),
        "mean_weight_std_across_members": float(np.mean(weights.std(axis=0))),
    }


def lb_submission_direction_table(id55_test: np.ndarray) -> pd.DataFrame:
    query = """
        SELECT id, submission_name, file_path, lb_mae, lb_spearman, submitted_at
          FROM lb_submissions
         WHERE track = 'activity'
           AND lb_mae IS NOT NULL
         ORDER BY id
    """
    rows = pd.read_sql(query, get_engine())
    out_rows = []
    id55_df = load_submission(DEFAULT_ANCHOR)
    id55_lb = rows.loc[rows["id"] == 55, "lb_mae"]
    id55_lb_mae = float(id55_lb.iloc[0]) if len(id55_lb) else np.nan
    for record in rows.to_dict(orient="records"):
        path = REPO_ROOT / str(record["file_path"])
        if not path.exists():
            continue
        try:
            cand = load_submission(path)
            aligned = align_submission(cand, id55_df)
        except Exception as exc:
            out_rows.append(
                {
                    "id": record["id"],
                    "submission_name": record["submission_name"],
                    "file_path": record["file_path"],
                    "lb_mae": record["lb_mae"],
                    "error": str(exc),
                }
            )
            continue
        pred = aligned["candidate"].to_numpy(dtype=np.float64)
        delta = pred - id55_test
        row: dict[str, float | str] = {
            "id": int(record["id"]),
            "submission_name": str(record["submission_name"]),
            "file_path": str(record["file_path"]),
            "lb_mae": float(record["lb_mae"]),
            "delta_lb_mae_vs_id55": float(record["lb_mae"] - id55_lb_mae),
            "lb_spearman": float(record["lb_spearman"])
            if pd.notna(record["lb_spearman"])
            else np.nan,
            "id55_abs_delta_mean": float(np.mean(np.abs(delta))),
            "id55_abs_delta_p90": float(np.quantile(np.abs(delta), 0.90)),
            "id55_abs_delta_max": float(np.max(np.abs(delta))),
            "id55_spearman": float(stats.spearmanr(id55_test, pred).statistic),
            "error": "",
        }
        for axis in bad_axis_correlations(delta):
            row[f"{axis.label}_projection"] = axis.candidate_projection
            row[f"{axis.label}_pearson"] = axis.pearson
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def summarize_weight_rows(names: list[str], weights_by_setting: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for setting_name, weights in weights_by_setting.items():
        for member, weight in zip(names, weights):
            rows.append({"setting": setting_name, "member": member, "weight": weight})
    return pd.DataFrame(rows).sort_values(["setting", "weight"], ascending=[True, False])


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_train_smiles_with_counter()
    test_df = load_test_smiles()
    y = train_df["pec50"].to_numpy(dtype=np.float64)
    selectivity = y - train_df["counter_pec50"].to_numpy(dtype=np.float64)
    train_smiles = train_df["smiles"].tolist()
    test_smiles = test_df["smiles"].tolist()

    weights_map = load_latest_caruana_weight_map()
    names = list(weights_map.keys())
    X, X_test = load_pool_by_names(names, y, n_test=len(test_df))
    anchor_w = normalize_weight_map(weights_map, names)
    anchor_oof = X @ anchor_w
    anchor_test = X_test @ anchor_w
    id55_test = load_id55_anchor()

    print(f"Loaded pool: {len(names)} members, train={X.shape}, test={X_test.shape}")
    print(f"Raw anchor MAE={mae(y, anchor_oof):.6f}")

    row_weights = build_row_weights(train_smiles, test_smiles, y, selectivity, X)
    splits = build_splits(train_smiles, test_smiles, y, selectivity)

    rows = []
    weights_by_setting = {}
    for split_name, split in splits.items():
        for weight_name, row_weight in row_weights.items():
            for l2 in (0.1, 0.3, 1.0, 3.0):
                setting = BlendSetting(split_name, weight_name, l2_anchor=l2)
                row, full_w = run_setting(
                    setting=setting,
                    X=X,
                    X_test=X_test,
                    y=y,
                    split=split,
                    row_weight=row_weight,
                    anchor_w=anchor_w,
                    anchor_oof=anchor_oof,
                    anchor_test=anchor_test,
                    id55_test=id55_test,
                )
                rows.append(row)
                weights_by_setting[str(row["name"])] = full_w
                print(
                    f"{row['name']}: MAE={row['MAE']:.6f} "
                    f"dMAE={row['delta_mae_vs_raw_anchor']:.6f} "
                    f"id55_p90={row['id55_abs_delta_p90']:.4f} "
                    f"bad56proj={row.get('id56_minus_id55_projection', np.nan):.3f}"
                )

    summary = pd.DataFrame(rows).sort_values(
        ["MAE", "id55_abs_delta_p90", "id56_minus_id55_projection"]
    )
    summary_path = args.out_dir / "oof_proxy_diagnostics_summary.csv"
    summary.to_csv(summary_path, index=False)

    weights_path = args.out_dir / "oof_proxy_diagnostics_weights.csv"
    summarize_weight_rows(names, weights_by_setting).to_csv(weights_path, index=False)

    lb_table = lb_submission_direction_table(id55_test)
    lb_path = args.out_dir / "lb_submission_direction_table.csv"
    lb_table.to_csv(lb_path, index=False)

    boot_rows = []
    for weight_name in ("uniform", "importance", "potent46_soft", "importance_x_potent46"):
        for l2 in (0.3, 1.0, 3.0):
            boot_rows.append(
                bootstrap_setting(
                    name=f"{weight_name}__l2_{str(l2).replace('.', 'p')}",
                    X=X,
                    X_test=X_test,
                    y=y,
                    row_weight=row_weights[weight_name],
                    anchor_w=anchor_w,
                    anchor_test=anchor_test,
                    l2_anchor=l2,
                    n_bootstrap=args.bootstrap,
                    seed=args.seed,
                )
            )
    boot = pd.DataFrame(boot_rows)
    boot_path = args.out_dir / "simplex_bootstrap_stability.csv"
    boot.to_csv(boot_path, index=False)

    print("\n=== Best by OOF MAE ===")
    display_cols = [
        "name",
        "MAE",
        "delta_mae_vs_raw_anchor",
        "id55_abs_delta_p90",
        "id56_minus_id55_projection",
        "weight_l1_from_anchor",
        "weight_max",
    ]
    print(summary[display_cols].head(20).to_markdown(index=False, floatfmt=".6f"))
    print(f"\nWrote {summary_path}")
    print(f"Wrote {weights_path}")
    print(f"Wrote {lb_path}")
    print(f"Wrote {boot_path}")


if __name__ == "__main__":
    main()
