#!/usr/bin/env python
"""Probe SHAP-suggested gates for small top500 deltas around id55.

This complements the log2fc-gated probe with non-log2fc features that are
available on train and test: ensemble-family disagreement and simple molecular
descriptors highlighted by the existing residual SHAP diagnostics.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "track1_activity" / "src"
SCRIPTS_DIR = REPO_ROOT / "track1_activity" / "scripts"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from data import load_test_descriptors, load_train_descriptors  # noqa: E402
from evaluate import compute_metrics  # noqa: E402
from run_conservative_blend_probes import (  # noqa: E402
    load_latest_caruana_weight_map,
    load_pool_by_names,
    normalize_weight_map,
)
from submission_preflight import (  # noqa: E402
    DEFAULT_ANCHOR,
    bad_axis_correlations,
    load_submission,
)

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "shap_gated_top500"

SEED10_TOP500 = "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap"
OPTUNA_TOP500 = (
    "tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_top500_umap"
)

DESCRIPTOR_GATES = [
    "logp",
    "exactmw",
    "tpsa",
    "fractioncsp3",
    "hba",
    "hbd",
    "num_heavy_atoms",
    "num_heteroatoms",
    "num_rotatable_bonds",
    "num_rings",
    "num_aromatic_rings",
]
META_GATES = ["family_gap", "abs_family_gap", "member_std", "member_range"]


@dataclass(frozen=True)
class DeltaSource:
    name: str
    oof_delta: np.ndarray
    test_delta: np.ndarray


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def load_oof_and_test_for_name(
    name: str, y: np.ndarray, n_test: int
) -> tuple[np.ndarray, np.ndarray]:
    X, X_test = load_pool_by_names([name], y, n_test)
    return X[:, 0], X_test[:, 0]


def soft_between(values: np.ndarray, lo: float, hi: float, *, direction: str) -> np.ndarray:
    if direction == "high":
        return np.clip((values - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    if direction == "low":
        return np.clip((hi - values) / max(hi - lo, 1e-9), 0.0, 1.0)
    raise ValueError(direction)


def add_ensemble_meta(
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    names: list[str],
    X: np.ndarray,
    X_test: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    family_mask = np.array(
        [
            ("chemprop" in name) or ("cheme_2d_full_boltz_log2fc_pred" in name)
            for name in names
        ],
        dtype=bool,
    )
    if family_mask.all() or (~family_mask).all():
        raise RuntimeError("family mask cannot define both family and non-family")
    out_train = train_feat.copy()
    out_test = test_feat.copy()
    for out, mat in ((out_train, X), (out_test, X_test)):
        out["member_std"] = mat.std(axis=1)
        out["member_range"] = mat.max(axis=1) - mat.min(axis=1)
        out["family_gap"] = mat[:, family_mask].mean(axis=1) - mat[
            :, ~family_mask
        ].mean(axis=1)
        out["abs_family_gap"] = np.abs(out["family_gap"])
    return out_train, out_test


def build_gates(train_feat: pd.DataFrame, test_feat: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    gates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for col in DESCRIPTOR_GATES + META_GATES:
        train_values = train_feat[col].to_numpy(dtype=np.float64)
        test_values = test_feat[col].to_numpy(dtype=np.float64)
        if not np.isfinite(train_values).all() or not np.isfinite(test_values).all():
            continue
        for direction in ("high", "low"):
            for q in (0.20, 0.30, 0.50, 0.70, 0.80):
                if direction == "high":
                    lo = float(np.quantile(train_values, q))
                    hi = float(np.quantile(train_values, 0.95))
                else:
                    lo = float(np.quantile(train_values, 0.05))
                    hi = float(np.quantile(train_values, q))
                name = f"{col}_{direction}_soft_q{int(q * 100)}"
                gates[name] = (
                    soft_between(train_values, lo, hi, direction=direction),
                    soft_between(test_values, lo, hi, direction=direction),
                )
            for q in (0.20, 0.30, 0.50, 0.70, 0.80):
                threshold = float(np.quantile(train_values, q))
                if direction == "high":
                    train_gate = (train_values >= threshold).astype(np.float64)
                    test_gate = (test_values >= threshold).astype(np.float64)
                else:
                    train_gate = (train_values <= threshold).astype(np.float64)
                    test_gate = (test_values <= threshold).astype(np.float64)
                gates[f"{col}_{direction}_hard_q{int(q * 100)}"] = (
                    train_gate,
                    test_gate,
                )
    return gates


def evaluate_source(
    *,
    source: DeltaSource,
    gates: dict[str, tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    raw_oof: np.ndarray,
    id55_test: np.ndarray,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    raw_mae = mae(y, raw_oof)
    raw_sp = float(stats.spearmanr(y, raw_oof).statistic)
    for gate_name, (train_gate, test_gate) in gates.items():
        for positive_only in (False, True):
            if positive_only:
                train_mask = source.oof_delta > 0.0
                test_mask = source.test_delta > 0.0
                train_gate_use = train_gate * train_mask
                test_gate_use = test_gate * test_mask
                full_gate_name = f"positive_delta_x_{gate_name}"
            else:
                train_gate_use = train_gate
                test_gate_use = test_gate
                full_gate_name = gate_name
            if test_gate_use.mean() == 0.0:
                continue
            for gamma in (0.15, 0.25, 0.35, 0.50):
                corrected = raw_oof + gamma * train_gate_use * source.oof_delta
                metrics = compute_metrics(y, corrected)
                candidate = id55_test + gamma * test_gate_use * source.test_delta
                delta = candidate - id55_test
                row: dict[str, float | str] = {
                    "source": source.name,
                    "gate": full_gate_name,
                    "gamma": gamma,
                    "MAE": float(metrics["MAE"]),
                    "delta_mae_vs_raw": float(metrics["MAE"] - raw_mae),
                    "Spearman_R": float(metrics["Spearman_R"]),
                    "delta_spearman_vs_raw": float(metrics["Spearman_R"] - raw_sp),
                    "train_gate_mean": float(np.mean(train_gate_use)),
                    "test_gate_mean": float(np.mean(test_gate_use)),
                    "id55_abs_delta_mean": float(np.mean(np.abs(delta))),
                    "id55_abs_delta_p90": float(np.quantile(np.abs(delta), 0.90)),
                    "id55_abs_delta_max": float(np.max(np.abs(delta))),
                    "id55_spearman": float(stats.spearmanr(id55_test, candidate).statistic),
                }
                for axis in bad_axis_correlations(delta):
                    row[f"{axis.label}_projection"] = axis.candidate_projection
                    row[f"{axis.label}_pearson"] = axis.pearson
                rows.append(row)
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_feat = load_train_descriptors()
    test_feat = load_test_descriptors()
    y = train_feat["pec50"].to_numpy(dtype=np.float64)

    weights_map = load_latest_caruana_weight_map()
    names = list(weights_map.keys())
    X, X_test = load_pool_by_names(names, y, n_test=len(test_feat))
    anchor_w = normalize_weight_map(weights_map, names)
    raw_oof = X @ anchor_w
    raw_test = X_test @ anchor_w
    train_feat, test_feat = add_ensemble_meta(train_feat, test_feat, names, X, X_test)

    seed10_oof, seed10_test = load_oof_and_test_for_name(SEED10_TOP500, y, len(test_feat))
    optuna_oof, optuna_test = load_oof_and_test_for_name(OPTUNA_TOP500, y, len(test_feat))
    sources = [
        DeltaSource("seed10_top500_minus_current_raw", seed10_oof - raw_oof, seed10_test - raw_test),
        DeltaSource("optuna_top500_minus_current_raw", optuna_oof - raw_oof, optuna_test - raw_test),
    ]
    id55_test = load_submission(DEFAULT_ANCHOR)["pEC50"].to_numpy(dtype=np.float64)
    gates = build_gates(train_feat, test_feat)

    rows: list[dict[str, float | str]] = []
    for source in sources:
        rows.extend(
            evaluate_source(
                source=source,
                gates=gates,
                y=y,
                raw_oof=raw_oof,
                id55_test=id55_test,
            )
        )
    summary = pd.DataFrame(rows).sort_values(
        ["delta_mae_vs_raw", "id55_abs_delta_p90", "id56_minus_id55_projection"]
    )
    summary.to_csv(OUT_DIR / "shap_gated_top500_summary.csv", index=False)

    safe = summary[
        (summary["delta_mae_vs_raw"] < -0.0005)
        & (summary["id55_abs_delta_p90"] <= 0.035)
        & (summary["id55_abs_delta_max"] <= 0.12)
    ].copy()
    safe.to_csv(OUT_DIR / "shap_gated_top500_safeish.csv", index=False)

    display_cols = [
        "source",
        "gate",
        "gamma",
        "MAE",
        "delta_mae_vs_raw",
        "id55_abs_delta_p90",
        "id55_abs_delta_max",
        "id56_minus_id55_projection",
        "test_gate_mean",
    ]
    print(f"Raw current OOF MAE={mae(y, raw_oof):.6f}")
    print("\n=== Best OOF ===")
    print(summary[display_cols].head(25).to_markdown(index=False, floatfmt=".6f"))
    print("\n=== Safe-ish ===")
    print(safe[display_cols].head(25).to_markdown(index=False, floatfmt=".6f"))
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
