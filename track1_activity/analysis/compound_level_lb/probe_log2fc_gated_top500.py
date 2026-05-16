#!/usr/bin/env python
"""Probe log2fc-gated top500 deltas around the id55 anchor.

No submission CSVs are written. This is a cheap follow-up to the OOF proxy
diagnostics: if log2fc-derived features are the useful axis, check whether
borrowing top500 movement only in high predicted-log2fc regions is safer than a
global ensemble reweight.
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

from data import get_engine, load_test_smiles, load_train_smiles_with_counter  # noqa: E402
from evaluate import compute_metrics  # noqa: E402
from run_conservative_blend_probes import (  # noqa: E402
    load_latest_caruana_weight_map,
    load_pool_by_names,
    normalize_weight_map,
)
from splits import _morgan_fp_matrix  # noqa: E402
from submission_preflight import (  # noqa: E402
    DEFAULT_ANCHOR,
    bad_axis_correlations,
    load_submission,
)

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "log2fc_gated_top500"
LF_PATH = (
    REPO_ROOT
    / "data"
    / "chemprop_pretrain_log2fc_predictions_optuna_trial10_seed5ens.parquet"
)

SEED10_TOP500 = "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap"
OPTUNA_TOP500 = (
    "tabpfn_cheme_2d_full_boltz_log2fc_pred_optuna_trial10_seed5ens_top500_umap"
)


@dataclass(frozen=True)
class DeltaSource:
    name: str
    oof_delta: np.ndarray
    test_delta: np.ndarray


def load_train_test_ids() -> tuple[list[int], list[int]]:
    train_ids = pd.read_sql(
        "SELECT compound_id FROM train_activity ORDER BY id", get_engine()
    )["compound_id"].astype(int)
    test_ids = pd.read_sql(
        "SELECT compound_id FROM test_activity ORDER BY id", get_engine()
    )["compound_id"].astype(int)
    return train_ids.tolist(), test_ids.tolist()


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def load_oof_and_test_for_name(
    name: str, y: np.ndarray, n_test: int
) -> tuple[np.ndarray, np.ndarray]:
    return tuple(arr[:, 0] for arr in load_pool_by_names([name], y, n_test))  # type: ignore[return-value]


def tanimoto_max(query: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    query_bool = query.astype(bool)
    anchors_bool = anchors.astype(bool)
    inter = query_bool.astype(np.uint16) @ anchors_bool.astype(np.uint16).T
    union = query_bool.sum(axis=1, keepdims=True) + anchors_bool.sum(axis=1) - inter
    return np.divide(
        inter, union, out=np.zeros_like(inter, dtype=np.float64), where=union > 0
    ).max(axis=1)


def potent46_soft_gate(
    train_df: pd.DataFrame, test_smiles: list[str], *, threshold: float = 0.40
) -> tuple[np.ndarray, np.ndarray]:
    y = train_df["pec50"].to_numpy(dtype=np.float64)
    selectivity = y - train_df["counter_pec50"].to_numpy(dtype=np.float64)
    potent = (y >= 6.0) & (np.nan_to_num(selectivity, nan=-np.inf) >= 1.5)
    train_fp = _morgan_fp_matrix(train_df["smiles"].tolist())
    test_fp = _morgan_fp_matrix(test_smiles)
    anchors = train_fp[potent]
    train_nn = tanimoto_max(train_fp, anchors)
    test_nn = tanimoto_max(test_fp, anchors)
    return (
        np.clip((train_nn - threshold) / 0.15, 0.0, 1.0),
        np.clip((test_nn - threshold) / 0.15, 0.0, 1.0),
    )


def log2fc_frame(train_ids: list[int], test_ids: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    lf = pd.read_parquet(LF_PATH)
    if not set(train_ids).issubset(lf.index) or not set(test_ids).issubset(lf.index):
        raise RuntimeError(f"{LF_PATH} does not cover current train/test ids")
    train_lf = lf.loc[train_ids].copy()
    test_lf = lf.loc[test_ids].copy()
    for df in (train_lf, test_lf):
        df["lf_mean"] = 0.5 * (df["log2fc_8p25_pred"] + df["log2fc_33_pred"])
        df["lf_slope"] = df["log2fc_33_pred"] - df["log2fc_8p25_pred"]
        df["lf_max"] = df[["log2fc_8p25_pred", "log2fc_33_pred"]].max(axis=1)
    return train_lf, test_lf


def soft_above(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi <= lo:
        return (values >= lo).astype(np.float64)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def build_gates(
    train_lf: pd.DataFrame,
    test_lf: pd.DataFrame,
    potent_train: np.ndarray,
    potent_test: np.ndarray,
    source: DeltaSource,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    gates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for col in ("log2fc_33_pred", "lf_mean", "lf_slope"):
        train_values = train_lf[col].to_numpy(dtype=np.float64)
        test_values = test_lf[col].to_numpy(dtype=np.float64)
        for q in (0.50, 0.60, 0.70, 0.80, 0.90):
            thr = float(np.quantile(train_values, q))
            name = f"{col}_hard_q{int(q * 100)}"
            gates[name] = (
                (train_values >= thr).astype(np.float64),
                (test_values >= thr).astype(np.float64),
            )
        for q in (0.50, 0.60, 0.70, 0.80):
            lo = float(np.quantile(train_values, q))
            hi = float(np.quantile(train_values, 0.95))
            name = f"{col}_soft_q{int(q * 100)}_to_q95"
            gates[name] = (soft_above(train_values, lo, hi), soft_above(test_values, lo, hi))

    base_keys = list(gates.keys())
    for key in base_keys:
        train_gate, test_gate = gates[key]
        gates[f"potent46_x_{key}"] = (potent_train * train_gate, potent_test * test_gate)
        gates[f"positive_delta_x_{key}"] = (
            train_gate * (source.oof_delta > 0.0),
            test_gate * (source.test_delta > 0.0),
        )
        gates[f"potent46_x_positive_delta_x_{key}"] = (
            potent_train * train_gate * (source.oof_delta > 0.0),
            potent_test * test_gate * (source.test_delta > 0.0),
        )
    return gates


def source_rows(
    source: DeltaSource,
    y: np.ndarray,
    raw_oof: np.ndarray,
    id55_test: np.ndarray,
    gates: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    raw_mae = mae(y, raw_oof)
    raw_sp = float(stats.spearmanr(y, raw_oof).statistic)
    for gate_name, (train_gate, test_gate) in gates.items():
        for gamma in (0.15, 0.25, 0.35, 0.50):
            shift_oof = gamma * train_gate * source.oof_delta
            corrected = raw_oof + shift_oof
            metrics = compute_metrics(y, corrected)
            shift_test = gamma * test_gate * source.test_delta
            candidate = id55_test + shift_test
            delta = candidate - id55_test
            row: dict[str, float | str] = {
                "source": source.name,
                "gate": gate_name,
                "gamma": gamma,
                "MAE": float(metrics["MAE"]),
                "delta_mae_vs_raw": float(metrics["MAE"] - raw_mae),
                "Spearman_R": float(metrics["Spearman_R"]),
                "delta_spearman_vs_raw": float(metrics["Spearman_R"] - raw_sp),
                "train_gate_mean": float(np.mean(train_gate)),
                "test_gate_mean": float(np.mean(test_gate)),
                "test_shift_mean": float(np.mean(shift_test)),
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
    train_df = load_train_smiles_with_counter()
    test_df = load_test_smiles()
    y = train_df["pec50"].to_numpy(dtype=np.float64)
    train_ids, test_ids = load_train_test_ids()

    weights_map = load_latest_caruana_weight_map()
    names = list(weights_map.keys())
    X, X_test = load_pool_by_names(names, y, n_test=len(test_df))
    anchor_w = normalize_weight_map(weights_map, names)
    raw_oof = X @ anchor_w
    raw_test = X_test @ anchor_w

    seed10_oof, seed10_test = load_oof_and_test_for_name(SEED10_TOP500, y, len(test_df))
    optuna_oof, optuna_test = load_oof_and_test_for_name(OPTUNA_TOP500, y, len(test_df))
    sources = [
        DeltaSource("seed10_top500_minus_current_raw", seed10_oof - raw_oof, seed10_test - raw_test),
        DeltaSource("optuna_top500_minus_current_raw", optuna_oof - raw_oof, optuna_test - raw_test),
    ]

    id55_test = load_submission(DEFAULT_ANCHOR)["pEC50"].to_numpy(dtype=np.float64)
    train_lf, test_lf = log2fc_frame(train_ids, test_ids)
    potent_train, potent_test = potent46_soft_gate(train_df, test_df["smiles"].tolist())

    all_rows = []
    for source in sources:
        gates = build_gates(train_lf, test_lf, potent_train, potent_test, source)
        all_rows.extend(source_rows(source, y, raw_oof, id55_test, gates))

    summary = pd.DataFrame(all_rows)
    summary = summary.sort_values(
        ["delta_mae_vs_raw", "id55_abs_delta_p90", "id56_minus_id55_projection"]
    )
    summary.to_csv(OUT_DIR / "log2fc_gated_top500_summary.csv", index=False)

    safe = summary[
        (summary["delta_mae_vs_raw"] < -0.0005)
        & (summary["id55_abs_delta_p90"] <= 0.035)
        & (summary["id55_abs_delta_max"] <= 0.12)
    ].copy()
    safe.to_csv(OUT_DIR / "log2fc_gated_top500_safeish.csv", index=False)

    print(f"Raw current OOF MAE={mae(y, raw_oof):.6f}")
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
    print("\n=== Best OOF ===")
    print(summary[display_cols].head(25).to_markdown(index=False, floatfmt=".6f"))
    print("\n=== Safe-ish ===")
    print(safe[display_cols].head(25).to_markdown(index=False, floatfmt=".6f"))
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
