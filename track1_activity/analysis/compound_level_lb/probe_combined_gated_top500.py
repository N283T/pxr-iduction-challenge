#!/usr/bin/env python
"""Probe combined log2fc/SHAP gates for small top500 deltas around id55."""

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
from probe_log2fc_gated_top500 import (  # noqa: E402
    LF_PATH,
    OPTUNA_TOP500,
    SEED10_TOP500,
    load_train_test_ids,
)
from probe_shap_gated_top500 import (  # noqa: E402
    add_ensemble_meta,
    load_oof_and_test_for_name,
    soft_between,
)
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

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "combined_gated_top500"
SUB_DIR = REPO_ROOT / "track1_activity" / "submissions"


@dataclass(frozen=True)
class DeltaSource:
    name: str
    oof_delta: np.ndarray
    test_delta: np.ndarray


@dataclass(frozen=True)
class Gate:
    name: str
    train: np.ndarray
    test: np.ndarray


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def soft_above(values: np.ndarray, q_lo: float, q_hi: float = 0.95) -> np.ndarray:
    lo = float(np.quantile(values, q_lo))
    hi = float(np.quantile(values, q_hi))
    return np.clip((values - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def log2fc_gate(col: str, q_lo: float) -> Gate:
    train_ids, test_ids = load_train_test_ids()
    lf = pd.read_parquet(LF_PATH)
    if not set(train_ids).issubset(lf.index) or not set(test_ids).issubset(lf.index):
        raise RuntimeError(f"{LF_PATH} does not cover current train/test ids")
    train_lf = lf.loc[train_ids].copy()
    test_lf = lf.loc[test_ids].copy()
    for df in (train_lf, test_lf):
        df["lf_mean"] = 0.5 * (df["log2fc_8p25_pred"] + df["log2fc_33_pred"])
        df["lf_slope"] = df["log2fc_33_pred"] - df["log2fc_8p25_pred"]
    return Gate(
        name=f"{col}_soft_q{int(q_lo * 100)}_to_q95",
        train=soft_above(train_lf[col].to_numpy(dtype=np.float64), q_lo),
        test=soft_above(test_lf[col].to_numpy(dtype=np.float64), q_lo),
    )


def shap_gate(
    train_feat: pd.DataFrame, test_feat: pd.DataFrame, col: str, q_lo: float
) -> Gate:
    train_values = train_feat[col].to_numpy(dtype=np.float64)
    test_values = test_feat[col].to_numpy(dtype=np.float64)
    lo = float(np.quantile(train_values, q_lo))
    hi = float(np.quantile(train_values, 0.95))
    return Gate(
        name=f"{col}_high_soft_q{int(q_lo * 100)}",
        train=soft_between(train_values, lo, hi, direction="high"),
        test=soft_between(test_values, lo, hi, direction="high"),
    )


def combine_gates(a: Gate, b: Gate) -> list[Gate]:
    return [
        Gate(f"{a.name}_AND_{b.name}", a.train * b.train, a.test * b.test),
        Gate(
            f"{a.name}_MEAN_{b.name}",
            0.5 * (a.train + b.train),
            0.5 * (a.test + b.test),
        ),
        Gate(
            f"{a.name}_MAX_{b.name}",
            np.maximum(a.train, b.train),
            np.maximum(a.test, b.test),
        ),
    ]


def evaluate_gate(
    *,
    source: DeltaSource,
    gate: Gate,
    y: np.ndarray,
    raw_oof: np.ndarray,
    id55_test: np.ndarray,
    gamma: float,
) -> dict[str, float | str]:
    corrected = raw_oof + gamma * gate.train * source.oof_delta
    metrics = compute_metrics(y, corrected)
    candidate = id55_test + gamma * gate.test * source.test_delta
    delta = candidate - id55_test
    row: dict[str, float | str] = {
        "source": source.name,
        "gate": gate.name,
        "gamma": gamma,
        "MAE": float(metrics["MAE"]),
        "delta_mae_vs_raw": float(metrics["MAE"] - mae(y, raw_oof)),
        "Spearman_R": float(metrics["Spearman_R"]),
        "train_gate_mean": float(np.mean(gate.train)),
        "test_gate_mean": float(np.mean(gate.test)),
        "id55_abs_delta_mean": float(np.mean(np.abs(delta))),
        "id55_abs_delta_p90": float(np.quantile(np.abs(delta), 0.90)),
        "id55_abs_delta_max": float(np.max(np.abs(delta))),
        "id55_spearman": float(stats.spearmanr(id55_test, candidate).statistic),
    }
    for axis in bad_axis_correlations(delta):
        row[f"{axis.label}_projection"] = axis.candidate_projection
        row[f"{axis.label}_pearson"] = axis.pearson
    return row


def write_candidate_csv(name: str, pred: np.ndarray) -> Path:
    anchor = load_submission(DEFAULT_ANCHOR)
    out = anchor.copy()
    out["pEC50"] = pred
    path = SUB_DIR / f"{name}.csv"
    out.to_csv(path, index=False)
    return path


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

    seed10_oof, seed10_test = load_oof_and_test_for_name(
        SEED10_TOP500, y, len(test_feat)
    )
    optuna_oof, optuna_test = load_oof_and_test_for_name(
        OPTUNA_TOP500, y, len(test_feat)
    )
    sources = [
        DeltaSource(
            "seed10_top500_minus_current_raw",
            seed10_oof - raw_oof,
            seed10_test - raw_test,
        ),
        DeltaSource(
            "optuna_top500_minus_current_raw",
            optuna_oof - raw_oof,
            optuna_test - raw_test,
        ),
    ]

    id55_test = load_submission(DEFAULT_ANCHOR)["pEC50"].to_numpy(dtype=np.float64)
    lf_gates = [
        log2fc_gate("log2fc_33_pred", 0.50),
        log2fc_gate("log2fc_33_pred", 0.60),
        log2fc_gate("lf_mean", 0.50),
    ]
    nr_gate = shap_gate(train_feat, test_feat, "num_rings", 0.20)
    fg_gate = shap_gate(train_feat, test_feat, "family_gap", 0.20)

    rows: list[dict[str, float | str]] = []
    candidate_specs: list[tuple[str, DeltaSource, Gate, float]] = []
    for source in sources:
        pos = Gate(
            "positive_delta",
            (source.oof_delta > 0.0).astype(np.float64),
            (source.test_delta > 0.0).astype(np.float64),
        )
        shap_gates = [
            nr_gate,
            fg_gate,
            Gate(
                f"positive_delta_AND_{nr_gate.name}",
                pos.train * nr_gate.train,
                pos.test * nr_gate.test,
            ),
            Gate(
                f"positive_delta_AND_{fg_gate.name}",
                pos.train * fg_gate.train,
                pos.test * fg_gate.test,
            ),
        ]
        gates: list[Gate] = []
        gates.extend(lf_gates)
        gates.extend(shap_gates)
        for lf_gate in lf_gates:
            for other in shap_gates:
                gates.extend(combine_gates(lf_gate, other))
        gates.extend(
            [
                Gate(
                    f"{lf_gates[1].name}_AND_{nr_gate.name}_AND_{fg_gate.name}",
                    lf_gates[1].train * nr_gate.train * fg_gate.train,
                    lf_gates[1].test * nr_gate.test * fg_gate.test,
                ),
                Gate(
                    f"{lf_gates[1].name}_MEAN_positive_delta_AND_{nr_gate.name}_AND_{fg_gate.name}",
                    (lf_gates[1].train + pos.train * nr_gate.train + fg_gate.train)
                    / 3.0,
                    (lf_gates[1].test + pos.test * nr_gate.test + fg_gate.test) / 3.0,
                ),
            ]
        )
        for gate in gates:
            if gate.test.mean() == 0.0:
                continue
            for gamma in (0.15, 0.25, 0.35, 0.50):
                rows.append(
                    evaluate_gate(
                        source=source,
                        gate=gate,
                        y=y,
                        raw_oof=raw_oof,
                        id55_test=id55_test,
                        gamma=gamma,
                    )
                )
                candidate_specs.append((source.name, source, gate, gamma))

    summary = pd.DataFrame(rows).sort_values(
        ["delta_mae_vs_raw", "id55_abs_delta_p90", "id56_minus_id55_projection"]
    )
    summary.to_csv(OUT_DIR / "combined_gated_top500_summary.csv", index=False)
    safe = summary[
        (summary["delta_mae_vs_raw"] < -0.0005)
        & (summary["id55_abs_delta_p90"] <= 0.035)
        & (summary["id55_abs_delta_max"] <= 0.12)
    ].copy()
    safe.to_csv(OUT_DIR / "combined_gated_top500_safeish.csv", index=False)

    key_by_spec = {
        (source_name, gate.name, gamma): (source, gate, gamma)
        for source_name, source, gate, gamma in candidate_specs
    }
    selected = safe.head(3)
    written_rows = []
    for idx, row in enumerate(selected.itertuples(index=False), start=1):
        source, gate, gamma = key_by_spec[(row.source, row.gate, float(row.gamma))]
        short = f"ens_id55_combo_gate_rank{idx}"
        pred = id55_test + gamma * gate.test * source.test_delta
        path = write_candidate_csv(short, pred)
        written_rows.append(
            {
                "rank": idx,
                "candidate": str(path),
                "source": row.source,
                "gate": row.gate,
                "gamma": row.gamma,
            }
        )
    pd.DataFrame(written_rows).to_csv(
        OUT_DIR / "combined_gated_top500_candidates.csv", index=False
    )

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
    print("\n=== Candidate CSVs ===")
    print(pd.DataFrame(written_rows).to_markdown(index=False))
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
