#!/usr/bin/env python
"""Probe tiny high-activity lifts around the latest trusted LB anchor.

The pseudo-public retrain battery suggested underprediction in a test-like,
high-activity holdout. This script tests a deliberately simple correction:

    candidate = id57 + amount * gate(log2fc_pred, anchor_pred, optional potent46)

It avoids borrowing more top500 movement, because the latest combined top500
gate (id58) was LB-negative despite good local evidence.
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
SCRIPT_DIR = REPO_ROOT / "track1_activity" / "scripts"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(
    0, str(REPO_ROOT / "track1_activity" / "analysis" / "oof_reliability_audit")
)

from audit_pseudo_public_splits import build_frame, split_registry  # noqa: E402
from data import get_engine, load_test_smiles, load_train_smiles_with_counter  # noqa: E402
from evaluate import compute_metrics  # noqa: E402
from probe_log2fc_gated_top500 import (  # noqa: E402
    LF_PATH,
    SEED10_TOP500,
    load_oof_and_test_for_name,
    potent46_soft_gate,
)
from run_conservative_blend_probes import (  # noqa: E402
    load_latest_caruana_weight_map,
    load_pool_by_names,
    normalize_weight_map,
)
from submission_preflight import bad_axis_correlations, load_submission  # noqa: E402

SUB_DIR = REPO_ROOT / "track1_activity" / "submissions"
OUT_DIR = (
    Path(__file__).resolve().parent / "outputs" / "conservative_high_activity_lift"
)
ID57_PATH = SUB_DIR / "ens_id51_top500_potent46_t40_soft_g50.csv"

EVAL_SPLITS = (
    "umap_canonical",
    "public_adv_top513",
    "public_testnn_top513",
    "public_log2fc_top513",
    "public_hybrid_nolabel_top513",
    "public_hybrid_with_y_top513",
    "public_chembl_ext_nn_ge025",
)


@dataclass(frozen=True)
class Gate:
    name: str
    train: np.ndarray
    test: np.ndarray


def load_train_test_ids() -> tuple[list[int], list[int]]:
    train_ids = pd.read_sql(
        "SELECT compound_id FROM train_activity ORDER BY id", get_engine()
    )["compound_id"].astype(int)
    test_ids = pd.read_sql(
        "SELECT compound_id FROM test_activity ORDER BY id", get_engine()
    )["compound_id"].astype(int)
    return train_ids.tolist(), test_ids.tolist()


def soft_above(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((values - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def quantile_soft_gate(
    train_values: np.ndarray,
    test_values: np.ndarray,
    *,
    q_lo: float,
    q_hi: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    lo = float(np.quantile(train_values, q_lo))
    hi = float(np.quantile(train_values, q_hi))
    return soft_above(train_values, lo, hi), soft_above(test_values, lo, hi)


def log2fc_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_ids, test_ids = load_train_test_ids()
    lf = pd.read_parquet(LF_PATH)
    if not set(train_ids).issubset(lf.index) or not set(test_ids).issubset(lf.index):
        raise RuntimeError(f"{LF_PATH} does not cover current train/test ids")
    train_lf = lf.loc[train_ids].copy()
    test_lf = lf.loc[test_ids].copy()
    for frame in (train_lf, test_lf):
        frame["lf_mean"] = 0.5 * (frame["log2fc_8p25_pred"] + frame["log2fc_33_pred"])
        frame["lf_slope"] = frame["log2fc_33_pred"] - frame["log2fc_8p25_pred"]
        frame["lf_max"] = frame[["log2fc_8p25_pred", "log2fc_33_pred"]].max(axis=1)
    return train_lf, test_lf


def reconstruct_anchor_proxy(
    train_df: pd.DataFrame, test_df: pd.DataFrame, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weights_map = load_latest_caruana_weight_map()
    names = list(weights_map.keys())
    pool_oof, pool_test = load_pool_by_names(names, y, n_test=len(test_df))
    weights = normalize_weight_map(weights_map, names)
    raw_oof = pool_oof @ weights
    raw_test = pool_test @ weights

    seed10_oof, seed10_test = load_oof_and_test_for_name(SEED10_TOP500, y, len(test_df))
    potent_train, potent_test = potent46_soft_gate(
        train_df, test_df["smiles"].tolist(), threshold=0.40
    )

    # id57 is the same potent46 soft-gate as id55, with gamma 0.50.
    anchor_oof = raw_oof + 0.50 * potent_train * (seed10_oof - raw_oof)
    anchor_test_proxy = raw_test + 0.50 * potent_test * (seed10_test - raw_test)
    anchor_test = load_submission(ID57_PATH)["pEC50"].to_numpy(dtype=np.float64)
    return anchor_oof, anchor_test, anchor_test_proxy, potent_test


def build_gates(
    *,
    anchor_oof: np.ndarray,
    anchor_test: np.ndarray,
    train_lf: pd.DataFrame,
    test_lf: pd.DataFrame,
    potent_train: np.ndarray,
    potent_test: np.ndarray,
) -> list[Gate]:
    gates: list[Gate] = []

    pred_gates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for q in (0.50, 0.60, 0.70):
        pred_gates[f"pred_soft_q{int(q * 100)}"] = quantile_soft_gate(
            anchor_oof, anchor_test, q_lo=q, q_hi=0.95
        )

    lf_gates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for col in ("log2fc_33_pred", "lf_mean", "lf_max"):
        train_values = train_lf[col].to_numpy(dtype=np.float64)
        test_values = test_lf[col].to_numpy(dtype=np.float64)
        for q in (0.50, 0.60, 0.70, 0.80):
            lf_gates[f"{col}_soft_q{int(q * 100)}"] = quantile_soft_gate(
                train_values, test_values, q_lo=q, q_hi=0.95
            )

    for lf_name, (lf_train, lf_test) in lf_gates.items():
        gates.append(Gate(lf_name, lf_train, lf_test))
        gates.append(
            Gate(
                f"potent46_x_{lf_name}",
                potent_train * lf_train,
                potent_test * lf_test,
            )
        )
        for pred_name, (pred_train, pred_test) in pred_gates.items():
            gates.append(
                Gate(
                    f"{lf_name}_AND_{pred_name}",
                    lf_train * pred_train,
                    lf_test * pred_test,
                )
            )
            gates.append(
                Gate(
                    f"potent46_x_{lf_name}_AND_{pred_name}",
                    potent_train * lf_train * pred_train,
                    potent_test * lf_test * pred_test,
                )
            )
    return gates


def split_metric_delta(
    y: np.ndarray,
    anchor_oof: np.ndarray,
    candidate_oof: np.ndarray,
    registry: dict[str, list[tuple[np.ndarray, np.ndarray]]],
) -> dict[str, float]:
    rows: dict[str, float] = {}
    for split in EVAL_SPLITS:
        if split not in registry:
            continue
        base_maes = []
        cand_maes = []
        for _tr, va in registry[split]:
            base_maes.append(float(np.mean(np.abs(y[va] - anchor_oof[va]))))
            cand_maes.append(float(np.mean(np.abs(y[va] - candidate_oof[va]))))
        rows[f"{split}_mae"] = float(np.mean(cand_maes))
        rows[f"{split}_delta_mae"] = float(np.mean(cand_maes) - np.mean(base_maes))
    stress = [
        rows[f"{split}_delta_mae"]
        for split in EVAL_SPLITS
        if split.startswith("public_") and f"{split}_delta_mae" in rows
    ]
    rows["public_stress_mean_delta_mae"] = float(np.mean(stress))
    return rows


def evaluate_candidate(
    *,
    gate: Gate,
    amount: float,
    y: np.ndarray,
    anchor_oof: np.ndarray,
    anchor_test: np.ndarray,
    registry: dict[str, list[tuple[np.ndarray, np.ndarray]]],
) -> dict[str, float | str]:
    candidate_oof = anchor_oof + amount * gate.train
    candidate_test = anchor_test + amount * gate.test
    test_shift = candidate_test - anchor_test
    full_base = compute_metrics(y, anchor_oof)
    full_cand = compute_metrics(y, candidate_oof)
    row: dict[str, float | str] = {
        "gate": gate.name,
        "amount": amount,
        "full_mae": float(full_cand["MAE"]),
        "full_delta_mae": float(full_cand["MAE"] - full_base["MAE"]),
        "full_spearman": float(full_cand["Spearman_R"]),
        "full_delta_spearman": float(full_cand["Spearman_R"] - full_base["Spearman_R"]),
        "train_gate_mean": float(gate.train.mean()),
        "test_gate_mean": float(gate.test.mean()),
        "test_shift_mean": float(test_shift.mean()),
        "test_abs_shift_mean": float(np.abs(test_shift).mean()),
        "test_abs_shift_p90": float(np.quantile(np.abs(test_shift), 0.90)),
        "test_abs_shift_max": float(np.abs(test_shift).max()),
        "test_n_shift_gt_005": int((np.abs(test_shift) > 0.05).sum()),
        "id57_spearman": float(stats.spearmanr(anchor_test, candidate_test).statistic),
    }
    row.update(split_metric_delta(y, anchor_oof, candidate_oof, registry))
    for axis in bad_axis_correlations(test_shift):
        row[f"{axis.label}_projection"] = axis.candidate_projection
        row[f"{axis.label}_pearson"] = axis.pearson
    return row


def write_submission(name: str, pred: np.ndarray) -> Path:
    anchor = load_submission(ID57_PATH)
    out = anchor.copy()
    out["pEC50"] = pred
    path = SUB_DIR / f"{name}.csv"
    out.to_csv(path, index=False)
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df = load_train_smiles_with_counter()
    test_df = load_test_smiles()
    y = train_df["pec50"].to_numpy(dtype=np.float64)
    split_frame = build_frame()
    registry = split_registry(split_frame)

    anchor_oof, anchor_test, anchor_test_proxy, potent_test = reconstruct_anchor_proxy(
        train_df, test_df, y
    )
    potent_train, _ = potent46_soft_gate(
        train_df, test_df["smiles"].tolist(), threshold=0.40
    )
    train_lf, test_lf = log2fc_features()
    gates = build_gates(
        anchor_oof=anchor_oof,
        anchor_test=anchor_test,
        train_lf=train_lf,
        test_lf=test_lf,
        potent_train=potent_train,
        potent_test=potent_test,
    )

    rows: list[dict[str, float | str]] = []
    for gate in gates:
        if gate.test.mean() == 0:
            continue
        for amount in (0.01, 0.02, 0.03, 0.05, 0.08):
            rows.append(
                evaluate_candidate(
                    gate=gate,
                    amount=amount,
                    y=y,
                    anchor_oof=anchor_oof,
                    anchor_test=anchor_test,
                    registry=registry,
                )
            )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        [
            "public_hybrid_with_y_top513_delta_mae",
            "full_delta_mae",
            "test_abs_shift_p90",
        ]
    )
    summary.to_csv(OUT_DIR / "high_activity_lift_summary.csv", index=False)

    safe = summary[
        (summary["public_hybrid_with_y_top513_delta_mae"] < -0.0010)
        & (summary["full_delta_mae"] <= 0.0010)
        & (summary["public_stress_mean_delta_mae"] <= 0.0005)
        & (summary["test_abs_shift_p90"] <= 0.030)
        & (summary["test_abs_shift_max"] <= 0.080)
        & (summary["test_n_shift_gt_005"] <= 30)
    ].copy()
    safe.to_csv(OUT_DIR / "high_activity_lift_safeish.csv", index=False)

    written = []
    key = {
        (row.gate, float(row.amount)): row for row in summary.itertuples(index=False)
    }
    for idx, row in enumerate(safe.head(3).itertuples(index=False), start=1):
        gate = next(g for g in gates if g.name == row.gate)
        pred = anchor_test + float(row.amount) * gate.test
        path = write_submission(f"ens_id57_high_activity_lift_rank{idx}", pred)
        written.append(
            {
                "rank": idx,
                "path": str(path.relative_to(REPO_ROOT)),
                "gate": row.gate,
                "amount": row.amount,
                "public_hybrid_with_y_delta_mae": row.public_hybrid_with_y_top513_delta_mae,
                "full_delta_mae": row.full_delta_mae,
                "test_abs_shift_p90": row.test_abs_shift_p90,
                "test_abs_shift_max": row.test_abs_shift_max,
            }
        )
        key[(row.gate, float(row.amount))]
    written_df = pd.DataFrame(written)
    written_df.to_csv(OUT_DIR / "high_activity_lift_candidates.csv", index=False)

    base = compute_metrics(y, anchor_oof)
    report = [
        "# Conservative High-Activity Lift Probe",
        "",
        f"Anchor CSV: `{ID57_PATH.relative_to(REPO_ROOT)}`",
        f"Anchor OOF proxy MAE: `{base['MAE']:.6f}`",
        f"Anchor test/proxy mean absolute difference: `{np.mean(np.abs(anchor_test - anchor_test_proxy)):.6f}`",
        "",
        "Candidate formula:",
        "",
        "```text",
        "candidate = id57 + amount * gate(log2fc_pred, anchor_pred, optional potent46)",
        "```",
        "",
        "## Best By Pseudo High-Activity Holdout",
        "",
        summary[
            [
                "gate",
                "amount",
                "full_delta_mae",
                "public_stress_mean_delta_mae",
                "public_hybrid_with_y_top513_delta_mae",
                "public_log2fc_top513_delta_mae",
                "test_abs_shift_p90",
                "test_abs_shift_max",
                "id56_minus_id55_projection",
            ]
        ]
        .head(20)
        .to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Safe-Ish",
        "",
    ]
    if safe.empty:
        report.append("No candidate passed the conservative safe-ish filters.")
    else:
        report.append(
            safe[
                [
                    "gate",
                    "amount",
                    "full_delta_mae",
                    "public_stress_mean_delta_mae",
                    "public_hybrid_with_y_top513_delta_mae",
                    "test_abs_shift_p90",
                    "test_abs_shift_max",
                    "id56_minus_id55_projection",
                ]
            ]
            .head(20)
            .to_markdown(index=False, floatfmt=".6f")
        )
    report.extend(
        [
            "",
            "## Candidate CSVs",
            "",
            written_df.to_markdown(index=False, floatfmt=".6f")
            if not written_df.empty
            else "No CSVs written.",
            "",
        ]
    )
    (OUT_DIR / "report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"Anchor proxy MAE={base['MAE']:.6f}")
    print("\n=== Best by pseudo high-activity holdout ===")
    print(
        summary[
            [
                "gate",
                "amount",
                "full_delta_mae",
                "public_stress_mean_delta_mae",
                "public_hybrid_with_y_top513_delta_mae",
                "test_abs_shift_p90",
                "test_abs_shift_max",
                "id56_minus_id55_projection",
            ]
        ]
        .head(20)
        .to_markdown(index=False, floatfmt=".6f")
    )
    print("\n=== Safe-ish ===")
    if safe.empty:
        print("No safe-ish candidates.")
    else:
        print(
            safe[
                [
                    "gate",
                    "amount",
                    "full_delta_mae",
                    "public_stress_mean_delta_mae",
                    "public_hybrid_with_y_top513_delta_mae",
                    "test_abs_shift_p90",
                    "test_abs_shift_max",
                    "id56_minus_id55_projection",
                ]
            ]
            .head(20)
            .to_markdown(index=False, floatfmt=".6f")
        )
    print("\n=== Candidate CSVs ===")
    print(
        written_df.to_markdown(index=False, floatfmt=".6f")
        if not written_df.empty
        else "No CSVs written."
    )
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
