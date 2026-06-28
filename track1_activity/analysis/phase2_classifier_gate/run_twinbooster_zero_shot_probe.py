#!/usr/bin/env python
"""Probe TwinBooster assay-text zero-shot scores on the PXR challenge compounds.

TwinBooster returns an assay-conditioned active probability, not a pEC50 value.
This script therefore evaluates it as an independent ranking/gating prior:
Spearman to pEC50, high-tail ranking (pEC50 >= 6), low-tail ranking (pEC50 < 3),
and small id55-anchor gate scans on released AS1 labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TWINBOOSTER_ROOT = Path("/home/nagaet/twinbooster-pxr")
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "track1_activity/analysis/phase2_classifier_gate/outputs/twinbooster_zero_shot"
)
DEFAULT_ANCHOR = (
    REPO_ROOT / "track1_activity/submissions/ens_id51_top500_potent46_t40_soft_g35.csv"
)

PXR_ASSAY_DESCRIPTIONS = [
    (
        "pubchem_aid720659_exact",
        "qHTS assay for small molecule activators of the human pregnane X "
        "receptor (PXR) signaling pathway. The Pregnane X receptor (PXR) plays "
        "a critical role in the regulation of genes involved in drug metabolism "
        "and transporters. DPX-2 HepG2 cells are co-transfected with a PXR "
        "response element and a luciferase construct containing CYP3A4 promoter. "
        "Increase in luciferase activity identifies compounds that activate the "
        "PXR pathway.",
    ),
    (
        "pxr_reporter_activation",
        "Activation of human pregnane X receptor PXR NR1I2 in a cell-based "
        "reporter assay. Small molecule agonists induce PXR-mediated "
        "transcription and CYP3A4 induction; higher potency corresponds to "
        "lower EC50 and higher pEC50.",
    ),
    (
        "pxr_nuclear_receptor_agonist",
        "Human nuclear receptor pregnane X receptor PXR NR1I2 agonist activity. "
        "Compounds are active if they activate PXR signaling in ADMET screening.",
    ),
    (
        "cyp3a4_induction_pxr",
        "CYP3A4 induction mediated by human PXR activation. Identify compounds "
        "that induce xenobiotic response through pregnane X receptor agonism.",
    ),
    (
        "tox21_pxr_activator",
        "qHTS assay for activators of the pregnane X receptor signaling pathway, "
        "human PXR NR1I2 reporter assay.",
    ),
    (
        "pxr_transactivation_potency",
        "Concentration-response transactivation potency for human PXR. Active "
        "small molecules are potent PXR inducers with high pEC50 values.",
    ),
]


def fetch_compounds(host: str, port: int, dbname: str) -> pd.DataFrame:
    conn = psycopg2.connect(dbname=dbname, host=host, port=port)
    try:
        train = pd.read_sql(
            """
            SELECT 'train' AS split, t.compound_id, c.molecule_name, c.std_smiles, t.pec50
            FROM train_activity t
            JOIN compounds c ON c.id = t.compound_id
            WHERE c.std_smiles IS NOT NULL AND t.pec50 IS NOT NULL
            ORDER BY t.compound_id;
            """,
            conn,
        )
        test = pd.read_sql(
            """
            SELECT
                'test' AS split,
                t.compound_id,
                c.molecule_name,
                c.std_smiles,
                l.pec50
            FROM test_activity t
            JOIN compounds c ON c.id = t.compound_id
            LEFT JOIN test_activity_phase1_labels l ON l.compound_id = t.compound_id
            WHERE c.std_smiles IS NOT NULL
            ORDER BY t.compound_id;
            """,
            conn,
        )
    finally:
        conn.close()
    return pd.concat([train, test], ignore_index=True)


def safe_spearman(score: np.ndarray, y: np.ndarray) -> float:
    if len(score) < 2 or np.nanstd(score) == 0 or np.nanstd(y) == 0:
        return float("nan")
    return float(stats.spearmanr(score, y).statistic)


def maybe_auc_ap(target: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    if len(np.unique(target)) != 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(target, score)), float(
        average_precision_score(target, score)
    )


def score_metrics(df: pd.DataFrame, score_col: str) -> list[dict[str, object]]:
    rows = []
    for group, sub in [
        ("train", df[df["split"] == "train"]),
        ("as1", df[(df["split"] == "test") & df["pec50"].notna()]),
        ("train_plus_as1", df[df["pec50"].notna()]),
    ]:
        y = sub["pec50"].to_numpy(dtype=float)
        score = sub[score_col].to_numpy(dtype=float)
        high = (y >= 6.0).astype(int)
        low = (y < 3.0).astype(int)
        high_auc, high_ap = maybe_auc_ap(high, score)
        low_auc, low_ap = maybe_auc_ap(low, -score)
        slope, intercept = (
            np.polyfit(score, y, 1) if np.nanstd(score) > 0 else (np.nan, np.nan)
        )
        pred = slope * score + intercept
        rows.append(
            {
                "group": group,
                "n": len(sub),
                "spearman": safe_spearman(score, y),
                "fit_mae": float(np.mean(np.abs(pred - y))),
                "gte6_n": int(high.sum()),
                "gte6_auc": high_auc,
                "gte6_ap": high_ap,
                "lt3_n": int(low.sum()),
                "lt3_auc": low_auc,
                "lt3_ap": low_ap,
            }
        )
    return rows


def gate_scan(df: pd.DataFrame, score_col: str, anchor_path: Path) -> pd.DataFrame:
    if not anchor_path.exists():
        return pd.DataFrame()
    anchor = pd.read_csv(anchor_path)
    anchor = anchor.rename(
        columns={
            "pEC50": "anchor_pec50",
            "pec50": "anchor_pec50",
            "Molecule Name": "molecule_name",
        }
    )
    if "compound_id" not in anchor.columns and "molecule_name" in anchor.columns:
        join_cols = ["molecule_name"]
    elif "compound_id" in anchor.columns:
        join_cols = ["compound_id"]
    else:
        return pd.DataFrame()
    as1 = df[(df["split"] == "test") & df["pec50"].notna()].merge(
        anchor[[*join_cols, "anchor_pec50"]], on=join_cols, how="inner"
    )
    if as1.empty:
        return pd.DataFrame()

    y = as1["pec50"].to_numpy(dtype=float)
    base = as1["anchor_pec50"].to_numpy(dtype=float)
    score = as1[score_col].to_numpy(dtype=float)
    base_mae = float(np.mean(np.abs(base - y)))
    rows = []
    for mode, oriented, sign in [("high_lift", score, 1.0), ("low_drop", -score, -1.0)]:
        for q in [0.90, 0.93, 0.95, 0.97]:
            threshold = float(np.quantile(oriented, q))
            mask = oriented >= threshold
            for gamma in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
                pred = base.copy()
                pred[mask] += sign * gamma
                mae = float(np.mean(np.abs(pred - y)))
                rows.append(
                    {
                        "mode": mode,
                        "q": q,
                        "gamma": gamma,
                        "n_flag": int(mask.sum()),
                        "mae": mae,
                        "delta_mae": mae - base_mae,
                        "flag_true_high": int(((y >= 6.0) & mask).sum()),
                        "flag_true_low": int(((y < 3.0) & mask).sum()),
                        "flag_anchor_error_mean": float(np.mean(base[mask] - y[mask]))
                        if mask.any()
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["delta_mae", "n_flag"], ascending=[True, False]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--twinbooster-root", type=Path, default=DEFAULT_TWINBOOSTER_ROOT
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--anchor", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--db-host", default="/tmp")
    parser.add_argument("--db-port", type=int, default=5433)
    parser.add_argument("--db-name", default="pxr_challenge")
    args = parser.parse_args()

    sys.path.insert(0, str(args.twinbooster_root))
    from twinbooster.scripts.model import TwinBooster

    model_path = args.twinbooster_root / "twinbooster/scripts/barlow_twins/best_model"
    lgbm_path = (
        args.twinbooster_root / "twinbooster/scripts/lgbm/best_model/lgbm_model.joblib"
    )
    tb = TwinBooster(model_path=str(model_path), lgbm_path=str(lgbm_path))

    df = fetch_compounds(args.db_host, args.db_port, args.db_name)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    gate_rows = []
    smiles = df["std_smiles"].tolist()
    for prompt_name, description in PXR_ASSAY_DESCRIPTIONS:
        print(f"scoring {prompt_name} on {len(smiles)} compounds")
        pred, conf = tb.predict(smiles, description, get_confidence=True)
        score_col = f"tb_{prompt_name}"
        df[score_col] = np.asarray(pred, dtype=np.float64)
        df[f"{score_col}_confidence"] = np.asarray(conf, dtype=np.int8)
        for row in score_metrics(df, score_col):
            metric_rows.append(
                {"prompt": prompt_name, "description": description, **row}
            )
        gates = gate_scan(df, score_col, args.anchor)
        if not gates.empty:
            gates.insert(0, "prompt", prompt_name)
            gate_rows.append(gates)

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["group", "gte6_ap", "spearman"], ascending=[True, False, False]
    )
    gates = pd.concat(gate_rows, ignore_index=True) if gate_rows else pd.DataFrame()

    df.to_parquet(args.out_dir / "scores.parquet", index=False)
    metrics.to_csv(args.out_dir / "metrics.csv", index=False)
    if not gates.empty:
        gates.to_csv(args.out_dir / "id55_gate_scan.csv", index=False)
    report = {
        "n_rows": int(len(df)),
        "n_train": int((df["split"] == "train").sum()),
        "n_test": int((df["split"] == "test").sum()),
        "n_as1": int(((df["split"] == "test") & df["pec50"].notna()).sum()),
        "best_as1_gte6_ap": metrics[metrics["group"] == "as1"]
        .head(1)
        .to_dict("records"),
        "best_gate": gates.head(10).to_dict("records") if not gates.empty else [],
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    if not gates.empty:
        print("\nBest id55 gate rows:")
        print(gates.head(20).to_string(index=False))
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
