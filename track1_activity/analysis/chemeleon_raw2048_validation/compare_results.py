#!/usr/bin/env -S pixi run python
"""Compare legacy 300d and corrected raw-2048 CheMeleon TabPFN runs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sqlalchemy import create_engine, text


REPO_ROOT = Path(__file__).resolve().parents[3]
SUBMISSION_DIR = REPO_ROOT.joinpath("track1_activity", "submissions")
OUTPUT_DIR = REPO_ROOT.joinpath("data", "chemeleon_raw2048_validation")

MODELS = {
    "legacy300_single": {
        "experiment_id": 381,
        "submission": "tabpfn_chemeleon_umap.csv",
        "comparison_note": "Historical run used 10-trial tuned TabPFN parameters.",
    },
    "raw2048_single": {
        "experiment_id": 2491,
        "submission": "audit_chemeleon_raw2048_tabpfn_v2_6_umap.csv",
        "comparison_note": "Audit run used fixed TabPFN v2.6 defaults.",
    },
    "legacy300_mixed": {
        "experiment_id": 1608,
        "submission": (
            "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_umap_default.csv"
        ),
        "comparison_note": "Matched fixed TabPFN v2.6 defaults.",
    },
    "raw2048_mixed": {
        "experiment_id": 2492,
        "submission": ("audit_chemeleon_raw2048_mixed_full_tabpfn_v2_6_umap.csv"),
        "comparison_note": "Matched fixed TabPFN v2.6 defaults.",
    },
    "legacy300_mixed_top500": {
        "experiment_id": 1609,
        "submission": (
            "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap.csv"
        ),
        "comparison_note": (
            "Matched fold-local LightGBM top-500 and fixed TabPFN v2.6 defaults."
        ),
    },
    "raw2048_mixed_top500": {
        "experiment_id": 2493,
        "submission": ("audit_chemeleon_raw2048_mixed_top500_tabpfn_v2_6_umap.csv"),
        "comparison_note": (
            "Matched fold-local LightGBM top-500 and fixed TabPFN v2.6 defaults."
        ),
    },
    "raw2048_mixed_top300": {
        "experiment_id": 2494,
        "submission": ("audit_chemeleon_raw2048_mixed_top300_tabpfn_v2_6_umap.csv"),
        "comparison_note": "Raw-2048 fold-local top-K sweep.",
    },
    "raw2048_mixed_top400": {
        "experiment_id": 2495,
        "submission": ("audit_chemeleon_raw2048_mixed_top400_tabpfn_v2_6_umap.csv"),
        "comparison_note": "Raw-2048 fold-local top-K sweep.",
    },
    "raw2048_mixed_top600": {
        "experiment_id": 2496,
        "submission": ("audit_chemeleon_raw2048_mixed_top600_tabpfn_v2_6_umap.csv"),
        "comparison_note": "Raw-2048 fold-local top-K sweep.",
    },
    "raw2048_mixed_top700": {
        "experiment_id": 2497,
        "submission": ("audit_chemeleon_raw2048_mixed_top700_tabpfn_v2_6_umap.csv"),
        "comparison_note": "Raw-2048 fold-local top-K sweep.",
    },
    "raw2048_mixed_top800": {
        "experiment_id": 2498,
        "submission": ("audit_chemeleon_raw2048_mixed_top800_tabpfn_v2_6_umap.csv"),
        "comparison_note": "Raw-2048 fold-local top-K sweep.",
    },
    "raw2048_mixed_top1000": {
        "experiment_id": 2499,
        "submission": ("audit_chemeleon_raw2048_mixed_top1000_tabpfn_v2_6_umap.csv"),
        "comparison_note": "Raw-2048 fold-local top-K sweep.",
    },
}


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(np.mean(np.abs(y - prediction))),
        "bias": float(np.mean(prediction - y)),
        "spearman": float(spearmanr(y, prediction).statistic),
    }


def main() -> None:
    engine = create_engine("postgresql+psycopg2:///pxr_challenge?host=/tmp&port=5433")
    train = pd.read_sql_query(
        "SELECT id, pec50 AS y FROM train_activity ORDER BY id", engine
    )
    released_test = pd.read_sql_query(
        """
        SELECT t.id AS test_id, c.molecule_name, l.pec50 AS y
        FROM test_activity_phase1_labels l
        JOIN compounds c ON c.id = l.compound_id
        JOIN test_activity t ON t.compound_id = l.compound_id
        ORDER BY t.id
        """,
        engine,
    )

    rows: list[dict] = []
    oof_predictions: dict[str, np.ndarray] = {}
    for name, config in MODELS.items():
        experiment_id = int(config["experiment_id"])
        oof = pd.read_sql_query(
            text(
                """
                SELECT train_idx, oof_prediction
                FROM experiment_oof_predictions
                WHERE experiment_id = :experiment_id
                ORDER BY train_idx
                """
            ),
            engine,
            params={"experiment_id": experiment_id},
        )
        if len(oof) != len(train):
            raise ValueError(
                f"Experiment {experiment_id} has {len(oof)} OOF rows; "
                f"expected {len(train)}"
            )
        oof_values = oof["oof_prediction"].to_numpy(dtype=float)
        oof_predictions[name] = oof_values

        submission = pd.read_csv(SUBMISSION_DIR.joinpath(str(config["submission"])))
        submission = submission.rename(
            columns={"Molecule Name": "molecule_name", "pEC50": "prediction"}
        )[["molecule_name", "prediction"]]
        joined = released_test.merge(
            submission, on="molecule_name", how="inner", validate="one_to_one"
        )
        if len(joined) != len(released_test):
            raise ValueError(
                f"{name} matched {len(joined)}/{len(released_test)} released labels"
            )

        oof_metrics = metrics(train["y"].to_numpy(dtype=float), oof_values)
        test_metrics = metrics(
            joined["y"].to_numpy(dtype=float),
            joined["prediction"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "name": name,
                "experiment_id": experiment_id,
                "oof_mae": oof_metrics["mae"],
                "oof_spearman": oof_metrics["spearman"],
                "released_test_n": len(joined),
                "released_test_mae": test_metrics["mae"],
                "released_test_bias": test_metrics["bias"],
                "released_test_spearman": test_metrics["spearman"],
                "comparison_note": config["comparison_note"],
            }
        )

    comparisons = []
    for legacy, raw in (
        ("legacy300_single", "raw2048_single"),
        ("legacy300_mixed", "raw2048_mixed"),
        ("legacy300_mixed_top500", "raw2048_mixed_top500"),
    ):
        legacy_row = next(row for row in rows if row["name"] == legacy)
        raw_row = next(row for row in rows if row["name"] == raw)
        comparisons.append(
            {
                "legacy": legacy,
                "raw": raw,
                "delta_oof_mae_raw_minus_legacy": raw_row["oof_mae"]
                - legacy_row["oof_mae"],
                "delta_released_test_mae_raw_minus_legacy": raw_row["released_test_mae"]
                - legacy_row["released_test_mae"],
                "oof_prediction_pearson": float(
                    np.corrcoef(oof_predictions[legacy], oof_predictions[raw])[0, 1]
                ),
                "oof_mean_absolute_shift": float(
                    np.mean(np.abs(oof_predictions[legacy] - oof_predictions[raw]))
                ),
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    table = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR.joinpath(f"comparison_{stamp}.csv")
    json_path = OUTPUT_DIR.joinpath(f"comparison_{stamp}.json")
    table.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "released_test_label_count": len(released_test),
                "models": rows,
                "comparisons": comparisons,
            },
            indent=2,
        )
        + "\n"
    )

    print(table.to_string(index=False))
    print("\nComparisons (raw minus legacy):")
    print(pd.DataFrame(comparisons).to_string(index=False))
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
