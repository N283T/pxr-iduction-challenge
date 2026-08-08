#!/usr/bin/env -S pixi run python
"""Evaluate replacing the legacy-300d top-500 ensemble slot with raw-2048."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr
from sqlalchemy import create_engine


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT.joinpath("track1_activity", "src")
SCRIPT_DIR = REPO_ROOT.joinpath("track1_activity", "scripts")
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from data import DB_PARAMS, load_test_smiles, load_train_smiles_target  # noqa: E402
from run_ensemble import ENSEMBLE_MODELS, optimize_caruana  # noqa: E402


OLD_TOP500 = "tabpfn_cheme_2d_full_boltz_log2fc_pred_seed10ens_top500_umap"
NEW_MEMBERS = {
    "top500": "audit_chemeleon_raw2048_mixed_top500_tabpfn_v2_6_umap",
    "full": "audit_chemeleon_raw2048_mixed_full_tabpfn_v2_6_umap",
}
OUTPUT_DIR = REPO_ROOT.joinpath("data", "chemeleon_raw2048_validation")
SEEDS = (42, 43, 44, 45, 46)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--new-member",
        choices=tuple(NEW_MEMBERS),
        default="top500",
        help="Raw-2048 mixed-feature member used for the replacement",
    )
    return parser.parse_args()


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(np.mean(np.abs(y - prediction))),
        "bias": float(np.mean(prediction - y)),
        "spearman": float(spearmanr(y, prediction).statistic),
    }


def load_predictions(
    names: list[str], train_n: int, test_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, name, submission_path
            FROM experiments
            WHERE name = ANY(%s)
            """,
            (names,),
        )
        rows = cur.fetchall()
        by_name = {str(row[1]): row for row in rows}
        missing = [name for name in names if name not in by_name]
        if missing:
            raise ValueError(f"Missing experiments: {missing}")

        oofs: list[np.ndarray] = []
        tests: list[np.ndarray] = []
        expected_test_names = test_df["molecule_name"].astype(str).tolist()
        for name in names:
            experiment_id, _, submission_path = by_name[name]
            cur.execute(
                """
                SELECT train_idx, oof_prediction
                FROM experiment_oof_predictions
                WHERE experiment_id = %s
                ORDER BY train_idx
                """,
                (experiment_id,),
            )
            oof = np.asarray([row[1] for row in cur.fetchall()], dtype=np.float64)
            if oof.shape != (train_n,):
                raise ValueError(f"{name} OOF shape {oof.shape}; expected {(train_n,)}")

            path = REPO_ROOT.joinpath(str(submission_path))
            submission = pd.read_csv(path).rename(
                columns={"Molecule Name": "molecule_name", "pEC50": "prediction"}
            )
            prediction_by_name = submission.set_index("molecule_name")["prediction"]
            missing_test = [
                molecule_name
                for molecule_name in expected_test_names
                if molecule_name not in prediction_by_name.index
            ]
            if missing_test:
                raise ValueError(f"{name} missing test names: {missing_test[:10]}")
            test_prediction = prediction_by_name.loc[expected_test_names].to_numpy(
                dtype=np.float64
            )
            oofs.append(oof)
            tests.append(test_prediction)

    return np.column_stack(oofs), np.column_stack(tests)


def optimize(matrix: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    return optimize_caruana(
        matrix,
        y,
        n_iter=100,
        init_top_n=3,
        bag_frac=0.5,
        n_bags=20,
        seed=seed,
    )


def evaluate(
    label: str,
    names: list[str],
    oof_matrix: np.ndarray,
    test_matrix: np.ndarray,
    weights: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    oof_prediction = oof_matrix @ weights
    test_prediction = test_matrix @ weights
    return {
        "label": label,
        "oof": metrics(y_train, oof_prediction),
        "released_test": metrics(y_test, test_prediction),
        "weights": {
            name: float(weight) for name, weight in zip(names, weights, strict=True)
        },
        "oof_prediction": oof_prediction,
        "test_prediction": test_prediction,
    }


def serializable(result: dict) -> dict:
    return {
        key: value
        for key, value in result.items()
        if key not in {"oof_prediction", "test_prediction"}
    }


def main() -> None:
    args = parse_args()
    new_member = NEW_MEMBERS[args.new_member]
    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    y_train = train_df["pec50"].to_numpy(dtype=np.float64)

    engine = create_engine("postgresql+psycopg2:///pxr_challenge?host=/tmp&port=5433")
    released = pd.read_sql_query(
        """
        SELECT t.id AS test_id, c.molecule_name, l.pec50
        FROM test_activity_phase1_labels l
        JOIN test_activity t ON t.compound_id = l.compound_id
        JOIN compounds c ON c.id = l.compound_id
        ORDER BY t.id
        """,
        engine,
    )
    if len(released) != len(test_df):
        raise ValueError(
            f"Released labels cover {len(released)}/{len(test_df)} test compounds; "
            "this audit expects complete released labels"
        )
    if (
        released["molecule_name"].astype(str).tolist()
        != test_df["molecule_name"].astype(str).tolist()
    ):
        raise ValueError("Released-label order does not match test_activity order")
    y_test = released["pec50"].to_numpy(dtype=np.float64)

    baseline_names = list(ENSEMBLE_MODELS)
    if OLD_TOP500 not in baseline_names:
        raise ValueError(f"{OLD_TOP500} is not in ENSEMBLE_MODELS")
    swap_names = baseline_names.copy()
    swap_index = swap_names.index(OLD_TOP500)
    swap_names[swap_index] = new_member

    all_names = list(dict.fromkeys(baseline_names + [new_member]))
    all_oof, all_test = load_predictions(all_names, len(y_train), test_df)
    column_by_name = {name: idx for idx, name in enumerate(all_names)}
    baseline_columns = [column_by_name[name] for name in baseline_names]
    swap_columns = [column_by_name[name] for name in swap_names]
    baseline_oof = all_oof[:, baseline_columns]
    baseline_test = all_test[:, baseline_columns]
    swap_oof = all_oof[:, swap_columns]
    swap_test = all_test[:, swap_columns]

    per_seed = []
    baseline_weights = []
    swap_weights = []
    for seed in SEEDS:
        base_weight = optimize(baseline_oof, y_train, seed)
        swap_weight = optimize(swap_oof, y_train, seed)
        baseline_weights.append(base_weight)
        swap_weights.append(swap_weight)
        base_result = evaluate(
            f"baseline_seed{seed}",
            baseline_names,
            baseline_oof,
            baseline_test,
            base_weight,
            y_train,
            y_test,
        )
        swap_result = evaluate(
            f"swap_seed{seed}",
            swap_names,
            swap_oof,
            swap_test,
            swap_weight,
            y_train,
            y_test,
        )
        per_seed.append(
            {
                "seed": seed,
                "baseline": serializable(base_result),
                "swap": serializable(swap_result),
                "delta_oof_mae": swap_result["oof"]["mae"] - base_result["oof"]["mae"],
                "delta_released_test_mae": swap_result["released_test"]["mae"]
                - base_result["released_test"]["mae"],
            }
        )

    baseline_seed42_weight = baseline_weights[0]
    baseline_seed42 = evaluate(
        "baseline_seed42",
        baseline_names,
        baseline_oof,
        baseline_test,
        baseline_seed42_weight,
        y_train,
        y_test,
    )
    swap_seed42 = evaluate(
        "optimized_swap_seed42",
        swap_names,
        swap_oof,
        swap_test,
        swap_weights[0],
        y_train,
        y_test,
    )
    fixed_swap = evaluate(
        "fixed_weight_swap_seed42",
        swap_names,
        swap_oof,
        swap_test,
        baseline_seed42_weight,
        y_train,
        y_test,
    )

    baseline_bag5_weight = np.mean(np.stack(baseline_weights), axis=0)
    baseline_bag5_weight /= baseline_bag5_weight.sum()
    swap_bag5_weight = np.mean(np.stack(swap_weights), axis=0)
    swap_bag5_weight /= swap_bag5_weight.sum()
    baseline_bag5 = evaluate(
        "baseline_bag5",
        baseline_names,
        baseline_oof,
        baseline_test,
        baseline_bag5_weight,
        y_train,
        y_test,
    )
    swap_bag5 = evaluate(
        "swap_bag5",
        swap_names,
        swap_oof,
        swap_test,
        swap_bag5_weight,
        y_train,
        y_test,
    )

    summary_rows = []
    for result in (
        baseline_seed42,
        fixed_swap,
        swap_seed42,
        baseline_bag5,
        swap_bag5,
    ):
        summary_rows.append(
            {
                "label": result["label"],
                "oof_mae": result["oof"]["mae"],
                "oof_spearman": result["oof"]["spearman"],
                "released_test_mae": result["released_test"]["mae"],
                "released_test_spearman": result["released_test"]["spearman"],
                "swapped_slot_weight": result["weights"].get(
                    new_member,
                    result["weights"].get(OLD_TOP500, 0.0),
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))

    seed_deltas = pd.DataFrame(
        [
            {
                "seed": row["seed"],
                "delta_oof_mae": row["delta_oof_mae"],
                "delta_released_test_mae": row["delta_released_test_mae"],
            }
            for row in per_seed
        ]
    )
    print("\nPer-seed optimized SWAP deltas (swap minus baseline):")
    print(seed_deltas.to_string(index=False))
    print(
        "mean delta: "
        f"OOF={seed_deltas['delta_oof_mae'].mean():+.6f}, "
        f"released_test={seed_deltas['delta_released_test_mae'].mean():+.6f}"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"ensemble_swap_{args.new_member}"
    csv_path = OUTPUT_DIR.joinpath(f"{prefix}_{stamp}.csv")
    json_path = OUTPUT_DIR.joinpath(f"{prefix}_{stamp}.json")
    prediction_path = OUTPUT_DIR.joinpath(f"{prefix}_optimized_seed42_{stamp}.csv")
    fixed_prediction_path = OUTPUT_DIR.joinpath(
        f"{prefix}_fixed_weight_seed42_{stamp}.csv"
    )
    summary.to_csv(csv_path, index=False)
    pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": swap_seed42["test_prediction"],
        }
    ).to_csv(prediction_path, index=False)
    pd.DataFrame(
        {
            "SMILES": test_df["smiles"],
            "Molecule Name": test_df["molecule_name"],
            "pEC50": fixed_swap["test_prediction"],
        }
    ).to_csv(fixed_prediction_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "old_top500": OLD_TOP500,
                "new_member_kind": args.new_member,
                "new_member": new_member,
                "pool_size": len(baseline_names),
                "baseline_pool": baseline_names,
                "swap_pool": swap_names,
                "canonical_seed42": {
                    "baseline": serializable(baseline_seed42),
                    "fixed_weight_swap": serializable(fixed_swap),
                    "optimized_swap": serializable(swap_seed42),
                },
                "bag5": {
                    "baseline": serializable(baseline_bag5),
                    "swap": serializable(swap_bag5),
                },
                "per_seed_optimized": per_seed,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {prediction_path}")
    print(f"Wrote {fixed_prediction_path}")


if __name__ == "__main__":
    main()
