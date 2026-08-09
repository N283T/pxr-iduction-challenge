#!/usr/bin/env -S pixi run python
"""Fast OOF-only probe of random linear 2048-to-300 CheMeleon projections."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = REPO_ROOT.joinpath("track1_activity", "scripts")
SRC_DIR = REPO_ROOT.joinpath("track1_activity", "src")
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SRC_DIR))

from data import load_test_smiles, load_train_smiles_target  # noqa: E402
from evaluate import compute_metrics  # noqa: E402
from splits import umap_split_indices  # noqa: E402

import run_train  # noqa: E402
from run_cv import (  # noqa: E402
    impute_from_train,
    load_embedding_table,
    load_ids,
)


OUTPUT_DIR = REPO_ROOT.joinpath("data", "chemeleon_raw2048_validation")
RAW_TABLE = "compound_chemeleon_raw2048"
LEGACY_TABLE = "compound_chemeleon"
RAW_DIM = 2048
COMPRESSED_DIM = 300
PROJECTION_SEEDS = (0, 1, 2)
N_ESTIMATORS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-legacy",
        action="store_true",
        help="Run only new random projections when the legacy control is complete.",
    )
    return parser.parse_args()


def random_linear_projection(values: np.ndarray, seed: int) -> np.ndarray:
    """Match the untrained Linear(2048, 300) used by the legacy extractor."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        layer = torch.nn.Linear(RAW_DIM, COMPRESSED_DIM)
    layer.eval()
    projected = []
    with torch.inference_mode():
        for start in range(0, len(values), 512):
            projected.append(layer(torch.from_numpy(values[start : start + 512])))
    return torch.cat(projected).numpy().astype(np.float32, copy=False)


def save_checkpoint(path: Path, results: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {"results": results},
            indent=2,
            default=lambda value: (
                value.item() if isinstance(value, np.generic) else str(value)
            ),
        )
        + "\n"
    )


def run_variant(
    label: str,
    chemeleon: np.ndarray,
    base: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    model_path: Path,
) -> dict:
    from tabpfn import TabPFNRegressor

    X = np.concatenate([chemeleon, base], axis=1)
    X, _ = impute_from_train(X, X[:1])
    oof = np.zeros(len(y), dtype=np.float32)
    fold_metrics = []
    print(f"\n{label}: input_dim={X.shape[1]}", flush=True)
    for fold, (fit_idx, val_idx) in enumerate(splits):
        model = TabPFNRegressor(
            device="cuda",
            n_estimators=N_ESTIMATORS,
            softmax_temperature=0.9,
            random_state=42,
            ignore_pretraining_limits=True,
            model_path=model_path,
        )
        model.fit(X[fit_idx], y[fit_idx])
        oof[val_idx] = model.predict(X[val_idx])
        fold_result = compute_metrics(y[val_idx], oof[val_idx])
        fold_metrics.append(fold_result)
        print(
            f"  fold={fold} MAE={fold_result['MAE']:.6f} "
            f"Sp={fold_result['Spearman_R']:.6f}",
            flush=True,
        )
        del model
        torch.cuda.empty_cache()

    overall = compute_metrics(y, oof)
    print(
        f"  overall MAE={overall['MAE']:.6f} Sp={overall['Spearman_R']:.6f}",
        flush=True,
    )
    return {
        "label": label,
        "input_dim": int(X.shape[1]),
        "n_estimators": N_ESTIMATORS,
        "fold_metrics": fold_metrics,
        "overall": overall,
    }


def main() -> None:
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR.joinpath(f"random_linear_fast_{stamp}.json")

    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    train_ids, _ = load_ids()
    raw = load_embedding_table(RAW_TABLE, train_ids, RAW_DIM)
    legacy = load_embedding_table(LEGACY_TABLE, train_ids, COMPRESSED_DIM)
    base, _ = run_train.load_features(
        "2d_full_boltz_log2fc_pred_seed10ens", train_df, test_df
    )
    y = train_df["pec50"].to_numpy(dtype=np.float32)
    splits = umap_split_indices(
        train_df["smiles"].tolist(), n_splits=5, n_clusters=50, seed=42
    )
    model_path = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2_6
    ).model_path

    variants = {} if args.skip_legacy else {"legacy300": legacy}
    variants.update(
        {
            f"random_linear300_seed{seed}": random_linear_projection(raw, seed)
            for seed in PROJECTION_SEEDS
        }
    )

    results = []
    for label, values in variants.items():
        results.append(run_variant(label, values, base, y, splits, model_path))
        save_checkpoint(output_path, results)
        print(f"  checkpoint={output_path}", flush=True)

    print("\nSummary", flush=True)
    for result in results:
        print(
            f"{result['label']}: MAE={result['overall']['MAE']:.6f} "
            f"Sp={result['overall']['Spearman_R']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
