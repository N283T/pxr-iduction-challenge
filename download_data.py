"""Download PXR challenge datasets to data/ directory.

By default downloads every config. Pass --configs to restrict to a subset,
e.g. ``python download_data.py --configs structure`` to refresh Track 2 only.
"""

import argparse
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.joinpath("data")
DATA_DIR.mkdir(exist_ok=True)
HF_BASE_URL = (
    "https://huggingface.co/datasets/openadmet/pxr-challenge-train-test/resolve/main"
)

CONFIGS = {
    "default": {
        "train": "pxr-challenge_TRAIN.csv",
        "test": "pxr-challenge_TEST_BLINDED.csv",
    },
    "counter_assay": {"train": "pxr-challenge_counter-assay_TRAIN.csv"},
    "single_concentration": {"train": "pxr-challenge_single_concentration_TRAIN.csv"},
    "structure": {"test": "pxr-challenge_structure_TEST_BLINDED.csv"},
    "phase_1_unblinded": {"test": "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv"},
    "phase_2_unblinded": {"test": "pxr-challenge_TEST_PHASE_2_UNBLINDED.csv"},
    "crudes_htchem": {"train": "pxr-challenge_htchem-libraries_TRAIN.csv"},
    "semi_pure_htchem": {
        "train": "pxr-challenge_96-compound-uscale-semi-pure_TRAIN.csv"
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(CONFIGS.keys()),
        default=list(CONFIGS.keys()),
        help="Subset of configs to download (default: all).",
    )
    args = parser.parse_args()

    for config_name in args.configs:
        print(f"Downloading config: {config_name}")
        for split, source_name in CONFIGS[config_name].items():
            df = pd.read_csv(f"{HF_BASE_URL}/{source_name}")
            filename = f"{config_name}_{split}.parquet"
            output_path = DATA_DIR.joinpath(filename)
            df.to_parquet(output_path, index=False)
            print(f"  Saved {filename} ({len(df)} rows)")

    print("\nDone! Datasets saved to data/")


if __name__ == "__main__":
    main()
