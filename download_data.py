"""Download PXR challenge datasets to data/ directory.

By default downloads every config. Pass --configs to restrict to a subset,
e.g. ``python download_data.py --configs structure`` to refresh Track 2 only.
"""

import argparse
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).parent.joinpath("data")
DATA_DIR.mkdir(exist_ok=True)

CONFIGS = {
    "default": {"splits": ["train", "test"]},
    "counter_assay": {"splits": ["train"]},
    "single_concentration": {"splits": ["train"]},
    "structure": {"splits": ["test"]},
    "phase_1_unblinded": {"splits": ["test"]},
    "crudes_htchem": {"splits": ["train"]},
    "semi_pure_htchem": {"splits": ["train"]},
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
        info = CONFIGS[config_name]
        print(f"Downloading config: {config_name}")
        ds = load_dataset("openadmet/pxr-challenge-train-test", config_name)
        for split in info["splits"]:
            df = ds[split].to_pandas()
            filename = f"{config_name}_{split}.parquet"
            output_path = DATA_DIR.joinpath(filename)
            df.to_parquet(output_path, index=False)
            print(f"  Saved {filename} ({len(df)} rows)")

    print("\nDone! Datasets saved to data/")


if __name__ == "__main__":
    main()
