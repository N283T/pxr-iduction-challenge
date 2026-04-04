"""Download all PXR challenge datasets to data/ directory."""

from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATA_DIR = Path(__file__).parent.joinpath("data")
DATA_DIR.mkdir(exist_ok=True)

CONFIGS = {
    "default": {"splits": ["train", "test"]},
    "counter_assay": {"splits": ["train"]},
    "single_concentration": {"splits": ["train"]},
    "structure": {"splits": ["test"]},
}

for config_name, info in CONFIGS.items():
    print(f"Downloading config: {config_name}")
    ds = load_dataset("openadmet/pxr-challenge-train-test", config_name)
    for split in info["splits"]:
        df = ds[split].to_pandas()
        filename = f"{config_name}_{split}.parquet"
        output_path = DATA_DIR.joinpath(filename)
        df.to_parquet(output_path, index=False)
        print(f"  Saved {filename} ({len(df)} rows)")

print("\nDone! All datasets saved to data/")
