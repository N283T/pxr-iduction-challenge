"""Run inference on a single fold's val pkl with the best FT ckpt.

Inputs:
  - fold k (0-4)
  - save_dir containing version_0/checkpoint/epoch*_step*.ckpt
    (AffinityTrainer saves here when patience=-1 / save_topk set)

Reads the training log to pick the ckpt with lowest Validation MSE,
then invokes the fork's inference.py via subprocess to produce a CSV
aligned with val_cids from 03_build_fold_pkls.py.

Output: structures/gator/ft_runs/<run_name>/fold{k}_val_preds.csv
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
FORK_ROOT = Path("/home/nagaet/ghq/github.com/N283T/GatorAffinity")


def find_best_ckpt(save_dir: Path) -> Path:
    """AffinityTrainer writes `version_N/checkpoint/*.ckpt` and logs
    a line `Validation: {val_loss}, save path: .../epoch{e}_step{s}.ckpt`
    per improvement. Pick the ckpt with the lowest validation loss
    across training.
    """
    version_dirs = sorted(save_dir.glob("version_*"))
    if not version_dirs:
        raise FileNotFoundError(f"no version_* in {save_dir}")
    version = version_dirs[-1]
    ckpt_dir = version.joinpath("checkpoint")
    ckpts = sorted(ckpt_dir.glob("epoch*_step*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints in {ckpt_dir}")

    # Look for a global log next to the trainer log to pick best by val loss.
    # Fallback: pick the latest (usually best-so-far per save_topk behaviour).
    log_candidates = list(save_dir.parent.glob(f"{save_dir.name}.log"))
    if log_candidates:
        text = log_candidates[0].read_text()
        pattern = re.compile(
            r"Validation:\s*([0-9.]+).*save path:\s*(\S+\.ckpt)"
        )
        hits = pattern.findall(text)
        if hits:
            best_val, best_ckpt = min(hits, key=lambda kv: float(kv[0]))
            return Path(best_ckpt)

    # Fallback: last checkpoint on disk
    return ckpts[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument(
        "--save-dir",
        type=Path,
        required=True,
        help="Training save_dir (contains version_*/checkpoint/)",
    )
    ap.add_argument(
        "--val-pkl",
        type=Path,
        default=None,
        help="Override path to fold val pkl (default: structures/gator/folds/fold{fold}_val.pkl)",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output CSV (default: <save-dir>/fold{fold}_val_preds.csv)",
    )
    args = ap.parse_args()

    val_pkl = (
        args.val_pkl
        if args.val_pkl is not None
        else REPO_ROOT.joinpath(
            "structures", "gator", "folds", f"fold{args.fold}_val.pkl"
        )
    )
    out_csv = (
        args.out_csv
        if args.out_csv is not None
        else args.save_dir.joinpath(f"fold{args.fold}_val_preds.csv")
    )

    ckpt = find_best_ckpt(args.save_dir)
    print(f"Using ckpt: {ckpt}")
    print(f"Val pkl:    {val_pkl}")
    print(f"Output CSV: {out_csv}")

    cmd = [
        "pixi",
        "run",
        "python",
        "inference.py",
        "--model_ckpt",
        str(ckpt),
        "--test_set_path",
        str(val_pkl),
        "--output_csv",
        str(out_csv),
    ]
    ret = subprocess.run(cmd, cwd=FORK_ROOT)
    if ret.returncode != 0:
        sys.exit(ret.returncode)

    # Quick sanity print: compare count with val_cids.json
    cids_path = val_pkl.with_name(f"fold{args.fold}_val_cids.json")
    if cids_path.exists():
        cids = json.loads(cids_path.read_text())
        print(f"\nExpected val compounds: {len(cids)}")


if __name__ == "__main__":
    main()
