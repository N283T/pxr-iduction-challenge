#!/usr/bin/env -S pixi run python
# /// script
# requires-python = ">=3.12"
# ///
"""One-shot recovery script: extract pretrain.pt / scaler.json / meta.json from
a Lightning checkpoint that was written by run_chemprop_mtr_pretrain.py.

Use this when the pretrain script crashed *after* ModelCheckpoint wrote
best_val.ckpt but *before* it could save pretrain.pt / scaler.json / meta.json
(e.g. because PyTorch 2.6+ torch.load defaulted to weights_only=True and raised
an UnpicklingError on the Lightning AttributeDict).

Usage:
    pixi run python track1_activity/scripts/recover_chemprop_mtr_artifacts.py
    pixi run python track1_activity/scripts/recover_chemprop_mtr_artifacts.py --seed 42
    pixi run python track1_activity/scripts/recover_chemprop_mtr_artifacts.py --seed 42 --force
    pixi run python track1_activity/scripts/recover_chemprop_mtr_artifacts.py --seed 42 --keep-ckpt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "scripts")))

# Import helpers from the pretrain script (deterministic, no side effects)
from run_chemprop_mtr_pretrain import (  # noqa: E402
    DEFAULT_PARAMS,
    NAN_DROP_IDS,
    fit_scaler,
    load_descriptor_targets,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recover pretrain.pt / scaler.json / meta.json from best_val.ckpt"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used during the original pretrain run (default: 42)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite pretrain.pt / scaler.json / meta.json if they already exist",
    )
    p.add_argument(
        "--keep-ckpt",
        action="store_true",
        help="Keep best_val.ckpt after recovery (default: delete it to save space)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_dir = REPO_ROOT.joinpath(f"models/chemprop_mtr_seed{args.seed}")
    ckpt_path = ckpt_dir.joinpath("best_val.ckpt")
    pretrain_pt = ckpt_dir.joinpath("pretrain.pt")
    scaler_json = ckpt_dir.joinpath("scaler.json")
    meta_json = ckpt_dir.joinpath("meta.json")

    # --- pre-flight checks ------------------------------------------------
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"best_val.ckpt not found in {ckpt_dir}. Nothing to recover from."
        )

    if pretrain_pt.exists() and not args.force:
        raise FileExistsError(
            f"{pretrain_pt} already exists. "
            "Pass --force to overwrite, or delete it manually."
        )

    # --- step 1: load Lightning checkpoint --------------------------------
    print(f"Loading {ckpt_path} ...")
    # weights_only=False: Lightning checkpoints contain AttributeDict which is
    # not in PyTorch 2.6+ default safe-globals allowlist.  We trust our own
    # checkpoint here (written by run_chemprop_mtr_pretrain.py).
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    print(f"  state_dict keys (first 5): {list(state_dict.keys())[:5]}")

    # --- step 2: save pretrain.pt FIRST -----------------------------------
    torch.save(state_dict, pretrain_pt)
    print(f"  -> wrote {pretrain_pt}  ({pretrain_pt.stat().st_size // 1024} KB)")

    # --- step 3: re-derive scaler from raw descriptor targets -------------
    print("Loading descriptor targets from DB (deterministic) ...")
    _smiles, raw_targets, _ids, desc_names = load_descriptor_targets()
    scaler = fit_scaler(raw_targets)
    scaler_json.write_text(json.dumps(scaler, indent=2))
    print(f"  -> wrote {scaler_json}")

    # --- step 4: write meta.json ------------------------------------------
    # Attempt to recover best_val_loss from the ModelCheckpoint callback state
    # stored inside the Lightning checkpoint.
    best_val_loss: float | None = None
    cb_states = ckpt.get("callbacks", {})
    for _key, cb_val in cb_states.items():
        # Lightning stores a dict per callback class; ModelCheckpoint entry
        # contains "best_model_score" as a tensor.
        if isinstance(cb_val, dict) and "best_model_score" in cb_val:
            score = cb_val["best_model_score"]
            if score is not None:
                best_val_loss = float(score.item() if hasattr(score, "item") else score)
            break

    meta = {
        "seed": args.seed,
        "n_compounds": len(_smiles),
        "descriptor_names": desc_names,
        "params": DEFAULT_PARAMS,
        "best_val_loss": best_val_loss,
        "final_val_loss": None,  # not recoverable post-hoc
        "recovered_from_ckpt": True,
        "nan_drop_ids": sorted(NAN_DROP_IDS),
    }
    meta_json.write_text(json.dumps(meta, indent=2))
    print(f"  -> wrote {meta_json}")
    if best_val_loss is not None:
        print(f"  best_val_loss recovered: {best_val_loss:.6f}")
    else:
        print("  best_val_loss: not found in ckpt callbacks (set to null)")

    # --- step 5: optionally remove best_val.ckpt --------------------------
    if args.keep_ckpt:
        print(f"  keeping {ckpt_path} (--keep-ckpt passed)")
    else:
        ckpt_path.unlink()
        print(f"  deleted {ckpt_path} (pass --keep-ckpt to retain)")

    # --- summary ----------------------------------------------------------
    print("\nRecovery complete:")
    for p in [pretrain_pt, scaler_json, meta_json]:
        exists = "OK" if p.exists() else "MISSING"
        print(f"  [{exists}] {p}")


if __name__ == "__main__":
    main()
