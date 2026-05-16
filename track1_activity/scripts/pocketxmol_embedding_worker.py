"""Worker for extracting PocketXMol hidden-state embeddings.

This script is executed with the PocketXMol Python environment while the current
working directory is the PocketXMol checkout. It intentionally avoids molecule
reconstruction and the 100-step sampling loop; it only builds PocketXMol inputs,
applies noise at selected timesteps, runs one model forward per timestep, and
saves the captured variable-length hidden states. Fixed-length pooling is a
separate downstream step.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config-model", default="configs/sample/pxm.yml")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timesteps", nargs="+", type=float, default=[1.0, 0.5, 0.05])
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def load_manifest(path: Path, limit: int | None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def import_pocketxmol_modules():
    sys.path.append(".")
    from models.maskfill import PMAsymDenoiser
    from scripts.sample_use import get_input_data
    from scripts.train_pl import DataModule
    from utils.dataset import UseDataset
    from utils.misc import make_config, seed_all
    from utils.sample_noise import get_sample_noiser
    from utils.transforms import Compose, get_transforms

    return {
        "PMAsymDenoiser": PMAsymDenoiser,
        "get_input_data": get_input_data,
        "DataModule": DataModule,
        "UseDataset": UseDataset,
        "make_config": make_config,
        "seed_all": seed_all,
        "get_sample_noiser": get_sample_noiser,
        "Compose": Compose,
        "get_transforms": get_transforms,
    }


def load_train_config(config, make_config):
    cfg_dir = os.path.dirname(config.model.checkpoint).replace(
        "checkpoints", "train_config"
    )
    train_config_files = os.listdir(cfg_dir)
    return make_config(os.path.join(cfg_dir, "".join(train_config_files)))


def make_transforms(config, train_config, mods):
    for samp_trans in config.get("transforms", {}).keys():
        if samp_trans in train_config.transforms.keys():
            train_config.transforms.get(samp_trans).update(
                config.transforms.get(samp_trans)
            )

    dm = mods["DataModule"](train_config)
    featurizer_list = dm.get_featurizers()
    in_dims = dm.get_in_dims()
    task_trans = mods["get_transforms"](config.task.transform, mode="use")

    if "variable_mol_size" in getattr(config, "transforms", []):
        transforms = featurizer_list + [
            mods["get_transforms"](config.transforms.variable_mol_size),
            task_trans,
        ]
    elif "variable_sc_size" in getattr(config, "transforms", []):
        transforms = featurizer_list + [
            mods["get_transforms"](config.transforms.variable_sc_size),
            task_trans,
        ]
    else:
        transforms = featurizer_list + [task_trans]

    addition_transforms = [
        mods["get_transforms"](tr) for tr in config.data.get("transforms", [])
    ]
    transforms = mods["Compose"](transforms + addition_transforms)
    follow_batch = sum(
        [getattr(t, "follow_batch", []) for t in transforms.transforms], []
    )
    exclude_keys = sum(
        [getattr(t, "exclude_keys", []) for t in transforms.transforms], []
    )
    return transforms, follow_batch, exclude_keys, in_dims


def load_model(first_config, config_model: str, device: str, mods):
    config = mods["make_config"](first_config, config_model)
    mods["seed_all"](int(config.sample.seed))
    ckpt = torch.load(config.model.checkpoint, map_location=device, weights_only=False)
    train_config = load_train_config(config, mods["make_config"])
    _, _, _, in_dims = make_transforms(config, deepcopy(train_config), mods)
    model = mods["PMAsymDenoiser"](config=train_config.model, **in_dims).to(device)
    model.load_state_dict(
        {
            k[6:]: value
            for k, value in ckpt["state_dict"].items()
            if k.startswith("model.")
        }
    )
    model.eval()
    return model, train_config, in_dims


def make_batch(row, config_model: str, train_config_base, in_dims, device: str, mods):
    config = mods["make_config"](row["config_path"], config_model)
    train_config = deepcopy(train_config_base)
    transforms, follow_batch, exclude_keys, _ = make_transforms(
        config, train_config, mods
    )
    data_cfg = config.data
    is_pep = data_cfg.get("is_pep", None)
    if is_pep is None:
        is_pep = data_cfg.input_ligand.endswith(
            ".pdb"
        ) or data_cfg.input_ligand.startswith("pep")
    data, _, _ = mods["get_input_data"](
        protein_path=data_cfg.protein_path,
        input_ligand=data_cfg.get("input_ligand", None),
        is_pep=is_pep,
        pocket_args=data_cfg.get("pocket_args", {}),
        pocmol_args=data_cfg.get("pocmol_args", {}),
    )
    dataset = mods["UseDataset"](
        data, n=1, task=config.task.name, transforms=transforms
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        follow_batch=follow_batch,
        exclude_keys=exclude_keys,
    )
    batch = next(iter(loader)).to(device)
    noiser = mods["get_sample_noiser"](
        config.noise,
        in_dims["num_node_types"],
        in_dims["num_edge_types"],
        mode="sample",
        device=device,
        ref_config=train_config.noise,
    )
    return batch, noiser


def to_numpy(tensor: torch.Tensor, dtype: str) -> np.ndarray:
    array = tensor.detach().float().cpu().numpy()
    return array.astype(np.float16 if dtype == "float16" else np.float32)


def extract_one(
    batch, noiser, model, timesteps: list[float], dtype: str
) -> dict[str, np.ndarray]:
    hidden: dict[str, np.ndarray] = {}
    for step in timesteps:
        cache: dict[str, torch.Tensor | tuple[torch.Tensor, ...]] = {}

        def pocket_hook(_module, _inputs, output):
            cache["pocket"] = output

        def denoiser_hook(_module, _inputs, output):
            cache["denoiser"] = output

        handles = [
            model.pocket_encoder.register_forward_hook(pocket_hook),
            model.denoiser.register_forward_hook(denoiser_hook),
        ]
        try:
            step_batch = deepcopy(batch)
            step_batch = noiser(step_batch, step)
            with torch.no_grad():
                model(step_batch)
        finally:
            for handle in handles:
                handle.remove()

        h_pocket = cache["pocket"]
        h_node, _pos_node, h_edge = cache["denoiser"]
        step_prefix = f"t{step:.3f}".replace(".", "p")
        hidden[f"{step_prefix}_pocket"] = to_numpy(h_pocket, dtype)
        hidden[f"{step_prefix}_node"] = to_numpy(h_node, dtype)
        hidden[f"{step_prefix}_edge"] = to_numpy(h_edge, dtype)
    return hidden


def main() -> None:
    args = parse_args()
    mods = import_pocketxmol_modules()
    rows = load_manifest(args.manifest, args.limit)
    if not rows:
        raise SystemExit("empty manifest")

    model, train_config, in_dims = load_model(
        rows[0]["config_path"], args.config_model, args.device, mods
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = args.out_dir / "raw_hidden_manifest.csv"
    failures = []

    with manifest_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["compound_id", "cid", "split", "pec50", "hidden_path"]
        )
        writer.writeheader()
        for i, row in enumerate(rows, 1):
            cid = row["cid"]
            out_path = args.out_dir / f"{cid}.npz"
            print(f"[{i}/{len(rows)}] {cid}", flush=True)
            try:
                if not out_path.exists():
                    batch, noiser = make_batch(
                        row, args.config_model, train_config, in_dims, args.device, mods
                    )
                    hidden = extract_one(
                        batch, noiser, model, args.timesteps, args.dtype
                    )
                    np.savez_compressed(out_path, **hidden)
                writer.writerow(
                    {
                        "compound_id": row["compound_id"],
                        "cid": row["cid"],
                        "split": row["split"],
                        "pec50": row["pec50"],
                        "hidden_path": out_path,
                    }
                )
                f.flush()
            except Exception as exc:  # keep long runs moving
                failures.append((row["compound_id"], type(exc).__name__, str(exc)))

    if failures:
        failure_path = args.out_dir / "raw_hidden_failures.csv"
        with failure_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["compound_id", "error_type", "message"])
            writer.writerows(failures)


if __name__ == "__main__":
    main()
