"""Extract per-layer CLS embeddings from Uni-Mol v2.

DR (2026-05-02) reports intermediate layers often outperform the final
layer for downstream regression by 5-8% MAE. We hook each transformer
layer's output and dump per-layer CLS representations for the same
13136 compounds, then compare via TabPFN UMAP.

Usage (inside Uni-Mol pixi env):
    pixi run python 14_extract_intermediate_layers.py \\
        --csv  data/unimol/pretrain_all.csv \\
        --ckpt models/unimol_v2_log2fc/model_0.pth \\
        --out-dir data/unimol/layers/

Outputs: <out_dir>/layer_<i>.npz with keys (compound_id, cls_repr).
The final layer's CLS = same as get_repr()['cls_repr'] when no pair-track tweaks.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ckpt", required=True, help="finetuned .pth (force-loaded)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-size", default="84m")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from unimol_tools import UniMolRepr

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"loading on {device}", file=sys.stderr)
    repr_model = UniMolRepr(
        model_name="unimolv2", model_size=args.model_size, use_cuda=True,
        batch_size=args.batch_size,
    )
    # Force load finetuned weights (the public default loaded in __init__ is overridden)
    repr_model.model.load_pretrained_weights(path=args.ckpt, strict=False)
    repr_model.model.eval()

    # Locate the transformer encoder layers
    encoder = repr_model.model.encoder
    layers = encoder.layers
    n_layers = len(layers)
    print(f"encoder has {n_layers} layers", file=sys.stderr)

    # Forward hooks: capture each layer's x output (then x[:, 0, :] = CLS)
    captures: dict[int, list[np.ndarray]] = {i: [] for i in range(n_layers)}

    def make_hook(layer_idx: int):
        def _hook(module, inputs, output):
            x = output[0] if isinstance(output, tuple) else output
            cls = x[:, 0, :].detach().cpu().numpy().astype(np.float32)
            captures[layer_idx].append(cls)
        return _hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]

    df = pd.read_csv(args.csv)
    smiles = df["SMILES"].tolist()
    cids = df["compound_id"].astype(int).to_numpy()
    print(f"encoding {len(smiles)} smiles", file=sys.stderr)

    # Run get_repr in chunks of batch_size to avoid OOM
    BATCH = args.batch_size
    for start in tqdm(range(0, len(smiles), BATCH), desc="encode"):
        chunk = smiles[start : start + BATCH]
        with torch.no_grad():
            _ = repr_model.get_repr(data=chunk)

    for h in handles:
        h.remove()

    # Save per-layer
    for i in range(n_layers):
        if not captures[i]:
            print(f"layer {i}: no captures", file=sys.stderr)
            continue
        cls_all = np.concatenate(captures[i], axis=0)
        if cls_all.shape[0] != len(cids):
            print(
                f"layer {i}: shape mismatch {cls_all.shape[0]} vs {len(cids)}",
                file=sys.stderr,
            )
            continue
        out_path = out_dir.joinpath(f"layer_{i:02d}.npz")
        np.savez(out_path, compound_id=cids, cls_repr=cls_all)
        print(f"  wrote {out_path}  shape {cls_all.shape}")


if __name__ == "__main__":
    main()
