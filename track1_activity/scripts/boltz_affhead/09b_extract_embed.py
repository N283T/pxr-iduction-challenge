"""Phase 4 of Boltz strategy-3: forward all 13k Boltz trunk vectors
through the pretrained MLP encoder, extract the penultimate hidden
(256d for variants A/B/C), and save as a feature parquet.

Output: data/boltz_trunk_pretrain_embed_<variant>.parquet
  index: compound_id
  columns: emb_000..emb_255

Phase 5 (run_train.py) registers this parquet as a feature for TabPFN.

Usage
-----
    pixi run python track1_activity/scripts/boltz_affhead/09b_extract_embed.py --variant a
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(
    0, str(REPO_ROOT.joinpath("track1_activity", "scripts", "boltz_affhead"))
)
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))

import importlib.util  # noqa: E402

CKPT_ROOT = REPO_ROOT.joinpath("track1_activity", "checkpoints")
DATA_DIR = REPO_ROOT.joinpath("data")


def _load_pretrain_module():
    spec = importlib.util.spec_from_file_location(
        "_boltz_mlp_pretrain",
        REPO_ROOT.joinpath(
            "track1_activity", "scripts", "boltz_affhead", "09_mlp_pretrain.py"
        ),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_boltz_mlp_pretrain"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["a", "b", "c"], default="a")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument(
        "--layer",
        choices=["penult", "first"],
        default="penult",
        help=(
            "Which hidden layer to extract from. 'penult' = penultimate "
            "(256d for all variants, default). 'first' = first hidden, "
            "only meaningful for variant b (512d) where 1024->512->256->2 "
            "has a wider intermediate. For a/c 'first' equals 'penult'."
        ),
    )
    ap.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help=(
            "If the checkpoint was trained with a non-default hidden size, "
            "pass that value (e.g. 512) so the matching ckpt dir "
            "boltz_trunk_pretrain_<v>_h<HIDDEN> is loaded."
        ),
    )
    ap.add_argument(
        "--pool",
        choices=["mean", "concat", "cls"],
        default="mean",
        help="Variant C pool used at pretrain time (must match ckpt dir suffix).",
    )
    ap.add_argument(
        "--targets",
        choices=["both", "8p25", "33"],
        default="both",
        help="Subset of targets used at pretrain time (must match ckpt dir suffix).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output parquet path (default: data/boltz_trunk_pretrain_embed_<variant>[_h<HIDDEN>][_first].parquet)",
    )
    args = ap.parse_args()

    mod = _load_pretrain_module()

    parts = []
    if args.hidden_dim is not None:
        parts.append(f"h{args.hidden_dim}")
    if args.variant == "c" and args.pool != "mean":
        parts.append(args.pool)
    if args.targets != "both":
        parts.append(f"t{args.targets}")
    ckpt_suffix = ("_" + "_".join(parts)) if parts else ""
    ckpt_dir = CKPT_ROOT.joinpath(f"boltz_trunk_pretrain_{args.variant}{ckpt_suffix}")
    state_path = ckpt_dir.joinpath("pretrain.pt")
    meta_path = ckpt_dir.joinpath("metadata.json")
    if not state_path.exists():
        raise SystemExit(f"checkpoint not found: {state_path}")
    if not meta_path.exists():
        raise SystemExit(f"metadata not found: {meta_path}")
    meta = json.loads(meta_path.read_text())
    print(
        f"[{args.variant}] checkpoint best_val={meta['best_val_loss']:.4f} "
        f"@ epoch {meta['best_epoch']} (out of {meta['epochs_run']})"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {"hidden_dim": meta.get("hidden_dim")}
    if args.variant == "c":
        kwargs["pool"] = meta.get("pool", "mean")
    model = mod.VARIANTS[args.variant](**kwargs)
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval().to(device)

    # Layer extraction: variant B has a 512d intermediate before its 256d
    # penult; expose it via --layer first. For a/c the request degrades to
    # penult (their first hidden = 256d).
    use_first_layer = args.layer == "first" and args.variant == "b"
    if use_first_layer:
        # encoder[0:3] = Linear(1024, 512) -> GELU -> Dropout, output 512d
        encoder_prefix = torch.nn.Sequential(*list(model.encoder)[:3])

        def embed_fn(xb: torch.Tensor) -> torch.Tensor:
            return encoder_prefix(xb)

        hidden_dim = 512
    else:
        embed_fn = model.embed
        hidden_dim = model.HIDDEN
    print(
        f"  device={device}, hidden_dim={hidden_dim}, layer={'first' if use_first_layer else 'penult'}"
    )

    trunk, _, cids = mod.load_trunk_and_targets()
    print(f"  trunk loaded: {trunk.shape}, compounds={len(cids)}")

    n = trunk.shape[0]
    embeds = np.empty((n, hidden_dim), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            end = min(n, start + args.batch_size)
            xb = torch.from_numpy(trunk[start:end]).to(device, non_blocking=True)
            embeds[start:end] = embed_fn(xb).cpu().numpy()

    cols = [f"emb_{i:03d}" for i in range(hidden_dim)]
    df = pd.DataFrame(embeds, columns=cols)
    df.insert(0, "compound_id", cids)

    out_parts = []
    if args.hidden_dim is not None:
        out_parts.append(f"h{args.hidden_dim}")
    if args.variant == "c" and args.pool != "mean":
        out_parts.append(args.pool)
    if args.targets != "both":
        out_parts.append(f"t{args.targets}")
    if use_first_layer:
        out_parts.append("first")
    out_suffix = ("_" + "_".join(out_parts)) if out_parts else ""
    out = args.out or DATA_DIR.joinpath(
        f"boltz_trunk_pretrain_embed_{args.variant}{out_suffix}.parquet"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, compression="zstd")
    print(f"[{args.variant}] saved {len(df)}x{hidden_dim} -> {out}")


if __name__ == "__main__":
    main()
