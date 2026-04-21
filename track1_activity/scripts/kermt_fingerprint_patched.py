"""Patched fingerprint driver for KERMT.

KERMT's task/fingerprint.py has a bug: load_checkpoint() returns
(model, state) but the code assigns that tuple to ``model`` directly,
which then fails on ``model.eval()`` with
``AttributeError: 'tuple' object has no attribute 'eval'``.

Rather than patch the vendored third-party KERMT clone, this script
replicates the fingerprint branch of KERMT's main.py with the tuple
unpacking fixed. Intended to run inside the KERMT-local pixi env
(so ``import kermt`` resolves).

Usage (from run_kermt_embed_extract.sh):

    pixi run --manifest-path $KERMT_REPO/pixi.toml python \
        $PXR_REPO/track1_activity/scripts/kermt_fingerprint_patched.py \
        --data_path <smiles.csv> \
        --checkpoint_path <model.pt> \
        --output_path <embeddings.npz> \
        --fingerprint_source both \
        --batch_size 32

Output NPZ has a single key ``fps`` of shape (n_compounds, emb_dim),
matching the format written by KERMT's main.py.
"""

from __future__ import annotations

import random
import sys

import numpy as np
import torch
from rdkit import RDLogger


def setup(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(mode=True)


def main() -> None:
    # Suppress RDKit logger as main.py does.
    RDLogger.logger().setLevel(RDLogger.CRITICAL)

    # Import here so the MolVocab side-effect executes before any
    # data loading (matches main.py ordering).
    from kermt.data.torchvocab import MolVocab  # noqa: F401
    from kermt.data import MolCollator, MoleculeDataset
    from kermt.util.parsing import parse_args
    from kermt.util.utils import create_logger, get_data, load_checkpoint
    from torch.utils.data import DataLoader

    args = parse_args()
    print(f"Setting up with random seed: {args.seed}")
    setup(seed=args.seed)
    print(f"args: {args}")

    if args.parser_name != "fingerprint":
        print(
            "ERROR: this script only supports the 'fingerprint' subcommand",
            file=sys.stderr,
        )
        sys.exit(2)

    logger = create_logger(name="fingerprint", save_dir=None, quiet=False)

    print("Loading data")
    test_data = get_data(
        path=args.data_path,
        args=args,
        use_compound_names=False,
        max_data_size=float("inf"),
        skip_invalid_smiles=False,
    )
    test_data = MoleculeDataset(test_data)
    logger.info(f"Total size = {len(test_data):,}")
    logger.info("Generating...")

    checkpoint_path = args.checkpoint_paths[0]
    # FIX: load_checkpoint returns (model, state); the upstream
    # task/fingerprint.py forgets to unpack.
    model, _state = load_checkpoint(
        checkpoint_path,
        cuda=args.cuda,
        current_args=args,
        logger=logger,
    )
    model.eval()
    args.bond_drop_rate = 0

    mol_collator = MolCollator(args=args, shared_dict={})
    loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=mol_collator,
    )

    preds: list[np.ndarray] = []
    for item in loader:
        _, batch, features_batch, _, _ = item
        with torch.no_grad():
            batch_preds = model(batch, features_batch)
            preds.extend(batch_preds.data.cpu().numpy())

    feas = np.asarray(preds)
    print(f"Fingerprint array shape: {feas.shape}  dtype: {feas.dtype}")
    np.savez_compressed(args.output_path, fps=feas)
    print(f"Wrote {args.output_path}")


if __name__ == "__main__":
    main()
