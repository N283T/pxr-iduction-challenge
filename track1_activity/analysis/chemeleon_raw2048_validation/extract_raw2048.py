#!/usr/bin/env -S pixi run python
"""Extract the raw 2048d CheMeleon fingerprint into a separate DB table.

The legacy ``compound_chemeleon`` table is intentionally untouched. Its 300d
values were produced by ``MPNN.encoding()``, which applies a fresh predictor
projection after the pretrained message-passing fingerprint.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import psycopg2
import torch
from chemprop import featurizers, nn
from chemprop.data import BatchMolGraph
from psycopg2.extras import execute_values
from rdkit import Chem


DB_PARAMS = {"dbname": "pxr_challenge", "host": "/tmp", "port": 5433}
CHECKPOINT_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
CHECKPOINT_PATH = Path.home().joinpath(".chemprop", "chemeleon_mp.pt")
TABLE = "compound_chemeleon_raw2048"
EXPECTED_DIM = 2048

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


class RawCheMeleonFingerprint:
    """Pretrained message passing followed directly by mean aggregation."""

    def __init__(self) -> None:
        if not CHECKPOINT_PATH.exists():
            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = CHECKPOINT_PATH.with_suffix(".pt.tmp")
            logger.info("Downloading CheMeleon checkpoint to %s", CHECKPOINT_PATH)
            urlretrieve(CHECKPOINT_URL, tmp_path)
            tmp_path.replace(CHECKPOINT_PATH)

        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        self.message_passing = nn.BondMessagePassing(**checkpoint["hyper_parameters"])
        self.message_passing.load_state_dict(checkpoint["state_dict"], strict=True)
        self.message_passing.eval()
        self.aggregation = nn.MeanAggregation()
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        self.output_dim = self.message_passing.output_dim
        if self.output_dim != EXPECTED_DIM:
            raise ValueError(
                f"Expected CheMeleon output dim {EXPECTED_DIM}, got {self.output_dim}"
            )

    def __call__(self, smiles: list[str]) -> np.ndarray:
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        invalid = [i for i, mol in enumerate(mols) if mol is None]
        if invalid:
            raise ValueError(f"Invalid SMILES at batch offsets: {invalid[:10]}")
        bmg = BatchMolGraph([self.featurizer(mol) for mol in mols])
        with torch.inference_mode():
            atom_embeddings = self.message_passing(bmg)
            fingerprints = self.aggregation(atom_embeddings, bmg.batch)
        values = fingerprints.cpu().numpy().astype(np.float32, copy=False)
        if values.shape != (len(smiles), EXPECTED_DIM):
            raise ValueError(
                f"Unexpected fingerprint shape {values.shape}; "
                f"expected {(len(smiles), EXPECTED_DIM)}"
            )
        if not np.isfinite(values).all():
            raise ValueError("Non-finite value in CheMeleon fingerprints")
        return values


def checkpoint_sha256() -> str:
    digest = hashlib.sha256()
    with CHECKPOINT_PATH.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute existing rows. By default the extractor resumes missing rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = RawCheMeleonFingerprint()
    sha256 = checkpoint_sha256()
    logger.info(
        "Loaded raw CheMeleon encoder: dim=%d checkpoint_sha256=%s",
        generator.output_dim,
        sha256,
    )

    with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                compound_id INTEGER PRIMARY KEY REFERENCES compounds(id),
                embedding REAL[] NOT NULL,
                checkpoint_sha256 TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                CHECK (array_length(embedding, 1) = {EXPECTED_DIM})
            )
            """
        )
        if args.force:
            cur.execute(f"TRUNCATE {TABLE}")
        cur.execute(
            f"""
            SELECT c.id, c.std_smiles
            FROM compounds c
            LEFT JOIN {TABLE} e ON e.compound_id = c.id
            WHERE c.std_mol IS NOT NULL AND e.compound_id IS NULL
            ORDER BY c.id
            """
        )
        compounds = cur.fetchall()

    logger.info("Rows to compute: %d", len(compounds))
    total = 0
    for start in range(0, len(compounds), args.batch_size):
        batch = compounds[start : start + args.batch_size]
        ids = [int(row[0]) for row in batch]
        smiles = [str(row[1]) for row in batch]
        embeddings = generator(smiles)
        rows = [
            (compound_id, embedding.tolist(), sha256)
            for compound_id, embedding in zip(ids, embeddings, strict=True)
        ]
        with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
            execute_values(
                cur,
                f"""
                INSERT INTO {TABLE} (compound_id, embedding, checkpoint_sha256)
                VALUES %s
                ON CONFLICT (compound_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    checkpoint_sha256 = EXCLUDED.checkpoint_sha256,
                    created_at = now()
                """,
                rows,
                page_size=args.batch_size,
            )
        total += len(batch)
        logger.info("Computed %d/%d", total, len(compounds))

    with psycopg2.connect(**DB_PARAMS) as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT count(*), min(array_length(embedding, 1)),
                   max(array_length(embedding, 1)), count(DISTINCT checkpoint_sha256)
            FROM {TABLE}
            """
        )
        count, min_dim, max_dim, checkpoint_count = cur.fetchone()
        cur.execute("SELECT count(*) FROM compounds WHERE std_mol IS NOT NULL")
        expected_count = cur.fetchone()[0]

    if (count, min_dim, max_dim, checkpoint_count) != (
        expected_count,
        EXPECTED_DIM,
        EXPECTED_DIM,
        1,
    ):
        raise RuntimeError(
            "Validation failed: "
            f"count={count}/{expected_count}, dims={min_dim}..{max_dim}, "
            f"checkpoint_count={checkpoint_count}"
        )
    logger.info("Done: %d rows, dim=%d, legacy table untouched", count, min_dim)


if __name__ == "__main__":
    main()
