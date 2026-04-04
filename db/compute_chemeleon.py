"""Compute CheMeleon fingerprints and store in DB.

CheMeleon is a descriptor-based foundation model for molecular properties,
built on ChemProp MPNN. Pretrained checkpoint is downloaded from Zenodo.
"""

import logging
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import psycopg2
import torch
from chemprop import featurizers, nn
from chemprop.data import BatchMolGraph
from chemprop.models import MPNN
from chemprop.nn import RegressionFFN
from rdkit import Chem

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DB_PARAMS = {"dbname": "pxr_challenge", "host": "/tmp", "port": 5433}
CHECKPOINT_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
CHECKPOINT_DIR = Path.home().joinpath(".chemprop")
CHECKPOINT_PATH = CHECKPOINT_DIR.joinpath("chemeleon_mp.pt")
BATCH_SIZE = 256


class CheMeleonFingerprint:
    """Generate CheMeleon molecular fingerprints."""

    def __init__(self):
        # Download checkpoint if needed
        if not CHECKPOINT_PATH.exists():
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading CheMeleon checkpoint to {CHECKPOINT_PATH}...")
            urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)

        # Build model on CPU (BatchMolGraph.to() is broken in chemprop 2.2.x)
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)

        mp = nn.BondMessagePassing(**checkpoint["hyper_parameters"])
        mp.load_state_dict(checkpoint["state_dict"])
        agg = nn.MeanAggregation()
        ffn = RegressionFFN(input_dim=mp.output_dim)
        self.model = MPNN(mp, agg, ffn)
        self.model.eval()

        self.hidden_dim = mp.output_dim
        logger.info(f"CheMeleon loaded. Hidden dim: {self.hidden_dim}")

    def __call__(self, smiles_list: list[str]) -> np.ndarray:
        """Generate fingerprints for a list of SMILES."""
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        bmg = BatchMolGraph([self.featurizer(m) for m in mols])

        with torch.no_grad():
            encoding = self.model.encoding(bmg)

        return encoding.numpy()


def main():
    fp_gen = CheMeleonFingerprint()

    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS compound_chemeleon (
            compound_id INTEGER PRIMARY KEY REFERENCES compounds(id),
            embedding DOUBLE PRECISION[] NOT NULL
        )
    """)
    cur.execute("TRUNCATE compound_chemeleon")
    conn.commit()

    cur.execute(
        "SELECT id, std_smiles FROM compounds WHERE std_mol IS NOT NULL ORDER BY id"
    )
    compounds = cur.fetchall()
    logger.info(f"Computing CheMeleon fingerprints for {len(compounds)} compounds...")

    total = 0
    for batch_start in range(0, len(compounds), BATCH_SIZE):
        batch = compounds[batch_start : batch_start + BATCH_SIZE]
        ids = [c[0] for c in batch]
        smiles = [c[1] for c in batch]

        embeddings = fp_gen(smiles)

        for i, cid in enumerate(ids):
            cur.execute(
                "INSERT INTO compound_chemeleon (compound_id, embedding) VALUES (%s, %s)",
                (cid, embeddings[i].tolist()),
            )

        conn.commit()
        total += len(batch)
        if total % 2048 == 0 or total == len(compounds):
            logger.info(f"  {total}/{len(compounds)}")

    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_chemeleon_cid ON compound_chemeleon(compound_id)"
    )
    conn.commit()

    cur.execute("SELECT count(*) FROM compound_chemeleon")
    count = cur.fetchone()[0]
    cur.execute("SELECT array_length(embedding, 1) FROM compound_chemeleon LIMIT 1")
    dim = cur.fetchone()[0]
    logger.info(f"Done. {count} rows, dim={dim}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
