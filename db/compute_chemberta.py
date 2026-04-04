"""Compute ChemBERTa embeddings for all compounds and store in DB."""

import json
import logging

import numpy as np
import psycopg2
import torch
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DB_PARAMS = {"dbname": "pxr_challenge", "host": "/tmp", "port": 5433}
MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"
BATCH_SIZE = 128


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logger.info(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    hidden_size = model.config.hidden_size
    logger.info(f"Hidden size: {hidden_size}")

    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    # Create table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS compound_chemberta (
            compound_id INTEGER PRIMARY KEY REFERENCES compounds(id),
            embedding DOUBLE PRECISION[] NOT NULL
        )
    """)
    cur.execute("TRUNCATE compound_chemberta")
    conn.commit()

    # Fetch all compounds
    cur.execute("SELECT id, std_smiles FROM compounds WHERE std_mol IS NOT NULL ORDER BY id")
    compounds = cur.fetchall()
    logger.info(f"Computing embeddings for {len(compounds)} compounds...")

    total_inserted = 0

    for batch_start in range(0, len(compounds), BATCH_SIZE):
        batch = compounds[batch_start : batch_start + BATCH_SIZE]
        ids = [c[0] for c in batch]
        smiles_list = [c[1] for c in batch]

        # Tokenize and encode
        inputs = tokenizer(
            smiles_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Mean pooling over non-padding tokens
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        token_embeddings = outputs.last_hidden_state
        masked_embeddings = token_embeddings * attention_mask
        summed = masked_embeddings.sum(dim=1)
        counts = attention_mask.sum(dim=1)
        mean_pooled = (summed / counts).cpu().numpy()

        for i, cid in enumerate(ids):
            embedding = mean_pooled[i].tolist()
            cur.execute(
                "INSERT INTO compound_chemberta (compound_id, embedding) VALUES (%s, %s)",
                (cid, embedding),
            )

        conn.commit()
        total_inserted += len(batch)
        if total_inserted % 1024 == 0 or total_inserted == len(compounds):
            logger.info(f"  Processed {total_inserted}/{len(compounds)}")

    # Create index
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_chemberta_compound ON compound_chemberta(compound_id)"
    )
    conn.commit()

    # Verify
    cur.execute("SELECT count(*) FROM compound_chemberta")
    count = cur.fetchone()[0]
    cur.execute("SELECT array_length(embedding, 1) FROM compound_chemberta LIMIT 1")
    dim = cur.fetchone()[0]
    logger.info(f"Done. {count} rows, embedding dim={dim}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
