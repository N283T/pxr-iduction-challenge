"""Compute embeddings from multiple pretrained models and store in DB.

Usage:
    python db/compute_embeddings.py                    # all models
    python db/compute_embeddings.py chemberta-77m-mlm  # specific model
"""

import logging
import sys

import numpy as np
import psycopg2
import torch
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DB_PARAMS = {"dbname": "pxr_challenge", "host": "/tmp", "port": 5433}

MODELS = {
    "chemberta-77m-mlm": {
        "hf_id": "DeepChem/ChemBERTa-77M-MLM",
        "table": "compound_chemberta",
        "batch_size": 128,
    },
    "chemberta-77m-mtr": {
        "hf_id": "DeepChem/ChemBERTa-77M-MTR",
        "table": "compound_chemberta_mtr",
        "batch_size": 128,
    },
    "chemberta-100m-mlm": {
        "hf_id": "DeepChem/ChemBERTa-100M-MLM",
        "table": "compound_chemberta_100m",
        "batch_size": 64,
    },
    "chemberta-10m-mlm": {
        "hf_id": "DeepChem/ChemBERTa-10M-MLM",
        "table": "compound_chemberta_10m",
        "batch_size": 128,
    },
    "chemberta-10m-mtr": {
        "hf_id": "DeepChem/ChemBERTa-10M-MTR",
        "table": "compound_chemberta_10m_mtr",
        "batch_size": 128,
    },
    "chemberta-5m-mlm": {
        "hf_id": "DeepChem/ChemBERTa-5M-MLM",
        "table": "compound_chemberta_5m",
        "batch_size": 128,
    },
    "chemberta-5m-mtr": {
        "hf_id": "DeepChem/ChemBERTa-5M-MTR",
        "table": "compound_chemberta_5m_mtr",
        "batch_size": 128,
    },
    # MoLFormer models require transformers.onnx (removed in transformers>=5)
    # Disabled until upstream compatibility is fixed
    # "molformer-1.1b": {
    #     "hf_id": "DeepChem/MoLFormer-c3-1.1B",
    #     "table": "compound_molformer_1b",
    #     "batch_size": 16,
    #     "trust_remote_code": True,
    # },
    # "molformer-100m": {
    #     "hf_id": "DeepChem/MoLFormer-c3-100M",
    #     "table": "compound_molformer_100m",
    #     "batch_size": 64,
    #     "trust_remote_code": True,
    # },
}


def compute_embeddings(model_key: str):
    """Compute embeddings for a single model and store in DB."""
    config = MODELS[model_key]
    hf_id = config["hf_id"]
    table = config["table"]
    batch_size = config["batch_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[{model_key}] Loading {hf_id} on {device}")

    trust = config.get("trust_remote_code", False)
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=trust)
    model = AutoModel.from_pretrained(hf_id, trust_remote_code=trust).to(device)
    model.eval()
    hidden_size = model.config.hidden_size
    logger.info(f"[{model_key}] Hidden size: {hidden_size}")

    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            compound_id INTEGER PRIMARY KEY REFERENCES compounds(id),
            embedding DOUBLE PRECISION[] NOT NULL
        )
    """)
    cur.execute(f"TRUNCATE {table}")
    conn.commit()

    cur.execute(
        "SELECT id, std_smiles FROM compounds WHERE std_mol IS NOT NULL ORDER BY id"
    )
    compounds = cur.fetchall()
    logger.info(f"[{model_key}] Computing embeddings for {len(compounds)} compounds...")

    total_inserted = 0

    for batch_start in range(0, len(compounds), batch_size):
        batch = compounds[batch_start : batch_start + batch_size]
        ids = [c[0] for c in batch]
        smiles_list = [c[1] for c in batch]

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
            cur.execute(
                f"INSERT INTO {table} (compound_id, embedding) VALUES (%s, %s)",
                (cid, mean_pooled[i].tolist()),
            )

        conn.commit()
        total_inserted += len(batch)
        if total_inserted % 1024 == 0 or total_inserted == len(compounds):
            logger.info(f"[{model_key}]   {total_inserted}/{len(compounds)}")

    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_cid ON {table}(compound_id)")
    conn.commit()

    cur.execute(f"SELECT count(*) FROM {table}")
    count = cur.fetchone()[0]
    cur.execute(f"SELECT array_length(embedding, 1) FROM {table} LIMIT 1")
    dim = cur.fetchone()[0]
    logger.info(f"[{model_key}] Done. {count} rows, dim={dim}")

    cur.close()
    conn.close()

    # Free GPU memory
    del model
    torch.cuda.empty_cache()


def main():
    if len(sys.argv) > 1:
        keys = sys.argv[1:]
        for key in keys:
            if key not in MODELS:
                logger.error(f"Unknown model: {key}. Available: {list(MODELS.keys())}")
                return
    else:
        keys = list(MODELS.keys())

    logger.info(f"Models to compute: {keys}")

    for key in keys:
        try:
            compute_embeddings(key)
        except Exception as e:
            logger.error(f"[{key}] Failed: {e}")
            continue


if __name__ == "__main__":
    main()
