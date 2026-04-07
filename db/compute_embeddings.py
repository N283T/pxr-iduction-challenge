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
    # --- Non-DeepChem models ---
    "chemberta-zinc-v1": {
        "hf_id": "seyonec/ChemBERTa-zinc-base-v1",
        "table": "compound_chemberta_zinc_v1",
        "batch_size": 128,
    },
    "bert-base-smiles": {
        "hf_id": "unikei/bert-base-smiles",
        "table": "compound_bert_smiles",
        "batch_size": 64,
    },
    # MoLFormer / IBM models (patched for transformers>=5)
    "molformer-xl": {
        "hf_id": "ibm/MoLFormer-XL-both-10pct",
        "table": "compound_molformer",
        "batch_size": 64,
        "trust_remote_code": True,
        "deterministic_eval": True,
        "fix_rotary": True,
    },
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
    extra_kwargs = {}
    if "deterministic_eval" in config:
        extra_kwargs["deterministic_eval"] = config["deterministic_eval"]
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=trust)
    model = AutoModel.from_pretrained(
        hf_id, trust_remote_code=trust, **extra_kwargs
    ).to(device)
    model.eval()

    # Fix corrupted rotary embeddings (MoLFormer + transformers v5 bug)
    if config.get("fix_rotary", False):
        for layer in model.encoder.layer:
            rotary = layer.attention.self.rotary_embeddings
            rotary.inv_freq = (
                1.0
                / (rotary.base ** (torch.arange(0, rotary.dim, 2).float() / rotary.dim))
            ).to(device)
            rotary._set_cos_sin_cache(
                seq_len=rotary.max_position_embeddings,
                device=device,
                dtype=torch.float32,
            )
        logger.info(f"[{model_key}] Fixed rotary embeddings")

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

        # Use pooler_output if available (MoLFormer), else mean pooling
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output.cpu().numpy()
        else:
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state
            masked_embeddings = token_embeddings * attention_mask
            summed = masked_embeddings.sum(dim=1)
            counts = attention_mask.sum(dim=1)
            pooled = (summed / counts).cpu().numpy()

        for i, cid in enumerate(ids):
            cur.execute(
                f"INSERT INTO {table} (compound_id, embedding) VALUES (%s, %s)",
                (cid, pooled[i].tolist()),
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
