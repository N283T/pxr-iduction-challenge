"""Compute ChemFM embeddings for all compounds and store in DB.

ChemFM is a Llama-style causal LM pretrained on UniChem SMILES.
Two sizes supported (TheLuoFengLab, Nature Comm Chem 2025):
    1b: 22 layers / hidden=2048 / 969M params
    3b: 30 layers / hidden=3072 / 3.0B params

Both are decoder-only causal LMs, so we extract TWO pooled representations:
    embedding_last: last non-padding token hidden state
                    (right-most position has attended to full sequence)
    embedding_mean: attention-mask-weighted mean over last hidden state
                    (BERT-style, included for downstream A/B)

Usage:
    pixi run python db/compute_chemfm.py            # default 1b
    pixi run python db/compute_chemfm.py --size 3b
"""

from __future__ import annotations

import argparse
import logging
import time

import psycopg2
import torch
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DB_PARAMS = {"dbname": "pxr_challenge", "host": "/tmp", "port": 5433}

# Per-size defaults. Batch sizes chosen to leave >=4 GB VRAM headroom on
# an RTX 5080 (16 GB) at max_length=512 in bf16.
CONFIGS = {
    "1b": {
        "hf_id": "ChemFM/ChemFM-1B",
        "table": "compound_chemfm_1b",
        "batch_size": 32,
    },
    "3b": {
        "hf_id": "ChemFM/ChemFM-3B",
        "table": "compound_chemfm_3b",
        "batch_size": 16,
    },
}

MAX_LENGTH = 512


def compute(size: str) -> None:
    cfg = CONFIGS[size]
    hf_id = cfg["hf_id"]
    table = cfg["table"]
    batch_size = cfg["batch_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading {hf_id} on {device} (bf16)")

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModel.from_pretrained(hf_id, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    hidden_size = model.config.hidden_size
    logger.info(
        f"Loaded. hidden_size={hidden_size}, layers={model.config.num_hidden_layers}"
    )
    if torch.cuda.is_available():
        logger.info(f"VRAM after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute(f"TRUNCATE {table}")
    conn.commit()

    cur.execute(
        "SELECT id, std_smiles FROM compounds WHERE std_mol IS NOT NULL ORDER BY id"
    )
    compounds = cur.fetchall()
    n_total = len(compounds)
    logger.info(f"Computing embeddings for {n_total} compounds (batch={batch_size})")

    start = time.time()
    n_done = 0

    for batch_start in range(0, n_total, batch_size):
        batch = compounds[batch_start : batch_start + batch_size]
        ids = [c[0] for c in batch]
        smiles = [c[1] for c in batch]

        inputs = tokenizer(
            smiles,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)

        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=False)

        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"]

        lengths = mask.sum(dim=1)
        last_idx = (lengths - 1).clamp(min=0)
        batch_idx = torch.arange(hidden.size(0), device=device)
        emb_last = hidden[batch_idx, last_idx].float().cpu().numpy()

        mask_f = mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask_f).sum(dim=1)
        counts = mask_f.sum(dim=1).clamp(min=1)
        emb_mean = (summed / counts).float().cpu().numpy()

        rows = [
            (cid, emb_last[i].tolist(), emb_mean[i].tolist())
            for i, cid in enumerate(ids)
        ]
        cur.executemany(
            f"INSERT INTO {table} (compound_id, embedding_last, embedding_mean) "
            "VALUES (%s, %s, %s)",
            rows,
        )
        conn.commit()

        n_done += len(batch)
        if n_done % (batch_size * 16) == 0 or n_done == n_total:
            elapsed = time.time() - start
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (n_total - n_done) / rate if rate > 0 else 0
            logger.info(
                f"  {n_done}/{n_total} ({100 * n_done / n_total:.1f}%) "
                f"| {rate:.1f} cmpd/s | ETA {eta:.0f}s"
            )

    cur.execute(f"SELECT count(*) FROM {table}")
    (count,) = cur.fetchone()
    cur.execute(f"SELECT array_length(embedding_last, 1) FROM {table} LIMIT 1")
    (dim,) = cur.fetchone()
    logger.info(f"Done. {count} rows, dim={dim}")

    cur.close()
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=list(CONFIGS), default="1b")
    args = parser.parse_args()
    compute(args.size)


if __name__ == "__main__":
    main()
