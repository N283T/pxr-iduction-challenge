#!/usr/bin/env -S pixi run python
"""Train a lightweight learned head over Boltz core-pocket pair tensors.

This is the first non-tabular distogram experiment in this branch. For each
compound it loads the saved Boltz trunk pair representation ``z`` and the
predicted pose, extracts ``z[core_pocket_residue, ligand_atom, :]``, adds a
learned distance-bin embedding, then trains a tiny Transformer/attention
pooling head under canonical UMAP CV.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import psycopg2
import torch
from rdkit import Chem
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
sys.path.insert(
    0, str(REPO_ROOT.joinpath("track1_activity", "boltz2", "src", "boltz2"))
)

from constants import PXR_CORE_POCKET_RESIDUES, PROTEIN_CHAIN_ID, PXR_SEQUENCE  # noqa: E402
from data import DB_PARAMS, load_train_smiles_target  # noqa: E402
from evaluate import (  # noqa: E402
    compute_metrics,
    print_fold_summary,
    print_metrics,
    record_experiment,
    save_oof_predictions,
)
from splits import umap_split_indices  # noqa: E402

PROTEIN_N_RES = len(PXR_SEQUENCE)
CORE_IDX = np.asarray([r - 1 for r in PXR_CORE_POCKET_RESIDUES], dtype=np.int64)
DIST_BINS = np.concatenate(
    [
        np.asarray([0.0], dtype=np.float32),
        np.linspace(2.0, 22.0, 63, dtype=np.float32),
        np.asarray([np.inf], dtype=np.float32),
    ]
)
DEFAULT_CACHE = REPO_ROOT.joinpath("data", "boltz_affhead", "pairlite_core_cache.npz")
DEFAULT_CACHE_WORKERS = 8


@dataclass
class Config:
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 1
    dropout: float = 0.10
    batch_size: int = 24
    epochs: int = 60
    patience: int = 12
    lr: float = 2e-4
    weight_decay: float = 1e-4
    seed: int = 42


def _train_rows() -> pd.DataFrame:
    sql = """
    SELECT (row_number() OVER (ORDER BY t.id) - 1)::int AS train_idx,
           t.compound_id,
           t.pec50,
           b.embeddings_npz_path,
           b.pose_cif_path,
           b.ligand_pkl_path,
           b.ligand_atom_count
    FROM train_activity t
    LEFT JOIN compound_boltz2 b ON b.compound_id = t.compound_id
    ORDER BY t.id
    """
    with psycopg2.connect(**DB_PARAMS) as conn:
        return pd.read_sql(sql, conn)


def _protein_rep_coords(cif_path: str) -> np.ndarray:
    structure = gemmi.read_structure(cif_path)
    coords = np.full((PROTEIN_N_RES, 3), np.nan, dtype=np.float32)
    for model in structure:
        chain = model[PROTEIN_CHAIN_ID]
        for residue in chain:
            idx = int(residue.seqid.num) - 1
            if idx < 0 or idx >= PROTEIN_N_RES:
                continue
            fallback = None
            for atom in residue:
                if atom.element.name == "H":
                    continue
                coord = (atom.pos.x, atom.pos.y, atom.pos.z)
                if fallback is None:
                    fallback = coord
                if atom.name.strip() == "CA":
                    coords[idx] = coord
                    break
            if np.isnan(coords[idx, 0]) and fallback is not None:
                coords[idx] = fallback
        break
    return coords


def _ligand_coords(pkl_path: str) -> np.ndarray:
    with open(pkl_path, "rb") as fh:
        mol = pickle.load(fh)
    if not isinstance(mol, Chem.Mol):
        raise TypeError(f"Expected RDKit Mol in {pkl_path}, got {type(mol).__name__}")
    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append((pos.x, pos.y, pos.z))
    return np.asarray(coords, dtype=np.float32)


def _cache_one(row) -> dict:
    (
        train_idx_value,
        compound_id_value,
        pec50_value,
        embeddings_npz_path,
        pose_cif_path,
        ligand_pkl_path,
        ligand_atom_count,
    ) = row
    try:
        data = np.load(Path(embeddings_npz_path), allow_pickle=False)
        z = data["z"][0].astype(np.float32)
        ligand = _ligand_coords(str(ligand_pkl_path))
        n_lig = min(int(ligand_atom_count), ligand.shape[0], z.shape[0] - PROTEIN_N_RES)
        if n_lig <= 0:
            raise ValueError("empty ligand span")
        protein = _protein_rep_coords(str(pose_cif_path))[CORE_IDX]
        dist = np.linalg.norm(protein[:, None, :] - ligand[None, :n_lig, :], axis=2)
        bins = np.digitize(dist, DIST_BINS[1:-1], right=False).astype(np.uint8)
        lig_idx = np.arange(PROTEIN_N_RES, PROTEIN_N_RES + n_lig)
        z_block = z[CORE_IDX[:, None], lig_idx[None, :], :].astype(np.float16)
        return {
            "train_idx": int(train_idx_value),
            "compound_id": int(compound_id_value),
            "y": float(pec50_value),
            "n_lig": int(n_lig),
            "z_block": z_block,
            "bins": bins,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "compound_id": int(compound_id_value),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _store_cache_result(
    result: dict,
    out_i: int,
    z_core: np.ndarray,
    dist_bin: np.ndarray,
    mask: np.ndarray,
    train_idx: np.ndarray,
    compound_id: np.ndarray,
    y: np.ndarray,
) -> int:
    n_lig = int(result["n_lig"])
    z_core[out_i, :, :n_lig, :] = result["z_block"]
    dist_bin[out_i, :, :n_lig] = result["bins"]
    mask[out_i, :, :n_lig] = True
    train_idx[out_i] = int(result["train_idx"])
    compound_id[out_i] = int(result["compound_id"])
    y[out_i] = float(result["y"])
    return out_i + 1


def build_cache(
    path: Path, workers: int = DEFAULT_CACHE_WORKERS
) -> dict[str, np.ndarray]:
    rows = _train_rows()
    covered = rows[
        rows["embeddings_npz_path"].notna()
        & rows["pose_cif_path"].notna()
        & rows["ligand_pkl_path"].notna()
        & rows["ligand_atom_count"].notna()
    ].copy()
    max_lig = int(covered["ligand_atom_count"].max())
    n = len(covered)
    z_core = np.zeros((n, len(CORE_IDX), max_lig, 128), dtype=np.float16)
    dist_bin = np.zeros((n, len(CORE_IDX), max_lig), dtype=np.uint8)
    mask = np.zeros((n, len(CORE_IDX), max_lig), dtype=bool)
    train_idx = np.zeros(n, dtype=np.int64)
    compound_id = np.zeros(n, dtype=np.int64)
    y = np.zeros(n, dtype=np.float32)

    errors: list[tuple[int, str]] = []
    out_i = 0
    row_iter = list(covered.itertuples(index=False, name=None))
    if workers <= 1:
        iterator = map(_cache_one, row_iter)
    else:
        pool = Pool(workers)
        iterator = pool.imap_unordered(_cache_one, row_iter, chunksize=8)
    try:
        for result in tqdm(iterator, total=n, desc="pairlite-cache"):
            if "error" in result:
                errors.append((int(result["compound_id"]), str(result["error"])))
                continue
            out_i = _store_cache_result(
                result, out_i, z_core, dist_bin, mask, train_idx, compound_id, y
            )
    finally:
        if workers > 1:
            pool.close()
            pool.join()

    if errors:
        print(f"cache errors: {len(errors)}")
        for err in errors[:5]:
            print(err)
    arrays = {
        "z_core": z_core[:out_i],
        "dist_bin": dist_bin[:out_i],
        "mask": mask[:out_i],
        "train_idx": train_idx[:out_i],
        "compound_id": compound_id[:out_i],
        "y": y[:out_i],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"wrote cache {path}: n={out_i}, max_lig={max_lig}")
    return arrays


def load_cache(
    path: Path, rebuild: bool = False, workers: int = DEFAULT_CACHE_WORKERS
) -> dict[str, np.ndarray]:
    if rebuild or not path.exists():
        return build_cache(path, workers=workers)
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


class PairDataset(Dataset):
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        indices: np.ndarray,
        y_mean: float,
        y_std: float,
    ):
        self.z = arrays["z_core"][indices]
        self.dist = arrays["dist_bin"][indices].astype(np.int64)
        self.mask = arrays["mask"][indices]
        self.y = (arrays["y"][indices].astype(np.float32) - y_mean) / y_std

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.z[idx].astype(np.float32)),
            torch.from_numpy(self.dist[idx]),
            torch.from_numpy(self.mask[idx]),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


class PairLiteHead(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.z_proj = nn.Linear(128, cfg.d_model)
        self.dist_emb = nn.Embedding(64, cfg.d_model)
        if cfg.n_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.d_model * 4,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        else:
            self.encoder = nn.Identity()
        self.score = nn.Linear(cfg.d_model, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(
        self, z: torch.Tensor, dist_bin: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        bsz = z.shape[0]
        x = self.z_proj(z) + self.dist_emb(dist_bin)
        x = x.reshape(bsz, -1, x.shape[-1])
        flat_mask = mask.reshape(bsz, -1)
        if isinstance(self.encoder, nn.TransformerEncoder):
            x = self.encoder(x, src_key_padding_mask=~flat_mask)
        else:
            x = self.encoder(x)
        logits = self.score(x).squeeze(-1).masked_fill(~flat_mask, -1e4)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return self.head(pooled).squeeze(-1)


def _train_one_fold(
    arrays: dict[str, np.ndarray],
    train_local: np.ndarray,
    val_local: np.ndarray,
    cfg: Config,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    torch.manual_seed(cfg.seed)
    y_train = arrays["y"][train_local].astype(np.float32)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std() if y_train.std() > 1e-6 else 1.0)
    train_ds = PairDataset(arrays, train_local, y_mean, y_std)
    val_ds = PairDataset(arrays, val_local, y_mean, y_std)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=0
    )

    model = PairLiteHead(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_mae = float("inf")
    best_preds: np.ndarray | None = None
    bad_epochs = 0

    for epoch in range(cfg.epochs):
        model.train()
        for z, dist, mask, target in train_loader:
            z = z.to(device)
            dist = dist.to(device)
            mask = mask.to(device)
            target = target.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = model(z, dist, mask)
                loss = loss_fn(pred, target)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

        preds = []
        model.eval()
        with torch.no_grad():
            for z, dist, mask, _target in val_loader:
                z = z.to(device)
                dist = dist.to(device)
                mask = mask.to(device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    pred = model(z, dist, mask)
                preds.append(pred.float().cpu().numpy())
        val_pred = np.concatenate(preds) * y_std + y_mean
        val_true = arrays["y"][val_local]
        mae = float(np.mean(np.abs(val_true - val_pred)))
        if mae < best_mae - 1e-4:
            best_mae = mae
            best_preds = val_pred.copy()
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch % 5 == 0 or bad_epochs == 0:
            print(f"    epoch {epoch + 1:03d}: val_mae={mae:.4f} best={best_mae:.4f}")
        if bad_epochs >= cfg.patience:
            break

    assert best_preds is not None
    metrics = compute_metrics(arrays["y"][val_local], best_preds)
    return best_preds, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--cache-workers", type=int, default=DEFAULT_CACHE_WORKERS)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--experiment-name", default="pairlite_core_z_dist_umap")
    args = parser.parse_args()

    cfg = Config(
        d_model=args.d_model,
        n_layers=args.layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
    )
    arrays = load_cache(
        args.cache, rebuild=args.rebuild_cache, workers=args.cache_workers
    )
    train_df = load_train_smiles_target()
    splits = umap_split_indices(train_df["smiles"].tolist(), seed=42, n_clusters=50)
    full_to_local = {int(full_idx): i for i, full_idx in enumerate(arrays["train_idx"])}
    oof = np.full(len(train_df), np.nan, dtype=np.float32)
    covered = np.zeros(len(train_df), dtype=bool)
    fold_metrics = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, cache_n={len(arrays['y'])}, cfg={cfg}")

    for fold, (tr_idx, va_idx) in enumerate(splits):
        train_local = np.asarray(
            [full_to_local[int(i)] for i in tr_idx if int(i) in full_to_local]
        )
        val_pairs = [
            (int(i), full_to_local[int(i)]) for i in va_idx if int(i) in full_to_local
        ]
        val_full = np.asarray([p[0] for p in val_pairs], dtype=np.int64)
        val_local = np.asarray([p[1] for p in val_pairs], dtype=np.int64)
        print(f"\nFold {fold}: train={len(train_local)} val={len(val_local)}")
        preds, metrics = _train_one_fold(arrays, train_local, val_local, cfg, device)
        oof[val_full] = preds
        covered[val_full] = True
        fold_metrics.append(metrics)
        print_metrics(metrics, f"Fold {fold}")

    print("\nOverall OOF (covered subset):")
    y_full = train_df["pec50"].to_numpy(dtype=np.float32)
    overall = compute_metrics(y_full[covered], oof[covered])
    print_metrics(overall)
    print_fold_summary(fold_metrics)
    exp_id = record_experiment(
        name=args.experiment_name,
        description="PairLite learned head on core-pocket Boltz z plus distance-bin embedding",
        model_type="pairlite",
        feature_set="core_z_dist_pair_tensor",
        hyperparameters=asdict(cfg),
        fold_metrics=fold_metrics,
        notes="Tiny TransformerEncoder over z[13 core residues x ligand atoms] + Boltz-like distance bins.",
        on_conflict_replace=True,
    )
    save_oof_predictions(exp_id, np.nan_to_num(oof, nan=0.0), covered_mask=covered)


if __name__ == "__main__":
    main()
