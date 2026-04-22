"""Phase 3 of Boltz strategy-3: pretrain an MLP head on Boltz trunk
(1024d) using single-concentration log2_fc weak labels.

Three variants per spec:
  A  baseline simple MLP   1024 -> 256 -> 2
  B  wider MLP             1024 -> 512 -> 256 -> 2
  C  PairFormer-inspired   split 1024 -> 4x256 tokens, 2-layer transformer, pool

Targets: [log2fc_8p25, log2fc_33], z-scored, NaN-masked MSE.
Split: 90/10 random within compounds, seed=42.
Output: pretrain.pt (state_dict + metadata) under
  track1_activity/checkpoints/boltz_trunk_pretrain_<variant>/

Phase 4 (09b_extract_embed.py) consumes the saved checkpoint.

Usage
-----
    pixi run python track1_activity/scripts/boltz_affhead/09_mlp_pretrain.py --variant a
    pixi run python track1_activity/scripts/boltz_affhead/09_mlp_pretrain.py --variant b
    pixi run python track1_activity/scripts/boltz_affhead/09_mlp_pretrain.py --variant c
    pixi run python track1_activity/scripts/boltz_affhead/09_mlp_pretrain.py --variant a --max-epochs 5  # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.joinpath("track1_activity", "src")))
from data import DB_PARAMS  # noqa: E402

CKPT_ROOT = REPO_ROOT.joinpath("track1_activity", "checkpoints")

INPUT_DIM = 1024
N_TARGETS = 2  # full two-head pretrain; reduced by --targets slicing

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_trunk_and_targets() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (trunk[N,1024], targets[N,2] with NaN, compound_ids[N])."""
    sql = """
    SELECT b.compound_id,
           b.s_prot_mean, b.s_lig_mean, b.z_if_mean, b.z_if_max,
           agg.log2fc_8p25, agg.log2fc_33
      FROM compound_boltz2_trunk_fast b
      LEFT JOIN (
        SELECT compound_id,
          AVG(CASE WHEN concentration_m BETWEEN 8.2e-6 AND 8.3e-6
                   THEN log2_fc_estimate END) AS log2fc_8p25,
          AVG(CASE WHEN concentration_m BETWEEN 3.28e-5 AND 3.32e-5
                   THEN log2_fc_estimate END) AS log2fc_33
        FROM single_concentration
        GROUP BY compound_id
      ) agg ON agg.compound_id = b.compound_id
     ORDER BY b.compound_id
    """
    with psycopg2.connect(**DB_PARAMS) as conn:
        df = pd.read_sql(sql, conn)

    n = len(df)
    trunk = np.empty((n, INPUT_DIM), dtype=np.float32)
    for i, row in enumerate(df.itertuples(index=False)):
        trunk[i, :384] = row.s_prot_mean
        trunk[i, 384:768] = row.s_lig_mean
        trunk[i, 768:896] = row.z_if_mean
        trunk[i, 896:1024] = row.z_if_max
    targets = df[["log2fc_8p25", "log2fc_33"]].to_numpy(dtype=np.float32)
    cids = df["compound_id"].to_numpy(dtype=np.int64)
    return trunk, targets, cids


def zscore_targets(
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.zeros(N_TARGETS, dtype=np.float32)
    stds = np.ones(N_TARGETS, dtype=np.float32)
    for i in range(N_TARGETS):
        valid = np.isfinite(targets[:, i])
        means[i] = float(np.nanmean(targets[valid, i]))
        stds[i] = float(np.nanstd(targets[valid, i])) or 1.0
    z = (targets - means) / stds  # NaN propagates
    return z, means, stds


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class MLPVariantA(nn.Module):
    """1024 -> hidden -> 2  (penultimate hidden = HIDDEN)."""

    DEFAULT_HIDDEN = 256

    def __init__(self, dropout: float = 0.2, hidden_dim: int | None = None):
        super().__init__()
        self.HIDDEN = hidden_dim if hidden_dim is not None else self.DEFAULT_HIDDEN
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, self.HIDDEN),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(self.HIDDEN, N_TARGETS)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


class MLPVariantB(nn.Module):
    """1024 -> hidden*2 -> hidden -> 2  (penultimate = HIDDEN, intermediate = HIDDEN*2)."""

    DEFAULT_HIDDEN = 256

    def __init__(self, dropout: float = 0.2, hidden_dim: int | None = None):
        super().__init__()
        self.HIDDEN = hidden_dim if hidden_dim is not None else self.DEFAULT_HIDDEN
        intermediate = self.HIDDEN * 2
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, intermediate),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate, self.HIDDEN),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(self.HIDDEN, N_TARGETS)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


class MLPVariantC(nn.Module):
    """PairFormer-inspired: split 1024d into 4 tokens (s_prot, s_lig,
    z_if_mean, z_if_max), project each to TOKEN_DIM, 2-layer self-
    attention, then pool.

    Pool strategies:
      mean   - mean over 4 attended tokens -> TOKEN_DIM (default)
      concat - concatenate 4 attended tokens -> 4*TOKEN_DIM (keeps all)
      cls    - prepend a learnable CLS token, pool by reading CLS out
    Penultimate hidden = HIDDEN. With hidden_dim that differs from the
    natural pool output, a projection Linear is appended.
    """

    DEFAULT_HIDDEN = 256
    N_TOKENS = 4
    TOKEN_DIM = 256

    def __init__(
        self,
        dropout: float = 0.1,
        n_layers: int = 2,
        n_heads: int = 4,
        hidden_dim: int | None = None,
        pool: str = "mean",
    ):
        super().__init__()
        if pool not in {"mean", "concat", "cls"}:
            raise ValueError(f"pool must be mean/concat/cls, got {pool}")
        self.pool = pool

        # Natural output dim of the pooling stage
        natural = self.TOKEN_DIM * self.N_TOKENS if pool == "concat" else self.TOKEN_DIM
        self.HIDDEN = hidden_dim if hidden_dim is not None else natural

        self.proj_s_prot = nn.Linear(384, self.TOKEN_DIM)
        self.proj_s_lig = nn.Linear(384, self.TOKEN_DIM)
        self.proj_z_mean = nn.Linear(128, self.TOKEN_DIM)
        self.proj_z_max = nn.Linear(128, self.TOKEN_DIM)
        self.token_norm = nn.LayerNorm(self.TOKEN_DIM)

        if pool == "cls":
            # Learnable CLS token prepended to 4-token sequence
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.TOKEN_DIM))
            nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.TOKEN_DIM,
            nhead=n_heads,
            dim_feedforward=self.TOKEN_DIM * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.proj_out = (
            nn.Sequential(
                nn.Linear(natural, self.HIDDEN), nn.GELU(), nn.Dropout(dropout)
            )
            if self.HIDDEN != natural
            else nn.Identity()
        )
        self.head = nn.Linear(self.HIDDEN, N_TARGETS)

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        s_prot = self.proj_s_prot(x[:, :384])
        s_lig = self.proj_s_lig(x[:, 384:768])
        z_mean = self.proj_z_mean(x[:, 768:896])
        z_max = self.proj_z_max(x[:, 896:1024])
        tokens = self.token_norm(torch.stack([s_prot, s_lig, z_mean, z_max], dim=1))
        if self.pool == "cls":
            cls = self.cls_token.expand(tokens.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)  # (B, 5, TOKEN_DIM)
        return tokens

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._tokens(x)
        attended = self.transformer(tokens)
        if self.pool == "mean":
            pooled = attended[:, -self.N_TOKENS :].mean(dim=1)
        elif self.pool == "concat":
            pooled = attended[:, -self.N_TOKENS :].reshape(
                attended.size(0), self.N_TOKENS * self.TOKEN_DIM
            )
        else:  # cls
            pooled = attended[:, 0]
        return self.proj_out(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


VARIANTS = {"a": MLPVariantA, "b": MLPVariantB, "c": MLPVariantC}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(target)
    if not mask.any():
        return torch.zeros((), device=pred.device, requires_grad=True)
    diff = (pred - torch.where(mask, target, torch.zeros_like(target))) * mask
    return (diff**2).sum() / mask.sum()


@dataclass
class Metadata:
    variant: str
    input_dim: int
    hidden_dim: int
    n_targets: int
    n_total: int
    n_train: int
    n_val: int
    n_valid_8p25: int
    n_valid_33: int
    target_means: list[float]
    target_stds: list[float]
    epochs_run: int
    best_epoch: int
    best_val_loss: float
    final_val_loss: float
    seed: int
    lr: float
    batch_size: int
    weight_decay: float
    pool: str = "mean"


def run(variant_key: str, args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    parts = []
    if args.hidden_dim is not None:
        parts.append(f"h{args.hidden_dim}")
    if variant_key == "c" and args.pool != "mean":
        parts.append(args.pool)
    if args.targets != "both":
        parts.append(f"t{args.targets}")
    suffix = ("_" + "_".join(parts)) if parts else ""
    out_dir = CKPT_ROOT.joinpath(f"boltz_trunk_pretrain_{variant_key}{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[variant={variant_key}] device={device}, out_dir={out_dir}")

    trunk, targets_raw, cids = load_trunk_and_targets()
    n_total = trunk.shape[0]
    n_valid_8 = int(np.isfinite(targets_raw[:, 0]).sum())
    n_valid_33 = int(np.isfinite(targets_raw[:, 1]).sum())
    print(
        f"  data: {n_total} compounds (1024d trunk), "
        f"log2fc_8p25 valid={n_valid_8}, log2fc_33 valid={n_valid_33}"
    )

    targets_z, means, stds = zscore_targets(targets_raw)
    print(f"  target means={means.tolist()}, stds={stds.tolist()}")

    # Optional: silence one of the two heads so pretrain focuses on the
    # higher-r concentration. NaN -> masked_mse ignores it in both
    # gradient and val metric.
    if args.targets == "8p25":
        targets_z[:, 1] = np.nan
        print("  --targets 8p25: masked log2fc_33 head (gradient zeroed)")
    elif args.targets == "33":
        targets_z[:, 0] = np.nan
        print("  --targets 33: masked log2fc_8p25 head")

    # 90/10 random split (compound-level)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_total)
    n_val = int(n_total * args.val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"  split: train={len(train_idx)}, val={len(val_idx)}")

    def make_loader(idx, shuffle):
        x = torch.from_numpy(trunk[idx])
        y = torch.from_numpy(targets_z[idx])
        return DataLoader(
            TensorDataset(x, y),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader = make_loader(val_idx, shuffle=False)

    model_cls = VARIANTS[variant_key]
    kwargs = {"dropout": args.dropout, "hidden_dim": args.hidden_dim}
    if variant_key == "c":
        kwargs["pool"] = args.pool
    model = model_cls(**kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model {model_cls.__name__}: {n_params:,} params")

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val = float("inf")
    best_epoch = 0
    patience = args.patience
    bad_epochs = 0
    epochs_run = 0
    history: list[dict] = []
    final_val = float("inf")

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_losses: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = masked_mse(pred, yb)
            if loss.requires_grad:
                loss.backward()
                opt.step()
            train_losses.append(float(loss.detach()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                val_losses.append(float(masked_mse(model(xb), yb)))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss})
        print(
            f"  epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}"
            + ("  <-- best" if val_loss < best_val else "")
        )

        epochs_run = epoch
        final_val = val_loss
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), out_dir.joinpath("pretrain.pt"))
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stop at epoch {epoch} (no improvement for {patience})")
                break

    meta = Metadata(
        variant=variant_key,
        input_dim=INPUT_DIM,
        hidden_dim=model.HIDDEN,
        n_targets=N_TARGETS,
        n_total=n_total,
        n_train=len(train_idx),
        n_val=len(val_idx),
        n_valid_8p25=n_valid_8,
        n_valid_33=n_valid_33,
        target_means=means.tolist(),
        target_stds=stds.tolist(),
        epochs_run=epochs_run,
        best_epoch=best_epoch,
        best_val_loss=best_val,
        final_val_loss=final_val,
        seed=args.seed,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        pool=args.pool if variant_key == "c" else "mean",
    )
    out_dir.joinpath("metadata.json").write_text(json.dumps(asdict(meta), indent=2))
    out_dir.joinpath("history.json").write_text(json.dumps(history, indent=2))
    print(
        f"\n[variant={variant_key}] DONE: best_val={best_val:.4f} @ epoch {best_epoch}"
    )
    print(f"  saved: {out_dir.joinpath('pretrain.pt')}")
    print(f"  meta:  {out_dir.joinpath('metadata.json')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), default="a")
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Override penultimate hidden dim (default 256 A/B/C-mean, 1024 C-concat).",
    )
    ap.add_argument(
        "--pool",
        choices=["mean", "concat", "cls"],
        default="mean",
        help="Variant C only: pool 4 attended tokens by mean/concat/cls.",
    )
    ap.add_argument(
        "--targets",
        choices=["both", "8p25", "33"],
        default="both",
        help="Subset of log2_fc heads to train on. '8p25' / '33' masks the other head.",
    )
    args = ap.parse_args()

    # Variant C uses lower lr per spec (attention is more sensitive)
    if args.variant == "c" and args.lr == 1e-3:
        args.lr = 3e-4
        print("Note: --variant c overrides default lr to 3e-4")

    run(args.variant, args)


if __name__ == "__main__":
    main()
