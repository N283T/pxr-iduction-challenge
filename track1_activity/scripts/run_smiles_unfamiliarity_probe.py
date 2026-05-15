#!/usr/bin/env python
"""Fold-safe SMILES reconstruction unfamiliarity probe for Track 1.

This is a lightweight PXR adaptation of the Joint Molecular Model idea from
van Tilborg et al. (2026): train a shared SMILES encoder for reconstruction
and property prediction, then use per-molecule reconstruction loss as an
unfamiliarity/OOD signal.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "track1_activity" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import load_test_smiles, load_train_smiles_target  # noqa: E402
from evaluate import compute_metrics, print_metrics  # noqa: E402
from splits import umap_split_indices  # noqa: E402

OUTPUT_ROOT = REPO_ROOT / "track1_activity" / "analysis" / "smiles_unfamiliarity"


@dataclass
class Tokenizer:
    stoi: dict[str, int]
    max_len: int
    pad_idx: int = 0
    bos_idx: int = 1
    eos_idx: int = 2
    unk_idx: int = 3

    @classmethod
    def fit(cls, smiles: list[str], max_len: int | None = None) -> "Tokenizer":
        chars = sorted({ch for smi in smiles for ch in smi})
        stoi = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        stoi.update({ch: i + 4 for i, ch in enumerate(chars)})
        if max_len is None:
            max_len = max(len(smi) for smi in smiles) + 2
        return cls(stoi=stoi, max_len=max_len)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, smiles: str) -> list[int]:
        ids = [self.bos_idx]
        ids.extend(self.stoi.get(ch, self.unk_idx) for ch in smiles)
        ids.append(self.eos_idx)
        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
            ids[-1] = self.eos_idx
        ids.extend([self.pad_idx] * (self.max_len - len(ids)))
        return ids


class SmilesDataset(Dataset):
    def __init__(
        self,
        smiles: list[str],
        tokenizer: Tokenizer,
        y: np.ndarray | None = None,
        y_mean: float = 0.0,
        y_std: float = 1.0,
    ) -> None:
        self.tokens = torch.tensor(
            [tokenizer.encode(s) for s in smiles], dtype=torch.long
        )
        self.y = None
        if y is not None:
            self.y = torch.tensor((y - y_mean) / y_std, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.tokens[idx]
        return self.tokens[idx], self.y[idx]


class SmilesJMM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        emb_dim: int,
        hidden_dim: int,
        z_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.encoder = nn.Sequential(
            nn.Conv1d(emb_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveMaxPool1d(1),
        )
        self.to_z = nn.Linear(hidden_dim, z_dim)
        self.z_to_hidden = nn.Linear(z_dim, hidden_dim)
        self.decoder = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.decoder_out = nn.Linear(hidden_dim, vocab_size)
        self.reg_head = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.embedding(tokens).transpose(1, 2)
        z = self.to_z(self.encoder(emb).squeeze(-1))
        h0 = torch.tanh(self.z_to_hidden(z)).unsqueeze(0)
        decoder_in = self.embedding(tokens[:, :-1])
        dec_out, _ = self.decoder(decoder_in, h0)
        logits = self.decoder_out(dec_out)
        y_hat = self.reg_head(z).squeeze(-1)
        return logits, y_hat, z


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reconstruction_loss_per_sample(
    logits: torch.Tensor, targets: torch.Tensor, pad_idx: int
) -> torch.Tensor:
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets[:, 1:].reshape(-1),
        ignore_index=pad_idx,
        reduction="none",
    ).reshape(targets.shape[0], -1)
    lengths = (targets[:, 1:] != pad_idx).sum(dim=1).clamp_min(1)
    return token_loss.sum(dim=1) / lengths


def make_loader(
    smiles: list[str],
    tokenizer: Tokenizer,
    y: np.ndarray | None,
    y_mean: float,
    y_std: float,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        SmilesDataset(smiles, tokenizer, y, y_mean, y_std),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_model(
    train_smiles: list[str],
    train_y: np.ndarray,
    tokenizer: Tokenizer,
    args: argparse.Namespace,
    fold: int,
) -> tuple[SmilesJMM, dict[str, float]]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed + fold)
    order = rng.permutation(len(train_smiles))
    n_val = max(1, int(round(len(order) * args.inner_val_frac)))
    inner_va = order[:n_val]
    inner_tr = order[n_val:]

    y_mean = float(train_y[inner_tr].mean())
    y_std = float(train_y[inner_tr].std() or 1.0)
    model = SmilesJMM(
        vocab_size=tokenizer.vocab_size,
        pad_idx=tokenizer.pad_idx,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        z_dim=args.z_dim,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    tr_loader = make_loader(
        [train_smiles[i] for i in inner_tr],
        tokenizer,
        train_y[inner_tr],
        y_mean,
        y_std,
        args.batch_size,
        True,
    )
    va_loader = make_loader(
        [train_smiles[i] for i in inner_va],
        tokenizer,
        train_y[inner_va],
        y_mean,
        y_std,
        args.batch_size,
        False,
    )

    best_state = None
    best_loss = float("inf")
    bad_epochs = 0
    for epoch in range(args.max_epochs):
        model.train()
        train_losses = []
        for tokens, yb in tr_loader:
            tokens = tokens.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits, y_hat, _z = model(tokens)
            recon = reconstruction_loss_per_sample(logits, tokens, tokenizer.pad_idx)
            reg = F.mse_loss(y_hat, yb, reduction="none")
            loss = recon.mean() + args.gamma * reg.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))

        val_loss = evaluate_loss(
            model, va_loader, tokenizer.pad_idx, args.gamma, device
        )
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.y_mean = y_mean  # type: ignore[attr-defined]
    model.y_std = y_std  # type: ignore[attr-defined]
    return model, {
        "epochs": epoch + 1,
        "best_inner_loss": best_loss,
        "inner_train_loss_last": float(np.mean(train_losses)),
        "y_mean": y_mean,
        "y_std": y_std,
    }


@torch.no_grad()
def evaluate_loss(
    model: SmilesJMM,
    loader: DataLoader,
    pad_idx: int,
    gamma: float,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for tokens, yb in loader:
        tokens = tokens.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits, y_hat, _z = model(tokens)
        recon = reconstruction_loss_per_sample(logits, tokens, pad_idx)
        reg = F.mse_loss(y_hat, yb, reduction="none")
        losses.extend((recon + gamma * reg).detach().cpu().numpy().tolist())
    return float(np.mean(losses))


@torch.no_grad()
def predict_unfamiliarity(
    model: SmilesJMM,
    smiles: list[str],
    tokenizer: Tokenizer,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    loader = make_loader(
        smiles,
        tokenizer,
        None,
        0.0,
        1.0,
        args.predict_batch_size,
        False,
    )
    model.eval()
    all_recon = []
    all_pred = []
    all_z = []
    for tokens in loader:
        tokens = tokens.to(device, non_blocking=True)
        logits, y_hat, z = model(tokens)
        recon = reconstruction_loss_per_sample(logits, tokens, tokenizer.pad_idx)
        all_recon.append(recon.detach().cpu().numpy())
        pred = y_hat.detach().cpu().numpy() * model.y_std + model.y_mean  # type: ignore[attr-defined]
        all_pred.append(pred)
        all_z.append(z.detach().cpu().numpy())
    return np.concatenate(all_recon), np.concatenate(all_pred), np.concatenate(all_z)


def maybe_load_anchor_oof() -> np.ndarray | None:
    try:
        scripts_dir = REPO_ROOT / "track1_activity" / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from run_ensemble_calibrate_importance import load_caruana_oof_and_test

        anchor_oof, _anchor_test, _anchor_df = load_caruana_oof_and_test()
        return np.asarray(anchor_oof, dtype=np.float64)
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        print(f"Could not load current ensemble OOF for diagnostics: {exc}")
        return None


def corr_or_nan(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    return float(spearmanr(a[mask], b[mask]).statistic)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", default="jmm_lite_seed42")
    p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--predict-batch-size", type=int, default=1024)
    p.add_argument("--emb-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--z-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--inner-val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fold-limit", type=int, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    run_dir = OUTPUT_ROOT / "outputs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_train_smiles_target()
    test_df = load_test_smiles()
    train_smiles = train_df["smiles"].tolist()
    test_smiles = test_df["smiles"].tolist()
    y = train_df["pec50"].to_numpy(dtype=np.float32)
    tokenizer = Tokenizer.fit(train_smiles + test_smiles)
    folds = umap_split_indices(train_smiles)
    if args.fold_limit is not None:
        folds = folds[: args.fold_limit]

    print("SMILES unfamiliarity probe")
    print(f"  run_name={args.run_name}")
    print(f"  train={len(train_smiles)} test={len(test_smiles)} folds={len(folds)}")
    print(f"  vocab={tokenizer.vocab_size} max_len={tokenizer.max_len}")
    print(
        f"  z={args.z_dim} hidden={args.hidden_dim} gamma={args.gamma} "
        f"epochs={args.max_epochs}"
    )

    oof_unf = np.full(len(train_smiles), np.nan, dtype=np.float64)
    oof_jmm = np.full(len(train_smiles), np.nan, dtype=np.float64)
    test_unf_folds = []
    test_pred_folds = []
    fold_rows = []
    for fold, (tr_idx, va_idx) in enumerate(folds):
        print(f"\n[Fold {fold}] train={len(tr_idx)} val={len(va_idx)}")
        model, info = train_one_model(
            [train_smiles[i] for i in tr_idx],
            y[tr_idx],
            tokenizer,
            args,
            fold,
        )
        val_unf, val_pred, _val_z = predict_unfamiliarity(
            model, [train_smiles[i] for i in va_idx], tokenizer, args
        )
        test_unf, test_pred, _test_z = predict_unfamiliarity(
            model, test_smiles, tokenizer, args
        )
        oof_unf[va_idx] = np.log(np.clip(val_unf, 1e-8, None))
        oof_jmm[va_idx] = val_pred
        test_unf_folds.append(np.log(np.clip(test_unf, 1e-8, None)))
        test_pred_folds.append(test_pred)
        fold_metrics = compute_metrics(y[va_idx], val_pred)
        print_metrics(fold_metrics, label=f"Fold {fold} pEC50 head")
        print(
            f"  unfamiliarity mean={oof_unf[va_idx].mean():.4f} "
            f"std={oof_unf[va_idx].std():.4f} epochs={info['epochs']}"
        )
        fold_rows.append(
            {
                "fold": fold,
                "n_train": len(tr_idx),
                "n_val": len(va_idx),
                **info,
                **{f"jmm_{k}": v for k, v in fold_metrics.items()},
                "val_unf_mean": float(oof_unf[va_idx].mean()),
                "val_unf_std": float(oof_unf[va_idx].std()),
            }
        )

    covered = np.isfinite(oof_unf) & np.isfinite(oof_jmm)
    test_unf_mean = np.mean(test_unf_folds, axis=0)
    test_pred_mean = np.mean(test_pred_folds, axis=0)
    jmm_metrics = compute_metrics(y[covered], oof_jmm[covered])
    print("\nOverall pEC50 head OOF:")
    print_metrics(jmm_metrics)

    anchor_oof = maybe_load_anchor_oof()
    anchor_abs_error_corr = float("nan")
    anchor_resid_corr = float("nan")
    if anchor_oof is not None and len(anchor_oof) == len(y):
        anchor_abs_error_corr = corr_or_nan(oof_unf, np.abs(anchor_oof - y))
        anchor_resid_corr = corr_or_nan(oof_unf, anchor_oof - y)

    summary = {
        "run_name": args.run_name,
        "coverage": int(covered.sum()),
        "train_n": len(train_smiles),
        "test_n": len(test_smiles),
        "vocab_size": tokenizer.vocab_size,
        "max_len": tokenizer.max_len,
        "oof_unf_mean": float(np.nanmean(oof_unf)),
        "oof_unf_std": float(np.nanstd(oof_unf)),
        "test_unf_mean": float(np.mean(test_unf_mean)),
        "test_unf_std": float(np.std(test_unf_mean)),
        "test_minus_oof_unf_mean": float(np.mean(test_unf_mean) - np.nanmean(oof_unf)),
        "unf_vs_y_spearman": corr_or_nan(oof_unf, y),
        "unf_vs_jmm_abs_error_spearman": corr_or_nan(oof_unf, np.abs(oof_jmm - y)),
        "unf_vs_anchor_abs_error_spearman": anchor_abs_error_corr,
        "unf_vs_anchor_residual_spearman": anchor_resid_corr,
        **{f"jmm_{k}": float(v) for k, v in jmm_metrics.items()},
    }

    pd.DataFrame(fold_rows).to_csv(run_dir / "fold_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(run_dir / "summary.csv", index=False)
    pd.DataFrame(
        {
            "train_idx": np.arange(len(train_smiles)),
            "smiles": train_smiles,
            "pec50": y,
            "oof_unfamiliarity": oof_unf,
            "oof_jmm_prediction": oof_jmm,
            "covered": covered,
        }
    ).to_csv(run_dir / "oof_unfamiliarity.csv", index=False)
    pd.DataFrame(
        {
            "smiles": test_smiles,
            "molecule_name": test_df["molecule_name"],
            "test_unfamiliarity": test_unf_mean,
            "test_jmm_prediction": test_pred_mean,
        }
    ).to_csv(run_dir / "test_unfamiliarity.csv", index=False)
    (run_dir / "tokenizer.json").write_text(json.dumps(asdict(tokenizer), indent=2))

    report = "\n".join(
        [
            "# SMILES Unfamiliarity Probe",
            "",
            "Lightweight PXR adaptation of JMM reconstruction unfamiliarity.",
            "",
            "## Setup",
            "",
            f"- Run name: `{args.run_name}`",
            f"- Coverage: `{summary['coverage']} / {len(train_smiles)}`",
            f"- Vocab size / max length: `{tokenizer.vocab_size}` / `{tokenizer.max_len}`",
            f"- Args: `{json.dumps(vars(args), sort_keys=True)}`",
            "",
            "## pEC50 Head OOF",
            "",
            f"- MAE: `{jmm_metrics['MAE']:.6f}`",
            f"- RAE: `{jmm_metrics['RAE']:.6f}`",
            f"- Spearman: `{jmm_metrics['Spearman_R']:.6f}`",
            "",
            "## Unfamiliarity Diagnostics",
            "",
            f"- OOF unfamiliarity mean/std: `{summary['oof_unf_mean']:.6f}` / `{summary['oof_unf_std']:.6f}`",
            f"- Test unfamiliarity mean/std: `{summary['test_unf_mean']:.6f}` / `{summary['test_unf_std']:.6f}`",
            f"- Test minus OOF mean: `{summary['test_minus_oof_unf_mean']:.6f}`",
            f"- Spearman(unfamiliarity, y): `{summary['unf_vs_y_spearman']:.6f}`",
            f"- Spearman(unfamiliarity, JMM abs error): `{summary['unf_vs_jmm_abs_error_spearman']:.6f}`",
            f"- Spearman(unfamiliarity, current ensemble abs error): `{summary['unf_vs_anchor_abs_error_spearman']:.6f}`",
            f"- Spearman(unfamiliarity, current ensemble residual): `{summary['unf_vs_anchor_residual_spearman']:.6f}`",
            "",
            "## Fold Metrics",
            "",
            pd.DataFrame(fold_rows).to_markdown(index=False),
            "",
            "## Initial Read",
            "",
            "Use this as an OOD/gating diagnostic, not as a direct submission model. "
            "A useful signal should correlate with current ensemble absolute error "
            "or separate test molecules into a plausible high-risk region.",
        ]
    )
    (run_dir / "report.md").write_text(report + "\n")
    print(f"\nSaved report: {run_dir / 'report.md'}")
    print(f"Saved summary: {run_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
