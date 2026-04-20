"""PEFT fine-tuning trainer for SMILES regression.

Wraps a Hugging Face backbone with a peft adapter (LoRA / etc.) and a
2-layer MLP regression head. Provides ``train_one_fold`` which returns
val and (optionally) test predictions for a single CV fold, and frees
GPU memory at the end so 5-fold runs stay inside 16GB VRAM.
"""

import numpy as np  # type: ignore[import-untyped]
import torch  # type: ignore[import-untyped]
import torch.nn as nn  # type: ignore[import-untyped]
from peft import get_peft_model  # type: ignore[import-untyped]
from torch.utils.data import DataLoader, Dataset  # type: ignore[import-untyped]
from transformers import AutoModel, AutoTokenizer  # type: ignore[import-untyped]

from peft_backbones import get_backbone
from peft_methods import get_peft_builder


class SmilesDataset(Dataset):
    """Tokenises SMILES on the fly, optionally returns regression targets."""

    def __init__(
        self,
        smiles_list: list[str],
        tokenizer,
        targets: np.ndarray | None = None,
        max_length: int = 202,
    ):
        self.smiles = smiles_list
        self.tokenizer = tokenizer
        self.targets = targets
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.smiles[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        if self.targets is not None:
            item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return item


class PeftRegressor(nn.Module):
    """Generic PEFT-wrapped backbone + 2-layer MLP regression head."""

    def __init__(
        self,
        backbone_name: str,
        peft_method: str,
        peft_params: dict,
        head_hidden_dim: int,
        head_dropout: float,
    ):
        super().__init__()
        meta = get_backbone(backbone_name)
        base = AutoModel.from_pretrained(
            meta["hf_id"], trust_remote_code=meta["trust_remote_code"]
        )
        # Fix corrupted rotary-embedding buffers (MoLFormer-XL + transformers v5
        # bug: inv_freq loaded from checkpoint contains garbage values).
        # Recompute inv_freq and rebuild the cos/sin cache before PEFT wrapping.
        # See compute_embeddings.py and issue #30 for the same fix.
        if meta.get("fix_rotary", False):
            for layer in base.encoder.layer:
                rotary = layer.attention.self.rotary_embeddings
                rotary.inv_freq = 1.0 / (
                    rotary.base ** (torch.arange(0, rotary.dim, 2).float() / rotary.dim)
                )
                rotary._set_cos_sin_cache(
                    seq_len=rotary.max_position_embeddings,
                    device=rotary.inv_freq.device,
                    dtype=torch.float32,
                )
        peft_config = get_peft_builder(peft_method)(meta, peft_params)
        self.backbone = get_peft_model(base, peft_config)

        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(meta["hidden_dim"], head_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token embedding (index 0).
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_emb).squeeze(-1)


def _predict(
    model: PeftRegressor, loader: DataLoader, device: torch.device
) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            preds = model(input_ids, attention_mask)
            chunks.append(preds.cpu().numpy())
    return np.concatenate(chunks)


def get_tokenizer(backbone_name: str):
    """Load the tokenizer for a backbone."""
    meta = get_backbone(backbone_name)
    return AutoTokenizer.from_pretrained(
        meta["hf_id"], trust_remote_code=meta["trust_remote_code"]
    )


def train_one_fold(
    params: dict,
    backbone_name: str,
    peft_method: str,
    tokenizer,
    train_smiles: list[str],
    train_targets: np.ndarray,
    val_smiles: list[str],
    val_targets: np.ndarray,
    test_smiles: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Train one CV fold; return (val_preds, test_preds_or_None).

    ``params`` keys consumed:
        lora_rank, lora_alpha, lora_dropout, lora_target  (peft method args)
        head_hidden_dim, head_dropout                     (head args)
        backbone_lr, head_lr, weight_decay                (optimizer)
        batch_size, max_epochs, patience                  (training loop)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = get_backbone(backbone_name)

    train_ds = SmilesDataset(
        train_smiles, tokenizer, train_targets, max_length=meta["max_length"]
    )
    val_ds = SmilesDataset(
        val_smiles, tokenizer, val_targets, max_length=meta["max_length"]
    )
    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False)

    peft_params = {
        "lora_rank": params["lora_rank"],
        "lora_alpha": params["lora_alpha"],
        "lora_dropout": params["lora_dropout"],
        "lora_target": params["lora_target"],
    }
    model = PeftRegressor(
        backbone_name=backbone_name,
        peft_method=peft_method,
        peft_params=peft_params,
        head_hidden_dim=params["head_hidden_dim"],
        head_dropout=params["head_dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.backbone.parameters() if p.requires_grad],
                "lr": params["backbone_lr"],
            },
            {"params": model.head.parameters(), "lr": params["head_lr"]},
        ],
        weight_decay=params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params["max_epochs"], eta_min=1e-7
    )
    criterion = nn.MSELoss()

    best_val_mae = float("inf")
    best_state = None
    patience_counter = 0

    for _epoch in range(params["max_epochs"]):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            preds = model(input_ids, attention_mask)
            loss = criterion(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        val_preds = _predict(model, val_loader, device)
        val_mae = float(np.mean(np.abs(val_targets - val_preds)))

        if np.isnan(val_mae):
            del model
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"Val MAE became NaN at epoch {_epoch}. "
                "Likely diverged training; check learning rate or batch_size."
            )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= params["patience"]:
                break

    if best_state is None:
        # Should not happen unless max_epochs=0; fail loudly so the caller
        # does not silently use a random-init model.
        del model
        torch.cuda.empty_cache()
        raise RuntimeError(
            f"Training never improved val MAE; best={best_val_mae}. "
            "Check learning rate / max_epochs / data."
        )

    model.load_state_dict(best_state)
    val_preds = _predict(model, val_loader, device)

    test_preds = None
    if test_smiles is not None:
        test_ds = SmilesDataset(test_smiles, tokenizer, max_length=meta["max_length"])
        test_loader = DataLoader(
            test_ds, batch_size=params["batch_size"], shuffle=False
        )
        test_preds = _predict(model, test_loader, device)

    del model
    torch.cuda.empty_cache()

    return val_preds, test_preds
