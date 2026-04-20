"""Backbone registry for PEFT fine-tuning.

Each entry describes a Hugging Face model that can be wrapped with peft
(LoRA / adapter / etc.). The registry is intentionally pure data: no
torch / transformers / peft imports here so callers can read metadata
without paying the model-loading cost.

Adding a new backbone:
1. Append a new entry below.
2. Verify the LoRA target_modules names by inspecting
   ``dict(AutoModel.from_pretrained(hf_id).named_modules()).keys()``
   once during smoke test, and update lora_target_modules_* if needed.
"""

BACKBONES: dict[str, dict] = {
    "molformer_xl": {
        "hf_id": "ibm/MoLFormer-XL-both-10pct",
        "hidden_dim": 768,
        "max_length": 202,
        "trust_remote_code": True,
        # LoRA target submodule name fragments. peft matches these as
        # substrings against module.named_modules() keys, so partial names
        # are fine. These BERT-style names are placeholders -- verify and
        # update during smoke test (see Task 5).
        "lora_target_modules_qv": ["query", "value"],
        "lora_target_modules_qkvo": ["query", "key", "value", "dense"],
    },
}


def get_backbone(name: str) -> dict:
    """Return the backbone metadata dict, or raise KeyError with a helpful list."""
    if name not in BACKBONES:
        available = ", ".join(sorted(BACKBONES))
        raise KeyError(f"Unknown backbone '{name}'. Available: {available}")
    return BACKBONES[name]
