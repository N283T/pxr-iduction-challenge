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
        # are fine. Verified during smoke test (Task 5): MoLFormer-XL uses
        # standard BERT-style naming (Case A -- query/key/value/dense).
        "lora_target_modules_qv": ["query", "value"],
        "lora_target_modules_qkvo": ["query", "key", "value", "dense"],
        # MoLFormer-XL ships with corrupted rotary-embedding inv_freq buffers
        # when loaded via transformers >= v5 (see issue #30). Set this flag so
        # the trainer recomputes inv_freq + cos/sin cache after from_pretrained.
        "fix_rotary": True,
    },
    "molformer_c3_1_1b": {
        "hf_id": "DeepChem/MoLFormer-c3-1.1B",
        "hidden_dim": 768,
        "max_length": 202,
        "trust_remote_code": True,
        # Architecture identical to ibm/MoLFormer-XL-both-10pct (verified
        # config.json). The "1.1B" refers to pretrain token count; actual
        # model is ~80M params. HF auto_map references ibm modeling code,
        # so trust_remote_code=True pulls from the ibm repo.
        "lora_target_modules_qv": ["query", "value"],
        "lora_target_modules_qkvo": ["query", "key", "value", "dense"],
        # Same rotary embedding bug as ibm/MoLFormer-XL (inherited
        # architecture); PeftRegressor recomputes inv_freq + cos/sin cache.
        "fix_rotary": True,
    },
    "chemberta_5m_mtr": {
        "hf_id": "DeepChem/ChemBERTa-5M-MTR",
        "hidden_dim": 384,
        "max_length": 202,
        "trust_remote_code": False,
        # RoBERTa-3L architecture, ~5M params. Phase B1 audit
        # (2026-04-24): best raw-embedding candidate among 9 BERT-family
        # tables -- single-model OOF MAE 0.5287 (pool weakest + 0.043,
        # passes gate 2), min residual r 0.77 vs 9-pool (passes gate 1).
        # But caruana ADD Δ only -0.0020 (below the -0.003 threshold
        # tightened after the tier-0 LB regression). Phase B = continued
        # pretrain on log2fc via LoRA to adapt the embedding to the
        # task before retesting.
        "lora_target_modules_qv": ["query", "value"],
        "lora_target_modules_qkvo": ["query", "key", "value", "dense"],
        "fix_rotary": False,
    },
}


def get_backbone(name: str) -> dict:
    """Return the backbone metadata dict, or raise KeyError with a helpful list."""
    if name not in BACKBONES:
        available = ", ".join(sorted(BACKBONES))
        raise KeyError(f"Unknown backbone '{name}'. Available: {available}")
    return BACKBONES[name]
