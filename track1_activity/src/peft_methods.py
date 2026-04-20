"""PEFT method registry.

Each entry maps a method name to a builder that turns hyperparameters
into a peft Config object. Only LoRA is implemented in this PR; adapter
and last-k-layer FT will be added in follow-up PRs.
"""

from typing import Callable

from peft import LoraConfig, PeftConfig  # type: ignore[import-untyped]


def build_lora_config(backbone_meta: dict, params: dict) -> LoraConfig:
    """Build a LoraConfig from hyperparameters.

    ``params`` keys:
        lora_rank: int
        lora_alpha: int
        lora_dropout: float
        lora_target: "qv" or "qkvo"
    """
    target_key = f"lora_target_modules_{params['lora_target']}"
    target_modules = backbone_meta[target_key]
    return LoraConfig(
        r=params["lora_rank"],
        lora_alpha=params["lora_alpha"],
        lora_dropout=params["lora_dropout"],
        target_modules=target_modules,
        bias="none",
        # Custom regression head -- do not let peft inject a task head.
        task_type=None,
    )


PEFT_METHODS: dict[str, Callable[[dict, dict], PeftConfig]] = {
    "lora": build_lora_config,
}


def get_peft_builder(method: str) -> Callable[[dict, dict], PeftConfig]:
    if method not in PEFT_METHODS:
        available = ", ".join(sorted(PEFT_METHODS))
        raise KeyError(f"Unknown PEFT method '{method}'. Available: {available}")
    return PEFT_METHODS[method]
