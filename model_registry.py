"""VLM registry: which base checkpoint + processor/LoRA settings each model
key maps to. Add an entry here to bring a new base VLM into both train_sft.py
and train_grpo.py.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path


@dataclass
class VLMSpec:
    model_id: str
    lora_target_modules: str | list[str] = "all-linear"
    trust_remote_code: bool = True
    attn_implementation: str | None = None
    processor_kwargs: dict = field(default_factory=dict)


MODEL_REGISTRY: dict[str, VLMSpec] = {
    "qwen3.5-4b": VLMSpec(
        "Qwen/Qwen3.5-4B",
        processor_kwargs={
            "min_pixels": 256 * 32 * 32,
            "max_pixels": 2048 * 32 * 32,
        },
    ),
    "internvl3.5-4b": VLMSpec(
        "OpenGVLab/InternVL3_5-4B-HF",
        processor_kwargs={"max_patches": 6},
    ),
}


def sft_run_name(model_key: str, reasoning_format: str) -> str:
    return f"{model_key}-sft-{reasoning_format}"


def grpo_run_name(model_key: str, reasoning_format: str) -> str:
    return f"{model_key}-grpo-{reasoning_format}"


def sft_merged_dir(outputs_dir: Path, model_key: str, reasoning_format: str) -> Path:
    return outputs_dir / sft_run_name(model_key, reasoning_format) / "merged"


def spec_from_checkpoint(model_key: str, checkpoint_dir: Path) -> VLMSpec:
    """GRPO continues training from an SFT checkpoint rather than the base
    model_id, but keeps that model key's LoRA/processor settings."""
    base = MODEL_REGISTRY[model_key]
    return replace(base, model_id=str(checkpoint_dir))
