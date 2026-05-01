"""LoRA/QLoRA dry-run planning without model downloads or GPU requirements."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.experiments.config import load_yaml_config, resolve_path
from llm4rec.io.artifacts import ensure_dir, write_json


REQUIRED_LORA_FIELDS = [
    "base_model",
    "adapter_output_dir",
    "quantization",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "target_modules",
    "max_seq_length",
    "batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "num_train_epochs",
    "save_steps",
]


class LoRAConfigError(ValueError):
    """Raised for invalid LoRA dry-run configs."""


def validate_lora_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate LoRA/QLoRA fields without touching model weights."""

    training = dict(config.get("training", config.get("lora", config)))
    missing = [field for field in REQUIRED_LORA_FIELDS if field not in training]
    if missing:
        raise LoRAConfigError(f"Missing LoRA fields: {missing}")
    if training["quantization"] not in {"none", "4bit", "8bit"}:
        raise LoRAConfigError("quantization must be one of: none, 4bit, 8bit")
    if int(training["lora_r"]) <= 0 or int(training["lora_alpha"]) <= 0:
        raise LoRAConfigError("lora_r and lora_alpha must be positive")
    if not isinstance(training["target_modules"], list) or not training["target_modules"]:
        raise LoRAConfigError("target_modules must be a non-empty list")
    for field in ("max_seq_length", "batch_size", "gradient_accumulation_steps", "num_train_epochs", "save_steps"):
        if int(training[field]) <= 0:
            raise LoRAConfigError(f"{field} must be positive")
    if float(training["learning_rate"]) <= 0:
        raise LoRAConfigError("learning_rate must be positive")
    return training


def build_lora_training_plan(config_path: str | Path, *, output_dir: str | Path | None = None) -> Path:
    """Validate config and write a dry-run training_plan.json."""

    config = load_yaml_config(config_path)
    training = validate_lora_config(config)
    adapter_dir = resolve_path(output_dir or training["adapter_output_dir"])
    ensure_dir(adapter_dir)
    plan = {
        "dry_run": True,
        "gpu_required_for_this_command": False,
        "large_model_download_performed": False,
        "status": "planned_only",
        "training": training,
    }
    path = adapter_dir / "training_plan.json"
    write_json(path, plan)
    return path
