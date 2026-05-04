"""Configuration validation for local 8B LoRA/QLoRA experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class LoRA8BConfigError(ValueError):
    """Raised when a LoRA/QLoRA config is incomplete or unsafe."""


@dataclass(frozen=True)
class Local8BModelConfig:
    """Local base model configuration. No automatic downloads are allowed."""

    base_model_path: str
    model_family: str = "other"
    tokenizer_path: str | None = None
    trust_remote_code: bool = False
    torch_dtype: str = "auto"
    device_map: str = "auto"
    max_seq_length: int = 2048
    use_flash_attention: bool = False
    gradient_checkpointing: bool = True

    @property
    def resolved_model_path(self) -> Path:
        return Path(self.base_model_path).expanduser()


@dataclass(frozen=True)
class LoRA8BTrainingConfig:
    """PEFT LoRA/QLoRA and SFT training settings."""

    use_qlora: bool = True
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    num_train_epochs: float = 1.0
    max_steps: int | None = None
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int = 2
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    seed: int = 2026


def load_lora_8b_sections(config: dict[str, Any]) -> tuple[Local8BModelConfig, LoRA8BTrainingConfig]:
    """Validate nested model/training sections and return typed configs."""

    model_section = dict(config.get("model", config.get("llm", {})))
    training_section = dict(config.get("training", {}))
    if not model_section.get("base_model_path"):
        raise LoRA8BConfigError("model.base_model_path is required and must point to a local 8B model.")
    model = Local8BModelConfig(
        base_model_path=str(model_section["base_model_path"]),
        model_family=str(model_section.get("model_family", "other")),
        tokenizer_path=model_section.get("tokenizer_path"),
        trust_remote_code=bool(model_section.get("trust_remote_code", False)),
        torch_dtype=str(model_section.get("torch_dtype", "auto")),
        device_map=str(model_section.get("device_map", "auto")),
        max_seq_length=int(model_section.get("max_seq_length", 2048)),
        use_flash_attention=bool(model_section.get("use_flash_attention", False)),
        gradient_checkpointing=bool(model_section.get("gradient_checkpointing", True)),
    )
    training = LoRA8BTrainingConfig(
        use_qlora=bool(training_section.get("use_qlora", True)),
        load_in_4bit=bool(training_section.get("load_in_4bit", True)),
        load_in_8bit=bool(training_section.get("load_in_8bit", False)),
        bnb_4bit_quant_type=str(training_section.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_compute_dtype=str(training_section.get("bnb_4bit_compute_dtype", "bfloat16")),
        lora_r=int(training_section.get("lora_r", 16)),
        lora_alpha=int(training_section.get("lora_alpha", 32)),
        lora_dropout=float(training_section.get("lora_dropout", 0.05)),
        target_modules=[str(value) for value in training_section.get("target_modules", [])],
        bias=str(training_section.get("bias", "none")),
        task_type=str(training_section.get("task_type", "CAUSAL_LM")),
        per_device_train_batch_size=int(training_section.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(training_section.get("gradient_accumulation_steps", 16)),
        learning_rate=float(training_section.get("learning_rate", 2e-4)),
        weight_decay=float(training_section.get("weight_decay", 0.0)),
        warmup_ratio=float(training_section.get("warmup_ratio", 0.03)),
        num_train_epochs=float(training_section.get("num_train_epochs", 1.0)),
        max_steps=training_section.get("max_steps"),
        logging_steps=int(training_section.get("logging_steps", 10)),
        save_steps=int(training_section.get("save_steps", 200)),
        eval_steps=int(training_section.get("eval_steps", 200)),
        save_total_limit=int(training_section.get("save_total_limit", 2)),
        bf16=bool(training_section.get("bf16", True)),
        fp16=bool(training_section.get("fp16", False)),
        gradient_checkpointing=bool(training_section.get("gradient_checkpointing", True)),
        max_grad_norm=float(training_section.get("max_grad_norm", 1.0)),
        seed=int(training_section.get("seed", 2026)),
    )
    validate_lora_8b_config(model, training)
    return model, training


def validate_lora_8b_config(model: Local8BModelConfig, training: LoRA8BTrainingConfig) -> None:
    """Validate safe and meaningful LoRA settings."""

    if model.max_seq_length <= 0:
        raise LoRA8BConfigError("model.max_seq_length must be positive.")
    if model.trust_remote_code not in {True, False}:
        raise LoRA8BConfigError("model.trust_remote_code must be boolean.")
    if training.load_in_4bit and training.load_in_8bit:
        raise LoRA8BConfigError("load_in_4bit and load_in_8bit cannot both be true.")
    if training.lora_r <= 0 or training.lora_alpha <= 0:
        raise LoRA8BConfigError("lora_r and lora_alpha must be positive.")
    if not 0.0 <= training.lora_dropout < 1.0:
        raise LoRA8BConfigError("lora_dropout must be in [0, 1).")
    if not training.target_modules:
        raise LoRA8BConfigError("target_modules must be non-empty.")
    if training.per_device_train_batch_size <= 0 or training.gradient_accumulation_steps <= 0:
        raise LoRA8BConfigError("batch size and gradient accumulation must be positive.")
    if training.learning_rate <= 0:
        raise LoRA8BConfigError("learning_rate must be positive.")
    if training.max_steps is not None:
        max_steps = int(training.max_steps)
        if max_steps <= 0:
            raise LoRA8BConfigError("max_steps must be positive when set.")
