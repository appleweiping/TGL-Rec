"""Local 8B LoRA/QLoRA SFT trainer with explicit readiness gates."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

from llm4rec.experiments.config import load_yaml_config, resolve_path, save_resolved_config
from llm4rec.io.artifacts import ensure_dir, write_json
from llm4rec.trainers.gpu_guard import check_lora_readiness
from llm4rec.trainers.lora_config import load_lora_8b_sections
from llm4rec.utils.env import collect_environment


def train_lora_8b(
    config_path: str | Path,
    *,
    dry_run: bool = False,
    allow_scaffold: bool = False,
) -> dict[str, Any]:
    """Train a local 8B adapter only when all local prerequisites are present."""

    config = load_yaml_config(config_path)
    _guard_scaffold_training(config, allow_scaffold=allow_scaffold)
    variant = str(config.get("sft", {}).get("variant", "history_only_sft"))
    output_dir = ensure_dir(resolve_path(config["training_run"]["output_dir"]) / variant)
    save_resolved_config(config, output_dir / "resolved_config.yaml")
    write_json(output_dir / "environment.json", collect_environment(resolve_path(".")))
    readiness = check_lora_readiness(config_path, output_dir=output_dir)
    write_json(output_dir / "training_args.json", dict(config.get("training", {})))
    if dry_run or not readiness.get("feasible", False):
        report = {
            "adapter_saved": False,
            "blockers": readiness.get("blockers", []),
            "dry_run": dry_run,
            "status": "blocked" if not readiness.get("feasible", False) else "dry_run",
            "variant": variant,
        }
        write_json(output_dir / "failure_report.json", report)
        _append_log(output_dir, f"training not started: {report}")
        if not dry_run:
            raise RuntimeError(f"LoRA training blocked: {report['blockers']}")
        return report
    started = time.perf_counter()
    model_config, training_config = load_lora_8b_sections(config)
    try:
        _run_transformers_training(config, model_config, training_config, output_dir)
    except Exception as exc:
        report = {"adapter_saved": False, "error": str(exc), "status": "failed", "variant": variant}
        write_json(output_dir / "failure_report.json", report)
        _append_log(output_dir, f"training failed: {exc}")
        raise
    runtime = time.perf_counter() - started
    manifest = {
        "adapter_path": str(output_dir / "adapter"),
        "base_model_weights_saved": False,
        "runtime_seconds": runtime,
        "status": "succeeded",
        "variant": variant,
    }
    write_json(output_dir / "checkpoint_manifest.json", manifest)
    return manifest


def _guard_scaffold_training(config: dict[str, Any], *, allow_scaffold: bool) -> None:
    contract = dict(config.get("baseline_contract", {}))
    sft = dict(config.get("sft", {}))
    if not (contract.get("scaffold_only") or sft.get("scaffold_only")):
        return
    if allow_scaffold:
        return
    raise RuntimeError(
        "Refusing to train a scaffold reference baseline config. "
        "Use an official-code adapter config for reportable baselines, or pass "
        "--allow-scaffold for an explicitly non-reportable interface smoke run."
    )


def _run_transformers_training(
    config: dict[str, Any],
    model_config: Any,
    training_config: Any,
    output_dir: Path,
) -> None:
    """Run optional HF/PEFT training. Imports stay local so tests do not require these packages."""

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    seed = int(config.get("training", {}).get("seed", 2026))
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    sft_dir = resolve_path(config["sft"]["data_dir"])
    train_rows = [json.loads(line) for line in (sft_dir / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    valid_rows = [json.loads(line) for line in (sft_dir / "valid.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.tokenizer_path or model_config.base_model_path,
        trust_remote_code=model_config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = None
    if training_config.load_in_4bit or training_config.load_in_8bit:
        compute_dtype = getattr(torch, training_config.bnb_4bit_compute_dtype)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=training_config.load_in_4bit,
            load_in_8bit=training_config.load_in_8bit,
            bnb_4bit_quant_type=training_config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    model_kwargs = {
        "device_map": _training_device_map(model_config.device_map),
        "trust_remote_code": model_config.trust_remote_code,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = model_config.torch_dtype
    model = AutoModelForCausalLM.from_pretrained(
        model_config.base_model_path,
        **model_kwargs,
    )
    if training_config.use_qlora:
        model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=training_config.lora_r,
        lora_alpha=training_config.lora_alpha,
        lora_dropout=training_config.lora_dropout,
        target_modules=training_config.target_modules,
        bias=training_config.bias,
        task_type=training_config.task_type,
    )
    model = get_peft_model(model, peft_config)
    train_dataset = Dataset.from_list([_tokenize_sft(row, tokenizer, model_config.max_seq_length) for row in train_rows])
    eval_dataset = Dataset.from_list([_tokenize_sft(row, tokenizer, model_config.max_seq_length) for row in valid_rows])
    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        warmup_ratio=training_config.warmup_ratio,
        num_train_epochs=training_config.num_train_epochs,
        max_steps=-1 if training_config.max_steps is None else int(training_config.max_steps),
        logging_steps=training_config.logging_steps,
        save_steps=training_config.save_steps,
        eval_steps=training_config.eval_steps,
        save_total_limit=training_config.save_total_limit,
        bf16=training_config.bf16,
        fp16=training_config.fp16,
        gradient_checkpointing=training_config.gradient_checkpointing,
        max_grad_norm=training_config.max_grad_norm,
        report_to=[],
        seed=seed,
        data_seed=seed,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100, padding=True)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    result = trainer.train()
    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    write_json(output_dir / "training_metrics.json", result.metrics)


def _training_device_map(device_map: str) -> str | dict[str, int]:
    """Keep LoRA training on the visible GPU instead of CPU/GPU sharding."""

    normalized = str(device_map).strip().lower()
    if normalized in {"auto", "balanced", "balanced_low_0", "sequential"}:
        return {"": 0}
    return device_map


def _tokenize_sft(row: dict[str, Any], tokenizer: Any, max_seq_length: int) -> dict[str, Any]:
    prefix, assistant = _split_sft_text(row)
    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token:
        assistant = f"{assistant}{eos_token}"
    full_text = f"{prefix}{assistant}"
    encoded = tokenizer(full_text, truncation=True, max_length=max_seq_length, padding=False)
    prefix_encoded = tokenizer(prefix, truncation=True, max_length=max_seq_length, padding=False)
    input_ids = list(encoded["input_ids"])
    labels = list(input_ids)
    prefix_length = min(len(prefix_encoded["input_ids"]), len(labels))
    for index in range(prefix_length):
        labels[index] = -100
    attention_mask = encoded.get("attention_mask")
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    for index, token_id in enumerate(input_ids):
        if attention_mask is not None and int(attention_mask[index]) == 0:
            labels[index] = -100
        elif pad_token_id is not None and token_id == pad_token_id:
            labels[index] = -100
    encoded["labels"] = labels
    return encoded


def _split_sft_text(row: dict[str, Any]) -> tuple[str, str]:
    messages = list(row["messages"])
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("SFT row must end with an assistant message")
    prefix_messages = messages[:-1]
    assistant_message = messages[-1]
    prefix = "".join(f"{msg['role']}: {msg['content']}\n" for msg in prefix_messages)
    prefix = f"{prefix}assistant: "
    assistant = str(assistant_message["content"])
    return prefix, assistant


def _append_log(output_dir: Path, message: str) -> None:
    with (output_dir / "logs.txt").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}\n")
