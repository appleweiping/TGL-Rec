"""Local 8B LoRA/QLoRA SFT trainer with explicit readiness gates."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from llm4rec.experiments.config import load_yaml_config, resolve_path, save_resolved_config
from llm4rec.io.artifacts import ensure_dir, write_json
from llm4rec.trainers.gpu_guard import check_lora_readiness
from llm4rec.trainers.lora_config import load_lora_8b_sections
from llm4rec.utils.env import collect_environment


def train_lora_8b(config_path: str | Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Train a local 8B adapter only when all local prerequisites are present."""

    config = load_yaml_config(config_path)
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


def _run_transformers_training(
    config: dict[str, Any],
    model_config: Any,
    training_config: Any,
    output_dir: Path,
) -> None:
    """Run optional HF/PEFT training. Imports stay local so tests do not require these packages."""

    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    sft_dir = resolve_path(config["sft"]["data_dir"])
    train_rows = [json.loads(line) for line in (sft_dir / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    valid_rows = [json.loads(line) for line in (sft_dir / "valid.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.tokenizer_path or model_config.base_model_path,
        trust_remote_code=model_config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_config.base_model_path,
        device_map=model_config.device_map,
        trust_remote_code=model_config.trust_remote_code,
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
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset)
    result = trainer.train()
    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    write_json(output_dir / "training_metrics.json", result.metrics)


def _tokenize_sft(row: dict[str, Any], tokenizer: Any, max_seq_length: int) -> dict[str, Any]:
    text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in row["messages"])
    encoded = tokenizer(text, truncation=True, max_length=max_seq_length, padding="max_length")
    encoded["labels"] = list(encoded["input_ids"])
    return encoded


def _append_log(output_dir: Path, message: str) -> None:
    with (output_dir / "logs.txt").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}\n")
