"""GPU and dependency readiness checks for local 8B LoRA/QLoRA."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from llm4rec.experiments.config import load_yaml_config, resolve_path
from llm4rec.io.artifacts import ensure_dir, write_json
from llm4rec.trainers.lora_config import LoRA8BConfigError, load_lora_8b_sections


def check_lora_readiness(config_path: str | Path, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Check local model path, CUDA, VRAM, and optional LoRA dependencies."""

    config = load_yaml_config(config_path)
    report: dict[str, Any] = {
        "base_model_path_exists": False,
        "bitsandbytes_available": importlib.util.find_spec("bitsandbytes") is not None,
        "cuda_available": False,
        "feasible": False,
        "flash_attn_available": importlib.util.find_spec("flash_attn") is not None,
        "peft_available": importlib.util.find_spec("peft") is not None,
        "transformers_available": importlib.util.find_spec("transformers") is not None,
    }
    try:
        model_config, training_config = load_lora_8b_sections(config)
        model_path = resolve_path(model_config.base_model_path)
        report["base_model_path"] = str(model_path)
        report["base_model_path_exists"] = model_path.exists()
        report["use_qlora"] = training_config.use_qlora
        report["max_seq_length"] = model_config.max_seq_length
        report["recommended_batch_size"] = training_config.per_device_train_batch_size
        report["estimated_model_memory_gb"] = 5.0 if training_config.load_in_4bit else 16.0
        report["estimated_optimizer_adapter_memory_gb"] = 2.0
        report["qlora_required"] = True
    except LoRA8BConfigError as exc:
        report["config_error"] = str(exc)
        _write_readiness_report(config, report, output_dir)
        return report

    try:
        import torch

        report["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            report["gpu_name"] = torch.cuda.get_device_name(index)
            report["vram_free_gb"] = free_bytes / 1024**3
            report["vram_total_gb"] = total_bytes / 1024**3
        else:
            report["gpu_name"] = ""
            report["vram_free_gb"] = 0.0
            report["vram_total_gb"] = 0.0
    except Exception as exc:  # pragma: no cover - environment-specific.
        report["torch_error"] = str(exc)
        report["gpu_name"] = ""
        report["vram_free_gb"] = 0.0
        report["vram_total_gb"] = 0.0

    blockers: list[str] = []
    if not report["base_model_path_exists"]:
        blockers.append("base_model_path_missing")
    if not report["cuda_available"]:
        blockers.append("cuda_unavailable")
    if not report["transformers_available"]:
        blockers.append("transformers_missing")
    if not report["peft_available"]:
        blockers.append("peft_missing")
    if training_config.use_qlora and not report["bitsandbytes_available"]:
        blockers.append("bitsandbytes_missing_for_qlora")
    if model_config.use_flash_attention and not report["flash_attn_available"]:
        blockers.append("flash_attn_missing")
    if report.get("vram_total_gb", 0.0) and report["vram_total_gb"] < report["estimated_model_memory_gb"]:
        blockers.append("insufficient_vram")
    report["blockers"] = blockers
    report["feasible"] = not blockers
    report["recommendation"] = _recommendation(report)
    _write_readiness_report(config, report, output_dir)
    return report


def _recommendation(report: dict[str, Any]) -> str:
    blockers = set(report.get("blockers", []))
    if "base_model_path_missing" in blockers:
        return "Set model.base_model_path to an existing local 8B model directory. No download was attempted."
    if "cuda_unavailable" in blockers:
        return "Dry-run only: CUDA is unavailable, do not start training."
    if "bitsandbytes_missing_for_qlora" in blockers:
        return "Install bitsandbytes or disable QLoRA only if VRAM is sufficient."
    if "insufficient_vram" in blockers:
        return "Use QLoRA, lower max_seq_length, reduce batch size, or increase gradient accumulation."
    if blockers:
        return "Resolve blockers before training."
    return "Ready for controlled local LoRA/QLoRA training."


def _write_readiness_report(config: dict[str, Any], report: dict[str, Any], output_dir: str | Path | None) -> None:
    target = output_dir or config.get("readiness", {}).get(
        "output_dir",
        "outputs/paper_runs/protocol_v1/lora_8b/readiness",
    )
    out_dir = ensure_dir(resolve_path(target))
    write_json(out_dir / "lora_readiness.json", report)
