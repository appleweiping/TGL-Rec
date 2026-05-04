from pathlib import Path

from llm4rec.trainers.lora_sft import train_lora_8b


def test_lora_8b_dry_run_writes_failure_report(tmp_path: Path):
    config = tmp_path / "cfg.yaml"
    config.write_text(
        f"""
model:
  base_model_path: {tmp_path / "missing-model"}
training:
  target_modules: [q_proj]
sft:
  variant: history_only_sft
training_run:
  output_dir: {tmp_path / "runs"}
""",
        encoding="utf-8",
    )

    result = train_lora_8b(config, dry_run=True)

    assert result["status"] in { "blocked", "dry_run" }
    assert (tmp_path / "runs" / "history_only_sft" / "failure_report.json").is_file()
