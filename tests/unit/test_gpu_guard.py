from pathlib import Path

from llm4rec.trainers.gpu_guard import check_lora_readiness


def test_gpu_guard_reports_missing_model_path(tmp_path: Path):
    config = tmp_path / "lora.yaml"
    config.write_text(
        f"""
model:
  base_model_path: {tmp_path / "missing-model"}
training:
  target_modules: [q_proj]
readiness:
  output_dir: {tmp_path / "readiness"}
""",
        encoding="utf-8",
    )

    report = check_lora_readiness(config)

    assert report["base_model_path_exists"] is False
    assert "base_model_path_missing" in report["blockers"]
    assert (tmp_path / "readiness" / "lora_readiness.json").is_file()
