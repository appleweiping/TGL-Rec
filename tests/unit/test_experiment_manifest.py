import pytest

from llm4rec.experiments.validate import ExperimentValidationError, validate_experiment_config


def test_experiment_manifest_validation_passes_main_accuracy():
    result = validate_experiment_config("configs/experiments/main_accuracy.yaml")
    assert result["status"] == "pass"


def test_experiment_manifest_validation_rejects_reportable_mock(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
manifest:
  dataset: tiny
  split_strategy: leave_one_out
  candidate_strategy: full_catalog
  methods: [mock]
  seeds: [1]
  metrics: [Recall@5]
  output_dir: outputs/runs/bad
  run_mode: smoke
  reportable: true
""",
        encoding="utf-8",
    )
    with pytest.raises(ExperimentValidationError):
        validate_experiment_config(path)
