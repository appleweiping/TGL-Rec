import pytest

from llm4rec.experiments.validate import ExperimentValidationError, validate_experiment_config


def test_pilot_reportable_true_is_rejected(tmp_path):
    config = tmp_path / "bad_pilot.yaml"
    config.write_text(
        """
manifest:
  dataset: movielens_style_sampled_pilot
  split_strategy: leave_one_out
  candidate_strategy: sampled_fixed
  methods: [random]
  seeds: [0]
  metrics: [Recall@5]
  output_dir: outputs/runs
  run_mode: pilot
  reportable: false
experiment:
  run_id: bad
  output_dir: outputs/runs
  run_mode: pilot
pilot_reportable: true
dataset:
  candidate_protocol: sampled_fixed
pilot:
  split_artifact: shared
  candidate_artifact: shared
evaluation:
  candidate_protocol: sampled_fixed
training:
  enable_lora_training: false
llm:
  allow_api_calls: false
""",
        encoding="utf-8",
    )
    with pytest.raises(ExperimentValidationError, match="pilot_reportable=false"):
        validate_experiment_config(config)


def test_pilot_api_enabled_is_rejected(tmp_path):
    config = tmp_path / "bad_api.yaml"
    config.write_text(
        """
manifest:
  dataset: movielens_style_sampled_pilot
  split_strategy: leave_one_out
  candidate_strategy: sampled_fixed
  methods: [random]
  seeds: [0]
  metrics: [Recall@5]
  output_dir: outputs/runs
  run_mode: pilot
  reportable: false
experiment:
  run_id: bad
  output_dir: outputs/runs
  run_mode: pilot
pilot_reportable: false
dataset:
  candidate_protocol: sampled_fixed
pilot:
  split_artifact: shared
  candidate_artifact: shared
evaluation:
  candidate_protocol: sampled_fixed
training:
  enable_lora_training: false
llm:
  allow_api_calls: true
""",
        encoding="utf-8",
    )
    with pytest.raises(ExperimentValidationError, match="API"):
        validate_experiment_config(config)
