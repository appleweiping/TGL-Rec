from llm4rec.experiments.runner import run_experiment


def test_sequential_markov_smoke_outputs_schema(tmp_path):
    config = tmp_path / "seq.yaml"
    config.write_text(
        f"""
experiment:
  run_id: seq_smoke
  output_dir: {tmp_path.as_posix()}
  overwrite: true
  run_mode: smoke
  seed: 2026
dataset:
  config_path: configs/datasets/tiny.yaml
  preprocess: true
methods: [configs/baselines/markov_transition.yaml]
evaluation:
  config_path: configs/evaluation/default.yaml
""",
        encoding="utf-8",
    )
    result = run_experiment(config)
    assert (result.run_dir / "predictions.jsonl").is_file()
    assert "markov_transition" in result.metrics["by_method"]
