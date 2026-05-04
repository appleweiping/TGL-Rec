from pathlib import Path

from llm4rec.experiments.deepseek_llm import run_deepseek_matrix
from llm4rec.io.artifacts import write_jsonl


def test_deepseek_dry_run_writes_required_artifacts(tmp_path: Path):
    split = tmp_path / "splits.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    write_jsonl(
        split,
        [
            {"user_id": "u1", "item_id": "i1", "timestamp": 1, "split": "train", "domain": "movielens"},
            {"user_id": "u1", "item_id": "i2", "timestamp": 2, "split": "test", "domain": "movielens"},
        ],
    )
    write_jsonl(
        candidates,
        [
            {
                "candidate_items": ["i1", "i2", "i3"],
                "candidate_protocol": "sampled",
                "split": "test",
                "target_item": "i2",
                "user_id": "u1",
            }
        ],
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
llm:
  provider: deepseek
  base_url: https://api.deepseek.com
  model: deepseek-v4-flash
  api_key_env: DEEPSEEK_API_KEY
  max_tokens: 16
datasets: [movielens_full]
dataset_artifacts:
  movielens_full:
    split_artifact: {split}
    candidate_artifact: {candidates}
selection:
  top_m_candidates_for_llm: 2
prompt_variants: [history_only]
experiment:
  dry_run_output_dir: {tmp_path / "dry_run"}
  dry_run:
    max_users_per_dataset: 1
    max_requests: 1
""",
        encoding="utf-8",
    )

    result = run_deepseek_matrix(config, dry_run=True)

    run_dir = Path(result["run_dir"])
    assert (run_dir / "cost_estimate.json").is_file()
    assert (run_dir / "planned_requests.jsonl").is_file()
    assert (run_dir / "prompt_length_report.csv").is_file()
    assert (run_dir / "target_inclusion_audit.csv").is_file()
    assert (run_dir / "api_safety_report.json").is_file()
