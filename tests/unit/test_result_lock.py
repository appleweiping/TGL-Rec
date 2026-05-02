import pytest

from llm4rec.evaluation.result_lock import ResultLockError, lock_results


def test_result_lock_refuses_incomplete_run(tmp_path):
    with pytest.raises(ResultLockError, match="incomplete"):
        lock_results(tmp_path)


def test_result_lock_accepts_complete_run(tmp_path):
    for name in ["resolved_config.yaml", "predictions.jsonl", "metrics.json", "metrics.csv"]:
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    manifest = lock_results(tmp_path)
    assert manifest["locked"] is True
