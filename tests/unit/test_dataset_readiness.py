from llm4rec.data.readiness import NO_EXECUTION_FLAG, check_dataset_readiness


def test_dataset_readiness_missing_is_structured(tmp_path):
    output = tmp_path / "readiness.json"
    report = check_dataset_readiness("configs/datasets/movielens_full.yaml", output)
    assert output.is_file()
    assert report[NO_EXECUTION_FLAG] is True
    assert report["status"] in {"MISSING", "PARTIAL", "READY"}
    assert "counts" in report
