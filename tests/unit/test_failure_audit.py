from llm4rec.evaluation.failure_audit import audit_failures
from llm4rec.io.artifacts import write_csv_rows


def test_failure_audit_categorizes_skipped_dependency(tmp_path):
    write_csv_rows(
        tmp_path / "method_status.csv",
        [
            {"method": "random", "status": "succeeded", "message": ""},
            {"method": "sasrec", "status": "skipped", "message": "PyTorch unavailable"},
        ],
    )
    report = audit_failures(tmp_path)
    assert report["failure_count"] == 1
    assert report["failure_categories"]["skipped_dependency"] == 1
    assert (tmp_path / "failure_report.json").is_file()
