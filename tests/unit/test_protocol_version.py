from llm4rec.data.readiness import NO_EXECUTION_FLAG
from llm4rec.experiments.protocol_version import freeze_protocol


def test_protocol_freeze_dry_run_writes_manifests(tmp_path):
    manifest = freeze_protocol("protocol_test", tmp_path, dry_run=True)
    assert manifest[NO_EXECUTION_FLAG] is True
    assert manifest["status"] == "DRY_RUN_ONLY"
    assert (tmp_path / "protocol_manifest.json").is_file()
    assert (tmp_path / "frozen_split_manifest.json").is_file()
    assert (tmp_path / "frozen_candidate_manifest.json").is_file()
