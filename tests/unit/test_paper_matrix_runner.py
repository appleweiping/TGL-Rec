import json
from pathlib import Path

from llm4rec.experiments.paper_matrix import (
    DatasetBundle,
    _candidate_items_for_row,
    _assert_output_dir,
    normalize_method,
    sha256_file,
)


def test_normalize_method_uses_phase9b_directory_name():
    assert normalize_method("mf") == "mf_bpr"
    assert normalize_method("bpr") == "mf_bpr"
    assert normalize_method("sasrec") == "sasrec"


def test_shared_pool_candidate_expansion_includes_outside_target(tmp_path: Path):
    pool = {
        "candidate_items": ["i1", "i2", "i3"],
        "negative_pool_for_targets_outside_pool": ["i1", "i2"],
    }
    bundle = DatasetBundle(
        name="tiny",
        config_path=tmp_path / "config.yaml",
        split_artifact=tmp_path / "splits.jsonl",
        candidate_artifact=tmp_path / "candidates.jsonl",
        candidate_pool_artifact=tmp_path / "candidate_pool.json",
        candidate_protocol="fixed_sampled",
        split_strategy="leave_one_out",
        train_rows=[],
        item_rows=[],
        item_catalog=set(),
        history_by_user={},
        test_timestamp_by_user={},
        item_popularity={},
        long_tail=set(),
        candidate_pool=pool,
        artifact_checksums={},
    )
    row = {"candidate_size": 3, "candidate_storage": "shared_pool", "target_item": "i9"}

    assert _candidate_items_for_row(bundle, row) == ["i1", "i2", "i9"]


def test_sha256_file_is_stable(tmp_path: Path):
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps({"a": 1}, sort_keys=True), encoding="utf-8")

    assert sha256_file(path) == sha256_file(path)


def test_output_dir_guard_accepts_protocol_v1_outputs():
    _assert_output_dir(Path("outputs/paper_runs/protocol_v1/main_accuracy_seed0"))
