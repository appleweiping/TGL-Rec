import json

from llm4rec.data.artifact_freeze import plan_data_artifact_freeze


def test_materialize_data_artifacts_writes_checksummed_splits_and_candidates(tmp_path):
    interactions_path = tmp_path / "interactions.jsonl"
    items_path = tmp_path / "items.jsonl"
    split_path = tmp_path / "artifacts" / "splits.jsonl"
    candidate_path = tmp_path / "artifacts" / "candidates.jsonl"
    interactions = [
        {"user_id": "u1", "item_id": "i1", "timestamp": 1, "rating": 1.0, "domain": "tiny"},
        {"user_id": "u1", "item_id": "i2", "timestamp": 2, "rating": 1.0, "domain": "tiny"},
        {"user_id": "u1", "item_id": "i3", "timestamp": 3, "rating": 1.0, "domain": "tiny"},
        {"user_id": "u2", "item_id": "i2", "timestamp": 1, "rating": 1.0, "domain": "tiny"},
        {"user_id": "u2", "item_id": "i3", "timestamp": 2, "rating": 1.0, "domain": "tiny"},
        {"user_id": "u2", "item_id": "i4", "timestamp": 3, "rating": 1.0, "domain": "tiny"},
    ]
    items = [
        {"item_id": f"i{index}", "title": f"Item {index}", "domain": "tiny"}
        for index in range(1, 5)
    ]
    interactions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in interactions),
        encoding="utf-8",
    )
    items_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in items),
        encoding="utf-8",
    )
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        f"""
dataset:
  name: tiny_freeze
  adapter: generic_jsonl
  seed: 7
  paths:
    interactions: {interactions_path.as_posix()}
    items: {items_path.as_posix()}
  paper_artifacts:
    protocol_version: protocol_test
    split_artifact: {split_path.as_posix()}
    candidate_artifact: {candidate_path.as_posix()}
    split_protocol: leave_one_out
    candidate_protocol: fixed_sampled
    candidate_size: 3
    seed: 7
""",
        encoding="utf-8",
    )

    manifest = plan_data_artifact_freeze([config_path], tmp_path / "protocol", materialize=True)

    dataset = manifest["datasets"][0]
    assert dataset["materialized"] is True
    assert dataset["split_artifact"]["rows"] == 6
    assert dataset["split_artifact"]["sha256"]
    assert dataset["candidate_artifact"]["rows"] == 4
    assert dataset["candidate_artifact"]["candidate_size"] == 3
    assert dataset["candidate_artifact"]["target_included_rows"] == 4
    assert candidate_path.is_file()
    rows = [json.loads(line) for line in candidate_path.read_text(encoding="utf-8").splitlines()]
    assert all(row["target_item"] in row["candidate_items"] for row in rows)
    assert all(row["metadata"]["shared_across_methods"] is True for row in rows)
