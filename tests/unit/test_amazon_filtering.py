import json

from llm4rec.data.amazon_filtering import filter_amazon_multidomain
from llm4rec.io.artifacts import read_jsonl


def test_amazon_filtering_materializes_without_modifying_raw(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    interactions = source / "interactions.jsonl"
    items = source / "items.jsonl"
    interactions.write_text(
        "\n".join(
            [
                json.dumps({"domain": "Beauty", "item_id": "I1", "rating": 5, "timestamp": 1, "user_id": "U1"}),
                json.dumps({"domain": "Beauty", "item_id": "I2", "rating": 4, "timestamp": 2, "user_id": "U1"}),
                json.dumps({"domain": "Beauty", "item_id": "I1", "rating": 3, "timestamp": 3, "user_id": "U2"}),
                json.dumps({"domain": "Beauty", "item_id": "I2", "rating": 2, "timestamp": 4, "user_id": "U2"}),
                json.dumps({"domain": "Beauty", "item_id": "I1", "rating": 1, "timestamp": 5, "user_id": "U3"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    items.write_text(
        "\n".join(
            [
                json.dumps({"domain": "Beauty", "item_id": "I1", "title": "One"}),
                json.dumps({"domain": "Beauty", "item_id": "I2", "title": "Two"}),
                json.dumps({"domain": "Beauty", "item_id": "I3", "title": "Three"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    before_interactions = interactions.read_text(encoding="utf-8")
    before_items = items.read_text(encoding="utf-8")
    config = tmp_path / "filter.yaml"
    out = tmp_path / "processed"
    config.write_text(
        f"""
dataset:
  name: filtered
  min_user_interactions: 2
  min_item_interactions: 2
  paths:
    source_interactions: {interactions.as_posix()}
    source_items: {items.as_posix()}
    interactions: {(out / 'interactions.jsonl').as_posix()}
    items: {(out / 'items.jsonl').as_posix()}
    filtering_report: {(out / 'filtering_report.json').as_posix()}
filtering:
  strategy: iterative_k_core
  user_min_interactions: 2
  item_min_interactions: 2
""",
        encoding="utf-8",
    )

    report = filter_amazon_multidomain(config, materialize=True)

    assert report["status"] == "MATERIALIZED"
    assert report["raw_files_unchanged"] is True
    assert interactions.read_text(encoding="utf-8") == before_interactions
    assert items.read_text(encoding="utf-8") == before_items
    assert report["output_users"] == 2
    assert report["output_items"] == 2
    assert report["output_interactions"] == 4
    assert report["users_still_below_threshold"] == 0
    assert report["items_still_below_threshold"] == 0
    assert len(read_jsonl(out / "interactions.jsonl")) == 4
    assert len(read_jsonl(out / "items.jsonl")) == 2
