import json

from llm4rec.data.amazon_converter import prepare_amazon_multidomain
from llm4rec.io.artifacts import read_jsonl


def test_amazon_converter_materializes_sample(tmp_path):
    raw = tmp_path / "beauty"
    raw.mkdir()
    (raw / "All_Beauty.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"rating": 5, "parent_asin": "I1", "timestamp": 100, "user_id": "U1"}),
                json.dumps({"rating": 4, "parent_asin": "I1", "timestamp": 100, "user_id": "U1"}),
                json.dumps({"rating": 3, "parent_asin": "I2", "timestamp": None, "user_id": "U2"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (raw / "meta_All_Beauty.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"parent_asin": "I1", "title": "Item 1"}),
                json.dumps({"parent_asin": "I1", "title": "Item 1 duplicate"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source = tmp_path / "source.yaml"
    source.write_text(
        f"""
dataset:
  raw_domains:
    Beauty: {raw.as_posix()}
""",
        encoding="utf-8",
    )
    config = tmp_path / "sampled.yaml"
    interactions = tmp_path / "out" / "interactions.jsonl"
    items = tmp_path / "out" / "items.jsonl"
    report_path = tmp_path / "out" / "conversion_report.json"
    config.write_text(
        f"""
dataset:
  source_config: {source.as_posix()}
  sample:
    max_interactions_per_domain: 10
  paths:
    interactions: {interactions.as_posix()}
    items: {items.as_posix()}
    conversion_report: {report_path.as_posix()}
""",
        encoding="utf-8",
    )
    report = prepare_amazon_multidomain(config, materialize=True)
    assert report["summary"]["valid_interactions"] == 1
    assert report["summary"]["dropped_rows"] == 1
    assert report["summary"]["duplicate_interactions_removed"] == 1
    assert len(read_jsonl(interactions)) == 1
    assert len(read_jsonl(items)) == 1


def test_amazon_converter_preflight_writes_plan(tmp_path):
    raw = tmp_path / "beauty"
    raw.mkdir()
    (raw / "All_Beauty.jsonl").write_text(
        json.dumps({"rating": 5, "parent_asin": "I1", "timestamp": 100, "user_id": "U1"}) + "\n",
        encoding="utf-8",
    )
    (raw / "meta_All_Beauty.jsonl").write_text(
        json.dumps({"parent_asin": "I1", "title": "Item 1"}) + "\n",
        encoding="utf-8",
    )
    source = tmp_path / "source.yaml"
    source.write_text(
        f"""
dataset:
  raw_domains:
    Beauty: {raw.as_posix()}
""",
        encoding="utf-8",
    )
    config = tmp_path / "full.yaml"
    preflight = tmp_path / "out" / "conversion_preflight.json"
    config.write_text(
        f"""
dataset:
  source_config: {source.as_posix()}
  paths:
    interactions: {(tmp_path / 'out' / 'interactions.jsonl').as_posix()}
    items: {(tmp_path / 'out' / 'items.jsonl').as_posix()}
    conversion_preflight: {preflight.as_posix()}
    schema_report: {(tmp_path / 'out' / 'schema_report.json').as_posix()}
""",
        encoding="utf-8",
    )
    report = prepare_amazon_multidomain(config, preflight=True)
    assert report["conversion_mode"] == "full"
    assert report["estimated_input_review_rows"] >= 1
    assert report["not_an_experiment_warning"]
    assert preflight.is_file()
