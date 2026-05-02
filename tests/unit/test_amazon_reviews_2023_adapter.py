import json

from llm4rec.data.amazon_reviews_2023 import inspect_amazon_reviews_2023


def test_amazon_reviews_2023_inspection_detects_candidates(tmp_path):
    domain_dir = tmp_path / "beauty"
    domain_dir.mkdir()
    (domain_dir / "All_Beauty.jsonl").write_text(
        json.dumps({"rating": 5, "parent_asin": "I1", "timestamp": 100, "user_id": "U1"}) + "\n",
        encoding="utf-8",
    )
    (domain_dir / "meta_All_Beauty.jsonl").write_text(
        json.dumps({"parent_asin": "I1", "title": "Item 1", "store": "Brand"}) + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "amazon.yaml"
    config.write_text(
        f"""
dataset:
  name: amazon_reviews_2023
  adapter: amazon_reviews_2023
  raw_domains:
    Beauty: {domain_dir.as_posix()}
""",
        encoding="utf-8",
    )
    report = inspect_amazon_reviews_2023(config, tmp_path / "schema_report.json")
    assert report["overall_status"] == "convertible"
    assert report["domains"]["Beauty"]["can_convert"] is True
    assert report["domains"]["Beauty"]["compression_format"]["reviews"] == "jsonl"
