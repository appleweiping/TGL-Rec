import json
import subprocess
import sys
from pathlib import Path


def test_amazon_sample_conversion_smoke(tmp_path):
    root = Path(__file__).resolve().parents[2]
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
    config = tmp_path / "sampled.yaml"
    config.write_text(
        f"""
dataset:
  source_config: {source.as_posix()}
  sample:
    max_interactions_per_domain: 5
  paths:
    interactions: {(tmp_path / 'out' / 'interactions.jsonl').as_posix()}
    items: {(tmp_path / 'out' / 'items.jsonl').as_posix()}
    conversion_report: {(tmp_path / 'out' / 'conversion_report.json').as_posix()}
""",
        encoding="utf-8",
    )
    subprocess.run([sys.executable, "scripts/check_amazon_schema.py", "--config", str(source)], cwd=root, check=True)
    subprocess.run([sys.executable, "scripts/prepare_amazon_multidomain.py", "--config", str(config), "--materialize"], cwd=root, check=True)
    assert (tmp_path / "out" / "interactions.jsonl").is_file()
    assert (tmp_path / "out" / "items.jsonl").is_file()
