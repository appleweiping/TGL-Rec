import json
import subprocess
import sys
from pathlib import Path


def test_amazon_filtered_readiness_smoke(tmp_path):
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "raw"
    out = tmp_path / "processed"
    source.mkdir()
    interactions = source / "interactions.jsonl"
    items = source / "items.jsonl"
    interactions.write_text(
        "\n".join(
            [
                json.dumps({"domain": "Beauty", "item_id": "I1", "rating": 5, "timestamp": 1, "user_id": "U1"}),
                json.dumps({"domain": "Beauty", "item_id": "I2", "rating": 4, "timestamp": 2, "user_id": "U1"}),
                json.dumps({"domain": "Beauty", "item_id": "I1", "rating": 5, "timestamp": 3, "user_id": "U2"}),
                json.dumps({"domain": "Beauty", "item_id": "I2", "rating": 4, "timestamp": 4, "user_id": "U2"}),
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
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "filtered.yaml"
    readiness = tmp_path / "readiness.json"
    config.write_text(
        f"""
dataset:
  name: filtered_smoke
  adapter: generic_jsonl
  min_user_interactions: 2
  min_item_interactions: 2
  required_interaction_columns: [user_id, item_id, timestamp, rating_or_interaction, domain]
  required_item_columns: [item_id, text_field, domain]
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
readiness:
  allow_download: false
  output_path: {readiness.as_posix()}
""",
        encoding="utf-8",
    )

    subprocess.run(
        [sys.executable, "scripts/filter_amazon_multidomain.py", "--config", str(config), "--materialize"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        [sys.executable, "scripts/check_dataset_readiness.py", "--config", str(config)],
        cwd=root,
        check=True,
    )
    report = json.loads(readiness.read_text(encoding="utf-8"))
    assert report["status"] == "READY"
    assert report["leave_one_out_feasible"] is True
    assert report["users_with_too_few_interactions"] == 0
