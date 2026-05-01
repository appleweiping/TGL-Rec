from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


def test_tiny_preprocess_script_writes_artifacts(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = {
        "dataset": {
            "name": "tiny_test",
            "adapter": "tiny_jsonl",
            "paths": {
                "interactions": str(root / "data" / "tiny" / "interactions.jsonl"),
                "items": str(root / "data" / "tiny" / "items.jsonl"),
            },
            "output_dir": str(tmp_path / "processed"),
            "split_strategy": "leave_one_out",
            "candidate_protocol": "full_catalog",
            "seed": 2026,
        }
    }
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    subprocess.run(
        [sys.executable, str(root / "scripts" / "preprocess.py"), "--config", str(config_path)],
        cwd=root,
        check=True,
    )
    output = tmp_path / "processed"
    assert (output / "train.jsonl").is_file()
    assert (output / "valid.jsonl").is_file()
    assert (output / "test.jsonl").is_file()
    assert (output / "items.jsonl").is_file()
    assert (output / "candidates.jsonl").is_file()
    assert (output / "metadata.json").is_file()
