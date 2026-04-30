from pathlib import Path

import pytest

from tglrec.utils.config import ConfigError, load_config, write_config


def test_load_config_reads_mapping(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("seed: 2026\nnested:\n  value: true\n", encoding="utf-8")

    assert load_config(path) == {"seed": 2026, "nested": {"value": True}}


def test_load_config_rejects_sequence(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(path)


def test_write_config_is_stable(tmp_path: Path):
    path = tmp_path / "config.yaml"
    write_config({"b": 2, "a": 1}, path)

    assert path.read_text(encoding="utf-8").splitlines() == ["a: 1", "b: 2"]
