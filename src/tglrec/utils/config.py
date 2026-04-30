"""YAML config loading with deterministic, explicit behavior."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a config file is missing or malformed."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dictionary.

    The loader rejects non-mapping top-level YAML because repository configs are
    intended to be explicit experiment or dataset records, not arbitrary values.
    """

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"Config must have a mapping at the top level: {config_path}")
    return dict(data)


def write_config(config: Mapping[str, Any], path: str | Path) -> None:
    """Write a config deterministically with sorted keys."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=True, allow_unicode=False)

