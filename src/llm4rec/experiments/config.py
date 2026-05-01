"""YAML config loading and resolution."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

try:  # PyYAML is the preferred parser when the project dependencies are installed.
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised by bare Python smoke commands.
    yaml = None

from llm4rec.io.artifacts import write_yaml


def project_root() -> Path:
    """Return repository root inferred from this source file."""

    return Path(__file__).resolve().parents[3]


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and require a mapping at the top level."""

    config_path = resolve_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    data = yaml.safe_load(text) if yaml is not None else _simple_yaml_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    data.setdefault("_meta", {})
    data["_meta"]["config_path"] = str(config_path)
    return data


def resolve_path(path: str | Path, *, base_dir: str | Path | None = None) -> Path:
    """Resolve a path relative to the repository root or a provided base directory."""

    value = Path(path)
    if value.is_absolute():
        return value
    base = Path(base_dir) if base_dir is not None else project_root()
    return (base / value).resolve()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries without mutating inputs."""

    output = deepcopy(base)
    for key, value in override.items():
        if (
            key in output
            and isinstance(output[key], dict)
            and isinstance(value, dict)
        ):
            output[key] = deep_merge(output[key], value)
        else:
            output[key] = deepcopy(value)
    return output


def resolve_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment config and inline referenced dataset/evaluation configs."""

    config = load_yaml_config(path)
    dataset_section = config.get("dataset", {})
    if isinstance(dataset_section, dict) and dataset_section.get("config_path"):
        dataset_config = load_yaml_config(dataset_section["config_path"])
        merged_dataset = deep_merge(dataset_config.get("dataset", {}), dataset_section)
        merged_dataset.pop("config_path", None)
        config["dataset"] = merged_dataset

    evaluation_section = config.get("evaluation", {})
    if isinstance(evaluation_section, dict) and evaluation_section.get("config_path"):
        eval_config = load_yaml_config(evaluation_section["config_path"])
        merged_evaluation = deep_merge(eval_config.get("evaluation", {}), evaluation_section)
        merged_evaluation.pop("config_path", None)
        config["evaluation"] = merged_evaluation
    return config


def save_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    """Save a resolved config artifact."""

    write_yaml(path, config)


def _simple_yaml_load(text: str) -> dict[str, Any]:
    """Load the small YAML subset used by Phase 1 configs.

    This fallback intentionally supports only nested mappings with two-space indentation and
    inline scalar/list values. It exists so `python scripts/...` can fail less mysteriously in a
    bare interpreter; reportable runs should still install the project dependencies.
    """

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            raise ValueError(f"Unsupported YAML line without key/value separator: {raw_line!r}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
        else:
            current[key] = _parse_simple_scalar(value)
    return root


def _parse_simple_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if value == "{}":
        return {}
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_simple_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
