"""Evaluate local 8B LoRA outputs when adapters exist."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.evaluation.lora_export import export_lora_table  # noqa: E402
from llm4rec.experiments.config import load_yaml_config, resolve_path
from llm4rec.io.artifacts import ensure_dir, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    run_dir = ensure_dir(resolve_path(config["evaluation_run"]["output_dir"]))
    adapters = [resolve_path(path) for path in config["evaluation_run"].get("adapter_paths", [])]
    missing = [str(path) for path in adapters if not path.exists()]
    if missing:
        write_json(
            run_dir / "failure_report.json",
            {"adapter_paths_missing": missing, "blocks_evaluation": True, "status": "blocked"},
        )
        raise RuntimeError(f"LoRA evaluation blocked; missing adapters: {missing}")
    export_lora_table(run_dir)
    print(f"lora evaluation completed: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
