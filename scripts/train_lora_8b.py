"""Train local 8B LoRA/QLoRA adapters when readiness passes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.experiments.config import load_yaml_config, deep_merge  # noqa: E402
from llm4rec.trainers.lora_sft import train_lora_8b  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, help="Override sft.data_dir")
    parser.add_argument("--base-model-path", type=str, help="Override model.base_model_path")
    parser.add_argument("--output-dir", type=Path, help="Override training_run.output_dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-scaffold",
        action="store_true",
        help="Allow explicitly non-reportable scaffold/reference interface smoke training.",
    )
    args = parser.parse_args()

    # Apply CLI overrides to config
    config = load_yaml_config(args.config)
    if args.data_dir:
        config.setdefault("sft", {})["data_dir"] = str(args.data_dir)
    if args.base_model_path:
        config.setdefault("model", {})["base_model_path"] = args.base_model_path
    if args.output_dir:
        config.setdefault("training_run", {})["output_dir"] = str(args.output_dir)

    # Write merged config to temp file for train_lora_8b
    import tempfile, yaml
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        merged_config_path = Path(f.name)

    try:
        result = train_lora_8b(merged_config_path, dry_run=args.dry_run, allow_scaffold=args.allow_scaffold)
        print(f"lora training status: {result.get('status')} variant={result.get('variant')}")
    finally:
        merged_config_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
