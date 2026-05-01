"""Train a smoke-scale baseline and write a checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.experiments.config import deep_merge, load_yaml_config, resolve_experiment_config  # noqa: E402
from llm4rec.trainers.lora import build_lora_training_plan  # noqa: E402
from llm4rec.trainers.registry import build_trainer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = resolve_experiment_config(args.config)
    training = dict(config.get("training", {}))
    if training.get("config_path"):
        loaded = load_yaml_config(training["config_path"])
        config["training"] = deep_merge(dict(loaded.get("training", loaded)), {k: v for k, v in training.items() if k != "config_path"})
    config.setdefault("training", {})
    if not config["training"]:
        config["training"] = {
            "baseline": {"params": {"epochs": 3, "factors": 4}},
            "run_id": str(config.get("experiment", {}).get("run_id", "smoke")),
            "type": "traditional_mf",
        }
    if config["training"].get("type") in {"lora", "qlora"} or args.dry_run:
        path = build_lora_training_plan(config["training"].get("config_path", args.config))
        print(f"LoRA dry-run training plan written: {path}")
        return 0
    result = build_trainer(config).train()
    print(f"training run completed: {result.run_dir}")
    print(f"checkpoint: {result.checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
