"""Merge per-domain LoRA SFT artifacts into a four-domain training set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.trainers.sft_merge import merge_lora_sft_data  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--protocol-version", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = merge_lora_sft_data(
        input_root=args.input_root,
        output_dir=args.output_dir,
        variant=args.variant,
        datasets=[str(dataset) for dataset in args.datasets],
        protocol_version=args.protocol_version,
        dry_run=args.dry_run,
    )
    print(
        "merged sft data: "
        f"variant={result.manifest['variant']} "
        f"datasets={','.join(result.manifest['datasets'])} "
        f"train={result.manifest['num_train_rows']} "
        f"valid={result.manifest['num_valid_rows']} "
        f"output={result.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
