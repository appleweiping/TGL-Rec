"""Export paper-table-ready artifacts from metrics files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.evaluation.export import export_paper_tables  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = export_paper_tables(args.input, args.output)
    print(f"table manifest written: {args.output / 'table_manifest.json'}")
    print(f"metric_file_count={manifest['metric_file_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
