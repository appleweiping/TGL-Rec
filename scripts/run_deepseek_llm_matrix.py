"""Run or dry-run Phase 9D DeepSeek API LLM matrices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.experiments.deepseek_llm import run_deepseek_matrix  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage", choices=["dry_run", "pilot", "full"], default=None)
    args = parser.parse_args()
    result = run_deepseek_matrix(args.config, dry_run=bool(args.dry_run), stage=args.stage)
    print(f"deepseek matrix {result.get('mode', args.stage or 'pilot')} completed: {result['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
