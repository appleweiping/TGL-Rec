"""Estimate Phase 9D DeepSeek API cost without API calls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.experiments.deepseek_llm import estimate_deepseek_cost  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    result = estimate_deepseek_cost(args.config)
    print(
        "deepseek cost estimate: "
        f"requests={result['estimated_requests']} "
        f"tokens={result['estimated_prompt_tokens'] + result['estimated_completion_tokens']} "
        f"usd_cache_miss={result['estimated_cost_usd_cache_miss']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
