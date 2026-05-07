"""Generate the server-side four-domain same-candidate run plan."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.experiments.four_domain_plan import cli_main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(cli_main())
