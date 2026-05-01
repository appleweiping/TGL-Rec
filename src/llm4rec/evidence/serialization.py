"""Evidence JSONL serialization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from llm4rec.evidence.base import Evidence
from llm4rec.io.artifacts import read_jsonl, write_jsonl


def write_evidence_jsonl(path: str | Path, evidence: Iterable[Evidence]) -> None:
    """Write evidence rows to JSONL."""

    write_jsonl(path, [row.to_dict() for row in evidence])


def read_evidence_jsonl(path: str | Path) -> list[Evidence]:
    """Read evidence rows from JSONL."""

    return [Evidence.from_dict(row) for row in read_jsonl(path)]
