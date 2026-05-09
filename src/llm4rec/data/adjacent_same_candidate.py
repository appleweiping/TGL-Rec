"""Normalize adjacent legacy same-candidate artifacts into importable task dirs."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from llm4rec.io.artifacts import ensure_dir, iter_jsonl, sha256_file, write_json


DEFAULT_PROTOCOL_VERSION = "protocol_uncertainty_legacy_local"
DOMAIN_ALIASES = {
    "amazon_beauty": "beauty",
    "beauty": "beauty",
    "amazon_book": "books",
    "amazon_books": "books",
    "amazon_books_small": "books",
    "book": "books",
    "books": "books",
    "amazon_electronic": "electronics",
    "amazon_electronics": "electronics",
    "amazon_electronics_small": "electronics",
    "electronic": "electronics",
    "electronics": "electronics",
    "amazon_movie": "movies",
    "amazon_movies": "movies",
    "amazon_movies_small": "movies",
    "movie": "movies",
    "movies": "movies",
}


@dataclass(frozen=True)
class AdjacentTask:
    """One normalized adjacent-source task directory."""

    domain: str
    split: str
    source_dir: Path
    output_dir: Path
    rows: int


def normalize_adjacent_same_candidate_tasks(
    *,
    source_root: str | Path,
    output_root: str | Path = "outputs/staging/adjacent_same_candidate",
    protocol_version: str = DEFAULT_PROTOCOL_VERSION,
    domains: Iterable[str] = ("beauty", "books", "electronics", "movies"),
    splits: Iterable[str] = ("valid", "test"),
) -> list[AdjacentTask]:
    """Normalize legacy adjacent ranking files into Week8-compatible task dirs."""

    source = Path(source_root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Missing adjacent processed root: {source}")
    requested_domains = [canonical_domain_name(domain) for domain in domains]
    requested_splits = [str(split) for split in splits]
    directory_by_domain = _find_domain_directories(source)
    tasks = []
    for domain in requested_domains:
        if domain not in directory_by_domain:
            raise FileNotFoundError(f"Could not find adjacent directory for domain={domain} under {source}")
        source_dir = directory_by_domain[domain]
        for split in requested_splits:
            ranking_path = source_dir / f"ranking_{split}.jsonl"
            if not ranking_path.is_file():
                raise FileNotFoundError(f"Missing ranking file for domain={domain} split={split}: {ranking_path}")
            tasks.append(
                _normalize_one_task(
                    source_dir=source_dir,
                    ranking_path=ranking_path,
                    output_root=Path(output_root),
                    protocol_version=protocol_version,
                    domain=domain,
                    split=split,
                )
            )
    return tasks


def canonical_domain_name(value: str) -> str:
    """Return canonical domain name for messy adjacent-project aliases."""

    key = str(value).strip().lower().replace("-", "_")
    key = key.removesuffix("_medium").removesuffix("_20neg").removesuffix("_2000").removesuffix("_5neg")
    if key in DOMAIN_ALIASES:
        return DOMAIN_ALIASES[key]
    raise ValueError(f"Unsupported adjacent domain alias: {value}")


def _find_domain_directories(root: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        try:
            domain = canonical_domain_name(path.name)
        except ValueError:
            continue
        if (path / "ranking_valid.jsonl").is_file() or (path / "ranking_test.jsonl").is_file():
            output.setdefault(domain, path)
    return output


def _normalize_one_task(
    *,
    source_dir: Path,
    ranking_path: Path,
    output_root: Path,
    protocol_version: str,
    domain: str,
    split: str,
) -> AdjacentTask:
    output_dir = ensure_dir(
        output_root / protocol_version / f"{domain}_large10000_100neg_{split}_same_candidate"
    )
    ranking_rows = [_normalize_ranking_row(row, domain=domain, split=split) for row in iter_jsonl(ranking_path)]
    _write_ranking(output_dir / f"ranking_{split}.jsonl", ranking_rows)
    _write_candidate_items(output_dir / "candidate_items.csv", ranking_rows)
    _copy_interactions(source_dir / "interactions.csv", output_dir / "train_interactions.csv")
    _copy_items(source_dir / "items.csv", output_dir / "item_metadata.csv")
    metadata = {
        "domain": domain,
        "import_warning": (
            "Normalized from adjacent uncertainty-llm4rec processed artifacts. "
            "Use for diagnostic/local runs unless candidate_count and protocol match the final paper setting."
        ),
        "protocol_version": protocol_version,
        "source_dir": str(source_dir),
        "source_layout": "uncertainty_llm4rec_processed",
        "source_ranking_file": str(ranking_path),
        "source_ranking_sha256": sha256_file(ranking_path),
        "split": split,
    }
    write_json(output_dir / "metadata.json", metadata)
    return AdjacentTask(
        domain=domain,
        split=split,
        source_dir=source_dir,
        output_dir=output_dir,
        rows=len(ranking_rows),
    )


def _normalize_ranking_row(row: dict[str, Any], *, domain: str, split: str) -> dict[str, Any]:
    candidates = [str(item) for item in row.get("candidate_item_ids") or row.get("candidate_items") or []]
    labels = list(row.get("candidate_labels") or [])
    if not candidates:
        raise ValueError(f"ranking row has no candidate_item_ids: {row}")
    target = str(row.get("positive_item_id") or row.get("target_item") or "")
    if not target and labels:
        positives = [item for item, label in zip(candidates, labels, strict=False) if str(label) in {"1", "1.0", "true", "True"}]
        if len(positives) != 1:
            raise ValueError(f"Expected exactly one positive label, found {len(positives)}")
        target = positives[0]
    if target not in candidates:
        raise ValueError(f"target missing from candidates: target={target}")
    source_event_id = str(row.get("source_event_id") or row.get("event_id") or "")
    if not source_event_id:
        source_event_id = f"{domain}:{split}:{row.get('user_id')}:{row.get('timestamp')}"
    return {
        "candidate_items": candidates,
        "domain": str(row.get("domain") or domain),
        "event_id": str(row.get("event_id") or source_event_id),
        "history": [str(item) for item in row.get("history", [])],
        "positive_item": target,
        "source_event_id": source_event_id,
        "split": str(row.get("split") or row.get("split_name") or split),
        "target_item": target,
        "timestamp": row.get("timestamp"),
        "user_id": str(row.get("user_id") or ""),
    }


def _write_ranking(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _write_candidate_items(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["event_id", "source_event_id", "user_id", "candidate_items", "target_item"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "candidate_items": json.dumps(row["candidate_items"], ensure_ascii=True),
                    "event_id": row["event_id"],
                    "source_event_id": row["source_event_id"],
                    "target_item": row["target_item"],
                    "user_id": row["user_id"],
                }
            )


def _copy_interactions(source_path: Path, output_path: Path) -> None:
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing interactions.csv: {source_path}")
    output_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")


def _copy_items(source_path: Path, output_path: Path) -> None:
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing items.csv: {source_path}")
    output_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", default="outputs/staging/adjacent_same_candidate")
    parser.add_argument("--protocol-version", default=DEFAULT_PROTOCOL_VERSION)
    parser.add_argument("--domains", nargs="+", default=["beauty", "books", "electronics", "movies"])
    parser.add_argument("--splits", nargs="+", default=["valid", "test"])
    args = parser.parse_args(argv)
    tasks = normalize_adjacent_same_candidate_tasks(
        source_root=args.source_root,
        output_root=args.output_root,
        protocol_version=args.protocol_version,
        domains=args.domains,
        splits=args.splits,
    )
    print(
        json.dumps(
            {
                "normalized": [
                    {
                        "domain": task.domain,
                        "output_dir": str(task.output_dir),
                        "rows": task.rows,
                        "source_dir": str(task.source_dir),
                        "split": task.split,
                    }
                    for task in tasks
                ],
                "status": "succeeded",
            },
            sort_keys=True,
        )
    )
    return 0

