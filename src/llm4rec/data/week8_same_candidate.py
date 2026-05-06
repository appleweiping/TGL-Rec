"""Importer for frozen Week8 large same-candidate ranking tasks."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from llm4rec.io.artifacts import ensure_dir, iter_jsonl, sha256_file, write_json, write_jsonl


DEFAULT_PROTOCOL_VERSION = "protocol_week8_large10000_same_candidate"


@dataclass(frozen=True)
class Week8ImportResult:
    """Imported artifact locations and manifest for one domain/split task."""

    output_dir: Path
    manifest: dict[str, Any]


def import_week8_same_candidate_task(
    task_dir: str | Path,
    *,
    output_root: str | Path = "outputs/artifacts",
    protocol_version: str = DEFAULT_PROTOCOL_VERSION,
    domain: str | None = None,
    split: str | None = None,
) -> Week8ImportResult:
    """Import one immutable Week8 same-candidate task directory.

    The importer preserves the external candidate sets and event identifiers.
    It does not resample users, positives, or negatives.
    """

    task_path = Path(task_dir).expanduser().resolve()
    if not task_path.is_dir():
        raise FileNotFoundError(f"Missing Week8 task directory: {task_path}")
    inferred_domain, inferred_split = _infer_domain_split(task_path)
    domain = str(domain or inferred_domain)
    split = str(split or inferred_split)
    ranking_path = _find_ranking_path(task_path, split)
    candidate_path = task_path / "candidate_items.csv"
    train_path = task_path / "train_interactions.csv"
    metadata_path = task_path / "item_metadata.csv"
    protocol_meta_path = task_path / "metadata.json"
    for required in (ranking_path, candidate_path, train_path, metadata_path):
        if not required.is_file():
            raise FileNotFoundError(f"Missing required Week8 input file: {required}")

    output_dir = Path(output_root) / protocol_version / domain
    ensure_dir(output_dir)
    train_rows = _read_train_interactions(train_path, domain=domain)
    ranking_rows = list(iter_jsonl(ranking_path))
    candidate_groups = _read_candidate_groups(candidate_path)
    candidates = _build_candidate_rows(
        ranking_rows,
        candidate_groups=candidate_groups,
        domain=domain,
        protocol_version=protocol_version,
        split=split,
    )
    item_rows = _read_item_metadata(metadata_path, domain=domain)
    split_rows = [*train_rows, *_target_split_rows(candidates)]

    split_path = output_dir / "splits.jsonl"
    candidate_out_path = output_dir / "candidates.jsonl"
    item_out_path = output_dir / "item_metadata.jsonl"
    manifest_path = output_dir / f"week8_{split}_import_manifest.json"
    write_jsonl(split_path, _merge_split_rows(split_path, split_rows, replace_split=split))
    write_jsonl(candidate_out_path, _merge_candidate_rows(candidate_out_path, candidates, replace_split=split))
    write_jsonl(item_out_path, item_rows)
    protocol_metadata = _read_optional_json(protocol_meta_path)
    manifest = {
        "candidate_artifact": str(candidate_out_path),
        "candidate_rows": len(candidates),
        "domain": domain,
        "external_files": {
            "candidate_items_csv": str(candidate_path),
            "candidate_items_sha256": sha256_file(candidate_path),
            "item_metadata_csv": str(metadata_path),
            "item_metadata_sha256": sha256_file(metadata_path),
            "ranking_jsonl": str(ranking_path),
            "ranking_sha256": sha256_file(ranking_path),
            "train_interactions_csv": str(train_path),
            "train_interactions_sha256": sha256_file(train_path),
        },
        "external_metadata": protocol_metadata,
        "import_rules": {
            "candidate_alignment": "preserved_from_external_task",
            "negative_resampling": False,
            "split_resampling": False,
            "user_resampling": False,
        },
        "item_metadata_artifact": str(item_out_path),
        "protocol_version": protocol_version,
        "source_task_dir": str(task_path),
        "split": split,
        "split_artifact": str(split_path),
        "status": "succeeded",
        "train_interactions": len(train_rows),
    }
    write_json(manifest_path, manifest)
    write_json(output_dir / "latest_import_manifest.json", manifest)
    return Week8ImportResult(output_dir=output_dir, manifest=manifest)


def import_week8_same_candidate_tasks(
    task_dirs: Iterable[str | Path],
    *,
    output_root: str | Path = "outputs/artifacts",
    protocol_version: str = DEFAULT_PROTOCOL_VERSION,
) -> list[Week8ImportResult]:
    """Import multiple Week8 task directories."""

    return [
        import_week8_same_candidate_task(
            task_dir,
            output_root=output_root,
            protocol_version=protocol_version,
        )
        for task_dir in task_dirs
    ]


def _infer_domain_split(task_path: Path) -> tuple[str, str]:
    name = task_path.name
    split = "test" if "_test_" in name or name.endswith("_test_same_candidate") else "valid"
    domain = name.split("_large", 1)[0] if "_large" in name else name.split("_", 1)[0]
    return domain, split


def _find_ranking_path(task_path: Path, split: str) -> Path:
    preferred = task_path / f"ranking_{split}.jsonl"
    if preferred.is_file():
        return preferred
    matches = sorted(task_path.glob("ranking_*.jsonl"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not identify ranking_{split}.jsonl in {task_path}")


def _read_train_interactions(path: Path, *, domain: str) -> list[dict[str, Any]]:
    rows = []
    for index, row in enumerate(_read_csv_dicts(path), start=1):
        rows.append(
            {
                "domain": str(row.get("domain") or domain),
                "item_id": _first_present(row, ("item_id", "item", "item:token")),
                "rating": _optional_float(row.get("rating")),
                "source": "week8_same_candidate_train_interactions",
                "split": "train",
                "timestamp": _optional_float(row.get("timestamp") or row.get("time") or row.get("timestamp:float")),
                "user_id": _first_present(row, ("user_id", "user", "user:token")),
            }
        )
        if not rows[-1]["user_id"] or not rows[-1]["item_id"]:
            raise ValueError(f"train_interactions row {index} is missing user_id or item_id: {path}")
    rows.sort(key=lambda value: (str(value["user_id"]), float(value.get("timestamp") or 0.0), str(value["item_id"])))
    return rows


def _read_candidate_groups(path: Path) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(_read_csv_dicts(path), start=1):
        event_id = _event_id(row)
        if not event_id:
            raise ValueError(f"candidate_items row {index} is missing event_id/source_event_id: {path}")
        group = grouped.setdefault(
            event_id,
            {
                "candidate_items": [],
                "source_event_id": _source_event_id(row),
                "target_item": None,
                "user_id": str(row.get("user_id") or ""),
            },
        )
        candidates = _candidate_list_from_row(row)
        if candidates:
            group["candidate_items"].extend(candidates)
        target = _target_from_row(row)
        if target:
            group["target_item"] = target
        if row.get("user_id") and not group.get("user_id"):
            group["user_id"] = str(row["user_id"])
    for event_id, group in grouped.items():
        group["candidate_items"] = _dedupe_preserve_order(group["candidate_items"])
        if not group["candidate_items"]:
            raise ValueError(f"candidate event has no candidates: {event_id}")
    return grouped


def _build_candidate_rows(
    ranking_rows: list[dict[str, Any]],
    *,
    candidate_groups: dict[str, dict[str, Any]],
    domain: str,
    protocol_version: str,
    split: str,
) -> list[dict[str, Any]]:
    output = []
    for index, ranking in enumerate(ranking_rows, start=1):
        event_id = _event_id(ranking) or f"{split}:{index}"
        group = candidate_groups.get(event_id)
        source_event_id = _source_event_id(ranking) or (group or {}).get("source_event_id") or event_id
        candidates = _candidate_list_from_row(ranking)
        if not candidates and group:
            candidates = list(group["candidate_items"])
        target = _target_from_row(ranking) or (group or {}).get("target_item")
        if not target:
            target = _target_from_candidates(candidates, ranking)
        if not target:
            raise ValueError(f"ranking row {index} is missing target item: event={event_id}")
        if target not in candidates:
            raise ValueError(f"target missing from candidates: event={event_id} target={target}")
        user_id = str(ranking.get("user_id") or (group or {}).get("user_id") or "")
        if not user_id:
            raise ValueError(f"ranking row {index} is missing user_id: event={event_id}")
        row = {
            "candidate_items": candidates,
            "candidate_source": "week8_same_candidate_external_task",
            "domain": str(ranking.get("domain") or domain),
            "event_id": str(event_id),
            "history": _history_from_row(ranking),
            "protocol_version": protocol_version,
            "source_event_id": str(source_event_id),
            "split": str(ranking.get("split") or split),
            "target_item": str(target),
            "user_id": user_id,
        }
        output.append(row)
    return output


def _read_item_metadata(path: Path, *, domain: str) -> list[dict[str, Any]]:
    rows = []
    for index, row in enumerate(_read_csv_dicts(path), start=1):
        item_id = _first_present(row, ("item_id", "item", "item:token"))
        if not item_id:
            raise ValueError(f"item_metadata row {index} is missing item_id: {path}")
        rows.append(
            {
                "brand": _optional_str(row.get("brand")),
                "category": _optional_str(row.get("category") or row.get("categories")),
                "description": _optional_str(row.get("description") or row.get("desc")),
                "domain": str(row.get("domain") or domain),
                "item_id": item_id,
                "raw_text": _optional_str(row.get("raw_text") or row.get("text")),
                "title": str(row.get("title") or item_id),
            }
        )
    return rows


def _target_split_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "domain": row.get("domain"),
            "event_id": row.get("event_id"),
            "item_id": row["target_item"],
            "rating": 1.0,
            "source": "week8_same_candidate_ranking_task",
            "source_event_id": row.get("source_event_id"),
            "split": row["split"],
            "timestamp": None,
            "user_id": row["user_id"],
        }
        for row in candidates
    ]


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {"value": value}


def _merge_candidate_rows(path: Path, new_rows: list[dict[str, Any]], *, replace_split: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return new_rows
    kept = [row for row in iter_jsonl(path) if str(row.get("split", "")) != replace_split]
    return [*kept, *new_rows]


def _merge_split_rows(path: Path, new_rows: list[dict[str, Any]], *, replace_split: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return new_rows
    kept = [
        row
        for row in iter_jsonl(path)
        if str(row.get("split", "")) not in {"train", replace_split}
    ]
    return [*kept, *new_rows]


def _event_id(row: dict[str, Any]) -> str:
    return str(row.get("event_id") or row.get("source_event_id") or row.get("event") or "").strip()


def _source_event_id(row: dict[str, Any]) -> str:
    return str(row.get("source_event_id") or row.get("event_id") or row.get("event") or "").strip()


def _candidate_list_from_row(row: dict[str, Any]) -> list[str]:
    for key in ("candidate_items", "candidates", "candidate_item_ids", "items"):
        value = row.get(key)
        parsed = _parse_list_like(value)
        if parsed:
            return parsed
    item = row.get("item_id") or row.get("candidate_item_id") or row.get("candidate") or row.get("item")
    return [str(item)] if item not in (None, "") else []


def _target_from_row(row: dict[str, Any]) -> str:
    for key in ("target_item", "positive_item", "positive_item_id", "label_item", "ground_truth_item", "item_id_pos"):
        if row.get(key) not in (None, ""):
            return str(row[key])
    label = str(row.get("label") or row.get("is_positive") or row.get("target") or "").strip().lower()
    if label in {"1", "1.0", "true", "yes", "positive"}:
        item = row.get("item_id") or row.get("candidate_item_id") or row.get("candidate") or row.get("item")
        return "" if item in (None, "") else str(item)
    return ""


def _target_from_candidates(candidates: list[str], row: dict[str, Any]) -> str:
    labels = _parse_list_like(row.get("labels") or row.get("candidate_labels"))
    if labels and len(labels) == len(candidates):
        for item, label in zip(candidates, labels, strict=False):
            if str(label).strip().lower() in {"1", "1.0", "true", "yes", "positive"}:
                return item
    return ""


def _history_from_row(row: dict[str, Any]) -> list[str]:
    for key in ("history", "history_items", "train_plus_valid_history", "user_history"):
        parsed = _parse_list_like(row.get(key))
        if parsed:
            return parsed
    return []


def _parse_list_like(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str):
        return [str(value)]
    text = value.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    if " " in text:
        return [part.strip() for part in text.split() if part.strip()]
    return [text]


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if row.get(key) not in (None, ""):
            return str(row[key])
    return ""


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    output = []
    seen = set()
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        output.append(key)
    return output


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", action="append", required=True, help="Week8 task directory. Repeatable.")
    parser.add_argument("--output-root", default="outputs/artifacts")
    parser.add_argument("--protocol-version", default=DEFAULT_PROTOCOL_VERSION)
    args = parser.parse_args(argv)
    results = import_week8_same_candidate_tasks(
        args.task_dir,
        output_root=args.output_root,
        protocol_version=args.protocol_version,
    )
    print(json.dumps({"imported": [result.manifest for result in results], "status": "succeeded"}, sort_keys=True))
    return 0
