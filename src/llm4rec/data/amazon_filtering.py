"""Paper-ready filtering for converted Amazon multidomain artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TextIO

from llm4rec.data.filtering import item_key, user_key
from llm4rec.data.readiness import NO_EXECUTION_FLAG
from llm4rec.experiments.config import load_yaml_config, resolve_path
from llm4rec.io.artifacts import ensure_dir, write_json
from llm4rec.utils.env import current_git_commit


class AmazonFilteringError(ValueError):
    """Raised when filtering cannot be performed safely."""


def filter_amazon_multidomain(
    config_path: str | Path,
    *,
    dry_run: bool = False,
    materialize: bool = False,
) -> dict[str, Any]:
    """Filter converted Amazon JSONL artifacts without modifying raw conversion files."""

    if dry_run and materialize:
        raise AmazonFilteringError("--dry-run and --materialize are mutually exclusive")
    if not dry_run and not materialize:
        dry_run = True
    plan = _filtering_plan(config_path)
    raw_snapshot_before = _raw_snapshot(plan)
    progress = _ProgressLogger(plan["progress_log_path"])
    progress.log(f"start status={'materialize' if materialize else 'dry_run'} strategy={plan['strategy']}")

    before_interactions = _summarize_interactions(plan["source_interactions_path"], progress=progress)
    before_items = _summarize_items(plan["source_items_path"])
    active_users, active_items, iterations, converged = _compute_retained_sets(plan, progress)
    after_counts = _count_active_interactions(
        plan["source_interactions_path"],
        active_users=active_users,
        active_items=active_items,
        progress=progress,
    )
    after_interactions = _summarize_interactions(
        plan["source_interactions_path"],
        active_users=active_users,
        active_items=active_items,
        progress=progress,
    )
    after_items = _summarize_items(plan["source_items_path"], active_items=active_items)

    if materialize:
        _assert_safe_outputs(plan)
        _materialize(plan, active_users, active_items, progress)

    raw_snapshot_after = _raw_snapshot(plan)
    report = _filtering_report(
        plan,
        before_interactions,
        before_items,
        after_interactions,
        after_items,
        after_counts,
        iterations,
        converged,
        raw_snapshot_before,
        raw_snapshot_after,
        dry_run=dry_run,
        materialized=materialize,
    )
    write_json(plan["report_path"], report)
    if plan["history_report_path"] is not None:
        write_json(plan["history_report_path"], report)
    progress.log(f"completed status={report['status']}")
    return report


def _filtering_plan(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    dataset = dict(config.get("dataset", config))
    filtering = dict(config.get("filtering", dataset.get("filtering", {})))
    paths = dict(dataset.get("paths", {}))
    source_interactions = paths.get("source_interactions") or dataset.get("source_interactions_path")
    source_items = paths.get("source_items") or dataset.get("source_items_path")
    interactions = paths.get("interactions") or dataset.get("interactions_path")
    items = paths.get("items") or dataset.get("items_path")
    if not source_interactions or not source_items:
        raise AmazonFilteringError("Filtering config must provide source_interactions and source_items")
    if not interactions or not items:
        raise AmazonFilteringError("Filtering config must provide output interactions and items paths")
    output_dir = ensure_dir(resolve_path(dataset.get("filtered_output_dir") or Path(interactions).parent))
    report_path = _path_value(paths.get("filtering_report"), output_dir / "filtering_report.json")
    history_report = paths.get("filtering_history_report")
    strategy = str(filtering.get("strategy", "iterative_k_core"))
    configured_k = int(filtering.get("k", 3))
    user_min = int(filtering.get("user_min_interactions", configured_k))
    item_min = int(filtering.get("item_min_interactions", configured_k))
    if strategy == "user_min_interactions" and "item_min_interactions" not in filtering:
        item_min = 1
    if strategy == "item_min_interactions" and "user_min_interactions" not in filtering:
        user_min = 1
    return {
        "config": config,
        "config_path": resolve_path(config_path),
        "dataset": dataset,
        "history_report_path": resolve_path(history_report) if history_report else None,
        "interactions_path": resolve_path(interactions),
        "item_min_interactions": item_min,
        "items_path": resolve_path(items),
        "max_iterations": int(filtering.get("max_iterations", 50)),
        "output_dir": output_dir,
        "overwrite_existing": bool(filtering.get("overwrite_existing", False)),
        "progress_log_path": _path_value(paths.get("filtering_progress"), output_dir / "filtering_progress.log"),
        "report_path": report_path,
        "source_interactions_path": resolve_path(source_interactions),
        "source_items_path": resolve_path(source_items),
        "strategy": strategy,
        "user_min_interactions": user_min,
    }


def _compute_retained_sets(
    plan: dict[str, Any],
    progress: "_ProgressLogger",
) -> tuple[set[str], set[tuple[str, str]], list[dict[str, Any]], bool]:
    strategy = plan["strategy"]
    user_min = plan["user_min_interactions"]
    item_min = plan["item_min_interactions"]
    active_users: set[str] | None = None
    active_items: set[tuple[str, str]] | None = None
    iterations: list[dict[str, Any]] = []
    max_iterations = plan["max_iterations"] if strategy == "iterative_k_core" else 1
    converged = False
    for iteration in range(1, max_iterations + 1):
        counts = _count_active_interactions(
            plan["source_interactions_path"],
            active_users=active_users,
            active_items=active_items,
            progress=progress,
        )
        next_users = {key for key, count in counts["user_counts"].items() if count >= user_min}
        next_items = {key for key, count in counts["item_counts"].items() if count >= item_min}
        removed_users = len(counts["user_counts"]) - len(next_users)
        removed_items = len(counts["item_counts"]) - len(next_items)
        iteration_row = {
            "input_interactions": counts["interactions"],
            "input_items": len(counts["item_counts"]),
            "input_users": len(counts["user_counts"]),
            "iteration": iteration,
            "output_items": len(next_items),
            "output_users": len(next_users),
            "removed_items": removed_items,
            "removed_users": removed_users,
        }
        iterations.append(iteration_row)
        progress.log(f"iteration {iteration}: {json.dumps(iteration_row, sort_keys=True)}")
        active_users = next_users
        active_items = next_items
        if strategy != "iterative_k_core" or (removed_users == 0 and removed_items == 0):
            converged = True
            break
    return active_users or set(), active_items or set(), iterations, converged


def _summarize_interactions(
    path: Path,
    *,
    active_users: set[str] | None = None,
    active_items: set[tuple[str, str]] | None = None,
    progress: "_ProgressLogger" | None = None,
) -> dict[str, Any]:
    users: set[str] = set()
    items: set[tuple[str, str]] = set()
    domains: dict[str, dict[str, Any]] = defaultdict(lambda: {"interactions": 0, "items": set(), "users": set()})
    interactions = 0
    timestamp_min: float | None = None
    timestamp_max: float | None = None
    for row in _iter_jsonl(path):
        user = user_key(row)
        item = item_key(row)
        if active_users is not None and user not in active_users:
            continue
        if active_items is not None and item not in active_items:
            continue
        interactions += 1
        users.add(user)
        items.add(item)
        domain = str(row.get("domain") or "")
        domains[domain]["interactions"] += 1
        domains[domain]["users"].add(user)
        domains[domain]["items"].add(item)
        timestamp = _to_float(row.get("timestamp"))
        if timestamp is not None:
            timestamp_min = timestamp if timestamp_min is None else min(timestamp_min, timestamp)
            timestamp_max = timestamp if timestamp_max is None else max(timestamp_max, timestamp)
        if progress is not None and interactions and interactions % 1_000_000 == 0:
            progress.log(f"summarized {interactions} retained interactions from {path.name}")
    return {
        "domains": {
            domain: {
                "interactions": stats["interactions"],
                "items": len(stats["items"]),
                "users": len(stats["users"]),
            }
            for domain, stats in sorted(domains.items())
        },
        "interactions": interactions,
        "items": len(items),
        "timestamp_max": timestamp_max,
        "timestamp_min": timestamp_min,
        "users": len(users),
    }


def _summarize_items(path: Path, *, active_items: set[tuple[str, str]] | None = None) -> dict[str, Any]:
    items: set[tuple[str, str]] = set()
    domains: dict[str, set[tuple[str, str]]] = defaultdict(set)
    missing_text = 0
    duplicate_items = 0
    for row in _iter_jsonl(path):
        key = item_key(row)
        if active_items is not None and key not in active_items:
            continue
        if key in items:
            duplicate_items += 1
            continue
        items.add(key)
        domains[key[0]].add(key)
        if not _has_item_text(row):
            missing_text += 1
    return {
        "domains": {domain: {"items": len(values)} for domain, values in sorted(domains.items())},
        "duplicate_items": duplicate_items,
        "items": len(items),
        "missing_text_items": missing_text,
    }


def _count_active_interactions(
    path: Path,
    *,
    active_users: set[str] | None,
    active_items: set[tuple[str, str]] | None,
    progress: "_ProgressLogger" | None = None,
) -> dict[str, Any]:
    user_counts: Counter[str] = Counter()
    item_counts: Counter[tuple[str, str]] = Counter()
    interactions = 0
    for row in _iter_jsonl(path):
        user = user_key(row)
        item = item_key(row)
        if active_users is not None and user not in active_users:
            continue
        if active_items is not None and item not in active_items:
            continue
        interactions += 1
        user_counts[user] += 1
        item_counts[item] += 1
        if progress is not None and interactions and interactions % 1_000_000 == 0:
            progress.log(f"counted {interactions} active interactions from {path.name}")
    return {"interactions": interactions, "item_counts": item_counts, "user_counts": user_counts}


def _materialize(
    plan: dict[str, Any],
    active_users: set[str],
    active_items: set[tuple[str, str]],
    progress: "_ProgressLogger",
) -> None:
    ensure_dir(plan["interactions_path"].parent)
    ensure_dir(plan["items_path"].parent)
    interactions_tmp = Path(str(plan["interactions_path"]) + ".tmp")
    items_tmp = Path(str(plan["items_path"]) + ".tmp")
    interactions_tmp.unlink(missing_ok=True)
    items_tmp.unlink(missing_ok=True)
    try:
        with interactions_tmp.open("w", encoding="utf-8", newline="\n") as interactions_handle:
            written = _write_filtered_interactions(
                plan["source_interactions_path"],
                interactions_handle,
                active_users,
                active_items,
            )
            progress.log(f"wrote {written} filtered interactions to temp")
        with items_tmp.open("w", encoding="utf-8", newline="\n") as items_handle:
            written_items = _write_filtered_items(plan["source_items_path"], items_handle, active_items)
            progress.log(f"wrote {written_items} filtered items to temp")
        interactions_tmp.replace(plan["interactions_path"])
        items_tmp.replace(plan["items_path"])
    except Exception:
        progress.log("filtering failed before atomic rename; final processed files left untouched")
        raise


def _write_filtered_interactions(
    path: Path,
    handle: TextIO,
    active_users: set[str],
    active_items: set[tuple[str, str]],
) -> int:
    written = 0
    for row in _iter_jsonl(path):
        if user_key(row) in active_users and item_key(row) in active_items:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            written += 1
    return written


def _write_filtered_items(path: Path, handle: TextIO, active_items: set[tuple[str, str]]) -> int:
    written = 0
    seen: set[tuple[str, str]] = set()
    for row in _iter_jsonl(path):
        key = item_key(row)
        if key not in active_items or key in seen:
            continue
        seen.add(key)
        handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
        written += 1
    return written


def _filtering_report(
    plan: dict[str, Any],
    before_interactions: dict[str, Any],
    before_items: dict[str, Any],
    after_interactions: dict[str, Any],
    after_items: dict[str, Any],
    after_counts: dict[str, Any],
    iterations: list[dict[str, Any]],
    converged: bool,
    raw_snapshot_before: dict[str, dict[str, Any]],
    raw_snapshot_after: dict[str, dict[str, Any]],
    *,
    dry_run: bool,
    materialized: bool,
) -> dict[str, Any]:
    user_min = plan["user_min_interactions"]
    item_min = plan["item_min_interactions"]
    users_below = sum(1 for count in after_counts["user_counts"].values() if count < user_min)
    items_below = sum(1 for count in after_counts["item_counts"].values() if count < item_min)
    report = {
        NO_EXECUTION_FLAG: True,
        "code_commit": current_git_commit(resolve_path(".")),
        "config_hash": _file_sha256(plan["config_path"]),
        "converged": converged,
        "dry_run": bool(dry_run),
        "filtering_strategy": plan["strategy"],
        "input_domains": len(before_interactions["domains"]),
        "input_interactions": before_interactions["interactions"],
        "input_items": before_items["items"],
        "input_timestamp_max": before_interactions["timestamp_max"],
        "input_timestamp_min": before_interactions["timestamp_min"],
        "input_users": before_interactions["users"],
        "item_min_interactions": item_min,
        "items_still_below_threshold": items_below,
        "iterations": iterations,
        "materialized": bool(materialized),
        "output_domains": len(after_interactions["domains"]),
        "output_interactions": after_interactions["interactions"],
        "output_items": after_items["items"],
        "output_paths": {
            "filtering_report": str(plan["report_path"]),
            "interactions": str(plan["interactions_path"]),
            "items": str(plan["items_path"]),
        },
        "output_timestamp_max": after_interactions["timestamp_max"],
        "output_timestamp_min": after_interactions["timestamp_min"],
        "output_users": after_interactions["users"],
        "per_domain": _domain_retention(
            before_interactions["domains"],
            before_items["domains"],
            after_interactions["domains"],
            after_items["domains"],
        ),
        "raw_files_unchanged": raw_snapshot_before == raw_snapshot_after,
        "raw_snapshot_after": raw_snapshot_after,
        "raw_snapshot_before": raw_snapshot_before,
        "removed_items": before_items["items"] - after_items["items"],
        "removed_users": before_interactions["users"] - after_interactions["users"],
        "retained_interaction_ratio": _ratio(after_interactions["interactions"], before_interactions["interactions"]),
        "retained_item_ratio": _ratio(after_items["items"], before_items["items"]),
        "retained_user_ratio": _ratio(after_interactions["users"], before_interactions["users"]),
        "status": "MATERIALIZED" if materialized else "DRY_RUN",
        "user_min_interactions": user_min,
        "users_still_below_threshold": users_below,
        "warnings": _report_warnings(converged, users_below, items_below, after_items),
    }
    return report


def _domain_retention(
    before_interactions: dict[str, Any],
    before_items: dict[str, Any],
    after_interactions: dict[str, Any],
    after_items: dict[str, Any],
) -> dict[str, Any]:
    domains = sorted(set(before_interactions) | set(before_items) | set(after_interactions) | set(after_items))
    rows = {}
    for domain in domains:
        before = {
            "interactions": int(before_interactions.get(domain, {}).get("interactions", 0)),
            "items": int(before_items.get(domain, {}).get("items", before_interactions.get(domain, {}).get("items", 0))),
            "users": int(before_interactions.get(domain, {}).get("users", 0)),
        }
        after = {
            "interactions": int(after_interactions.get(domain, {}).get("interactions", 0)),
            "items": int(after_items.get(domain, {}).get("items", after_interactions.get(domain, {}).get("items", 0))),
            "users": int(after_interactions.get(domain, {}).get("users", 0)),
        }
        rows[domain] = {
            "after": after,
            "before": before,
            "retained_interaction_ratio": _ratio(after["interactions"], before["interactions"]),
            "retained_item_ratio": _ratio(after["items"], before["items"]),
            "retained_user_ratio": _ratio(after["users"], before["users"]),
        }
    return rows


def _report_warnings(
    converged: bool,
    users_below: int,
    items_below: int,
    after_items: dict[str, Any],
) -> list[str]:
    warnings = []
    if not converged:
        warnings.append("iterative filtering reached max_iterations before convergence")
    if users_below:
        warnings.append(f"{users_below} users remain below the configured threshold")
    if items_below:
        warnings.append(f"{items_below} items remain below the configured threshold")
    if after_items.get("missing_text_items"):
        warnings.append(f"{after_items['missing_text_items']} retained items are missing text")
    return warnings


def _assert_safe_outputs(plan: dict[str, Any]) -> None:
    source_paths = {plan["source_interactions_path"], plan["source_items_path"]}
    final_paths = [plan["interactions_path"], plan["items_path"]]
    if any(path in source_paths for path in final_paths):
        raise AmazonFilteringError("Refusing to write filtered outputs over raw converted Amazon files")
    existing = [str(path) for path in final_paths if path.exists()]
    if existing and not plan["overwrite_existing"]:
        raise AmazonFilteringError(
            "Refusing to overwrite existing filtered files without overwrite_existing=true: " + ", ".join(existing)
        )
    if existing and plan["overwrite_existing"]:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for path in final_paths:
            if path.exists():
                path.replace(Path(str(path) + f".bak.{timestamp}"))


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise AmazonFilteringError(f"JSONL row {line_number} in {path} is not an object")
            yield value


def _raw_snapshot(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    snapshot = {}
    for key in ["source_interactions_path", "source_items_path"]:
        path = plan[key]
        stat = path.stat()
        snapshot[str(path)] = {"mtime_ns": stat.st_mtime_ns, "size_bytes": stat.st_size}
    return snapshot


def _has_item_text(row: dict[str, Any]) -> bool:
    return any(str(row.get(field) or "").strip() for field in ("title", "raw_text", "description"))


def _path_value(value: Any, default: Path) -> Path:
    return resolve_path(value) if value else default


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _ProgressLogger:
    def __init__(self, path: Path):
        self.path = path
        ensure_dir(path.parent)

    def log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(f"{timestamp}\t{message}\n")
