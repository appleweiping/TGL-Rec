"""Paper split and candidate artifact freezing for launch protocols."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from llm4rec.data.movielens_adapter import load_movielens_style, remap_user_item_ids
from llm4rec.data.readiness import NO_EXECUTION_FLAG
from llm4rec.data.splits import leave_one_out_split
from llm4rec.experiments.config import load_yaml_config, resolve_path
from llm4rec.io.artifacts import ensure_dir, write_json
from llm4rec.utils.env import current_git_commit


def plan_data_artifact_freeze(
    dataset_config_paths: list[str | Path],
    output_dir: str | Path = "outputs/launch/paper_v1/protocol",
    *,
    materialize: bool = False,
) -> dict[str, Any]:
    """Create split/candidate freeze manifests and optionally materialize artifacts.

    ``materialize=False`` preserves the Phase 8 behavior of recording planned paths.
    ``materialize=True`` writes deterministic leave-one-out split artifacts and
    fixed shared candidate artifacts without running any model or evaluator.
    """

    output = ensure_dir(resolve_path(output_dir))
    datasets = []
    split_entries = []
    candidate_entries = []
    for config_path in dataset_config_paths:
        spec = _artifact_spec(config_path)
        if materialize:
            entry = _materialize_dataset(spec)
        else:
            entry = _planned_dataset(spec)
        datasets.append(entry)
        split_entries.append(entry["split_artifact"])
        candidate_entries.append(entry["candidate_artifact"])
    manifest = {
        NO_EXECUTION_FLAG: True,
        "code_commit": current_git_commit(resolve_path(".")),
        "datasets": datasets,
        "materialize_requested": bool(materialize),
        "protocol_version": _shared_protocol_version(datasets),
        "status": "planned_only" if not materialize else "materialized",
    }
    write_json(output / "data_artifact_freeze_plan.json", manifest)
    write_json(
        output / "frozen_split_manifest.json",
        {
            NO_EXECUTION_FLAG: True,
            "materialized": bool(materialize),
            "planned_split_artifacts": split_entries,
            "protocol_version": manifest["protocol_version"],
            "split_protocol": "leave_one_out",
            "split_artifacts": split_entries if materialize else [],
            "status": manifest["status"],
        },
    )
    write_json(
        output / "frozen_candidate_manifest.json",
        {
            NO_EXECUTION_FLAG: True,
            "candidate_protocol": "fixed_shared_candidates",
            "candidate_artifacts": candidate_entries if materialize else [],
            "materialized": bool(materialize),
            "planned_candidate_artifacts": candidate_entries,
            "protocol_version": manifest["protocol_version"],
            "status": manifest["status"],
        },
    )
    return manifest


def _artifact_spec(config_path: str | Path) -> dict[str, Any]:
    config_file = resolve_path(config_path)
    config = load_yaml_config(config_file)
    if "manifest" in config and isinstance(config.get("dataset"), dict) and config["dataset"].get("config_path"):
        dataset_config_path = resolve_path(config["dataset"]["config_path"])
        dataset_config = load_yaml_config(dataset_config_path)
        dataset = dict(dataset_config.get("dataset", dataset_config))
        artifact_config = dict(dataset_config.get("paper_artifacts", dataset.get("paper_artifacts", {})))
        split_artifact = config.get("split_artifact") or artifact_config.get("split_artifact")
        candidate_artifact = config.get("candidate_artifact") or artifact_config.get("candidate_artifact")
        protocol_version = str(config.get("protocol_version", artifact_config.get("protocol_version", "protocol_v1")))
    else:
        dataset_config_path = config_file
        dataset = dict(config.get("dataset", config))
        artifact_config = dict(config.get("paper_artifacts", dataset.get("paper_artifacts", {})))
        protocol_version = str(artifact_config.get("protocol_version", "protocol_v1"))
        split_artifact = artifact_config.get("split_artifact")
        candidate_artifact = artifact_config.get("candidate_artifact")

    name = str(dataset.get("name", dataset_config_path.stem))
    artifact_dir = artifact_config.get("output_dir", f"outputs/artifacts/{protocol_version}/{name}")
    return {
        "candidate_artifact": resolve_path(candidate_artifact or f"{artifact_dir}/candidates.jsonl"),
        "candidate_pool_artifact": resolve_path(
            artifact_config.get("candidate_pool_artifact", f"{artifact_dir}/candidate_pool.json")
        ),
        "candidate_protocol": str(artifact_config.get("candidate_protocol", dataset.get("candidate_protocol", "fixed_sampled"))),
        "candidate_scope": str(artifact_config.get("candidate_scope", "global")),
        "candidate_size": artifact_config.get("candidate_size"),
        "candidate_storage": str(artifact_config.get("candidate_storage", "expanded")),
        "config_path": config_file,
        "dataset": dataset,
        "dataset_config_path": dataset_config_path,
        "name": name,
        "protocol_version": protocol_version,
        "seed": int(artifact_config.get("seed", dataset.get("seed", 0))),
        "split_artifact": resolve_path(split_artifact or f"{artifact_dir}/splits.jsonl"),
        "split_protocol": str(artifact_config.get("split_protocol", dataset.get("split_strategy", "leave_one_out"))),
    }


def _planned_dataset(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_artifact": _artifact_entry(spec, "candidate", status="planned"),
        "config_path": str(spec["dataset_config_path"]),
        "dataset": spec["name"],
        "materialized": False,
        "protocol_version": spec["protocol_version"],
        "split_artifact": _artifact_entry(spec, "split", status="planned"),
    }


def _materialize_dataset(spec: dict[str, Any]) -> dict[str, Any]:
    _progress(f"loading dataset {spec['name']}")
    interactions, items = _load_dataset(spec["dataset"])
    if spec["split_protocol"] != "leave_one_out":
        raise ValueError(f"Unsupported paper split protocol: {spec['split_protocol']}")
    _progress(f"building leave-one-out split for {spec['name']}: interactions={len(interactions)} items={len(items)}")
    labeled = leave_one_out_split(interactions)
    item_ids = sorted({str(row["item_id"]) for row in items})
    split_counts = _split_counts(labeled)
    _progress(f"writing split artifact for {spec['name']}: rows={len(labeled)}")
    split_result = _write_jsonl_atomic(spec["split_artifact"], labeled)
    _progress(f"writing candidate artifact for {spec['name']}: items={len(item_ids)}")
    candidate_pool_result = None
    if spec["candidate_storage"] == "shared_pool":
        candidate_pool_result = _write_json_atomic(
            spec["candidate_pool_artifact"],
            _candidate_pool_payload(
                item_ids,
                candidate_size=_candidate_size(spec["candidate_size"], len(item_ids)),
                dataset_name=spec["name"],
                protocol_version=spec["protocol_version"],
                seed=spec["seed"],
            ),
        )
    candidate_result = _write_jsonl_atomic(
        spec["candidate_artifact"],
        _candidate_rows(
            labeled,
            item_ids,
            candidate_protocol=spec["candidate_protocol"],
            candidate_size=_candidate_size(spec["candidate_size"], len(item_ids)),
            candidate_pool_artifact=spec["candidate_pool_artifact"],
            candidate_storage=spec["candidate_storage"],
            dataset_name=spec["name"],
            protocol_version=spec["protocol_version"],
            seed=spec["seed"],
        ),
    )
    candidate_size = candidate_result["candidate_size_min"]
    return {
        "candidate_artifact": {
            **_artifact_entry(spec, "candidate", status="materialized"),
            **candidate_result,
            "candidate_protocol": spec["candidate_protocol"],
            "candidate_scope": spec["candidate_scope"],
            "candidate_size": candidate_size,
            "candidate_storage": spec["candidate_storage"],
            "candidate_pool_artifact": candidate_pool_result,
            "target_included_rows": candidate_result["target_included_rows"],
        },
        "config_path": str(spec["dataset_config_path"]),
        "dataset": spec["name"],
        "item_count": len(item_ids),
        "materialized": True,
        "protocol_version": spec["protocol_version"],
        "split_artifact": {
            **_artifact_entry(spec, "split", status="materialized"),
            **split_result,
            "split_counts": split_counts,
            "split_protocol": spec["split_protocol"],
        },
        "user_count": len({str(row["user_id"]) for row in labeled}),
    }


def _load_dataset(dataset: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    adapter = str(dataset.get("adapter", "generic_jsonl"))
    paths = dict(dataset.get("paths", {}))
    if adapter == "movielens_style":
        interactions, items = load_movielens_style(paths)
        item_ids_in_data = {str(row["item_id"]) for row in interactions}
        items = [row for row in items if str(row["item_id"]) in item_ids_in_data]
        return remap_user_item_ids(interactions, items)
    interactions_path = paths.get("interactions")
    items_path = paths.get("items")
    if not interactions_path or not items_path:
        raise FileNotFoundError("Dataset config must provide paths.interactions and paths.items")
    return _read_jsonl(resolve_path(interactions_path)), _read_jsonl(resolve_path(items_path))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} in {path} is not an object")
            rows.append(row)
    return rows


def _candidate_rows(
    labeled: list[dict[str, Any]],
    item_ids: list[str],
    *,
    candidate_protocol: str,
    candidate_size: int,
    dataset_name: str,
    protocol_version: str,
    seed: int,
    candidate_pool_artifact: Path | None = None,
    candidate_storage: str = "expanded",
):
    catalog = sorted({str(item_id) for item_id in item_ids})
    _progress(
        f"preparing candidates for {dataset_name}: protocol={candidate_protocol} "
        f"candidate_size={candidate_size} catalog={len(catalog)}"
    )
    sampling_catalog = _sampling_catalog(catalog, seed=seed, dataset_name=dataset_name)
    yielded = 0
    for row in labeled:
        if row.get("split") not in {"valid", "test"}:
            continue
        target = str(row["item_id"])
        if yielded == 0:
            _progress(f"building first candidate row for {dataset_name}: user={row['user_id']} split={row['split']}")
        candidates = None
        if candidate_storage == "expanded":
            candidates = _candidate_items(
                catalog,
                target,
                protocol=candidate_protocol,
                candidate_size=candidate_size,
                seed=seed,
                row_key=f"{dataset_name}|{row['split']}|{row['user_id']}|{target}",
                sampling_catalog=sampling_catalog,
            )
        if yielded == 0:
            size = len(candidates) if candidates is not None else candidate_size
            _progress(f"first candidate row ready for {dataset_name}: size={size}")
        yielded += 1
        row_payload = {
            "candidate_protocol": candidate_protocol,
            "candidate_size": len(candidates) if candidates is not None else candidate_size,
            "candidate_storage": candidate_storage,
            "domain": row.get("domain"),
            "metadata": {
                NO_EXECUTION_FLAG: True,
                "candidate_pool_artifact": None if candidate_pool_artifact is None else str(candidate_pool_artifact),
                "dataset": dataset_name,
                "protocol_version": protocol_version,
                "seed": seed,
                "shared_across_methods": True,
                "shared_across_seeds": True,
                "target_included": True if candidates is None else target in candidates,
            },
            "split": str(row["split"]),
            "target_item": target,
            "target_included": True if candidates is None else target in candidates,
            "user_id": str(row["user_id"]),
        }
        if candidates is None:
            row_payload["candidate_pool_id"] = _candidate_pool_id(dataset_name, seed, candidate_size)
            row_payload["candidate_pool_artifact"] = None if candidate_pool_artifact is None else str(candidate_pool_artifact)
            row_payload["target_inclusion_rule"] = "pool_if_present_else_first_k_minus_one_plus_target"
        else:
            row_payload["candidate_items"] = candidates
        yield row_payload


def _candidate_items(
    catalog: list[str],
    target: str,
    *,
    protocol: str,
    candidate_size: int,
    seed: int,
    row_key: str,
    sampling_catalog: list[str],
) -> list[str]:
    if target not in catalog:
        catalog.append(target)
        catalog.sort()
    if protocol == "full_catalog":
        return catalog
    if protocol not in {"fixed_sampled", "fixed_shared_candidates"}:
        raise ValueError(f"Unsupported paper candidate protocol: {protocol}")
    size = min(int(candidate_size), len(catalog))
    if size <= 0:
        raise ValueError("candidate_size must be positive")
    negatives_needed = max(0, size - 1)
    negatives = _sample_without_target(sampling_catalog, target, negatives_needed, seed=seed, row_key=row_key)
    candidates = sorted([*negatives, target])
    if len(candidates) != size or target not in candidates:
        raise ValueError(f"Invalid candidate set for {row_key}: size={len(candidates)} target_included={target in candidates}")
    return candidates


def _sample_without_target(
    catalog: list[str],
    target: str,
    count: int,
    *,
    seed: int,
    row_key: str,
) -> list[str]:
    if count >= len(catalog) - 1:
        return [item for item in catalog if item != target]
    digest = hashlib.sha256(f"{seed}|{row_key}".encode("utf-8")).digest()
    start = int.from_bytes(digest[:8], byteorder="big", signed=False) % len(catalog)
    chosen: set[str] = set()
    offset = 0
    while len(chosen) < count:
        item = catalog[(start + offset) % len(catalog)]
        offset += 1
        if item != target:
            chosen.add(item)
    return sorted(chosen)


def _candidate_pool_payload(
    item_ids: list[str],
    *,
    candidate_size: int,
    dataset_name: str,
    protocol_version: str,
    seed: int,
) -> dict[str, Any]:
    catalog = sorted({str(item_id) for item_id in item_ids})
    sampling_catalog = _sampling_catalog(catalog, seed=seed, dataset_name=dataset_name)
    pool = sampling_catalog[: min(candidate_size, len(sampling_catalog))]
    return {
        NO_EXECUTION_FLAG: True,
        "candidate_items": pool,
        "candidate_pool_id": _candidate_pool_id(dataset_name, seed, len(pool)),
        "candidate_size": len(pool),
        "dataset": dataset_name,
        "negative_pool_for_targets_outside_pool": pool[: max(0, len(pool) - 1)],
        "protocol_version": protocol_version,
        "seed": seed,
        "target_inclusion_rule": "pool_if_present_else_first_k_minus_one_plus_target",
    }


def _candidate_pool_id(dataset_name: str, seed: int, candidate_size: int) -> str:
    return f"{dataset_name}_seed{seed}_candidate_pool_{candidate_size}"


def _sampling_catalog(item_ids: list[str], *, seed: int, dataset_name: str) -> list[str]:
    return [
        item
        for _, item in sorted(
            (hashlib.sha256(f"{seed}|{dataset_name}|{item}".encode("utf-8")).digest(), str(item))
            for item in item_ids
        )
    ]


def _candidate_size(value: Any, item_count: int) -> int:
    if value in (None, "", "full", "FULL"):
        return int(item_count)
    return int(value)


def _write_jsonl_atomic(path: Path, rows: Any) -> dict[str, Any]:
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    digest = hashlib.sha256()
    rows_written = 0
    target_included_rows = 0
    candidate_size_min: int | None = None
    candidate_size_max: int | None = None
    with tmp.open("wb") as handle:
        for row in rows:
            if "candidate_items" in row:
                size = len(row["candidate_items"])
                candidate_size_min = size if candidate_size_min is None else min(candidate_size_min, size)
                candidate_size_max = size if candidate_size_max is None else max(candidate_size_max, size)
                if str(row.get("target_item")) in {str(item) for item in row["candidate_items"]}:
                    target_included_rows += 1
            elif "candidate_size" in row:
                size = int(row["candidate_size"])
                candidate_size_min = size if candidate_size_min is None else min(candidate_size_min, size)
                candidate_size_max = size if candidate_size_max is None else max(candidate_size_max, size)
                if bool(row.get("target_included")):
                    target_included_rows += 1
            line = (json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n").encode("utf-8")
            handle.write(line)
            digest.update(line)
            rows_written += 1
            if rows_written % 50000 == 0:
                _progress(f"wrote {rows_written} rows to {path}")
    tmp.replace(path)
    result = {
        "bytes": path.stat().st_size,
        "rows": rows_written,
        "sha256": digest.hexdigest(),
    }
    if candidate_size_min is not None:
        result["candidate_size_min"] = candidate_size_min
        result["candidate_size_max"] = candidate_size_max
        result["target_included_rows"] = target_included_rows
    return result


def _write_json_atomic(path: Path, data: Any) -> dict[str, Any]:
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    payload = (json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")
    tmp.write_bytes(payload)
    tmp.replace(path)
    return {
        "artifact_type": "candidate_pool",
        "bytes": path.stat().st_size,
        "path": str(path),
        "rows": 1,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "status": "materialized",
    }


def _progress(message: str) -> None:
    print(f"[artifact_freeze] {message}", file=sys.stderr, flush=True)


def _artifact_entry(spec: dict[str, Any], artifact_type: str, *, status: str) -> dict[str, Any]:
    path = spec["split_artifact"] if artifact_type == "split" else spec["candidate_artifact"]
    return {
        "artifact_type": artifact_type,
        "dataset": spec["name"],
        "path": str(path),
        "protocol_version": spec["protocol_version"],
        "status": status,
    }


def _split_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"train": 0, "valid": 0, "test": 0}
    for row in rows:
        split = str(row.get("split"))
        counts[split] = counts.get(split, 0) + 1
    return counts


def _split_order(split: str) -> int:
    return {"valid": 0, "test": 1}.get(split, 2)


def _shared_protocol_version(datasets: list[dict[str, Any]]) -> str:
    versions = sorted({str(row.get("protocol_version", "protocol_v1")) for row in datasets})
    return versions[0] if len(versions) == 1 else ",".join(versions)
