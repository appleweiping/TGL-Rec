"""Prepare domain data for TGL-Rec gate training from imported artifacts.

Bridges the Week8 import format to the gate trainer's expected format:
- data/domains/<domain>/train_interactions.jsonl
- data/domains/<domain>/item_records.jsonl
- data/domains/<domain>/same_candidate/ranking_valid.jsonl
- data/domains/<domain>/same_candidate/ranking_test.jsonl

Usage:
    python scripts/prepare_domain_data.py \
        --artifacts-root outputs/artifacts/protocol_week8_large10000_same_candidate \
        --output-root data/domains \
        --domains beauty books electronics movies
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm4rec.io.artifacts import ensure_dir, write_json, write_jsonl


def prepare_domain(artifacts_dir: Path, output_dir: Path, domain: str) -> dict:
    """Convert imported artifacts to gate-trainer format for one domain."""

    ensure_dir(output_dir)
    ensure_dir(output_dir / "same_candidate")

    splits_path = artifacts_dir / "splits.jsonl"
    items_path = artifacts_dir / "item_metadata.jsonl"
    candidates_path = artifacts_dir / "candidates.jsonl"

    if not splits_path.exists():
        return {"domain": domain, "status": "skipped", "reason": f"No splits.jsonl in {artifacts_dir}"}

    train_interactions = []
    valid_targets = []
    test_targets = []

    with open(splits_path) as f:
        for line in f:
            row = json.loads(line)
            split = row.get("split", "train")
            if split == "train":
                train_interactions.append({
                    "user_id": row["user_id"],
                    "item_id": row["item_id"],
                    "timestamp": row.get("timestamp", 0),
                    "domain": domain,
                })
            elif split == "valid":
                valid_targets.append(row)
            elif split == "test":
                test_targets.append(row)

    item_records = []
    if items_path.exists():
        with open(items_path) as f:
            for line in f:
                item_records.append(json.loads(line))

    valid_ranking = []
    test_ranking = []
    if candidates_path.exists():
        with open(candidates_path) as f:
            for line in f:
                row = json.loads(line)
                split = row.get("split", "valid")
                entry = {
                    "user_id": row["user_id"],
                    "candidate_items": row.get("candidate_items", []),
                    "target_item": row.get("target_item", row.get("positive_item", "")),
                    "source_event_id": row.get("source_event_id", row.get("event_id", "")),
                }
                if split == "valid":
                    valid_ranking.append(entry)
                elif split == "test":
                    test_ranking.append(entry)

    write_jsonl(output_dir / "train_interactions.jsonl", train_interactions)
    write_jsonl(output_dir / "item_records.jsonl", item_records)
    if valid_ranking:
        write_jsonl(output_dir / "same_candidate" / "ranking_valid.jsonl", valid_ranking)
    if test_ranking:
        write_jsonl(output_dir / "same_candidate" / "ranking_test.jsonl", test_ranking)

    manifest = {
        "domain": domain,
        "status": "prepared",
        "train_interactions": len(train_interactions),
        "item_records": len(item_records),
        "valid_ranking_rows": len(valid_ranking),
        "test_ranking_rows": len(test_ranking),
        "source_artifacts_dir": str(artifacts_dir),
    }
    write_json(output_dir / "prepare_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare domain data for gate training")
    parser.add_argument("--artifacts-root", required=True, help="Root of imported artifacts")
    parser.add_argument("--output-root", required=True, help="Output root for prepared data")
    parser.add_argument("--domains", nargs="+", default=["beauty", "books", "electronics", "movies"])
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts_root)
    output_root = Path(args.output_root)

    results = []
    for domain in args.domains:
        domain_key = domain
        artifacts_dir = artifacts_root / domain_key
        if not artifacts_dir.exists():
            for candidate in artifacts_root.iterdir():
                if candidate.is_dir() and domain in candidate.name:
                    artifacts_dir = candidate
                    domain_key = candidate.name
                    break

        output_dir = output_root / domain
        print(f"[prepare] {domain}: {artifacts_dir} -> {output_dir}")
        result = prepare_domain(artifacts_dir, output_dir, domain)
        results.append(result)
        print(f"  Status: {result['status']}, train={result.get('train_interactions', 0)}, "
              f"valid={result.get('valid_ranking_rows', 0)}, test={result.get('test_ranking_rows', 0)}")

    write_json(output_root / "prepare_summary.json", results)
    print(f"\n[prepare] Done. Summary written to {output_root / 'prepare_summary.json'}")


if __name__ == "__main__":
    main()
