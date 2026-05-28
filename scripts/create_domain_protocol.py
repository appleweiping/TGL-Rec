"""Generate same-candidate evaluation protocol for new Amazon 2023 domains.

Aligns with existing Pony protocol:
- Leave-one-out split (last interaction = test, second-to-last = valid)
- 101 candidates per user (1 positive + 100 popularity-sampled negatives)
- Min 3 interactions per user
- Random user selection (up to 10000 users)

Usage:
    python scripts/create_domain_protocol.py \
        --reviews-path data/raw/Video_Games.jsonl \
        --meta-path data/raw/meta_Video_Games.jsonl \
        --domain video_games \
        --output-dir outputs/baselines/external_tasks/video_games_large10000_100neg \
        --max-users 10000 \
        --num-negatives 100 \
        --min-interactions 3 \
        --seed 20260528
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm4rec.io.artifacts import ensure_dir, write_json, write_jsonl


def load_reviews(path: Path, min_interactions: int = 3) -> dict[str, list[dict]]:
    """Load reviews and filter users with min interactions."""
    user_interactions: dict[str, list[dict]] = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            uid = str(row.get("user_id", ""))
            iid = str(row.get("parent_asin", row.get("asin", "")))
            ts = row.get("timestamp", 0)
            if not uid or not iid:
                continue
            if uid not in user_interactions:
                user_interactions[uid] = []
            user_interactions[uid].append({
                "user_id": uid,
                "item_id": iid,
                "timestamp": ts,
                "rating": row.get("rating", 0),
            })

    # Filter by min interactions
    filtered = {
        uid: sorted(ints, key=lambda x: x["timestamp"])
        for uid, ints in user_interactions.items()
        if len(ints) >= min_interactions
    }
    return filtered


def load_metadata(path: Path) -> dict[str, dict]:
    """Load item metadata."""
    items = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            iid = str(row.get("parent_asin", row.get("asin", "")))
            if iid:
                items[iid] = {
                    "item_id": iid,
                    "title": row.get("title", ""),
                    "category": ", ".join(row.get("categories", row.get("main_category", [""]))),
                    "price": row.get("price", ""),
                }
    return items


def create_protocol(
    user_interactions: dict[str, list[dict]],
    item_metadata: dict[str, dict],
    *,
    max_users: int = 10000,
    num_negatives: int = 100,
    seed: int = 42,
    domain: str = "unknown",
) -> dict:
    """Create same-candidate evaluation protocol."""
    random.seed(seed)

    # Compute item popularity for negative sampling
    item_counts: Counter = Counter()
    all_items: set[str] = set()
    for uid, ints in user_interactions.items():
        for interaction in ints:
            item_counts[interaction["item_id"]] += 1
            all_items.add(interaction["item_id"])

    items_by_popularity = sorted(item_counts.keys(), key=lambda x: item_counts[x], reverse=True)

    # Select users
    eligible_users = list(user_interactions.keys())
    random.shuffle(eligible_users)
    selected_users = eligible_users[:max_users]

    # Create splits and candidates
    train_interactions = []
    valid_ranking = []
    test_ranking = []
    candidate_items_set: set[str] = set()

    for uid in selected_users:
        ints = user_interactions[uid]
        if len(ints) < 3:
            continue

        # Leave-one-out: last = test, second-to-last = valid
        train_ints = ints[:-2]
        valid_target = ints[-2]
        test_target = ints[-1]

        # Train interactions
        for interaction in train_ints:
            train_interactions.append(interaction)

        # User's interacted items (for negative filtering)
        user_items = {i["item_id"] for i in ints}

        # Sample negatives (popularity-based, excluding user's items)
        negatives = []
        for item in items_by_popularity:
            if item not in user_items:
                negatives.append(item)
            if len(negatives) >= num_negatives:
                break

        # Valid ranking
        valid_candidates = [valid_target["item_id"]] + negatives
        random.shuffle(valid_candidates)
        candidate_items_set.update(valid_candidates)
        valid_ranking.append({
            "user_id": uid,
            "target_item": valid_target["item_id"],
            "candidate_items": valid_candidates,
            "source_event_id": f"{domain}_valid_{uid}",
        })

        # Test ranking (include valid in history)
        test_candidates = [test_target["item_id"]] + negatives
        random.shuffle(test_candidates)
        candidate_items_set.update(test_candidates)
        test_ranking.append({
            "user_id": uid,
            "target_item": test_target["item_id"],
            "candidate_items": test_candidates,
            "source_event_id": f"{domain}_test_{uid}",
        })

    return {
        "train_interactions": train_interactions,
        "valid_ranking": valid_ranking,
        "test_ranking": test_ranking,
        "candidate_items": sorted(candidate_items_set),
        "item_metadata": {iid: item_metadata.get(iid, {"item_id": iid, "title": iid}) for iid in candidate_items_set},
        "stats": {
            "domain": domain,
            "selected_users": len(selected_users),
            "train_interactions": len(train_interactions),
            "valid_events": len(valid_ranking),
            "test_events": len(test_ranking),
            "candidate_items": len(candidate_items_set),
            "num_negatives": num_negatives,
        },
    }


def save_protocol(protocol: dict, output_dir: Path, domain: str, split: str) -> None:
    """Save protocol in Pony-compatible format."""
    ensure_dir(output_dir)

    # Train interactions CSV
    train_path = output_dir / "train_interactions.csv"
    with open(train_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["user_id", "item_id", "timestamp", "rating"])
        writer.writeheader()
        for row in protocol["train_interactions"]:
            writer.writerow(row)

    # Ranking JSONL
    ranking_data = protocol["valid_ranking"] if split == "valid" else protocol["test_ranking"]
    ranking_path = output_dir / f"ranking_{split}.jsonl"
    write_jsonl(ranking_path, ranking_data)

    # Candidate items CSV
    cand_path = output_dir / "candidate_items.csv"
    with open(cand_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id"])
        for item in protocol["candidate_items"]:
            writer.writerow([item])

    # Item metadata CSV
    meta_path = output_dir / "item_metadata.csv"
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "title", "category", "price"])
        writer.writeheader()
        for iid, meta in protocol["item_metadata"].items():
            writer.writerow(meta)

    # Metadata JSON
    write_json(output_dir / "metadata.json", {
        **protocol["stats"],
        "split_name": split,
        "protocol": "large_scale_leave_one_out_same_candidate_sampled_ranking",
        "num_candidates": 101,
        "negative_sampling": "popularity over items not interacted by the selected user",
        "required_score_schema": ["source_event_id", "user_id", "item_id", "score"],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Create same-candidate protocol for new domain")
    parser.add_argument("--reviews-path", required=True, type=Path)
    parser.add_argument("--meta-path", required=True, type=Path)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-users", type=int, default=10000)
    parser.add_argument("--num-negatives", type=int, default=100)
    parser.add_argument("--min-interactions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260528)
    args = parser.parse_args()

    print(f"[protocol] Loading reviews from {args.reviews_path}...")
    user_interactions = load_reviews(args.reviews_path, min_interactions=args.min_interactions)
    print(f"[protocol] {len(user_interactions)} users with >= {args.min_interactions} interactions")

    print(f"[protocol] Loading metadata from {args.meta_path}...")
    item_metadata = load_metadata(args.meta_path)
    print(f"[protocol] {len(item_metadata)} items with metadata")

    print(f"[protocol] Creating protocol (max_users={args.max_users}, negatives={args.num_negatives})...")
    protocol = create_protocol(
        user_interactions,
        item_metadata,
        max_users=args.max_users,
        num_negatives=args.num_negatives,
        seed=args.seed,
        domain=args.domain,
    )

    # Save valid and test splits
    valid_dir = args.output_dir / f"{args.domain}_large{args.max_users}_100neg_valid_same_candidate"
    test_dir = args.output_dir / f"{args.domain}_large{args.max_users}_100neg_test_same_candidate"

    print(f"[protocol] Saving valid split to {valid_dir}...")
    save_protocol(protocol, valid_dir, args.domain, "valid")

    print(f"[protocol] Saving test split to {test_dir}...")
    save_protocol(protocol, test_dir, args.domain, "test")

    print(f"\n[protocol] Done! Stats:")
    for k, v in protocol["stats"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
