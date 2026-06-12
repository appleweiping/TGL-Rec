#!/usr/bin/env python3
"""Build CC-PACE user profile slots from TRAIN-history titles (no future leakage).

Reads the same-candidate ranking jsonl (full schema: history = pre-cutoff TRAIN
titles), aggregates each user's history with the title-lexicon miners in
llm4rec.methods.cc_pace.text_facets, and writes profiles.json keyed by user_id:

  {user_id: {top_categories, liked_brands, concerns, routine_step,
             ingredient_prefs}}            # price_band omitted: no price data

These are the long-term profile slots schema.py renders into the judge's user
block -- mandatory for beauty, where the SOTA baseline (promax) wins precisely
because beauty preference is profile-expressible.

Usage:
  python scripts/build_cc_pace_profiles.py \
      --task <ranking_test.jsonl> --out outputs/cc_pace_beauty/profiles.json
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm4rec.methods.cc_pace.text_facets import profile_from_history  # noqa: E402


def parse_listish(x):
    if isinstance(x, list):
        return x
    if not x:
        return []
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []


def main(argv=None) -> str:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="same-candidate ranking jsonl (full schema)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", default="beauty")
    ap.add_argument("--max-per-slot", type=int, default=3)
    args = ap.parse_args(argv)

    profiles: dict[str, dict] = {}
    slot_fill: Counter = Counter()
    n_rows = 0
    with open(args.task, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            n_rows += 1
            user_id = str(d.get("user_id"))
            history_titles = [str(t) for t in parse_listish(d.get("history"))]
            prof = profile_from_history(
                history_titles, domain=args.domain, max_per_slot=args.max_per_slot
            )
            if prof:
                profiles[user_id] = prof
                for slot in prof:
                    slot_fill[slot] += 1

    payload = {
        "provenance": {
            "builder": "build_cc_pace_profiles.py",
            "task": os.path.abspath(args.task),
            "domain": args.domain,
            "n_task_rows": n_rows,
            "n_users_with_profile": len(profiles),
            "slot_fill_counts": dict(slot_fill),
            "source": "TRAIN history titles only (no candidate/target fields)",
        },
        "profiles": profiles,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    print(
        f"wrote {args.out}: {len(profiles)}/{n_rows} users with >=1 slot; "
        f"fill={dict(slot_fill)}"
    )
    return args.out


if __name__ == "__main__":
    main()
