#!/usr/bin/env python3
"""GO / KILL verdict for the CC-PACE beauty zero-shot probe.

Reads the variant outputs of scripts/cc_pace_beauty.py (full + text_only, with
their .per_user.jsonl checkpoint files) and applies the thresholds from
docs/method_v2_decision_CC-PACE.md / docs/HOW_TO_RUN_CC_PACE.md:

  GO to LoRA:  full zero-shot NDCG@10 >= 0.13  AND  full > text_only (positive
               CF-token ablation gap; paired user bootstrap p reported).
  Otherwise:   KILL/reframe -> re-run the 3-seat ARIS discussion.

Usage:
  python scripts/cc_pace_go_verdict.py --dir outputs/cc_pace_beauty --out outputs/cc_pace_beauty/go_verdict.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

GO_NDCG10 = 0.13
SOTA_NDCG10 = 0.1506


def load_variant(dir_: str, name: str):
    with open(os.path.join(dir_, f"{name}.json"), encoding="utf-8") as fh:
        summary = json.load(fh)
    per_user = {}
    pu_path = os.path.join(dir_, f"{name}.json.per_user.jsonl")
    if os.path.exists(pu_path):
        with open(pu_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    per_user[d["user_id"]] = d
    return summary, per_user


def paired_bootstrap_p(a: dict, b: dict, key: str = "NDCG@10", n_boot: int = 5000, seed: int = 13):
    """P(mean(a-b) <= 0) over common users; small p -> a reliably above b."""
    common = sorted(set(a) & set(b))
    if not common:
        return None, 0
    d = np.array([a[u][key] - b[u][key] for u in common])
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    return float((boots <= 0).mean()), len(common)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    full, full_pu = load_variant(args.dir, "full")
    text, text_pu = load_variant(args.dir, "text_only")
    n_full = full["n_examples"]
    n_text = text["n_examples"]
    f10 = full["metrics"]["NDCG@10"]
    t10 = text["metrics"]["NDCG@10"]
    p_gap, n_common = paired_bootstrap_p(full_pu, text_pu)

    go = (f10 >= GO_NDCG10) and (f10 > t10)
    verdict = {
        "decision": "GO" if go else "KILL_OR_REFRAME",
        "criteria": {
            "full_ndcg10_ge_0.13": f10 >= GO_NDCG10,
            "full_gt_text_only": f10 > t10,
        },
        "full": {"NDCG@10": f10, "n": n_full, "mock": full.get("mock")},
        "text_only": {"NDCG@10": t10, "n": n_text, "mock": text.get("mock")},
        "cf_gap": {"delta_ndcg10": f10 - t10, "paired_bootstrap_p": p_gap, "n_paired": n_common},
        "context": {
            "go_threshold": GO_NDCG10,
            "sota_bar": SOTA_NDCG10,
            "note_if_go": "proceed to scripts/train_cc_pace_lora.py, then evaluate with --adapter; "
                          "STRONG GO needs post-LoRA >= 0.1506 (p<0.05) + panel-corruption >= 30% + "
                          "text_only below the CF baseline",
            "note_if_kill": "re-run the 3-seat ARIS discussion (CLAUDE.md rule 9) and redesign "
                            "before spending GPU on LoRA",
        },
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, indent=2)
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
