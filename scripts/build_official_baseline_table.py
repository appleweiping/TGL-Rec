#!/usr/bin/env python3
"""Build the master official-baseline comparison table for TGL-Rec.

Reads each per-(domain,baseline) ``same_candidate_external_baseline_summary.csv``
under ``data/pony_official_baselines/domains`` and emits a single tidy CSV with
one row per (domain, baseline). Only the eight official LLM4Rec baselines are
included; no method rows of our own are present in the source dirs, but we still
assert that nothing resembling an internal method leaks in.
"""
from __future__ import annotations

import csv
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "pony_official_baselines"
DOMAINS_DIR = ROOT / "domains"

DOMAINS = ["sports", "toys", "home", "tools", "beauty", "books", "electronics", "movies"]
BASELINES = ["elmrec", "irllrec", "llm2rec", "llmemb", "llmesr", "proex", "promax", "rlmrec"]

OUT_COLS = [
    "domain", "baseline", "sample_count", "avg_candidates", "score_coverage_rate",
    "MRR", "HR@5", "NDCG@5", "HR@10", "NDCG@10", "HR@20", "NDCG@20",
    "status_label", "implementation_status", "comparison_variant", "source_summary_path",
]


def read_summary(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"empty summary: {path}")
    return rows[0]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    out_rows = []
    manifest_files = []
    missing = []
    for d in DOMAINS:
        for b in BASELINES:
            pair = DOMAINS_DIR / d / b
            summ = pair / "same_candidate_external_baseline_summary.csv"
            if not summ.exists():
                missing.append(f"{d}/{b}")
                continue
            r = read_summary(summ)
            out_rows.append({
                "domain": d,
                "baseline": b,
                "sample_count": r.get("sample_count", ""),
                "avg_candidates": r.get("avg_candidates", ""),
                "score_coverage_rate": r.get("score_coverage_rate", ""),
                "MRR": r.get("MRR", ""),
                "HR@5": r.get("HR@5", ""),
                "NDCG@5": r.get("NDCG@5", ""),
                "HR@10": r.get("HR@10", ""),
                "NDCG@10": r.get("NDCG@10", ""),
                "HR@20": r.get("HR@20", ""),
                "NDCG@20": r.get("NDCG@20", ""),
                "status_label": r.get("status_label", ""),
                "implementation_status": r.get("implementation_status", ""),
                "comparison_variant": r.get("comparison_variant", ""),
                "source_summary_path": r.get("scores_path", ""),
            })
            for f in sorted(pair.iterdir()):
                if f.is_file():
                    manifest_files.append({
                        "path": str(f.relative_to(ROOT)).replace("\\", "/"),
                        "bytes": f.stat().st_size,
                        "sha256": sha256(f),
                    })

    # --- assertions ---
    assert not missing, f"missing pairs: {missing}"
    assert len(out_rows) == 64, f"expected 64 rows, got {len(out_rows)}"
    leaked = [r for r in out_rows if "ccrp" in (r["baseline"] + r["domain"]).lower()]
    assert not leaked, f"internal/ccrp rows leaked: {leaked}"

    out_csv = ROOT / "baseline_comparison_8domains.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_COLS)
        w.writeheader()
        w.writerows(out_rows)

    manifest = {
        "description": "Integrity manifest for official-baseline comparison evidence.",
        "protocol": "same-candidate ranking, 101 candidates/user, Qwen3-8B backbone",
        "domains": DOMAINS,
        "baselines": BASELINES,
        "n_pairs": len(out_rows),
        "files": manifest_files,
    }
    (ROOT / "IMPORT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"OK rows={len(out_rows)} files={len(manifest_files)} -> {out_csv.name}")


if __name__ == "__main__":
    main()
