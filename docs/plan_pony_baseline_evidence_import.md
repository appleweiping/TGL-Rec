# Plan: Import Pony-Rec 8-domain × 8-official-baseline lightweight evidence into TGL-Rec

## Goal (user's intent)
Keep a copy, inside TGL-Rec, of the **lightweight** official-baseline evidence produced by
Pony-Rec / Uncertainty — covering **8 domains × 8 official baselines ONLY**, explicitly
**excluding Pony's own C-CRP method rows**. TGL-Rec reuses these as its comparison baselines
(same 101-candidate protocol, same Qwen3-8B backbone) instead of re-running any baseline.

## Verified facts (audited 2026-06-07)
- 8 domains = two groups:
  - NEW strict: `sports, toys, home, tools` (10,000 users each)
  - EARLY/supplementary: `beauty, books, electronics, movies` (beauty 973 users, rest 10,000)
- 8 standard official baselines: `elmrec, irllrec, llm2rec, llmemb, llmesr, proex, promax, rlmrec`
  - (NOT `setrec` — that was TGL's old 8th baseline; Pony's canonical 8th is `llmemb`. We standardize on Pony's 8.)
- Server `pony-rec-gpu:/home/ajifang/projects/pony-rec-rescue-shadow-v6` reachable; all
  8 domains × 8 baselines have per-baseline dirs there.
- New 4 domains: full per-baseline lightweight evidence present (provenance, run_summary,
  score_audit, same_candidate summary csv, coverage).
- Early 4 domains: only 4/8 baselines have full `fairness_provenance.json` locally on Pony;
  but **summary CSVs (HR/NDCG/MRR) exist for all 8** per domain. Server is the source of truth.
- Current TGL `outputs/pony_official_baselines/` = 4 GB tarballs, **gitignored, not tracked** —
  i.e. effectively NOT preserved. This plan replaces it with a tracked lightweight copy.

## Decisions (user-approved)
1. Granularity = **provenance + metric tables** per domain×baseline (~tens of KB/dir; no 42MB
   ranking_eval_records.csv).
2. Early-domain source = **pull uniformly from Pony server** (consistent granularity across all 8).
3. **Track in git** (place under already-tracked `data/`, not the ignored `outputs/`).

## Target layout (new, git-tracked)
```
data/pony_official_baselines/
  README.md                              # provenance, what/where-from, exclusions, protocol
  baseline_comparison_8domains.csv       # MASTER: 64 rows (8 domains × 8 baselines), C-CRP excluded
  domains/
    <domain>/<baseline>/
      fairness_provenance.json
      <baseline>_official_run_summary.json
      <baseline>_official_score_audit.json
      same_candidate_external_baseline_summary.csv
      external_score_coverage.csv
  IMPORT_MANIFEST.json                   # source host/paths, sha256, pull timestamp, file list
```

## Steps
1. **Pre-clean**: leave the old 4GB `outputs/pony_official_baselines/` as-is (gitignored); new
   tracked copy goes to `data/pony_official_baselines/`. (Optionally delete tarballs later — ask.)
2. **Server-side stage**: on pony-rec-gpu, for each of 8 domains × 8 baselines, locate the
   per-baseline dir and copy ONLY the lightweight files (whitelist above) into a temp staging
   tree `/tmp/tgl_pony_evidence/<domain>/<baseline>/`. Skip scores.csv, *.pt, predictions/,
   ranking_eval_records.csv. Compute sha256 for each copied file.
3. **Pull**: `scp -r` (or rsync over the configured ssh) the staging tree to
   `data/pony_official_baselines/domains/`. Expected total: a few MB.
4. **Build master CSV**: parse each `same_candidate_external_baseline_summary.csv` (+ ledger /
   main_comparison_table for early domains as cross-check) into one
   `baseline_comparison_8domains.csv` with columns:
   `domain, baseline, n_users, HR@5, HR@10, HR@20, NDCG@5, NDCG@10, NDCG@20, MRR,
    sample_count, avg_candidates, score_coverage_rate, official_repo, pinned_commit, source_path`.
   **Assert: exactly 64 rows, zero rows containing "ccrp".**
5. **Manifest + README**: write IMPORT_MANIFEST.json (source paths, sha256, timestamp) and a
   README documenting protocol (101 candidates, Qwen3-8B, same-candidate), the 8 baselines,
   the C-CRP exclusion, and that heavy artifacts stay server-side by design.
6. **Verify**: assert 8 domains × 8 baselines = 64 dirs present; each dir has the whitelist files;
   master CSV has 64 non-C-CRP rows; numbers in master CSV match per-dir summaries (spot check).
7. **Git**: stage `data/pony_official_baselines/`, commit on a new branch, push.
   Commit msg notes source provenance + exclusions. (No secrets in these JSONs — verified small.)

## Cross-check / honesty notes (carried forward, not part of import)
- This import only makes the **comparison baselines** available. It does NOT establish that
  TGL-Rec's method beats them — prior audit showed TGL's temporal evidence lost to
  popularity/history-only on its own runs. Method runs on these 8 domains are a SEPARATE,
  later task (needs server training-history data; tracked in memory `tglrec-phase10-status`,
  which is currently over-optimistic and should be corrected separately).
- Early-domain caveat preserved in README: beauty=973 users (supplementary), not SOTA-grade.

## Out of scope (explicitly NOT in this task)
- Running TGL-Rec's own method on the 8 domains.
- Pulling training-history / heavy data.
- Editing the paper's results tables.
