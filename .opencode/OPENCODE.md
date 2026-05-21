# OPENCODE.md — TGL-Rec

Read `AGENTS.md` first. That is the authoritative contract.

## Context
TGL-Rec investigates whether LLM-based recommenders actually use temporal signals.
Proposed method: TDIG + graph-to-language evidence + need-aware gated reranker.

## Your Role
Implementation, testing, documentation updates.

## Quick Commands
- Run unit tests: `python -m pytest tests/unit/ -x`
- Run smoke tests: `python -m pytest tests/smoke/ -x`
- Lint: `ruff check src/ scripts/`
- CLI: `tglrec --help`
- Preprocess: `python scripts/preprocess_amazon.py --dataset beauty`

## Key Paths
- Active framework: `src/llm4rec/`
- Legacy CPU tools: `src/tglrec/`
- Scripts: `scripts/` (61 files)
- Configs: `configs/`
- Outputs: `outputs/` (experiment artifacts)

## Current State
- Phase 10: four-domain same-candidate experiments
- 7 official baselines completed
- LoRA fine-tuning implemented
- Paper outline exists but no results written yet

## Server Access

Remote GPU server `pony-rec-gpu` is directly accessible via SSH (key-based auth):
- **SSH command**: `ssh pony-rec-gpu`
- **Host**: `125.71.97.70:15302`, User `ajifang`
- **GPU**: NVIDIA RTX 4090 (49GB VRAM)
- **Server project path**: `~/projects/pony-rec-rescue-shadow-v6`
- **Local project path**: `D:\Research\TGL-Rec`

Agents can execute server commands directly: `ssh pony-rec-gpu "<command>"`
