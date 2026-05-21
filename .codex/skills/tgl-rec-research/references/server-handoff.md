# Server Handoff

Use this reference when running commands on the shared GPU server or interpreting results.

## Access

Server `pony-rec-gpu` is directly accessible via SSH (key-based auth configured):
```bash
ssh pony-rec-gpu "<command>"
```
- Host: `125.71.97.70:15302`, User: `ajifang`
- GPU: NVIDIA RTX 4090 (49GB VRAM)
- Server project path: `~/projects/pony-rec-rescue-shadow-v6`

## Protocol

Agents can now run server commands directly. Do not guess server state — always verify with a command before claiming status.

Before long jobs, check:

```bash
ssh pony-rec-gpu "cd ~/projects/pony-rec-rescue-shadow-v6 && git status --short && nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader"
```

- current repo path and branch;
- `git status --short`;
- `nvidia-smi`;
- expected input files with `test -f` or `test -d`;
- output directory preservation before reruns.

Do not commit private server configs, model checkpoints, copied paper text, or output artifacts.

## Command Style

Prefer copy-pasteable shell blocks with `set -euo pipefail` when appropriate. Preserve existing outputs by moving them to timestamped directories before reruns.

Example structure:

```bash
cd /path/to/TGL-Rec
git status --short
git pull --ff-only
nvidia-smi
test -d ~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks
```

Then run the specific script/config for the task. Include the exact output files the user should check and paste back.

## After Logs Arrive

Diagnose only from the pasted evidence. If commands change, update `docs/server_runbook.md` so the next agent does not repeat stale instructions.

If a run succeeds, record the real artifact paths and any observed metrics or diagnostics in the appropriate durable docs. If it fails, record the blocker and the next concrete command.
