# Server Handoff

Use this reference when preparing shared-server commands or interpreting pasted logs.

## Protocol

Codex cannot inspect the shared GPU server directly. Give the user exact commands, wait for pasted logs or errors, and do not infer success without evidence.

Before long jobs, include checks for:

- current repo path and branch;
- `git status --short`;
- `git pull` or the exact commit expected;
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
