# Scripts

This folder is for thin orchestration scripts only. Core logic should live under `src/tglrec/` and be exposed through the `tglrec` CLI.

Planned scripts:

- `bootstrap_env.sh`: create a minimal CPU/dev environment.
- dataset download helpers after source URLs/licenses are verified.
- experiment launch wrappers that call the Python CLI and save logs under `runs/`.
