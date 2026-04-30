# GitHub publishing setup

This workspace uses a portable GitHub CLI when a system install is unavailable.

From the repository root:

```powershell
.\scripts\github_env.ps1
gh auth login
gh auth status
```

If `gh auth login` succeeds, future publish runs can use:

```powershell
git status --short --branch
git add <files>
git commit -m "implement tdig direct transition graph"
git push -u origin main
```

Notes:

- The portable binary is expected at `tools/gh/bin/gh.exe` and is ignored by Git.
- This script only updates `PATH` for the current shell. It does not require admin rights.
- If Git cannot initialize or write `.git/config`, fix the Windows ACL/sandbox state first, then
  rerun `git init -b main` or clone the remote repository into a clean directory.
