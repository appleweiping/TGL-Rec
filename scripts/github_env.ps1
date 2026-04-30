# Prepare this shell for GitHub publishing from the local workspace.
#
# Usage from the repository root:
#   .\scripts\github_env.ps1
#   gh auth login
#   gh auth status
#
# The project keeps a portable GitHub CLI under tools/gh/bin when system gh
# cannot be installed. This script prepends that path for the current shell
# only; it does not edit the Windows registry or require administrator rights.

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PortableGhBin = Join-Path $RepoRoot "tools\gh\bin"

if (Test-Path (Join-Path $PortableGhBin "gh.exe")) {
    if (($env:Path -split ";") -notcontains $PortableGhBin) {
        $env:Path = "$PortableGhBin;$env:Path"
    }
}

Write-Host "Git:"
git --version

Write-Host ""
Write-Host "GitHub CLI:"
gh --version

Write-Host ""
Write-Host "GitHub auth status:"
gh auth status
