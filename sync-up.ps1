<#
.SYNOPSIS
    Push current training progress (best_model.zip, vec_normalize.pkl,
    current_stage.txt) to GitHub. Run this AFTER stopping train_ppo.py,
    or periodically during a long session, so the other machine can pick
    up from here via sync-down.ps1.
#>
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

git fetch origin main
$behind = (git rev-list --count HEAD..origin/main).Trim()
if ([int]$behind -gt 0) {
    Write-Warning "Local branch is behind origin/main by $behind commit(s). Run sync-down.ps1 first, then retry."
    exit 1
}

$status = git status --porcelain best_model.zip vec_normalize.pkl current_stage.txt
if (-not $status) {
    Write-Output "Nothing to sync -- best_model.zip, vec_normalize.pkl, and current_stage.txt are already up to date."
    exit 0
}

git add best_model.zip vec_normalize.pkl current_stage.txt
$stage     = if (Test-Path current_stage.txt) { (Get-Content current_stage.txt -Raw).Trim() } else { "?" }
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"
git commit -m "Sync progress: stage $stage ($timestamp)"
git push origin main

Write-Output "Pushed. Stage $stage synced to GitHub -- run sync-down.ps1 on the other machine before resuming."
