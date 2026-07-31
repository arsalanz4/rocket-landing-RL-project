<#
.SYNOPSIS
    Pull the latest training progress from GitHub. Run this BEFORE starting
    train_ppo.py, whichever machine you're on.
#>
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$dirty = git status --porcelain best_model.zip vec_normalize.pkl current_stage.txt
if ($dirty) {
    Write-Warning "Local best_model.zip / vec_normalize.pkl / current_stage.txt have uncommitted changes:"
    Write-Output $dirty
    Write-Warning "Run sync-up.ps1 first to push them, otherwise they may conflict with or be overwritten by the pull."
    exit 1
}

git pull origin main

$stage = if (Test-Path current_stage.txt) { (Get-Content current_stage.txt -Raw).Trim() } else { "1" }
if (Test-Path best_model.zip) {
    Write-Output "Ready to resume: stage $stage, best_model.zip present."
} else {
    Write-Output "No best_model.zip yet -- training will start fresh from stage $stage."
}
