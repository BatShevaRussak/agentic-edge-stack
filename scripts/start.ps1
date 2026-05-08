<#
.SYNOPSIS
    One-shot launcher: deploy -> models -> running API.

.DESCRIPTION
    Pure orchestrator: delegates every stage to the existing scripts so
    no logic is duplicated. Each delegated step is independently
    skippable for fast restarts.

    Stages (each is a separate script):
      1. deploy.ps1               (Python, venv, deps, Ollama LLM)
      2. import_embed_model.ps1   (BGE-small embedding model)
      3. run_api.ps1              (venv probe, Ollama probe, uvicorn)

.PARAMETER SkipDeploy
    Skip deploy.ps1. Use after the first successful run.

.PARAMETER SkipEmbedModel
    Skip the embedding-model download (BGE-small).

.PARAMETER BindHost
    Forwarded to run_api.ps1 (default: 0.0.0.0).

.PARAMETER Port
    Forwarded to run_api.ps1 (default: 8000).

.PARAMETER Reload
    Forwarded to run_api.ps1 (--reload, dev only).

.EXAMPLE
    .\scripts\start.ps1
    .\scripts\start.ps1 -SkipDeploy -SkipEmbedModel        # fast restart
    .\scripts\start.ps1 -Port 9000 -Reload                 # dev mode
#>

param(
    [switch] $SkipDeploy,
    [switch] $SkipEmbedModel,
    [string] $BindHost = "0.0.0.0",
    [int]    $Port     = 8000,
    [switch] $Reload
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

function Section([string]$title) {
    Write-Host ""
    Write-Host "===== $title =====" -ForegroundColor Yellow
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Agentic Edge Stack - Unified Launcher" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# --- 1. deploy.ps1 (Python, venv, deps, Ollama LLM) ------------------------
Section "Stage 1/3: deploy.ps1"
if ($SkipDeploy) {
    Write-Host "  skipped (-SkipDeploy)" -ForegroundColor DarkGray
} else {
    & "$PSScriptRoot\deploy.ps1"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: deploy.ps1 exited with $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# --- 2. import_embed_model.ps1 (BGE-small) ---------------------------------
Section "Stage 2/3: import_embed_model.ps1"
if ($SkipEmbedModel) {
    Write-Host "  skipped (-SkipEmbedModel)" -ForegroundColor DarkGray
} elseif (Test-Path ".\models\bge-small-en-v1.5\model.safetensors") {
    Write-Host "  embedding model already present - skipping" -ForegroundColor DarkGray
} else {
    & "$PSScriptRoot\import_embed_model.ps1"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: import_embed_model.ps1 exited with $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# --- 3. run_api.ps1 (uvicorn) ----------------------------------------------
Section "Stage 3/3: run_api.ps1"
$apiArgs = @{
    BindHost = $BindHost
    Port     = $Port
}
if ($Reload.IsPresent) { $apiArgs.Reload = $true }

& "$PSScriptRoot\run_api.ps1" @apiArgs
exit $LASTEXITCODE
