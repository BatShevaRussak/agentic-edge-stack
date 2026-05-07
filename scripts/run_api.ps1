<#
.SYNOPSIS
    Launches the Part 4 FastAPI server (uvicorn) that exposes the agent
    over HTTP with Server-Sent Events streaming.

.DESCRIPTION
    1. Verifies the project virtual environment exists.
    2. Verifies the local Ollama server is reachable.
    3. Activates the venv and starts uvicorn against ``app.api.main:app``.

    Host / port / reload-mode are read from environment variables (loaded
    by Pydantic Settings from ``.env``); the script accepts overrides via
    parameters.

.PARAMETER BindHost
    Override API_HOST (defaults to 0.0.0.0). Named ``BindHost`` because
    ``$Host`` is a PowerShell automatic variable.

.PARAMETER Port
    Override API_PORT (defaults to 8000).

.PARAMETER Reload
    Pass --reload to uvicorn (auto-restart on code changes). For dev only.

.EXAMPLE
    .\scripts\run_api.ps1
    .\scripts\run_api.ps1 -Port 9000 -Reload
#>

param(
    [Parameter(Mandatory = $false)] [string] $BindHost = "0.0.0.0",
    [Parameter(Mandatory = $false)] [int] $Port = 8000,
    [Parameter(Mandatory = $false)] [switch] $Reload
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Agentic Edge Stack - API Server (Part 4)" -ForegroundColor Cyan
Write-Host "============================================================`n" -ForegroundColor Cyan

# Step 1: venv
Write-Host "[1/3] Checking virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path ".\venv\Scripts\Activate.ps1")) {
    Write-Host "ERROR: venv not found at .\venv. Run scripts\deploy.ps1 first." -ForegroundColor Red
    exit 1
}
& ".\venv\Scripts\Activate.ps1"
Write-Host "  venv activated.`n" -ForegroundColor Green

# Step 2: Ollama reachability
Write-Host "[2/3] Probing Ollama server..." -ForegroundColor Yellow
try {
    $null = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 5 -UseBasicParsing
    Write-Host "  Ollama responding on http://localhost:11434`n" -ForegroundColor Green
} catch {
    Write-Host "WARNING: Ollama not reachable - the API will start but /chat will fail." -ForegroundColor Yellow
    Write-Host "         Run 'ollama serve' in a separate terminal.`n" -ForegroundColor Yellow
}

# Step 3: launch uvicorn
Write-Host "[3/3] Launching FastAPI via uvicorn..." -ForegroundColor Yellow
Write-Host "  Bind:    http://${BindHost}:${Port}" -ForegroundColor White
Write-Host "  Docs:    http://${BindHost}:${Port}/docs" -ForegroundColor White
Write-Host "  Health:  http://${BindHost}:${Port}/health" -ForegroundColor White
Write-Host "  Chat:    POST http://${BindHost}:${Port}/chat  (SSE)`n" -ForegroundColor White

$uvicornArgs = @(
    "app.api.main:app",
    "--host", $BindHost,
    "--port", $Port.ToString(),
    "--log-level", "info"
)
if ($Reload.IsPresent) {
    $uvicornArgs += "--reload"
    Write-Host "  --reload enabled (development mode).`n" -ForegroundColor Yellow
}

uvicorn @uvicornArgs
exit $LASTEXITCODE
