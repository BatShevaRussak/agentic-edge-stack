<#
.SYNOPSIS
    Deploys the Agentic Edge Stack: sets up Python venv, installs dependencies,
    and prepares the Ollama model.

.DESCRIPTION
    This script automates the local development setup:
    1. Verifies Python and Ollama installations.
    2. Creates a Python virtual environment.
    3. Installs project dependencies via pyproject.toml.
    4. Pulls the configured Ollama model.
    5. Verifies the Ollama server is responsive.

.EXAMPLE
    .\scripts\deploy.ps1
#>

$ErrorActionPreference = "Stop"

# Move to the project root (parent of the scripts folder)
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Agentic Edge Stack - Deployment Script" -ForegroundColor Cyan
Write-Host "============================================================`n" -ForegroundColor Cyan

# Step 1: Verify Python
Write-Host "[1/5] Checking Python installation..." -ForegroundColor Yellow
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Python not found." -ForegroundColor Red
    Write-Host "Install Python 3.10+ from https://www.python.org/downloads/"
    exit 1
}
$pythonVersion = (python --version) -replace "Python ", ""
Write-Host "  Python $pythonVersion detected.`n" -ForegroundColor Green

# Step 2: Verify Ollama
Write-Host "[2/5] Checking Ollama installation..." -ForegroundColor Yellow
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Ollama not found." -ForegroundColor Red
    Write-Host "Install Ollama from https://ollama.com/download"
    exit 1
}
Write-Host "  Ollama detected.`n" -ForegroundColor Green

# Step 3: Create virtual environment
Write-Host "[3/5] Setting up Python virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "  Virtual environment created."
} else {
    Write-Host "  Virtual environment already exists."
}

# Activate venv and install dependencies
& ".\venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip --quiet
pip install -e . --quiet
Write-Host "  Dependencies installed.`n" -ForegroundColor Green

# Step 4: Ensure model is available (skip pull if already imported / pulled)
Write-Host "[4/5] Checking Ollama model 'llama3.2:1b'..." -ForegroundColor Yellow
$existing = (ollama list | Out-String)
if ($existing -like "*llama3.2:1b*") {
    Write-Host "  Model already registered - skipping pull.`n" -ForegroundColor Green
} else {
    Write-Host "  Model not found locally. Running 'ollama pull llama3.2:1b'..."
    Write-Host "  (For offline / air-gapped setups, run .\scripts\import_model.ps1 with a local GGUF.)"
    ollama pull llama3.2:1b
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Model pull failed." -ForegroundColor Red
        Write-Host "  Offline alternative: place a GGUF in Downloads and run:" -ForegroundColor Yellow
        Write-Host "    .\scripts\import_model.ps1" -ForegroundColor Yellow
        exit 1
    }
    Write-Host "  Model ready.`n" -ForegroundColor Green
}

# Step 5: Verify server
Write-Host "[5/5] Verifying Ollama server..." -ForegroundColor Yellow
try {
    $null = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 5 -UseBasicParsing
    Write-Host "  Ollama server is responding.`n" -ForegroundColor Green
} catch {
    Write-Host "WARNING: Ollama server not reachable. Run 'ollama serve' in a separate terminal." -ForegroundColor Yellow
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Deployment complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "`nNext step: Run the verification script:" -ForegroundColor White
Write-Host "  python tests/verify_ollama.py" -ForegroundColor White
