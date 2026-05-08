<#
.SYNOPSIS
    Imports a local GGUF file into Ollama (offline alternative to `ollama pull`).
.DESCRIPTION
    Useful for air-gapped or offline deployments where `ollama pull` is
    not an option. The GGUF file is supplied manually, then this script:
      1. Locates the source GGUF (defaults to ~/Downloads).
      2. Copies it into the project's models/ folder under a normalized name.
      3. Runs `ollama create` against the project's Modelfile.
      4. Verifies the model appears in `ollama list`.

    The Modelfile in the repo root provides the official Llama 3.2 chat
    template and stop tokens, so the imported model behaves identically
    to one fetched via `ollama pull llama3.2:1b`.

.PARAMETER SourcePath
    Full path to the source .gguf file. Defaults to the standard
    Llama-3.2-1B-Instruct-Q4_K_M filename in the user's Downloads folder.

.PARAMETER ModelName
    The Ollama tag to register the model under. Defaults to llama3.2:1b.

.EXAMPLE
    .\scripts\import_model.ps1
    .\scripts\import_model.ps1 -SourcePath "D:\models\my-model.gguf"
#>

param(
    [string]$SourcePath = "$env:USERPROFILE\Downloads\Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    [string]$ModelName = "llama3.2:1b"
)

$ErrorActionPreference = "Stop"

# Move to project root (parent of the scripts folder)
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Ollama Model Importer (offline GGUF)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Source:       $SourcePath" -ForegroundColor White
Write-Host "Target name:  $ModelName`n" -ForegroundColor White

# Step 1: Verify source file exists
Write-Host "[1/4] Locating source GGUF file..." -ForegroundColor Yellow
if (-not (Test-Path -LiteralPath $SourcePath)) {
    Write-Host "ERROR: Source file not found at:" -ForegroundColor Red
    Write-Host "  $SourcePath" -ForegroundColor Red
    Write-Host "`nMake sure the .gguf file is in your Downloads folder," -ForegroundColor Yellow
    Write-Host "or pass -SourcePath '<full-path-to-file.gguf>'." -ForegroundColor Yellow
    exit 1
}
$sizeMB = [math]::Round((Get-Item -LiteralPath $SourcePath).Length / 1MB, 1)
Write-Host "  Found ($sizeMB MB).`n" -ForegroundColor Green

# Step 2: Copy + normalize filename into models/
Write-Host "[2/4] Copying to project models/ folder..." -ForegroundColor Yellow
$ModelsDir = Join-Path $ProjectRoot "models"
if (-not (Test-Path $ModelsDir)) {
    New-Item -ItemType Directory -Path $ModelsDir | Out-Null
    Write-Host "  Created models/ directory."
}
$TargetFile = Join-Path $ModelsDir "llama-3.2-1b-instruct-q4_k_m.gguf"
if (Test-Path $TargetFile) {
    Write-Host "  Target already exists - skipping copy." -ForegroundColor Cyan
} else {
    Copy-Item -LiteralPath $SourcePath -Destination $TargetFile
    Write-Host "  Copied to: $TargetFile"
}
Write-Host ""

# Step 3: Run ollama create
Write-Host "[3/4] Running 'ollama create $ModelName -f Modelfile'..." -ForegroundColor Yellow
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: 'ollama' command not found. Install from https://ollama.com/download" -ForegroundColor Red
    exit 1
}
ollama create $ModelName -f Modelfile
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: ollama create failed (exit code: $LASTEXITCODE)" -ForegroundColor Red
    exit 1
}
Write-Host "  Model registered.`n" -ForegroundColor Green

# Step 4: Verify the model is listed
Write-Host "[4/4] Verifying with 'ollama list'..." -ForegroundColor Yellow
$listOutput = (ollama list | Out-String)
Write-Host $listOutput
if ($listOutput -notlike "*$ModelName*") {
    Write-Host "WARNING: '$ModelName' not visible in ollama list output." -ForegroundColor Yellow
    exit 1
}
Write-Host "  '$ModelName' is registered.`n" -ForegroundColor Green

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Import complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "`nNext step: Run the verification script:" -ForegroundColor White
Write-Host "  python tests/verify_ollama.py" -ForegroundColor White
exit 0
