<#
.SYNOPSIS
    Mirrors the BGE-small-en-v1.5 sentence-transformers model into ./models/.

.DESCRIPTION
    Downloads the 10 files that make up `BAAI/bge-small-en-v1.5` directly
    from huggingface.co into a local folder, so the model can be loaded
    via `SentenceTransformer("./models/bge-small-en-v1.5")` with zero
    network access at runtime. Useful for reproducible setups and for
    offline / air-gapped deployments.

    Total download is ~134 MB (`model.safetensors` is 133 MB).
#>

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$modelDir    = Join-Path $projectRoot "models\bge-small-en-v1.5"
$poolDir     = Join-Path $modelDir "1_Pooling"
$normDir     = Join-Path $modelDir "2_Normalize"
$base        = "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main"

Write-Host "==> Target directory: $modelDir" -ForegroundColor Cyan

New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
New-Item -ItemType Directory -Force -Path $poolDir  | Out-Null
New-Item -ItemType Directory -Force -Path $normDir  | Out-Null

$files = @(
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "sentence_bert_config.json",
    "config_sentence_transformers.json",
    "modules.json"
)

foreach ($f in $files) {
    $url = "$base/$f"
    $dst = Join-Path $modelDir $f
    if (Test-Path $dst) {
        Write-Host "    skip (exists)  $f" -ForegroundColor DarkGray
        continue
    }
    Write-Host "    downloading    $f" -ForegroundColor Yellow
    Invoke-WebRequest -Uri $url -OutFile $dst -UseBasicParsing
}

$poolFile = Join-Path $poolDir "config.json"
if (Test-Path $poolFile) {
    Write-Host "    skip (exists)  1_Pooling/config.json" -ForegroundColor DarkGray
} else {
    Write-Host "    downloading    1_Pooling/config.json" -ForegroundColor Yellow
    Invoke-WebRequest -Uri "$base/1_Pooling/config.json" -OutFile $poolFile -UseBasicParsing
}

Write-Host ""
Write-Host "==> Verifying layout" -ForegroundColor Cyan
$expected = @(
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "sentence_bert_config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "1_Pooling\config.json"
)
$missing = @()
foreach ($rel in $expected) {
    $p = Join-Path $modelDir $rel
    if (Test-Path $p) {
        $size = (Get-Item $p).Length
        Write-Host ("    [OK]   {0,-40} {1,12:N0} bytes" -f $rel, $size) -ForegroundColor Green
    } else {
        Write-Host ("    [MISS] {0}" -f $rel) -ForegroundColor Red
        $missing += $rel
    }
}

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "Some files are missing. The model will not load until they are present." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "==> Done. Set EMBED_MODEL_NAME=./models/bge-small-en-v1.5 in .env" -ForegroundColor Green
exit 0
