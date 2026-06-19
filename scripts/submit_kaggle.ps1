param(
    [string]$Competition = $env:KAGGLE_COMPETITION,
    [string]$File = "outputs/submission_blend_v3_multiseed.csv",
    [string]$Message = "blend v3 multiseed"
)

$ErrorActionPreference = "Stop"

if (-not $Competition) {
    throw "KAGGLE_COMPETITION is required. Set it in .env or pass -Competition."
}

if (-not (Test-Path $File)) {
    throw "Submission file not found: $File"
}

$runningServices = docker compose ps --status running --services
if ($LASTEXITCODE -ne 0 -or -not ($runningServices -contains "ml")) {
    throw "Docker service 'ml' is not running. Start it with: docker compose up -d --build"
}

Write-Host "Submitting $File to competition '$Competition'..."
docker compose exec -T ml kaggle competitions submit -c $Competition -f $File -m $Message
if ($LASTEXITCODE -ne 0) {
    throw "Kaggle submission command failed."
}

Write-Host "Recent submissions:"
docker compose exec -T ml kaggle competitions submissions -c $Competition

