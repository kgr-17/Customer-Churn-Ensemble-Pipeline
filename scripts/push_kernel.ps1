param(
    [string]$KernelDir = "kaggle",
    [string]$NotebookSource = "notebooks/01_catboost_baseline.ipynb",
    [string]$NotebookTargetName = "01_catboost_baseline.ipynb"
)

$ErrorActionPreference = "Stop"

$kernelMetadataPath = Join-Path $KernelDir "kernel-metadata.json"
if (-not (Test-Path $kernelMetadataPath)) {
    throw "Missing kernel metadata: $kernelMetadataPath"
}

if (-not (Test-Path $NotebookSource)) {
    throw "Notebook file not found: $NotebookSource"
}

Copy-Item -Path $NotebookSource -Destination (Join-Path $KernelDir $NotebookTargetName) -Force

$runningServices = docker compose ps --status running --services
if ($LASTEXITCODE -ne 0 -or -not ($runningServices -contains "ml")) {
    throw "Docker service 'ml' is not running. Start it with: docker compose up -d --build"
}

Write-Host "Pushing Kaggle kernel from '$KernelDir'..."
docker compose exec -T ml kaggle kernels push -p $KernelDir
if ($LASTEXITCODE -ne 0) {
    throw "Kaggle kernels push failed."
}
