param(
  [switch]$Clean = $true
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
  $Py = $VenvPython
  Write-Host "Using venv python: $Py" -ForegroundColor DarkGray
} else {
  $Py = "python"
  Write-Host "Using system python (no .venv found)." -ForegroundColor Yellow
}

Write-Host "[1/3] Installing build dependency: pyinstaller" -ForegroundColor Cyan
& $Py -m pip install --upgrade pip
& $Py -m pip install pyinstaller

Write-Host "[2/3] Installing runtime dependencies" -ForegroundColor Cyan
& $Py -m pip install -r requirements.txt

Write-Host "[3/3] Building EXE (one-folder)" -ForegroundColor Cyan
$Args = @("banana_quality_grading.spec", "--noconfirm")
if ($Clean) { $Args += "--clean" }

& $Py -m PyInstaller @Args

if ($LASTEXITCODE -ne 0) {
  throw "PyInstaller failed with exit code $LASTEXITCODE"
}

Write-Host "Done. Output is under .\\dist\\BananaQualityGrading\\BananaQualityGrading.exe" -ForegroundColor Green
