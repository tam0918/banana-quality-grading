param(
  [switch]$Clean = $true
)

$ErrorActionPreference = "Stop"

Write-Host "[1/3] Installing build dependency: pyinstaller" -ForegroundColor Cyan
python -m pip install --upgrade pip
python -m pip install pyinstaller

Write-Host "[2/3] Installing runtime dependencies" -ForegroundColor Cyan
python -m pip install -r requirements.txt

Write-Host "[3/3] Building EXE (one-folder)" -ForegroundColor Cyan
$Args = @("banana_quality_grading.spec", "--noconfirm")
if ($Clean) { $Args += "--clean" }

pyinstaller @Args

Write-Host "Done. Output is under .\\dist\\BananaQualityGrading\\BananaQualityGrading.exe" -ForegroundColor Green
