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

$DistDir = Join-Path $RepoRoot "dist"
$OnedirDir = Join-Path $DistDir "BananaQualityGrading"
$OnefileExe = Join-Path $DistDir "BananaQualityGrading.exe"
$OnedirExe = Join-Path $OnedirDir "BananaQualityGrading.exe"

Write-Host "[4/4] Preparing release folder (.\\app exe\\)" -ForegroundColor Cyan
$ReleaseDir = Join-Path $RepoRoot "app exe"
Remove-Item -Recurse -Force $ReleaseDir -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $ReleaseDir | Out-Null

if (Test-Path $OnedirExe) {
  # onedir build: copy the whole folder (all DLLs/data)
  Copy-Item -Path (Join-Path $OnedirDir "*") -Destination $ReleaseDir -Recurse -Force
} elseif (Test-Path $OnefileExe) {
  # onefile build: copy the single EXE
  Copy-Item -Path $OnefileExe -Destination $ReleaseDir -Force
} else {
  throw "Build output not found. Expected '$OnefileExe' or '$OnedirExe'."
}

# Always copy runtime weights/assets next to the EXE.
# This makes distribution reliable even if some models are loaded from disk.
$ReleaseWeights = Join-Path $ReleaseDir "weights"
New-Item -ItemType Directory -Path $ReleaseWeights -ErrorAction SilentlyContinue | Out-Null
Copy-Item -Path (Join-Path $RepoRoot "weights\*.pt") -Destination $ReleaseWeights -Force -ErrorAction SilentlyContinue

foreach ($ModelName in @("yolov8n.pt", "yolo11n.pt", "yolov8n-cls.pt")) {
  $srcModel = Join-Path $RepoRoot $ModelName
  if (Test-Path $srcModel) {
    Copy-Item -Path $srcModel -Destination (Join-Path $ReleaseDir $ModelName) -Force
  }
}

$ReleaseFonts = Join-Path $ReleaseDir "assets\fonts"
New-Item -ItemType Directory -Path $ReleaseFonts -Force | Out-Null
Copy-Item -Path (Join-Path $RepoRoot "assets\fonts\*.ttf") -Destination $ReleaseFonts -Force -ErrorAction SilentlyContinue

Write-Host "Done." -ForegroundColor Green
Write-Host "- Build output: $DistDir" -ForegroundColor DarkGray
Write-Host "- Zip this folder: $ReleaseDir" -ForegroundColor Green
