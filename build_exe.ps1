# Build AI Hand Draw into a single standalone Windows executable.
#
#   ./build_exe.ps1
#
# Output: dist\AI Hand Draw.exe
# One file - copy it anywhere and run it; no Python install needed.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Ensuring build dependencies are installed..." -ForegroundColor Cyan
python -m pip install --quiet --upgrade pyinstaller pillow

Write-Host "Generating icon..." -ForegroundColor Cyan
python assets/make_icon.py

Write-Host "Cleaning previous build..." -ForegroundColor Cyan
if (Test-Path build) { Remove-Item -Recurse -Force build }
if (Test-Path dist)  { Remove-Item -Recurse -Force dist }

Write-Host "Building..." -ForegroundColor Cyan
python -m PyInstaller --noconfirm --clean handdraw.spec

$exe = Join-Path $PSScriptRoot "dist\AI Hand Draw.exe"
if (Test-Path $exe) {
    $mb = "{0:N0} MB" -f ((Get-Item $exe).Length / 1MB)
    Write-Host "`nBuild complete:" -ForegroundColor Green
    Write-Host "  $exe  ($mb)"
} else {
    Write-Error "Build finished but the executable was not found."
}
