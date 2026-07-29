# Build AI Hand Draw into a standalone Windows executable.
#
#   ./build_exe.ps1
#
# Output: dist\AI Hand Draw\AI Hand Draw.exe
# Run that .exe directly - no Python install needed on the target machine.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Ensuring PyInstaller is installed..." -ForegroundColor Cyan
python -m pip install --quiet --upgrade pyinstaller

Write-Host "Cleaning previous build..." -ForegroundColor Cyan
if (Test-Path build) { Remove-Item -Recurse -Force build }
if (Test-Path dist)  { Remove-Item -Recurse -Force dist }

Write-Host "Building..." -ForegroundColor Cyan
python -m PyInstaller --noconfirm --clean handdraw.spec

$exe = Join-Path $PSScriptRoot "dist\AI Hand Draw\AI Hand Draw.exe"
if (Test-Path $exe) {
    Write-Host "`nBuild complete:" -ForegroundColor Green
    Write-Host "  $exe"
} else {
    Write-Error "Build finished but the executable was not found."
}
