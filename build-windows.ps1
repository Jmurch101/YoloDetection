Param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

# Build a standalone Windows GUI app using PyInstaller
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\build-windows.ps1
#   powershell -ExecutionPolicy Bypass -File .\build-windows.ps1 -Clean

$APP_NAME = "YoloDetection"
$ENTRY = "gui_qt.py"

Set-Location $PSScriptRoot

if ($Clean) {
    Remove-Item -Recurse -Force build, dist, "$APP_NAME.spec" -ErrorAction SilentlyContinue
    Write-Host "Cleaned build artifacts."
    exit 0
}

python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt

# --noconsole for GUI apps on Windows
pyinstaller --noconsole --name "$APP_NAME" --onefile $ENTRY

Write-Host "Build complete. See .\dist"
