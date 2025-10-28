#!/usr/bin/env bash
set -euo pipefail

# Build a standalone macOS GUI app bundle using PyInstaller
# Usage:
#   ./build-macos.sh           # builds .app bundle
#   ./build-macos.sh clean     # remove dist/build

APP_NAME="YoloDetection"
ENTRY="gui_qt.py"
# Prefer Python 3.11 for PyInstaller stability
PY=${PY:-python3.11}
ICON=""

cd "$(dirname "$0")"

if [[ "${1:-}" == "clean" ]]; then
	rm -rf build dist "${APP_NAME}.spec"
	echo "Cleaned build artifacts."
	exit 0
fi

$PY -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt

# PyInstaller options:
# --windowed: no console window
# --name: app name
# --onefile: single binary (optional). For faster startup, you can remove --onefile to get an .app bundle
# --add-data if you need to include extra files

pyinstaller \
    --windowed \
    --name "${APP_NAME}" \
    ${ICON:+--icon "$ICON"} \
    "${ENTRY}"

# Dist output hints
# - Output will be dist/YoloDetection.app which users can double-click

echo "Build complete. See ./dist"
