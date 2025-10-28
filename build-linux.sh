#!/usr/bin/env bash
set -euo pipefail

# Build a standalone Linux GUI app using PyInstaller
# Usage:
#   ./build-linux.sh           # builds one-file binary
#   ./build-linux.sh clean     # remove dist/build

APP_NAME="YoloDetection"
ENTRY="gui_qt.py"

cd "$(dirname "$0")"

if [[ "${1:-}" == "clean" ]]; then
	rm -rf build dist "${APP_NAME}.spec"
	echo "Cleaned build artifacts."
	exit 0
fi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt

pyinstaller \
	--noconsole \
	--name "${APP_NAME}" \
	--onefile \
	"${ENTRY}"

echo "Build complete. See ./dist"
