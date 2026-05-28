#!/usr/bin/env bash
set -euo pipefail

# PyInstaller build script for Linux/WSL.
# Run from repo root with the venv activated:
#   source maixsense/bin/activate
#   bash build_executable_wsl.sh

APP_NAME="maixsense-grpc-server"
SPEC_FILE="maixsense-grpc-server.spec"

# Ultralytics tracking (BoT-SORT) needs 'lap' at runtime.
# Ensure it's installed in the build venv so PyInstaller can bundle it.
python -c "import importlib.util, sys; mods=('lap','lap._lapjv'); missing=[m for m in mods if importlib.util.find_spec(m) is None]; sys.exit(1 if missing else 0)" || {
  echo "[build] 'lap' missing in venv; installing..." >&2
  pip install 'setuptools<82' lap
}

python -m PyInstaller --clean --noconfirm "$SPEC_FILE"

echo ""
echo "Build done. Output:" 
if [[ -f "dist/${APP_NAME}" ]]; then
  ls -lh "dist/${APP_NAME}"
  echo "Test: ./dist/${APP_NAME} --help"
else
  echo "Expected dist/${APP_NAME} not found. Check PyInstaller logs above."
  exit 1
fi
