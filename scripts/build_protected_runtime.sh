#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 [source-dir] [work-root]" >&2
  exit 2
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_SOURCE=$(cd "$SCRIPT_DIR/.." && pwd)
SOURCE_DIR=$(realpath "${1:-$DEFAULT_SOURCE}")
WORK_ROOT=${2:-$HOME/maixsense-protected-build}

test "$#" -le 2 || usage
test -f "$SOURCE_DIR/setup_protected.py"
test -f "$SOURCE_DIR/maixsense-grpc-server-protected.spec"

mkdir -p "$WORK_ROOT"
WORK_ROOT=$(realpath "$WORK_ROOT")
BUILD_ID="maixsense-protected-$(date +%Y%m%d%H%M%S)"
RUN_DIR="$WORK_ROOT/runs/$BUILD_ID"
SOURCE_STAGE="$RUN_DIR/source"
PACKAGE_STAGE="$RUN_DIR/package"
RELEASE_DIR="$WORK_ROOT/releases/$BUILD_ID"

mkdir -p "$SOURCE_STAGE/scripts" "$SOURCE_STAGE/src/tof_pose"
mkdir -p "$PACKAGE_STAGE/scripts" "$PACKAGE_STAGE/src/tof_pose"
mkdir -p "$RELEASE_DIR"

cp "$SOURCE_DIR/setup_protected.py" "$SOURCE_STAGE/"
cp "$SOURCE_DIR/ai_pb2.py" "$SOURCE_STAGE/"
cp "$SOURCE_DIR/ai_pb2_grpc.py" "$SOURCE_STAGE/"
cp "$SOURCE_DIR/scripts/__init__.py" "$SOURCE_STAGE/scripts/"
cp "$SOURCE_DIR/scripts/grpc_server.py" "$SOURCE_STAGE/scripts/"
cp "$SOURCE_DIR/scripts/grpc_server_bootstrap.py" "$SOURCE_STAGE/scripts/"
cp "$SOURCE_DIR/src/tof_pose/__init__.py" "$SOURCE_STAGE/src/tof_pose/"

for module in realtime_service tracking person_distance pose_drawing object_storage model_bundle paths scene_rate_controller; do
  cp "$SOURCE_DIR/src/tof_pose/$module.py" "$SOURCE_STAGE/src/tof_pose/"
done

if [ -d "$SOURCE_DIR/src/tof_pose/assets" ]; then
  cp -a "$SOURCE_DIR/src/tof_pose/assets" "$SOURCE_STAGE/src/tof_pose/"
fi

cd "$SOURCE_STAGE"
python setup_protected.py build_ext

EXTENSION_DIR=$(find "$SOURCE_STAGE/build" -maxdepth 1 -type d -name 'lib.*' -print -quit)
test -n "$EXTENSION_DIR"
test -d "$EXTENSION_DIR"

find "$EXTENSION_DIR" -type f -name '*.so' -exec strip --strip-unneeded {} +

cp "$SOURCE_DIR/maixsense-grpc-server-protected.spec" "$PACKAGE_STAGE/"
cp "$SOURCE_STAGE/scripts/__init__.py" "$PACKAGE_STAGE/scripts/"
cp "$SOURCE_STAGE/scripts/grpc_server_bootstrap.py" "$PACKAGE_STAGE/scripts/"
cp "$SOURCE_STAGE/src/tof_pose/__init__.py" "$PACKAGE_STAGE/src/tof_pose/"
cp "$EXTENSION_DIR"/ai_pb2*.so "$PACKAGE_STAGE/"
cp "$EXTENSION_DIR/scripts"/grpc_server*.so "$PACKAGE_STAGE/scripts/"
cp "$EXTENSION_DIR/tof_pose"/*.so "$PACKAGE_STAGE/src/tof_pose/"

if [ -d "$SOURCE_STAGE/src/tof_pose/assets" ]; then
  cp -a "$SOURCE_STAGE/src/tof_pose/assets" "$PACKAGE_STAGE/src/tof_pose/"
fi

bash "$SOURCE_DIR/scripts/audit_protected_runtime.sh" --extensions-only "$PACKAGE_STAGE"

PYTHONPATH="$PACKAGE_STAGE:$PACKAGE_STAGE/src" python - <<'PY'
from scripts.grpc_server import main
from tof_pose.realtime_service import RealtimePoseEngine
from tof_pose.scene_rate_controller import SceneRateController

assert callable(main)
assert RealtimePoseEngine is not None
assert SceneRateController is not None
print("protected import smoke passed")
PY

cd "$PACKAGE_STAGE"
python -m PyInstaller \
  --clean \
  --noconfirm \
  maixsense-grpc-server-protected.spec \
  > "$RUN_DIR/pyinstaller-build.log" 2>&1

RUNTIME_DIR="$PACKAGE_STAGE/dist/maixsense-grpc-server"
bash "$SOURCE_DIR/scripts/audit_protected_runtime.sh" "$RUNTIME_DIR"
"$RUNTIME_DIR/maixsense-grpc-server" --help > "$RUN_DIR/help.txt"

mv "$RUNTIME_DIR" "$RELEASE_DIR/runtime"
cp "$RUN_DIR/pyinstaller-build.log" "$RELEASE_DIR/"
cp "$RUN_DIR/help.txt" "$RELEASE_DIR/"

python - <<'PY' > "$RELEASE_DIR/build-info.txt"
import importlib.metadata as md
import platform
import sys

print(f"python={sys.version.split()[0]}")
print(f"platform={platform.platform()}")
for name in (
    "Cython", "PyInstaller", "grpcio", "protobuf", "numpy",
    "opencv-python-headless", "torch", "ultralytics", "lap",
    "oss2", "boto3", "botocore", "cryptography",
):
    try:
        print(f"{name}={md.version(name)}")
    except md.PackageNotFoundError:
        pass
try:
    import tensorrt
    print(f"tensorrt={tensorrt.__version__}")
except Exception:
    pass
PY

python -m pip freeze --all > "$RELEASE_DIR/requirements-freeze.txt"

ARCHIVE="$RELEASE_DIR/$BUILD_ID.tar.gz"
tar -C "$RELEASE_DIR" -czf "$ARCHIVE" \
  runtime build-info.txt requirements-freeze.txt help.txt
(
  cd "$RELEASE_DIR"
  sha256sum "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE").sha256"
)

case "$RUN_DIR" in
  "$WORK_ROOT"/runs/maixsense-protected-*) rm -rf -- "$RUN_DIR" ;;
  *) echo "refusing to remove unexpected run directory: $RUN_DIR" >&2; exit 1 ;;
esac

echo "BUILD_ID=$BUILD_ID"
echo "RUNTIME=$RELEASE_DIR/runtime"
echo "ARCHIVE=$ARCHIVE"
echo "SHA256=$ARCHIVE.sha256"
