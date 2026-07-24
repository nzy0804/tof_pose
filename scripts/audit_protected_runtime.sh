#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --extensions-only <package-stage> | <runtime-dir>" >&2
  exit 2
}

fail() {
  echo "audit failed: $*" >&2
  exit 1
}

protected_paths=(
  "ai_pb2"
  "ai_pb2_grpc"
  "scripts/grpc_server"
  "tof_pose/realtime_service"
  "tof_pose/tracking"
  "tof_pose/object_storage"
  "tof_pose/model_bundle"
  "tof_pose/paths"
  "tof_pose/person_distance"
  "tof_pose/pose_drawing"
  "tof_pose/scene_rate_controller"
)

audit_extensions() {
  local root="$1"
  local path
  local matches

  test -d "$root" || fail "missing package stage: $root"
  for path in "${protected_paths[@]}"; do
    matches=$(find "$root" -type f -path "*/${path}*.so" -print)
    test -n "$matches" || fail "missing native extension: $path"
    test ! -e "$root/${path}.py" || fail "protected source remains: $path.py"
    test ! -e "$root/src/${path}.py" || fail "protected source remains: src/$path.py"
  done
}

audit_runtime() {
  local runtime="$1"
  local executable="$runtime/maixsense-grpc-server"
  local listing
  local path
  local module
  local native_count

  test -x "$executable" || fail "missing runtime executable: $executable"

  for path in "${protected_paths[@]}"; do
    native_count=$(find "$runtime" -type f -path "*/${path}*.so" | wc -l)
    test "$native_count" -ge 1 || fail "runtime missing native extension: $path"
  done

  listing=$(mktemp)
  trap 'rm -f "$listing"' RETURN
  pyi-archive_viewer -r "$executable" > "$listing"
  for path in "${protected_paths[@]}"; do
    module=${path//\//.}
    if grep -Fq "'$module'" "$listing"; then
      fail "protected module found in Python archive: $module"
    fi
  done

  if find "$runtime" -type f \
      \( -name 'grpc_server.py' \
      -o -name 'realtime_service.py' \
      -o -name 'tracking.py' \
      -o -name 'object_storage.py' \
      -o -name 'model_bundle.py' \
      -o -name 'person_distance.py' \
      -o -name 'pose_drawing.py' \
      -o -name 'paths.py' \
      -o -name 'scene_rate_controller.py' \
      -o -name 'ai_pb2.py' \
      -o -name 'ai_pb2_grpc.py' \
      -o -name 'setup_protected.py' \
      -o -name '*.spec' \) | grep -q .; then
    fail "runtime contains protected source or build metadata"
  fi
}

if [ "${1:-}" = "--extensions-only" ]; then
  test "$#" -eq 2 || usage
  audit_extensions "$(realpath "$2")"
  echo "extension audit passed"
  exit 0
fi

test "$#" -eq 1 || usage
audit_runtime "$(realpath "$1")"
echo "runtime audit passed"
