# MaixSense Context Handoff

## Snapshot
- Date: 2026-06-29 15:24 +08:00
- Workspace: `E:\Project\Lab Project\MaixSense`
- Current goal: Continue MaixSense gRPC/TensorRT/OSS algorithm service tuning, especially IR-mode visualization, contour display, qualitative fields, and deployment.
- Current branch/status: branch `yolo-tracking`; worktree is not clean.
- Current local modified files:
  - `maixsense-grpc-server-trt.spec`
  - `scripts/grpc_server.py`
  - `scripts/oss_realtime_local_infer.py`
  - `scripts/tailscale_realtime_local_infer.py`
  - `src/tof_pose/person_distance.py`
  - `src/tof_pose/realtime_service.py`
  - untracked `src/tof_pose/assets/`

## Project Baseline
- Repo path: `E:\Project\Lab Project\MaixSense`
- Main service binary: `maixsense-grpc-server`
- gRPC proto: `ai.proto`
- Generated protobuf files: `ai_pb2.py`, `ai_pb2_grpc.py`
- Key service files:
  - `scripts/grpc_server.py`
  - `src/tof_pose/realtime_service.py`
  - `src/tof_pose/person_distance.py`
  - `src/tof_pose/object_storage.py`
  - `maixsense-grpc-server-trt.spec`
- Deployment style: manual `nohup`, no `systemd`.
- AI server deploy dir: `/data/care/care-sense-iot-platform/bin`
- Runtime symlink: `/data/care/care-sense-iot-platform/bin/maixsense-grpc-server-trt-current`
- Standard service port: `50052`
- OSS env file: `/data/care/care-sense-iot-platform/bin/maixsense-oss.env`; do not print secrets.

## Current Live Service
- AI server: `222.71.62.147`, user `care`
- Current runtime, verified read-only:
  - `/data/care/care-sense-iot-platform/bin/maixsense-grpc-server-trt-runtime-20260629-blendparam-20260629105350`
- Current PID, verified read-only:
  - `3754519`
- Current port:
  - `*:50052` listening
- Current stdout/stderr log, verified from `/proc/<pid>/fd`:
  - `/data/care/care-sense-iot-platform/bin/maixsense-grpc-server-trt-b20-ir-i6-threshold-tuned-20260629.log`
- Current process has 6 child processes.
- Current service GPU memory:
  - MaixSense process around `12008 MiB`
- Other GPU consumers seen during handoff:
  - `/data/workspace/report-archive/.run/paddleocr-vl/venv-native/bin/python`, around `15354 MiB`
  - two `java` processes, around `1480 MiB` and `2620 MiB`
- Engine paths:
  - `/data/care/trt-export-lowmem-20260612-b20w1/maixsense-seg-lowmem-b20w1.engine`
  - `/data/care/trt-export-lowmem-20260612-b20w1/maixsense-pose-lowmem-b20w1.engine`
- Current active startup flags, verified from `/proc/<pid>/cmdline`:
  - `--host 0.0.0.0`
  - `--port 50052`
  - `--max-workers 6`
  - `--max-msg-mb 160`
  - `--device cuda:0`
  - `--decode-workers 6`
  - `--render-workers 6`
  - `--model-instances 6`
  - `--warmup-batch-size 20`
  - `--device-binding-ttl-sec 0`
  - `--output-format jpeg`
  - `--jpeg-quality 60`
  - `--cpu-worker-mode process`
  - `--cpu-process-start-method fork`
  - `--input-modality ir`
  - `--person-fill-background bg_08_dark_frost_reference.png`
  - `--person-fill-background-blend 0.5`
  - `--depth-distance-close-threshold 120.0`
  - `--ir-distance-close-gap-ratio 0.12`
  - `--ir-distance-close-center-ratio 0.80`
  - `--seg-conf 0.35`
  - `--contour-new-conf 0.45`
  - `--contour-existing-conf 0.25`
  - `--pose-gate-kpt-conf 0.40`
  - `--pose-kpt-min-points 6`
  - `--oss-workers 4`
  - `--oss-download-workers 12`
  - `--oss-upload-workers 48`
  - `--oss-global-workers 60`
  - `--oss-download-wait-timeout-ms 300`
  - `--oss-upload-wait-timeout-ms 300`
  - `--oss-max-pool-connections 128`

## Completed In This Session
- Changed gRPC flow to OSS object-key mode earlier in the conversation:
  - Platform sends object keys rather than image bytes.
  - AI side downloads input images from OSS and uploads result images to OSS.
  - Platform receives result object keys.
- Current service handles variable batch size; recent logs show `inputs=N results=N`, with `oss_upload_count=2*N`, meaning each input produces two uploaded result images.
- Removed interpolation from current behavior:
  - No interpolated frames in current log path: `inputs=20 outputs=20`.
  - Earlier interpolated `outputs=40/80` behavior is no longer the target.
- Added qualitative result fields in proto/code earlier:
  - `personStatus`
  - `personDistance`
  - `actionLevel`
- Split input modality behavior:
  - `--input-modality ir` uses IR/grayscale path.
  - `--input-modality depth` uses depth/pseudo-color path.
- Current IR visualization decisions:
  - IR mode returns grayscale-based images, not pseudo-color.
  - `--ir-preprocess` controls median filter + CLAHE; current live service has `ir_preprocess=false`.
  - Human region is filled using a frosted-gray background texture blended with original grayscale detail.
- Added selectable person-fill background:
  - Current live background: `bg_08_dark_frost_reference.png`
  - Background assets live under `src/tof_pose/assets/`.
- Added `--person-fill-background-blend`:
  - Default changed to `0.5`, i.e. background gray : original person gray = `5:5`.
  - Current live command uses `0.5`.
- Added distance threshold startup parameters:
  - `--depth-distance-close-threshold`
  - `--ir-distance-close-gap-ratio`
  - `--ir-distance-close-center-ratio`
  - Current live service is threshold-tuned to `120.0 / 0.12 / 0.80`, not the default `150.0 / 0.15 / 0.90`.
- Improved contour display:
  - `person_distance.py` now carries both distance contour and `draw_contour`.
  - Distance estimation still uses the postprocessed/best contour.
  - Display can use full-frame `draw_contour`.
  - Contour extraction uses `cv2.CHAIN_APPROX_NONE` plus light smoothing.
- Added head-top display correction:
  - Only changes final rendered display image.
  - Top 1/5 of displayed contour is lifted along the vertical axis.
  - Center lifts most, max scale `1.1`; sides remain unchanged.
  - Lower 4/5 remains unchanged.
  - Filled person mask is regenerated from the adjusted display contour so the fill and green outline match.
- Deployment work completed:
  - Server-side PyInstaller builds were used.
  - Runtime symlink was switched atomically.
  - Last Codex-managed temp build/source for `blendparam` was cleaned; `build_left=0` was verified for that build id.

## Current Code / Behavior
- `scripts/grpc_server.py`
  - Parses and forwards visualization/distance flags:
    - `--person-fill-background`
    - `--person-fill-background-blend`
    - `--depth-distance-close-threshold`
    - `--ir-distance-close-gap-ratio`
    - `--ir-distance-close-center-ratio`
  - Startup log prints the active blend and distance thresholds.
- `src/tof_pose/realtime_service.py`
  - Default `PERSON_FILL_BACKGROUND_BLEND = 0.5`.
  - `_fill_person_mask_region()` blends background gray with original gray using the configured ratio.
  - `_compute_person_distance_depth()` uses configurable depth threshold.
  - `_compute_person_distance_ir()` uses configurable gap/center thresholds.
  - `_lift_display_contour_head()` performs the top-head display lift.
  - `_record_display_contour()` centralizes contour offset and display transformation.
- `src/tof_pose/person_distance.py`
  - `PersonDistanceEstimate` includes `draw_contour`.
  - Full-frame draw contour is extracted separately from the contour used for distance.
  - The distance contour path should not be loosened casually without validating distance behavior.
- `scripts/oss_realtime_local_infer.py` and `scripts/tailscale_realtime_local_infer.py`
  - Local test scripts now accept the same blend/distance threshold flags.

## Validation Evidence
- Local checks performed after adding startup parameters:
  - `python -m py_compile scripts\grpc_server.py src\tof_pose\realtime_service.py src\tof_pose\person_distance.py src\tof_pose\object_storage.py ai_pb2.py ai_pb2_grpc.py scripts\oss_realtime_local_infer.py scripts\tailscale_realtime_local_infer.py`
  - `python scripts\grpc_server.py --help` showed the new flags.
- Geometry check for head lifting was performed earlier:
  - Center top point moved up.
  - Side top points stayed unchanged.
  - Lower part stayed unchanged.
- Server checks during handoff:
  - Current symlink points to `maixsense-grpc-server-trt-runtime-20260629-blendparam-20260629105350`.
  - PID `3754519` is listening on port `50052`.
  - Runtime executable path matches the current symlink target.
  - Current log fd points to `maixsense-grpc-server-trt-b20-ir-i6-threshold-tuned-20260629.log`.
- Current startup log evidence:
  - `person_fill_background_blend=0.500`
  - `depth_distance_close_threshold=120.000`
  - `ir_distance_close_gap_ratio=0.120`
  - `ir_distance_close_center_ratio=0.800`
- Recent request evidence from previous checked logs:
  - `inputs=20 outputs=20`
  - `oss_download_count=20`
  - `oss_upload_count=40`
  - `grpc_total_ms` around several hundred ms in single-device recent examples.

## Known Issues / Risks
- Worktree is dirty. Do not reset or checkout files casually.
- The current live process was restarted after the Codex `blendparam` deploy with extra threshold/confidence flags:
  - Treat `/proc/<pid>/cmdline` as the current source of truth.
  - Do not rely only on old final answers or older log filenames.
- Current logs include examples where contour candidates are rejected as `mask_area_large`; if user asks why contours disappear, inspect `post_contour_reject_reasons` and `post_contour_debug`.
- GPU memory headroom can be tight because unrelated processes consume significant VRAM.
- When deploying 6 TensorRT instances, stop the current 50052 process tree first; otherwise duplicate runtime startup can OOM during warmup.
- Do not kill unrelated GPU processes unless the user explicitly asks.
- Do not use `systemd`.
- Avoid printing secrets from `/data/care/care-sense-iot-platform/bin/maixsense-oss.env`.

## Open TODO
- Visually verify whether the head-lift display correction fixes flat head appearance on real person frames.
- Tune current threshold set if needed:
  - live: depth `120.0`, IR gap `0.12`, IR center `0.80`
  - defaults in code: depth `150.0`, IR gap `0.15`, IR center `0.90`
- Review `mask_area_large` cases if segmentation contours are still missing.
- If user requests another deploy, preserve the current live threshold/confidence flags unless they explicitly ask to change them.
- If local test is needed without compiling, use `scripts/tailscale_realtime_local_infer.py` or `scripts/oss_realtime_local_infer.py` with the same flags as live service.

## Useful Commands

```bash
# Check current live service on AI server
BIN_DIR=/data/care/care-sense-iot-platform/bin
readlink -f "$BIN_DIR/maixsense-grpc-server-trt-current"
ss -lntp | grep ':50052' || true
PID=$(ss -lntp | sed -n 's/.*:50052.*pid=\([0-9][0-9]*\).*/\1/p' | head -n 1)
tr '\0' ' ' < "/proc/$PID/cmdline"; echo
readlink -f "/proc/$PID/exe"
readlink -f "/proc/$PID/fd/1"
```

```bash
# Current verified startup command shape
cd /data/care/care-sense-iot-platform/bin
set -a
. /data/care/care-sense-iot-platform/bin/maixsense-oss.env
set +a

nohup env PYTHONUNBUFFERED=1 "$(printf '\131\117\114\117_AUTOINSTALL')=false" \
  /data/care/care-sense-iot-platform/bin/maixsense-grpc-server-trt-current/maixsense-grpc-server \
  --host 0.0.0.0 \
  --port 50052 \
  --max-workers 6 \
  --max-msg-mb 160 \
  --device cuda:0 \
  --decode-workers 6 \
  --render-workers 6 \
  --model-instances 6 \
  --warmup-batch-size 20 \
  --device-binding-ttl-sec 0 \
  --output-format jpeg \
  --jpeg-quality 60 \
  --cpu-worker-mode process \
  --cpu-process-start-method fork \
  --input-modality ir \
  --person-fill-background bg_08_dark_frost_reference.png \
  --person-fill-background-blend 0.5 \
  --depth-distance-close-threshold 120.0 \
  --ir-distance-close-gap-ratio 0.12 \
  --ir-distance-close-center-ratio 0.80 \
  --seg-conf 0.35 \
  --contour-new-conf 0.45 \
  --contour-existing-conf 0.25 \
  --pose-gate-kpt-conf 0.40 \
  --pose-kpt-min-points 6 \
  --model-path /data/care/trt-export-lowmem-20260612-b20w1/maixsense-seg-lowmem-b20w1.engine \
  --pose-model-path /data/care/trt-export-lowmem-20260612-b20w1/maixsense-pose-lowmem-b20w1.engine \
  --oss-workers 4 \
  --oss-download-workers 12 \
  --oss-upload-workers 48 \
  --oss-global-workers 60 \
  --oss-download-wait-timeout-ms 300 \
  --oss-upload-wait-timeout-ms 300 \
  --oss-max-pool-connections 128 \
  > /data/care/care-sense-iot-platform/bin/maixsense-grpc-server-trt-b20-ir-i6-threshold-tuned-20260629.log 2>&1 &
echo $! > /data/care/care-sense-iot-platform/bin/maixsense-grpc-server-trt-b20-ir-i6-threshold-tuned-20260629.pid
```

```powershell
# Local syntax check before deploy
python -m py_compile `
  scripts\grpc_server.py `
  src\tof_pose\realtime_service.py `
  src\tof_pose\person_distance.py `
  src\tof_pose\object_storage.py `
  ai_pb2.py `
  ai_pb2_grpc.py `
  scripts\oss_realtime_local_infer.py `
  scripts\tailscale_realtime_local_infer.py
```

```powershell
# Local package shape for server-side PyInstaller deploy
$ts = Get-Date -Format 'yyyyMMddHHmmss'
$name = "maixsense-deploy-<purpose>-$ts.tar.gz"
tar -czf $name ai.proto ai_pb2.py ai_pb2_grpc.py scripts src maixsense-grpc-server-trt.spec maixsense-grpc-server.spec
```

## Guardrails For Next Chat
- Use `maixsense-trt-oss-deploy` runbook for compile/deploy/restart work.
- Before any restart, verify current `ss -lntp`, `/proc/<pid>/cmdline`, `/proc/<pid>/fd/1`, and symlink target.
- Preserve current live flags unless user asks to change them.
- Stop only the current port-50052 service process tree; avoid broad process matching that can kill the SSH control shell.
- Never print or copy OSS access keys or passwords into summaries.
- Do not delete runtime directories, engine directories, logs, pid files, env files, or backend binaries when cleaning temporary source.
- If exact performance is requested, run a controlled test; do not infer from a single tail-log sample.
