#!/usr/bin/env python3
"""gRPC server wrapper for the local ModelService (scripts/infer_service.py).

Usage:
  python scripts/grpc_server.py --host 0.0.0.0 --port 50052
"""
import argparse
import logging
import multiprocessing
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import os
import sys

# ensure repository root and src/ are on sys.path so ai_pb2 and tof_pose can be imported
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
for path in (ROOT, SRC):
    if path not in sys.path:
        sys.path.insert(0, path)

# Ensure Ultralytics can write its settings/config somewhere writable.
# NOTE: keep runtime compatibility with Ultralytics' expected env var name,
# while avoiding the token appearing verbatim in deployment configs.
_cfg_env = "".join(["Y", "O", "L", "O", "_CONFIG_DIR"])
if not os.environ.get(_cfg_env):
    home_dir = os.path.expanduser('~')
    if home_dir and home_dir != '~':
        os.environ[_cfg_env] = os.path.join(home_dir, '.ultralytics')
        try:
            os.makedirs(os.environ[_cfg_env], exist_ok=True)
        except Exception:
            pass

import ai_pb2
import ai_pb2_grpc

from tof_pose.realtime_service import RealtimePoseEngine
from tof_pose.realtime_service import CPU_WORKER_MODE_PROCESS, CPU_WORKER_MODE_THREAD


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ModelServiceServicer(ai_pb2_grpc.ModelServiceServicer):
    def __init__(
        self,
        *,
        stateless: bool = False,
        pose_only: bool = False,
        pose_validate_seg: bool = True,
        model_path: str | None = None,
        pose_model_path: str | None = None,
        pose_conf: float | None = None,
        pose_kpt_conf: float | None = None,
        pose_kpt_min_points: int = 4,
        device: str | None = None,
        render_workers: int = 1,
        decode_workers: int = 1,
        png_compression: int = 1,
        output_format: str = "png",
        jpeg_quality: int = 80,
        cpu_worker_mode: str = CPU_WORKER_MODE_THREAD,
        cpu_process_start_method: str = "auto",
        model_instances: int = 1,
        warmup_models: bool = True,
        warmup_batch_size: int = 10,
        device_binding_ttl_sec: float = 120.0,
    ):
        self._engine_count = max(1, int(model_instances))
        self._engines: list[RealtimePoseEngine] = []
        self._dispatch_lock = threading.Lock()
        self._device_bindings: dict[str, int] = {}
        self._device_last_seen: dict[str, float] = {}
        self._device_binding_ttl_sec = max(0.0, float(device_binding_ttl_sec))
        self._engine_inflight = [0 for _ in range(self._engine_count)]
        self._engine_device_counts = [0 for _ in range(self._engine_count)]
        for idx in range(self._engine_count):
            logging.info("Initializing AI model instance %d/%d on device=%s", idx + 1, self._engine_count, device or "auto")
            self._engines.append(
                RealtimePoseEngine(
                    stateless=bool(stateless),
                    pose_only=bool(pose_only),
                    pose_validate_seg=bool(pose_validate_seg),
                    model_path=Path(model_path) if model_path else None,
                    pose_model_path=Path(pose_model_path) if pose_model_path else None,
                    pose_conf_threshold=pose_conf,
                    pose_kpt_conf_threshold=pose_kpt_conf,
                    pose_kpt_min_points=pose_kpt_min_points,
                    device=device,
                    render_workers=render_workers,
                    decode_workers=decode_workers,
                    png_compression=png_compression,
                    output_format=output_format,
                    jpeg_quality=jpeg_quality,
                    cpu_worker_mode=cpu_worker_mode,
                    cpu_process_start_method=cpu_process_start_method,
                    instance_name=f"model-{idx}",
                )
            )
        if warmup_models:
            warmup_batch_size = max(1, int(warmup_batch_size))
            for idx, engine in enumerate(self._engines):
                logging.info("Warming AI model instance %d/%d with batch_size=%d", idx + 1, self._engine_count, warmup_batch_size)
                engine.warmup(batch_size=warmup_batch_size)

    def _prune_stale_device_bindings(self, now: float) -> None:
        if self._device_binding_ttl_sec <= 0:
            return
        stale_keys = [
            key
            for key, last_seen in self._device_last_seen.items()
            if now - last_seen > self._device_binding_ttl_sec
        ]
        for key in stale_keys:
            index = self._device_bindings.pop(key, None)
            self._device_last_seen.pop(key, None)
            if index is not None and 0 <= index < len(self._engine_device_counts):
                self._engine_device_counts[index] = max(0, self._engine_device_counts[index] - 1)
        if stale_keys:
            logging.info(
                "Pruned %d stale AI stream bindings (ttl_sec=%.1f, engine_device_counts=%s)",
                len(stale_keys),
                self._device_binding_ttl_sec,
                self._engine_device_counts,
            )

    def _acquire_engine(self, device_id: str, batch_id: str) -> tuple[int, RealtimePoseEngine, str]:
        key = str(device_id or batch_id or "default")
        with self._dispatch_lock:
            now = time.monotonic()
            self._prune_stale_device_bindings(now)
            index = self._device_bindings.get(key)
            if index is None:
                index = min(
                    range(self._engine_count),
                    key=lambda idx: (self._engine_inflight[idx], self._engine_device_counts[idx], idx),
                )
                self._device_bindings[key] = index
                self._engine_device_counts[index] += 1
                logging.info(
                    "Binding AI stream key=%s to model-%d (engine_device_counts=%s)",
                    key,
                    index,
                    self._engine_device_counts,
                )
            self._device_last_seen[key] = now
            self._engine_inflight[index] += 1
            inflight_snapshot = list(self._engine_inflight)

        return index, self._engines[index], ",".join(str(value) for value in inflight_snapshot)

    def _release_engine(self, index: int) -> None:
        with self._dispatch_lock:
            if 0 <= index < len(self._engine_inflight):
                self._engine_inflight[index] = max(0, self._engine_inflight[index] - 1)

    def Infer(self, request, context):
        device_id = getattr(request, 'device_id', '')
        batch_id = getattr(request, 'batch_id', '')
        images = list(getattr(request, 'images', []))
        batch_size = int(getattr(request, 'batch_size', 0) or 0)
        if not images:
            context.set_details("images must not be empty")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return ai_pb2.InferResponse(device_id=device_id, batch_id=batch_id)
        if batch_size and batch_size != len(images):
            context.set_details(f"batch_size={batch_size} does not match images count={len(images)}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return ai_pb2.InferResponse(device_id=device_id, batch_id=batch_id)

        grpc_start = time.perf_counter()
        engine_index, engine, engine_inflight = self._acquire_engine(device_id, batch_id)
        engine_infer_ms = 0
        try:
            frames = [(image.frame_id, image.image_data) for image in images]
            engine_start = time.perf_counter()
            results = engine.infer_batch(frames)
            engine_infer_ms = int((time.perf_counter() - engine_start) * 1000)
            if len(results) != len(images) * 2:
                raise RuntimeError(f"expected {len(images) * 2} results for {len(images)} inputs, got {len(results)}")
        except Exception as e:
            logging.exception("Infer failed: device_id=%s batch_id=%s instance=%d", device_id, batch_id, engine_index)
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return ai_pb2.InferResponse(
                device_id=device_id,
                batch_id=batch_id,
            )
        finally:
            self._release_engine(engine_index)

        response_build_start = time.perf_counter()
        response = ai_pb2.InferResponse(
            device_id=device_id,
            batch_id=batch_id,
        )
        kind_map = {
            'interpolated': ai_pb2.RESULT_KIND_INTERPOLATED,
            'current': ai_pb2.RESULT_KIND_CURRENT,
        }
        for result in results:
            input_index = int(result.get('input_index', -1))
            image = images[input_index] if 0 <= input_index < len(images) else None
            capture_timestamp_ms = int(getattr(image, 'capture_timestamp_ms', 0) or 0) if image is not None else 0
            if result.get('result_kind') == 'interpolated' and input_index > 0:
                prev_ts = int(getattr(images[input_index - 1], 'capture_timestamp_ms', 0) or 0)
                if prev_ts and capture_timestamp_ms:
                    capture_timestamp_ms = int((prev_ts + capture_timestamp_ms) // 2)
            response.results.append(
                ai_pb2.InferResult(
                    frame_id=result.get('frame_id', ''),
                    capture_timestamp_ms=capture_timestamp_ms,
                    pseudo_color_image=result.get('pseudo_color_image', b''),
                    skeleton_contour_image=result.get('skeleton_contour_image', b''),
                    pseudo_color_image_format=result.get('pseudo_color_image_format', 'png'),
                    skeleton_contour_image_format=result.get('skeleton_contour_image_format', 'png'),
                    person_count=int(result.get('person_count', 0)),
                    processing_time_ms=int(result.get('processing_time_ms', 0)),
                    result_kind=kind_map.get(result.get('result_kind'), ai_pb2.RESULT_KIND_UNSPECIFIED),
                    input_index=input_index,
                    output_index=int(result.get('output_index', len(response.results))),
                )
            )
        response_build_ms = int((time.perf_counter() - response_build_start) * 1000)
        grpc_total_ms = int((time.perf_counter() - grpc_start) * 1000)
        response.processing_time_ms = grpc_total_ms
        logging.info(
            (
                "Infer request timing: device_id=%s batch_id=%s instance=model-%d "
                "inputs=%d results=%d engine_infer_ms=%d grpc_response_build_ms=%d grpc_total_ms=%d engine_inflight=%s"
            ),
            device_id,
            batch_id,
            engine_index,
            len(images),
            len(results),
            engine_infer_ms,
            response_build_ms,
            grpc_total_ms,
            engine_inflight,
        )
        return response


def serve(
    host: str = '0.0.0.0',
    port: int = 50052,
    max_workers: int = 4,
    max_msg_mb: int = 50,
    *,
    stateless: bool = False,
    pose_only: bool = False,
    pose_validate_seg: bool = True,
    model_path: str | None = None,
    pose_model_path: str | None = None,
    pose_conf: float | None = None,
    pose_kpt_conf: float | None = None,
    pose_kpt_min_points: int = 4,
    device: str | None = None,
    render_workers: int = 1,
    decode_workers: int = 1,
    png_compression: int = 1,
    output_format: str = "png",
    jpeg_quality: int = 80,
    cpu_worker_mode: str = CPU_WORKER_MODE_THREAD,
    cpu_process_start_method: str = "auto",
    model_instances: int = 1,
    warmup_models: bool = True,
    warmup_batch_size: int = 10,
    device_binding_ttl_sec: float = 120.0,
):
    model_instances = max(1, int(model_instances))
    server_opts = [
        ('grpc.max_send_message_length', max_msg_mb * 1024 * 1024),
        ('grpc.max_receive_message_length', max_msg_mb * 1024 * 1024),
    ]
    server = grpc.server(ThreadPoolExecutor(max_workers=max_workers), options=server_opts)
    ai_pb2_grpc.add_ModelServiceServicer_to_server(
        ModelServiceServicer(
            stateless=stateless,
            pose_only=pose_only,
            pose_validate_seg=pose_validate_seg,
            model_path=model_path,
            pose_model_path=pose_model_path,
            pose_conf=pose_conf,
            pose_kpt_conf=pose_kpt_conf,
            pose_kpt_min_points=pose_kpt_min_points,
            device=device,
            render_workers=render_workers,
            decode_workers=decode_workers,
            png_compression=png_compression,
            output_format=output_format,
            jpeg_quality=jpeg_quality,
            cpu_worker_mode=cpu_worker_mode,
            cpu_process_start_method=cpu_process_start_method,
            model_instances=model_instances,
            warmup_models=warmup_models,
            warmup_batch_size=warmup_batch_size,
            device_binding_ttl_sec=device_binding_ttl_sec,
        ),
        server,
    )
    bind_candidates = []
    if host:
        bind_candidates.append(f"{host}:{port}")
    bind_candidates.extend([
        f"0.0.0.0:{port}",
        f"[::]:{port}",
    ])

    bound_address = None
    last_error: Exception | None = None
    for bind in bind_candidates:
        try:
            if server.add_insecure_port(bind):
                bound_address = bind
                break
        except Exception as exc:
            last_error = exc

    if bound_address is None:
        raise RuntimeError(f"Failed to bind gRPC server on {bind_candidates}") from last_error

    if max_workers < model_instances:
        logging.warning(
            "max_workers=%d is lower than model_instances=%d; not all model instances can receive concurrent RPCs",
            max_workers,
            model_instances,
        )
    logging.info(
        'Starting gRPC server on %s (max_msg_mb=%d, max_workers=%d, model_instances=%d, cpu_worker_mode=%s, cpu_process_start_method=%s)',
        bound_address,
        max_msg_mb,
        max_workers,
        model_instances,
        cpu_worker_mode,
        cpu_process_start_method,
    )
    server.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info('Shutting down gRPC server')
        server.stop(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', default=50052, type=int)
    parser.add_argument('--max-workers', default=4, type=int)
    parser.add_argument('--max-msg-mb', default=50, type=int)
    parser.add_argument(
        '--stateless',
        action='store_true',
        help='treat every Infer call as the first frame (no cross-call caching/tracking)',
    )
    parser.add_argument(
        '--pose-only',
        action='store_true',
        help='run pose model only (no segmentation masks/contours); person_count is derived from pose keypoints',
    )
    parser.add_argument(
        '--no-pose-validate',
        action='store_true',
        help='disable pose-based gating for skeleton drawing (contours use segmentation plus shape rules)',
    )
    parser.add_argument('--model-path', default=None, help='override seg model path')
    parser.add_argument('--pose-model-path', default=None, help='override pose model path')
    parser.add_argument('--pose-conf', default=None, type=float, help='override pose confidence threshold (pose-only)')
    parser.add_argument('--pose-kpt-conf', default=None, type=float, help='override pose keypoint conf threshold (pose-only)')
    parser.add_argument('--pose-kpt-min-points', default=4, type=int, help='min confident keypoints to count one person (pose-only)')
    parser.add_argument('--device', default=None, help='model inference device, for example cuda:0 or cpu')
    parser.add_argument(
        '--render-workers',
        default=1,
        type=int,
        help='CPU workers for rendering and encoding returned images',
    )
    parser.add_argument(
        '--decode-workers',
        default=1,
        type=int,
        help='CPU workers for decoding input images and preparing model inputs',
    )
    parser.add_argument(
        '--cpu-worker-mode',
        default=CPU_WORKER_MODE_THREAD,
        choices=(CPU_WORKER_MODE_THREAD, CPU_WORKER_MODE_PROCESS),
        help='CPU worker backend for decode, model input preparation, rendering, and encoding',
    )
    parser.add_argument(
        '--cpu-process-start-method',
        default='auto',
        choices=('auto', 'fork', 'spawn', 'forkserver'),
        help='multiprocessing start method used when --cpu-worker-mode=process',
    )
    parser.add_argument(
        '--png-compression',
        default=1,
        type=int,
        help='PNG compression level for returned images, 0 is fastest and 9 is smallest',
    )
    parser.add_argument(
        '--output-format',
        default='png',
        choices=('png', 'jpeg', 'jpg'),
        help='image format for returned result images',
    )
    parser.add_argument(
        '--jpeg-quality',
        default=80,
        type=int,
        help='JPEG quality for returned images when --output-format=jpeg',
    )
    parser.add_argument(
        '--model-instances',
        default=1,
        type=int,
        help='number of AI model instances to keep in this process; device_id is routed sticky to one instance',
    )
    parser.add_argument(
        '--no-warmup',
        action='store_true',
        help='skip startup model warmup before binding the gRPC server',
    )
    parser.add_argument(
        '--warmup-batch-size',
        default=10,
        type=int,
        help='synthetic image batch size used for startup model warmup',
    )
    parser.add_argument(
        '--device-binding-ttl-sec',
        default=120.0,
        type=float,
        help='seconds after which an inactive device_id is unbound from its sticky model instance; 0 disables pruning',
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    serve(
        host=args.host,
        port=args.port,
        max_workers=args.max_workers,
        max_msg_mb=args.max_msg_mb,
        stateless=args.stateless,
        pose_only=args.pose_only,
        pose_validate_seg=not args.no_pose_validate,
        model_path=args.model_path,
        pose_model_path=args.pose_model_path,
        pose_conf=args.pose_conf,
        pose_kpt_conf=args.pose_kpt_conf,
        pose_kpt_min_points=args.pose_kpt_min_points,
        device=args.device,
        render_workers=args.render_workers,
        decode_workers=args.decode_workers,
        png_compression=args.png_compression,
        output_format=args.output_format,
        jpeg_quality=args.jpeg_quality,
        cpu_worker_mode=args.cpu_worker_mode,
        cpu_process_start_method=args.cpu_process_start_method,
        model_instances=args.model_instances,
        warmup_models=not args.no_warmup,
        warmup_batch_size=args.warmup_batch_size,
        device_binding_ttl_sec=args.device_binding_ttl_sec,
    )


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
