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
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
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

try:
    import numpy as _np
    import tensorrt as _trt

    if not hasattr(_trt, "__version__"):
        _trt.__version__ = "10.0.0"
    if hasattr(_trt, "Runtime") and not hasattr(_trt.Runtime, "__enter__"):
        _trt.Runtime.__enter__ = lambda self: self
        _trt.Runtime.__exit__ = lambda self, exc_type, exc, tb: None
    if not hasattr(_trt, "nptype"):
        _trt_dtype_map = {
            getattr(_trt, "float32", None): _np.float32,
            getattr(_trt, "float16", None): _np.float16,
            getattr(_trt, "int8", None): _np.int8,
            getattr(_trt, "int32", None): _np.int32,
            getattr(_trt, "int64", None): _np.int64,
            getattr(_trt, "uint8", None): _np.uint8,
            getattr(_trt, "bool", None): _np.bool_,
        }
        _trt_dtype_map.pop(None, None)
        _trt.nptype = lambda dtype: _trt_dtype_map[dtype]
except Exception:
    pass

from tof_pose.realtime_service import RealtimePoseEngine
from tof_pose.realtime_service import CPU_WORKER_MODE_PROCESS, CPU_WORKER_MODE_THREAD, INPUT_MODALITIES, INPUT_MODALITY_DEPTH
from tof_pose.object_storage import (
    ObjectStorageConfig,
    build_result_object_key,
    create_object_storage_client,
    image_content_type,
)


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _ObjectStorageStageStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._durations_ms: list[int] = []
        self._total_bytes = 0

    def add(self, duration_ms: int, byte_count: int) -> None:
        with self._lock:
            self._durations_ms.append(max(0, int(duration_ms)))
            self._total_bytes += max(0, int(byte_count))

    def summary(self) -> dict[str, int]:
        with self._lock:
            durations = sorted(self._durations_ms)
            total_bytes = self._total_bytes
        if not durations:
            return {
                "count": 0,
                "bytes": total_bytes,
                "p50_ms": 0,
                "p95_ms": 0,
                "max_ms": 0,
            }
        p50_index = len(durations) // 2
        p95_index = min(len(durations) - 1, int((len(durations) - 1) * 0.95 + 0.5))
        return {
            "count": len(durations),
            "bytes": total_bytes,
            "p50_ms": durations[p50_index],
            "p95_ms": durations[p95_index],
            "max_ms": durations[-1],
        }


class ModelServiceServicer(ai_pb2_grpc.ModelServiceServicer):
    def __init__(
        self,
        *,
        stateless: bool = False,
        pose_only: bool = False,
        pose_validate_seg: bool = True,
        pose_fallback: bool = True,
        model_path: str | None = None,
        pose_model_path: str | None = None,
        seg_conf: float | None = None,
        pose_conf: float | None = None,
        pose_kpt_conf: float | None = None,
        pose_gate_kpt_conf: float | None = None,
        pose_kpt_min_points: int = 4,
        mask_threshold: float = 0.5,
        mask_min_area_ratio: float | None = None,
        mask_max_area_ratio: float | None = None,
        contour_new_conf: float | None = None,
        contour_existing_conf: float | None = None,
        device: str | None = None,
        render_workers: int = 1,
        decode_workers: int = 1,
        png_compression: int = 1,
        output_format: str = "png",
        jpeg_quality: int = 80,
        input_modality: str = INPUT_MODALITY_DEPTH,
        cpu_worker_mode: str = CPU_WORKER_MODE_THREAD,
        cpu_process_start_method: str = "auto",
        model_instances: int = 1,
        parallel_models: bool = True,
        warmup_models: bool = True,
        warmup_batch_size: int = 10,
        device_binding_ttl_sec: float = 120.0,
        oss_config: ObjectStorageConfig | None = None,
        oss_workers: int | None = 4,
        oss_download_workers: int | None = None,
        oss_upload_workers: int | None = None,
        oss_global_workers: int = 8,
        oss_download_wait_timeout_ms: int = 150,
        oss_upload_wait_timeout_ms: int = 0,
        null_qualitative_results: bool = False,
    ):
        self._engine_count = max(1, int(model_instances))
        self._engines: list[RealtimePoseEngine] = []
        self._dispatch_lock = threading.Lock()
        self._device_bindings: dict[str, int] = {}
        self._device_last_seen: dict[str, float] = {}
        self._device_binding_ttl_sec = max(0.0, float(device_binding_ttl_sec))
        self._engine_inflight = [0 for _ in range(self._engine_count)]
        self._engine_device_counts = [0 for _ in range(self._engine_count)]
        self._capture_timestamp_lock = threading.Lock()
        self._last_capture_timestamp_by_device: dict[str, int] = {}
        self._oss_config = oss_config
        self._oss_client = create_object_storage_client(oss_config)
        self._oss_workers = max(0, int(oss_workers if oss_workers is not None else 4))
        self._oss_download_workers = max(
            0,
            int(self._oss_workers if oss_download_workers is None else oss_download_workers),
        )
        self._oss_upload_workers = max(
            0,
            int(self._oss_workers if oss_upload_workers is None else oss_upload_workers),
        )
        self._oss_global_workers = max(0, int(oss_global_workers))
        self._oss_download_wait_timeout_ms = max(0, int(oss_download_wait_timeout_ms))
        self._oss_upload_wait_timeout_ms = max(0, int(oss_upload_wait_timeout_ms))
        self._null_qualitative_results = bool(null_qualitative_results)
        self._oss_global_semaphore = (
            threading.BoundedSemaphore(self._oss_global_workers)
            if self._oss_global_workers > 0
            else None
        )
        if self._null_qualitative_results:
            logging.info("Qualitative result fields disabled: person_status/person_distance/action_level omitted")
        if self._oss_client is not None and self._oss_config is not None:
            logging.info(
                (
                    "Object storage enabled: provider=%s endpoint=%s bucket=%s output_prefix=%s "
                    "workers=%d download_workers=%d upload_workers=%d global_workers=%d pool_connections=%d "
                    "connect_timeout_sec=%s read_timeout_sec=%s request_retries=%d request_deadline_sec=%s "
                    "retry_backoff_ms=%d download_wait_timeout_ms=%d upload_wait_timeout_ms=%d"
                ),
                self._oss_config.provider,
                self._oss_config.endpoint,
                self._oss_config.bucket,
                self._oss_config.output_prefix,
                self._oss_workers,
                self._oss_download_workers,
                self._oss_upload_workers,
                self._oss_global_workers,
                self._oss_config.max_pool_connections,
                self._oss_config.connect_timeout_sec,
                self._oss_config.read_timeout_sec,
                self._oss_config.request_retries,
                self._oss_config.request_deadline_sec,
                self._oss_config.retry_backoff_ms,
                self._oss_download_wait_timeout_ms,
                self._oss_upload_wait_timeout_ms,
            )
        for idx in range(self._engine_count):
            logging.info("Initializing AI model instance %d/%d on device=%s", idx + 1, self._engine_count, device or "auto")
            self._engines.append(
                RealtimePoseEngine(
                    stateless=bool(stateless),
                    pose_only=bool(pose_only),
                    pose_validate_seg=bool(pose_validate_seg),
                    pose_fallback=bool(pose_fallback),
                    model_path=Path(model_path) if model_path else None,
                    pose_model_path=Path(pose_model_path) if pose_model_path else None,
                    seg_conf_threshold=seg_conf,
                    pose_conf_threshold=pose_conf,
                    pose_kpt_conf_threshold=pose_kpt_conf,
                    pose_gate_kpt_conf_threshold=pose_gate_kpt_conf,
                    pose_kpt_min_points=pose_kpt_min_points,
                    mask_threshold=mask_threshold,
                    mask_min_area_ratio=mask_min_area_ratio,
                    mask_max_area_ratio=mask_max_area_ratio,
                    contour_new_track_conf_threshold=contour_new_conf,
                    contour_existing_track_conf_threshold=contour_existing_conf,
                    device=device,
                    render_workers=render_workers,
                    decode_workers=decode_workers,
                    png_compression=png_compression,
                    output_format=output_format,
                    jpeg_quality=jpeg_quality,
                    input_modality=input_modality,
                    cpu_worker_mode=cpu_worker_mode,
                    cpu_process_start_method=cpu_process_start_method,
                    instance_name=f"model-{idx}",
                    parallel_models=parallel_models,
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
            self._last_capture_timestamp_by_device.pop(key, None)
            if index is not None and 0 <= index < len(self._engine_device_counts):
                self._engine_device_counts[index] = max(0, self._engine_device_counts[index] - 1)
        if stale_keys:
            logging.info(
                "Pruned %d stale AI stream bindings (ttl_sec=%.1f, engine_device_counts=%s)",
                len(stale_keys),
                self._device_binding_ttl_sec,
                self._engine_device_counts,
            )

    def _least_bound_engine_index(self) -> int:
        return min(
            range(self._engine_count),
            key=lambda idx: (self._engine_device_counts[idx], self._engine_inflight[idx], idx),
        )

    def _acquire_engine(self, device_id: str, batch_id: str) -> tuple[int, RealtimePoseEngine, str]:
        normalized_device_id = str(device_id or "").strip()
        if normalized_device_id:
            key = normalized_device_id
            route_by_device = True
        else:
            key = str(batch_id or "default").strip() or "default"
            route_by_device = False

        with self._dispatch_lock:
            now = time.monotonic()
            self._prune_stale_device_bindings(now)
            index = self._device_bindings.get(key)
            if index is None:
                if route_by_device:
                    index = self._least_bound_engine_index()
                else:
                    index = min(
                        range(self._engine_count),
                        key=lambda idx: (self._engine_inflight[idx], self._engine_device_counts[idx], idx),
                    )
                self._device_bindings[key] = index
                self._engine_device_counts[index] += 1
                logging.info(
                    "Binding AI stream key=%s to model-%d (routing=%s, engine_device_counts=%s)",
                    key,
                    index,
                    "device_id_balanced" if route_by_device else "fallback",
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

    def _oss_concurrency(self, item_count: int, worker_limit: int) -> int:
        count = max(1, int(item_count))
        if worker_limit <= 0:
            return count
        return min(worker_limit, count)

    def _run_object_storage_call(self, call):
        if self._oss_global_semaphore is None:
            return call()
        self._oss_global_semaphore.acquire()
        try:
            return call()
        finally:
            self._oss_global_semaphore.release()

    @staticmethod
    def _image_capture_timestamp_ms(images: list, index: int) -> int:
        if not (0 <= index < len(images)):
            return 0
        try:
            return max(0, int(getattr(images[index], "capture_timestamp_ms", 0) or 0))
        except (TypeError, ValueError):
            return 0

    def _annotate_result_capture_timestamps(self, device_id: str, images: list, results: list[dict]) -> None:
        input_timestamps = [self._image_capture_timestamp_ms(images, index) for index in range(len(images))]
        previous_batch_ts = 0
        last_input_ts = next((value for value in reversed(input_timestamps) if value > 0), 0)
        normalized_device_id = str(device_id or "").strip()
        if normalized_device_id:
            with self._capture_timestamp_lock:
                previous_batch_ts = self._last_capture_timestamp_by_device.get(normalized_device_id, 0)
                if last_input_ts:
                    self._last_capture_timestamp_by_device[normalized_device_id] = last_input_ts

        for result in results:
            try:
                input_index = int(result.get("input_index", -1))
            except (TypeError, ValueError):
                input_index = -1
            current_ts = input_timestamps[input_index] if 0 <= input_index < len(input_timestamps) else 0
            capture_timestamp_ms = current_ts
            if result.get("result_kind") == "interpolated":
                prev_ts = 0
                if input_index > 0:
                    prev_ts = input_timestamps[input_index - 1]
                elif input_index == 0:
                    prev_ts = previous_batch_ts
                if prev_ts and current_ts:
                    capture_timestamp_ms = int((prev_ts + current_ts) // 2)
            result["capture_timestamp_ms"] = int(capture_timestamp_ms or 0)

    def _download_request_images(
        self,
        images: list,
        *,
        device_id: str = "",
        batch_id: str = "",
    ) -> tuple[list[tuple[str, bytes]], int, dict[str, int]]:
        if self._oss_client is None:
            frames = []
            for image in images:
                image_data = bytes(getattr(image, "image_data", b"") or b"")
                if not image_data:
                    object_key = str(getattr(image, "object_key", "") or "").strip()
                    raise ValueError(f"image {getattr(image, 'frame_id', '')!r} has object_key={object_key!r} but object storage is not configured")
                frames.append((str(getattr(image, "frame_id", "") or ""), image_data))
            return frames, 0, _ObjectStorageStageStats().summary()

        download_start = time.perf_counter()
        frames: list[tuple[str, bytes] | None] = [None] * len(images)
        stage_stats = _ObjectStorageStageStats()

        def download_one(index: int, image) -> tuple[int, tuple[str, bytes]]:
            frame_id = str(getattr(image, "frame_id", "") or "")
            object_key = str(getattr(image, "object_key", "") or "").strip()
            if object_key:
                object_start = time.perf_counter()
                data = self._run_object_storage_call(lambda: self._oss_client.get_bytes(object_key))
                stage_stats.add(int((time.perf_counter() - object_start) * 1000), len(data))
                return index, (frame_id, data)
            image_data = bytes(getattr(image, "image_data", b"") or b"")
            if image_data:
                return index, (frame_id, image_data)
            raise ValueError(f"image {frame_id!r} missing object_key")

        futures = []
        executor = ThreadPoolExecutor(max_workers=self._oss_concurrency(len(images), self._oss_download_workers))
        try:
            futures = [executor.submit(download_one, index, image) for index, image in enumerate(images)]
            timeout_sec = (
                self._oss_download_wait_timeout_ms / 1000.0
                if self._oss_download_wait_timeout_ms > 0
                else None
            )
            if timeout_sec is None:
                done = set()
                for future in as_completed(futures):
                    done.add(future)
            else:
                done, pending = wait(futures, timeout=timeout_sec)
                if pending:
                    for future in pending:
                        future.cancel()
                    logging.warning(
                        (
                            "Object storage download wait timeout: device_id=%s batch_id=%s waited_ms=%d "
                            "completed=%d pending=%d"
                        ),
                        device_id,
                        batch_id,
                        self._oss_download_wait_timeout_ms,
                        len(done),
                        len(pending),
                    )
                    raise TimeoutError(
                        (
                            f"object storage download wait timeout after {self._oss_download_wait_timeout_ms}ms: "
                            f"completed={len(done)} pending={len(pending)}"
                        )
                    )

            for future in done:
                index, frame = future.result()
                frames[index] = frame
        finally:
            executor.shutdown(wait=self._oss_download_wait_timeout_ms <= 0, cancel_futures=True)

        summary = stage_stats.summary()
        summary["wait_timeout_ms"] = self._oss_download_wait_timeout_ms
        summary["pending"] = 0
        return [frame for frame in frames if frame is not None], int((time.perf_counter() - download_start) * 1000), summary

    def _upload_result_images(
        self,
        *,
        device_id: str,
        batch_id: str,
        results: list[dict],
    ) -> tuple[int, dict[str, int]]:
        if self._oss_client is None or self._oss_config is None:
            return 0, _ObjectStorageStageStats().summary()

        upload_start = time.perf_counter()
        stage_stats = _ObjectStorageStageStats()
        used_object_keys: set[str] = set()

        def build_upload_task(result_index: int, image_name: str) -> dict:
            result = results[result_index]
            data_key = "pseudo_color_image" if image_name == "pseudo_color" else "skeleton_contour_image"
            format_key = f"{data_key}_format"
            object_key_field = f"{image_name}_object_key"
            image_data = bytes(result.get(data_key, b"") or b"")
            if not image_data:
                raise ValueError(f"empty {image_name} image for result {result_index}")
            image_format = str(result.get(format_key, "") or "")
            try:
                capture_timestamp_ms = max(0, int(result.get("capture_timestamp_ms", 0) or 0))
            except (TypeError, ValueError):
                capture_timestamp_ms = 0

            image_identifier = image_name
            object_key = build_result_object_key(
                output_prefix=self._oss_config.output_prefix,
                device_id=device_id,
                timestamp_ms=capture_timestamp_ms,
                image_name=image_identifier,
                image_format=image_format,
            )
            if object_key in used_object_keys:
                try:
                    output_index = max(0, int(result.get("output_index", result_index)))
                except (TypeError, ValueError):
                    output_index = max(0, int(result_index))
                suffix = 0
                while object_key in used_object_keys:
                    suffix_text = f"{output_index:04d}" if suffix == 0 else f"{output_index:04d}_{suffix}"
                    image_identifier = f"{image_name}_{suffix_text}"
                    object_key = build_result_object_key(
                        output_prefix=self._oss_config.output_prefix,
                        device_id=device_id,
                        timestamp_ms=capture_timestamp_ms,
                        image_name=image_identifier,
                        image_format=image_format,
                    )
                    suffix += 1
            used_object_keys.add(object_key)
            return {
                "result_index": result_index,
                "image_name": image_name,
                "data_key": data_key,
                "object_key_field": object_key_field,
                "image_data": image_data,
                "image_format": image_format,
                "object_key": object_key,
            }

        def upload_one(task: dict) -> str:
            object_start = time.perf_counter()
            stored_key = self._run_object_storage_call(
                lambda: self._oss_client.put_bytes(
                    task["object_key"],
                    task["image_data"],
                    content_type=image_content_type(task["image_format"]),
                )
            )
            stage_stats.add(int((time.perf_counter() - object_start) * 1000), len(task["image_data"]))
            return stored_key

        def log_background_upload_failure(future, task: dict) -> None:
            if future.cancelled():
                logging.warning(
                    "Object storage upload skipped after response: key=%s",
                    task["object_key"],
                )
                return
            try:
                future.result()
            except Exception:
                logging.exception(
                    "Object storage upload failed after response: key=%s",
                    task["object_key"],
                )

        futures = []
        future_tasks = {}
        executor = ThreadPoolExecutor(max_workers=self._oss_concurrency(len(results) * 2, self._oss_upload_workers))
        try:
            for result_index in range(len(results)):
                for image_name in ("pseudo_color", "skeleton_contour"):
                    task = build_upload_task(result_index, image_name)
                    results[result_index][task["object_key_field"]] = task["object_key"]
                    results[result_index][task["data_key"]] = b""
                    future = executor.submit(upload_one, task)
                    futures.append(future)
                    future_tasks[future] = task

            timeout_sec = (
                self._oss_upload_wait_timeout_ms / 1000.0
                if self._oss_upload_wait_timeout_ms > 0
                else None
            )
            if timeout_sec is None:
                done = set()
                for future in as_completed(futures):
                    done.add(future)
            else:
                done, pending = wait(futures, timeout=timeout_sec)
                for future in pending:
                    future.add_done_callback(
                        lambda completed, task=future_tasks[future]: log_background_upload_failure(completed, task)
                    )
                if pending:
                    logging.warning(
                        (
                            "Object storage upload wait timeout: device_id=%s batch_id=%s waited_ms=%d "
                            "completed=%d pending=%d"
                        ),
                        device_id,
                        batch_id,
                        self._oss_upload_wait_timeout_ms,
                        len(done),
                        len(pending),
                    )

            for future in done:
                task = future_tasks[future]
                results[task["result_index"]][task["object_key_field"]] = future.result()

            summary = stage_stats.summary()
            summary["wait_timeout_ms"] = self._oss_upload_wait_timeout_ms
            summary["pending"] = len(futures) - len(done)
            return int((time.perf_counter() - upload_start) * 1000), summary
        finally:
            executor.shutdown(wait=self._oss_upload_wait_timeout_ms <= 0, cancel_futures=False)

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
        try:
            frames, oss_download_ms, oss_download_stats = self._download_request_images(
                images,
                device_id=device_id,
                batch_id=batch_id,
            )
            if len(frames) != len(images):
                raise RuntimeError(f"expected {len(images)} downloaded frames, got {len(frames)}")
        except Exception as e:
            logging.exception("Infer input load failed: device_id=%s batch_id=%s", device_id, batch_id)
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return ai_pb2.InferResponse(device_id=device_id, batch_id=batch_id)

        engine_index, engine, engine_inflight = self._acquire_engine(device_id, batch_id)
        engine_infer_ms = 0
        try:
            engine_start = time.perf_counter()
            results = engine.infer_batch(frames)
            engine_infer_ms = int((time.perf_counter() - engine_start) * 1000)
            if len(results) != len(images) * 2:
                raise RuntimeError(f"expected {len(images) * 2} results for {len(images)} inputs, got {len(results)}")
            self._annotate_result_capture_timestamps(device_id, images, results)
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

        try:
            oss_upload_ms, oss_upload_stats = self._upload_result_images(
                device_id=device_id,
                batch_id=batch_id,
                results=results,
            )
        except Exception as e:
            logging.exception("Infer output upload failed: device_id=%s batch_id=%s instance=%d", device_id, batch_id, engine_index)
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return ai_pb2.InferResponse(device_id=device_id, batch_id=batch_id)

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
            try:
                capture_timestamp_ms = max(0, int(result.get('capture_timestamp_ms', 0) or 0))
            except (TypeError, ValueError):
                capture_timestamp_ms = 0
            result_message = ai_pb2.InferResult(
                frame_id=result.get('frame_id', ''),
                capture_timestamp_ms=capture_timestamp_ms,
                pseudo_color_image=result.get('pseudo_color_image', b''),
                skeleton_contour_image=result.get('skeleton_contour_image', b''),
                pseudo_color_image_format=result.get('pseudo_color_image_format', 'png'),
                skeleton_contour_image_format=result.get('skeleton_contour_image_format', 'png'),
                pseudo_color_object_key=result.get('pseudo_color_object_key', ''),
                skeleton_contour_object_key=result.get('skeleton_contour_object_key', ''),
                person_count=int(result.get('person_count', 0)),
                processing_time_ms=int(result.get('processing_time_ms', 0)),
                result_kind=kind_map.get(result.get('result_kind'), ai_pb2.RESULT_KIND_UNSPECIFIED),
                input_index=input_index,
                output_index=int(result.get('output_index', len(response.results))),
            )
            if not self._null_qualitative_results:
                result_message.person_status = str(result.get('person_status', '') or '')
                result_message.person_distance = str(result.get('person_distance', '') or '')
                result_message.action_level = str(result.get('action_level', '') or '')
            response.results.append(result_message)
        response_build_ms = int((time.perf_counter() - response_build_start) * 1000)
        grpc_total_ms = int((time.perf_counter() - grpc_start) * 1000)
        response.processing_time_ms = grpc_total_ms
        logging.info(
            (
                "Infer request timing: device_id=%s batch_id=%s instance=model-%d "
                "inputs=%d results=%d "
                "oss_download_ms=%d oss_download_count=%d oss_download_bytes=%d "
                "oss_download_p50_ms=%d oss_download_p95_ms=%d oss_download_max_ms=%d "
                "oss_download_wait_timeout_ms=%d oss_download_pending_count=%d "
                "engine_infer_ms=%d "
                "oss_upload_ms=%d oss_upload_count=%d oss_upload_bytes=%d "
                "oss_upload_p50_ms=%d oss_upload_p95_ms=%d oss_upload_max_ms=%d "
                "oss_upload_wait_timeout_ms=%d oss_upload_pending_count=%d "
                "grpc_response_build_ms=%d grpc_total_ms=%d engine_inflight=%s"
            ),
            device_id,
            batch_id,
            engine_index,
            len(images),
            len(results),
            oss_download_ms,
            oss_download_stats["count"],
            oss_download_stats["bytes"],
            oss_download_stats["p50_ms"],
            oss_download_stats["p95_ms"],
            oss_download_stats["max_ms"],
            oss_download_stats.get("wait_timeout_ms", 0),
            oss_download_stats.get("pending", 0),
            engine_infer_ms,
            oss_upload_ms,
            oss_upload_stats["count"],
            oss_upload_stats["bytes"],
            oss_upload_stats["p50_ms"],
            oss_upload_stats["p95_ms"],
            oss_upload_stats["max_ms"],
            oss_upload_stats.get("wait_timeout_ms", 0),
            oss_upload_stats.get("pending", 0),
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
    pose_fallback: bool = True,
    model_path: str | None = None,
    pose_model_path: str | None = None,
    seg_conf: float | None = None,
    pose_conf: float | None = None,
    pose_kpt_conf: float | None = None,
    pose_gate_kpt_conf: float | None = None,
    pose_kpt_min_points: int = 4,
    mask_threshold: float = 0.5,
    mask_min_area_ratio: float | None = None,
    mask_max_area_ratio: float | None = None,
    contour_new_conf: float | None = None,
    contour_existing_conf: float | None = None,
    device: str | None = None,
    render_workers: int = 1,
    decode_workers: int = 1,
    png_compression: int = 1,
    output_format: str = "png",
    jpeg_quality: int = 80,
    input_modality: str = INPUT_MODALITY_DEPTH,
    cpu_worker_mode: str = CPU_WORKER_MODE_THREAD,
    cpu_process_start_method: str = "auto",
    model_instances: int = 1,
    parallel_models: bool = True,
    warmup_models: bool = True,
    warmup_batch_size: int = 10,
    device_binding_ttl_sec: float = 120.0,
    oss_config: ObjectStorageConfig | None = None,
    oss_workers: int | None = 4,
    oss_download_workers: int | None = None,
    oss_upload_workers: int | None = None,
    oss_global_workers: int = 8,
    oss_download_wait_timeout_ms: int = 150,
    oss_upload_wait_timeout_ms: int = 0,
    null_qualitative_results: bool = False,
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
            pose_fallback=pose_fallback,
            model_path=model_path,
            pose_model_path=pose_model_path,
            seg_conf=seg_conf,
            pose_conf=pose_conf,
            pose_kpt_conf=pose_kpt_conf,
            pose_gate_kpt_conf=pose_gate_kpt_conf,
            pose_kpt_min_points=pose_kpt_min_points,
            mask_threshold=mask_threshold,
            mask_min_area_ratio=mask_min_area_ratio,
            mask_max_area_ratio=mask_max_area_ratio,
            contour_new_conf=contour_new_conf,
            contour_existing_conf=contour_existing_conf,
            device=device,
            render_workers=render_workers,
            decode_workers=decode_workers,
            png_compression=png_compression,
            output_format=output_format,
            jpeg_quality=jpeg_quality,
            input_modality=input_modality,
            cpu_worker_mode=cpu_worker_mode,
            cpu_process_start_method=cpu_process_start_method,
            model_instances=model_instances,
            parallel_models=parallel_models,
            warmup_models=warmup_models,
            warmup_batch_size=warmup_batch_size,
            device_binding_ttl_sec=device_binding_ttl_sec,
            oss_config=oss_config,
            oss_workers=oss_workers,
            oss_download_workers=oss_download_workers,
            oss_upload_workers=oss_upload_workers,
            oss_global_workers=oss_global_workers,
            oss_download_wait_timeout_ms=oss_download_wait_timeout_ms,
            oss_upload_wait_timeout_ms=oss_upload_wait_timeout_ms,
            null_qualitative_results=null_qualitative_results,
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
        'Starting gRPC server on %s (max_msg_mb=%d, max_workers=%d, model_instances=%d, parallel_models=%s, cpu_worker_mode=%s, cpu_process_start_method=%s, input_modality=%s)',
        bound_address,
        max_msg_mb,
        max_workers,
        model_instances,
        "true" if parallel_models else "false",
        cpu_worker_mode,
        cpu_process_start_method,
        input_modality,
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
    parser.add_argument(
        '--no-pose-fallback',
        action='store_true',
        help='disable pose fallback when segmentation misses but pose keypoints are confident',
    )
    parser.add_argument('--model-path', default=None, help='override seg model path')
    parser.add_argument('--pose-model-path', default=None, help='override pose model path')
    parser.add_argument('--seg-conf', default=None, type=float, help='override segmentation confidence threshold')
    parser.add_argument('--pose-conf', default=None, type=float, help='override pose confidence threshold')
    parser.add_argument('--pose-kpt-conf', default=None, type=float, help='override pose keypoint conf threshold for pose-only person counting')
    parser.add_argument('--pose-gate-kpt-conf', default=None, type=float, help='override pose keypoint threshold used to validate segmentation tracks')
    parser.add_argument('--pose-kpt-min-points', default=4, type=int, help='min confident keypoints to count one person in pose-only mode')
    parser.add_argument('--mask-threshold', default=0.5, type=float, help='mask binarization threshold for contours')
    parser.add_argument('--mask-min-area-ratio', default=None, type=float, help='minimum mask area ratio accepted for contours')
    parser.add_argument('--mask-max-area-ratio', default=None, type=float, help='maximum mask area ratio accepted for contours')
    parser.add_argument('--contour-new-conf', default=None, type=float, help='confidence threshold for accepting a new contour track')
    parser.add_argument('--contour-existing-conf', default=None, type=float, help='confidence threshold for keeping a consistent existing contour track')
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
        '--input-modality',
        default=INPUT_MODALITY_DEPTH,
        choices=INPUT_MODALITIES,
        help='input image modality: ir applies grayscale median filtering plus CLAHE; depth directly applies pseudo color',
    )
    parser.add_argument(
        '--model-instances',
        default=1,
        type=int,
        help='number of AI model instances to keep in this process; device_id is routed sticky to one instance',
    )
    parser.add_argument(
        '--serial-models',
        action='store_true',
        help='run seg and pose model calls serially instead of in parallel',
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
    parser.add_argument(
        '--null-qualitative-results',
        action='store_true',
        help='omit unvalidated qualitative fields: person_status, person_distance, and action_level',
    )
    parser.add_argument('--oss-provider', default=None, choices=('aliyun', 's3'), help='object storage provider for request/result object keys')
    parser.add_argument('--oss-endpoint', default=None, help='object storage endpoint')
    parser.add_argument('--oss-bucket', default=None, help='object storage bucket name')
    parser.add_argument('--oss-access-key-id', default=None, help='object storage access key id')
    parser.add_argument('--oss-access-key-secret', default=None, help='object storage access key secret')
    parser.add_argument('--oss-region', default=None, help='object storage region, mainly for S3-compatible providers')
    parser.add_argument('--oss-security-token', default=None, help='optional temporary security token')
    parser.add_argument('--oss-output-prefix', default=None, help='prefix for AI result images uploaded by this service')
    parser.add_argument('--oss-max-pool-connections', default=128, type=int, help='max HTTP connection pool size for S3-compatible object storage')
    parser.add_argument('--oss-connect-timeout-sec', default=1.0, type=float, help='object storage connect timeout seconds; <=0 disables')
    parser.add_argument('--oss-read-timeout-sec', default=2.0, type=float, help='object storage read timeout seconds; <=0 disables')
    parser.add_argument('--oss-request-retries', default=2, type=int, help='object storage retry count after the first failed attempt')
    parser.add_argument('--oss-request-deadline-sec', default=3.0, type=float, help='best-effort per-object total deadline seconds; <=0 disables')
    parser.add_argument('--oss-retry-backoff-ms', default=100, type=int, help='initial object storage retry backoff in milliseconds')
    parser.add_argument(
        '--oss-workers',
        default=4,
        type=int,
        help='fallback per-request concurrent workers for object storage; 0 means one worker per object, still capped by --oss-global-workers',
    )
    parser.add_argument(
        '--oss-download-workers',
        default=None,
        type=int,
        help='per-request concurrent workers for object storage downloads; defaults to --oss-workers',
    )
    parser.add_argument(
        '--oss-upload-workers',
        default=None,
        type=int,
        help='per-request concurrent workers for object storage uploads; defaults to --oss-workers',
    )
    parser.add_argument(
        '--oss-global-workers',
        default=8,
        type=int,
        help='global concurrent object storage operations shared by all requests; 0 disables global limiting',
    )
    parser.add_argument(
        '--oss-download-wait-timeout-ms',
        default=150,
        type=int,
        help='maximum milliseconds to wait for request image downloads before failing the request; 0 waits for all downloads',
    )
    parser.add_argument(
        '--oss-upload-wait-timeout-ms',
        default=0,
        type=int,
        help='maximum milliseconds to wait for result uploads before returning object keys; 0 waits for all uploads',
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    oss_config = ObjectStorageConfig.from_values(
        provider=args.oss_provider,
        endpoint=args.oss_endpoint,
        bucket=args.oss_bucket,
        access_key_id=args.oss_access_key_id,
        access_key_secret=args.oss_access_key_secret,
        region=args.oss_region,
        security_token=args.oss_security_token,
        output_prefix=args.oss_output_prefix,
        max_pool_connections=args.oss_max_pool_connections,
        connect_timeout_sec=args.oss_connect_timeout_sec,
        read_timeout_sec=args.oss_read_timeout_sec,
        request_retries=args.oss_request_retries,
        request_deadline_sec=args.oss_request_deadline_sec,
        retry_backoff_ms=args.oss_retry_backoff_ms,
    )
    serve(
        host=args.host,
        port=args.port,
        max_workers=args.max_workers,
        max_msg_mb=args.max_msg_mb,
        stateless=args.stateless,
        pose_only=args.pose_only,
        pose_validate_seg=not args.no_pose_validate,
        pose_fallback=not args.no_pose_fallback,
        model_path=args.model_path,
        pose_model_path=args.pose_model_path,
        seg_conf=args.seg_conf,
        pose_conf=args.pose_conf,
        pose_kpt_conf=args.pose_kpt_conf,
        pose_gate_kpt_conf=args.pose_gate_kpt_conf,
        pose_kpt_min_points=args.pose_kpt_min_points,
        mask_threshold=args.mask_threshold,
        mask_min_area_ratio=args.mask_min_area_ratio,
        mask_max_area_ratio=args.mask_max_area_ratio,
        contour_new_conf=args.contour_new_conf,
        contour_existing_conf=args.contour_existing_conf,
        device=args.device,
        render_workers=args.render_workers,
        decode_workers=args.decode_workers,
        png_compression=args.png_compression,
        output_format=args.output_format,
        jpeg_quality=args.jpeg_quality,
        input_modality=args.input_modality,
        cpu_worker_mode=args.cpu_worker_mode,
        cpu_process_start_method=args.cpu_process_start_method,
        model_instances=args.model_instances,
        parallel_models=not args.serial_models,
        warmup_models=not args.no_warmup,
        warmup_batch_size=args.warmup_batch_size,
        device_binding_ttl_sec=args.device_binding_ttl_sec,
        oss_config=oss_config,
        oss_workers=args.oss_workers,
        oss_download_workers=args.oss_download_workers,
        oss_upload_workers=args.oss_upload_workers,
        oss_global_workers=args.oss_global_workers,
        oss_download_wait_timeout_ms=args.oss_download_wait_timeout_ms,
        oss_upload_wait_timeout_ms=args.oss_upload_wait_timeout_ms,
        null_qualitative_results=args.null_qualitative_results,
    )


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
