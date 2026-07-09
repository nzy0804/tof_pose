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
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
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
from tof_pose.realtime_service import (
    CPU_WORKER_MODE_PROCESS,
    CPU_WORKER_MODE_THREAD,
    INPUT_MODALITIES,
    INPUT_MODALITY_DEPTH,
    MODEL_INPUT_SIZES,
    MODEL_INPUT_SIZE_320,
    DEPTH_PERSON_DISTANCE_CLOSE_THRESHOLD,
    PERSON_FILL_BACKGROUND_DEFAULT,
    PERSON_FILL_BACKGROUND_BLEND,
    PERSON_FILL_BACKGROUND_NAMES,
    PERSON_DISTANCE_CLOSE_CENTER_RATIO,
    PERSON_DISTANCE_CLOSE_GAP_RATIO,
    POSE_STATUS_THIGH_TORSO_RATIO_THRESHOLD,
)
from tof_pose.object_storage import (
    ObjectStorageConfig,
    create_object_storage_client,
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


@dataclass
class _DynamicInferJob:
    request_id: int
    stream_key: str
    frame_id: str
    image_data: bytes
    input_index: int
    future: Future
    enqueued_at: float


class _DynamicInferBatcher:
    def __init__(
        self,
        *,
        engine: RealtimePoseEngine,
        instance_name: str,
        max_batch_size: int,
        max_wait_ms: int,
        max_queue_size: int,
    ) -> None:
        self._engine = engine
        self._instance_name = instance_name
        self._max_batch_size = max(1, int(max_batch_size))
        self._max_wait_ms = max(0, int(max_wait_ms))
        self._max_queue_size = max(1, int(max_queue_size))
        self._condition = threading.Condition()
        self._queue: deque[_DynamicInferJob] = deque()
        self._closed = False
        self._next_request_id = 0
        self._worker = threading.Thread(
            target=self._run,
            name=f"{instance_name}-dynamic-batch",
            daemon=True,
        )
        self._worker.start()

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    @property
    def max_wait_ms(self) -> int:
        return self._max_wait_ms

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @staticmethod
    def _normalize_request_results(results: list[dict]) -> list[dict]:
        for output_index, result in enumerate(results):
            result["input_index"] = output_index
            result["output_index"] = output_index
        return results

    def submit_async(self, *, stream_key: str, frames: list[tuple[str, bytes]]) -> Future:
        request_future: Future = Future()
        if not frames:
            request_future.set_result([])
            return request_future

        jobs: list[_DynamicInferJob] = []
        with self._condition:
            if self._closed:
                raise RuntimeError(f"dynamic batcher {self._instance_name} is closed")
            if len(self._queue) + len(frames) > self._max_queue_size:
                raise RuntimeError(
                    (
                        f"dynamic batch queue full for {self._instance_name}: "
                        f"queued={len(self._queue)} incoming={len(frames)} max_queue_size={self._max_queue_size}"
                    )
                )
            request_id = self._next_request_id
            self._next_request_id += 1
            enqueued_at = time.perf_counter()
            jobs = [
                _DynamicInferJob(
                    request_id=request_id,
                    stream_key=stream_key,
                    frame_id=frame_id,
                    image_data=image_data,
                    input_index=input_index,
                    future=Future(),
                    enqueued_at=enqueued_at,
                )
                for input_index, (frame_id, image_data) in enumerate(frames)
            ]
            self._queue.extend(jobs)
            self._condition.notify_all()

        remaining = len(jobs)
        aggregate_lock = threading.Lock()

        def complete_request(_completed: Future) -> None:
            nonlocal remaining
            with aggregate_lock:
                remaining -= 1
                if remaining > 0 or request_future.done():
                    return
                try:
                    results = [job.future.result() for job in jobs]
                    request_future.set_result(self._normalize_request_results(results))
                except Exception as exc:
                    request_future.set_exception(exc)

        for job in jobs:
            job.future.add_done_callback(complete_request)
        return request_future

    def submit(self, *, stream_key: str, frames: list[tuple[str, bytes]]) -> list[dict]:
        return self.submit_async(stream_key=stream_key, frames=frames).result()

    def _take_batch(self) -> tuple[list[_DynamicInferJob], int] | None:
        with self._condition:
            while not self._queue and not self._closed:
                self._condition.wait()
            if self._closed and not self._queue:
                return None

            first_enqueued_at = self._queue[0].enqueued_at
            while len(self._queue) < self._max_batch_size and not self._closed:
                if self._max_wait_ms <= 0:
                    break
                elapsed_ms = int((time.perf_counter() - first_enqueued_at) * 1000)
                remaining_ms = self._max_wait_ms - elapsed_ms
                if remaining_ms <= 0:
                    break
                self._condition.wait(timeout=remaining_ms / 1000.0)

            batch_size = min(len(self._queue), self._max_batch_size)
            batch = [self._queue.popleft() for _ in range(batch_size)]
            queue_remaining = len(self._queue)
            return batch, queue_remaining

    def _run(self) -> None:
        while True:
            taken = self._take_batch()
            if taken is None:
                return
            batch, queue_remaining = taken
            if not batch:
                continue

            wait_ms = int((time.perf_counter() - batch[0].enqueued_at) * 1000)
            stream_count = len({job.stream_key for job in batch})
            request_count = len({job.request_id for job in batch})
            logging.info(
                (
                    "Dynamic infer batch: instance=%s frames=%d requests=%d streams=%d "
                    "wait_ms=%d max_batch_size=%d max_wait_ms=%d queue_remaining=%d"
                ),
                self._instance_name,
                len(batch),
                request_count,
                stream_count,
                wait_ms,
                self._max_batch_size,
                self._max_wait_ms,
                queue_remaining,
            )
            try:
                items = [
                    {
                        "request_id": job.request_id,
                        "stream_key": job.stream_key,
                        "frame_id": job.frame_id,
                        "image_data": job.image_data,
                        "input_index": job.input_index,
                    }
                    for job in batch
                ]
                results = self._engine.infer_multi_stream_batch(items)
                if len(results) != len(batch):
                    raise RuntimeError(f"dynamic model batch returned {len(results)} results for {len(batch)} inputs")
                for job, result in zip(batch, results):
                    if not job.future.done():
                        job.future.set_result(result)
            except Exception as exc:
                logging.exception("Dynamic infer batch failed: instance=%s frames=%d", self._instance_name, len(batch))
                for job in batch:
                    if not job.future.done():
                        job.future.set_exception(exc)


@dataclass
class _AsyncInferJob:
    device_id: str
    batch_id: str
    sequence_id: int
    images: list
    accepted_at: float


@dataclass
class _AsyncDeviceState:
    inflight_batches: int = 0
    model_active: bool = False
    last_accepted_sequence_id: int | None = None
    pending: deque[_AsyncInferJob] = field(default_factory=deque)


class _ResultEventHub:
    def __init__(self, max_events: int) -> None:
        self._max_events = max(1, int(max_events))
        self._condition = threading.Condition()
        self._events: deque[tuple[int, ai_pb2.InferResultEvent]] = deque(maxlen=self._max_events)
        self._next_index = 0

    def publish(self, event: ai_pb2.InferResultEvent) -> None:
        with self._condition:
            self._events.append((self._next_index, event))
            self._next_index += 1
            self._condition.notify_all()

    def subscribe(self, *, device_ids: set[str], context):
        with self._condition:
            cursor = self._next_index
        while context.is_active():
            selected: list[ai_pb2.InferResultEvent] = []
            with self._condition:
                while context.is_active():
                    if self._events and cursor < self._events[0][0]:
                        cursor = self._events[0][0]
                    has_new = bool(self._events and self._events[-1][0] >= cursor)
                    if has_new:
                        break
                    self._condition.wait(timeout=1.0)
                if not context.is_active():
                    return
                for index, event in list(self._events):
                    if index < cursor:
                        continue
                    cursor = index + 1
                    if not device_ids or event.device_id in device_ids:
                        selected.append(event)
            for event in selected:
                yield event


class _AsyncInferenceManager:
    def __init__(
        self,
        *,
        servicer,
        device_window: int,
        result_buffer_size: int,
        prepare_workers: int,
        result_workers: int,
        retry_after_ms: int = 100,
    ) -> None:
        self._servicer = servicer
        self._device_window = max(1, int(device_window))
        self._retry_after_ms = max(0, int(retry_after_ms))
        self._lock = threading.Lock()
        self._device_states: dict[str, _AsyncDeviceState] = {}
        self._prepare_executor = ThreadPoolExecutor(
            max_workers=max(1, int(prepare_workers)),
            thread_name_prefix="async-infer-prepare",
        )
        self._result_executor = ThreadPoolExecutor(
            max_workers=max(1, int(result_workers)),
            thread_name_prefix="async-infer-result",
        )
        self.result_hub = _ResultEventHub(result_buffer_size)

    @property
    def device_window(self) -> int:
        return self._device_window

    def submit(self, request) -> ai_pb2.SubmitFramesResponse:
        device_id = str(getattr(request, "device_id", "") or "").strip()
        if not device_id:
            raise ValueError("device_id must not be empty")
        batch_id = str(getattr(request, "batch_id", "") or "").strip()
        try:
            sequence_id = int(getattr(request, "sequence_id", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("sequence_id must be an integer") from exc
        if sequence_id < 0:
            raise ValueError("sequence_id must be >= 0")

        images = list(getattr(request, "images", []))
        batch_size = int(getattr(request, "batch_size", 0) or 0)
        if not images:
            raise ValueError("images must not be empty")
        if batch_size and batch_size != len(images):
            raise ValueError(f"batch_size={batch_size} does not match images count={len(images)}")

        copied_images = []
        for index, image in enumerate(images):
            frame_id = str(getattr(image, "frame_id", "") or "")
            image_data = bytes(getattr(image, "image_data", b"") or b"")
            object_key = str(getattr(image, "object_key", "") or "").strip()
            if not image_data:
                raise ValueError(f"image {index} missing image_data; SubmitFrames requires raw image bytes")
            try:
                capture_timestamp_ms = max(0, int(getattr(image, "capture_timestamp_ms", 0) or 0))
            except (TypeError, ValueError):
                capture_timestamp_ms = 0
            copied_images.append(
                ai_pb2.InferImage(
                    frame_id=frame_id,
                    capture_timestamp_ms=capture_timestamp_ms,
                    image_data=image_data,
                    object_key=object_key,
                )
            )

        batch_id = batch_id or f"{device_id}-{sequence_id}"
        with self._lock:
            state = self._device_states.setdefault(device_id, _AsyncDeviceState())
            if state.inflight_batches >= self._device_window:
                return ai_pb2.SubmitFramesResponse(
                    device_id=device_id,
                    batch_id=batch_id,
                    sequence_id=sequence_id,
                    accepted=False,
                    accepted_count=0,
                    device_inflight_batches=state.inflight_batches,
                    retry_after_ms=self._retry_after_ms,
                    message="device_inflight_window_full",
                )
            if (
                state.last_accepted_sequence_id is not None
                and sequence_id <= state.last_accepted_sequence_id
            ):
                return ai_pb2.SubmitFramesResponse(
                    device_id=device_id,
                    batch_id=batch_id,
                    sequence_id=sequence_id,
                    accepted=False,
                    accepted_count=0,
                    device_inflight_batches=state.inflight_batches,
                    retry_after_ms=0,
                    message="sequence_id_must_increase_per_device",
                )

            job = _AsyncInferJob(
                device_id=device_id,
                batch_id=batch_id,
                sequence_id=sequence_id,
                images=copied_images,
                accepted_at=time.perf_counter(),
            )
            state.last_accepted_sequence_id = sequence_id
            state.inflight_batches += 1
            state.pending.append(job)
            inflight_batches = state.inflight_batches
            self._dispatch_next_locked(device_id, state)

        logging.info(
            "Async infer accepted: device_id=%s batch_id=%s sequence_id=%d images=%d inflight=%d window=%d",
            device_id,
            batch_id,
            sequence_id,
            len(copied_images),
            inflight_batches,
            self._device_window,
        )
        return ai_pb2.SubmitFramesResponse(
            device_id=device_id,
            batch_id=batch_id,
            sequence_id=sequence_id,
            accepted=True,
            accepted_count=len(copied_images),
            device_inflight_batches=inflight_batches,
            retry_after_ms=0,
            message="accepted",
        )

    def _dispatch_next_locked(self, device_id: str, state: _AsyncDeviceState) -> None:
        if state.model_active or not state.pending:
            return
        job = state.pending.popleft()
        state.model_active = True
        self._prepare_executor.submit(self._run_model_stage, job)

    def _mark_model_stage_done(self, job: _AsyncInferJob) -> None:
        with self._lock:
            state = self._device_states.get(job.device_id)
            if state is None:
                return
            state.model_active = False
            self._dispatch_next_locked(job.device_id, state)

    def _mark_result_done(self, job: _AsyncInferJob) -> None:
        with self._lock:
            state = self._device_states.get(job.device_id)
            if state is None:
                return
            state.inflight_batches = max(0, state.inflight_batches - 1)

    def _run_model_stage(self, job: _AsyncInferJob) -> None:
        try:
            frames, oss_download_ms, oss_download_stats = self._servicer._download_request_images(
                job.images,
                device_id=job.device_id,
                batch_id=job.batch_id,
            )
            if len(frames) != len(job.images):
                raise RuntimeError(f"expected {len(job.images)} downloaded frames, got {len(frames)}")

            engine_index, engine, stream_key, engine_inflight = self._servicer._acquire_engine(
                job.device_id,
                job.batch_id,
            )
            engine_start = time.perf_counter()
            try:
                if self._servicer._dynamic_batching:
                    if not (0 <= engine_index < len(self._servicer._dynamic_batchers)):
                        raise RuntimeError(f"dynamic batcher missing for model-{engine_index}")
                    model_future = self._servicer._dynamic_batchers[engine_index].submit_async(
                        stream_key=stream_key,
                        frames=frames,
                    )
                else:
                    model_future = Future()
                    try:
                        model_future.set_result(engine.infer_batch(frames))
                    except Exception as exc:
                        model_future.set_exception(exc)
                model_future.add_done_callback(
                    lambda completed: self._on_model_done(
                        job=job,
                        images=job.images,
                        model_future=completed,
                        engine_index=engine_index,
                        engine_inflight=engine_inflight,
                        engine_start=engine_start,
                        oss_download_ms=oss_download_ms,
                        oss_download_stats=oss_download_stats,
                    )
                )
            except Exception:
                self._servicer._release_engine(engine_index)
                raise
        except Exception as exc:
            logging.exception(
                "Async infer model stage failed before dispatch: device_id=%s batch_id=%s sequence_id=%d",
                job.device_id,
                job.batch_id,
                job.sequence_id,
            )
            self._mark_model_stage_done(job)
            self._publish_error_and_complete(job, str(exc))

    def _on_model_done(
        self,
        *,
        job: _AsyncInferJob,
        images: list,
        model_future: Future,
        engine_index: int,
        engine_inflight: str,
        engine_start: float,
        oss_download_ms: int,
        oss_download_stats: dict[str, int],
    ) -> None:
        engine_infer_ms = int((time.perf_counter() - engine_start) * 1000)
        try:
            results = model_future.result()
            if len(results) != len(images):
                raise RuntimeError(f"expected {len(images)} results for {len(images)} inputs, got {len(results)}")
            self._servicer._annotate_result_capture_timestamps(job.device_id, images, results)
        except Exception as exc:
            logging.exception(
                "Async infer model stage failed: device_id=%s batch_id=%s sequence_id=%d instance=%d",
                job.device_id,
                job.batch_id,
                job.sequence_id,
                engine_index,
            )
            self._servicer._release_engine(engine_index)
            self._mark_model_stage_done(job)
            self._publish_error_and_complete(job, str(exc))
            return

        self._servicer._release_engine(engine_index)
        self._mark_model_stage_done(job)
        self._result_executor.submit(
            self._publish_model_results,
            job,
            results,
            oss_download_ms,
            oss_download_stats,
            engine_infer_ms,
            engine_inflight,
        )

    def _result_message_from_dict(self, result: dict, fallback_output_index: int) -> ai_pb2.InferResult:
        kind_map = {
            "interpolated": ai_pb2.RESULT_KIND_INTERPOLATED,
            "current": ai_pb2.RESULT_KIND_CURRENT,
        }
        try:
            input_index = int(result.get("input_index", -1))
        except (TypeError, ValueError):
            input_index = -1
        try:
            capture_timestamp_ms = max(0, int(result.get("capture_timestamp_ms", 0) or 0))
        except (TypeError, ValueError):
            capture_timestamp_ms = 0
        result_message = ai_pb2.InferResult(
            frame_id=result.get("frame_id", ""),
            capture_timestamp_ms=capture_timestamp_ms,
            pseudo_color_image=result.get("pseudo_color_image", b""),
            skeleton_contour_image=result.get("skeleton_contour_image", b""),
            pseudo_color_image_format=result.get("pseudo_color_image_format", ""),
            skeleton_contour_image_format=result.get("skeleton_contour_image_format", "png"),
            pseudo_color_object_key=result.get("pseudo_color_object_key", ""),
            skeleton_contour_object_key=result.get("skeleton_contour_object_key", ""),
            person_count=int(result.get("person_count", 0)),
            processing_time_ms=int(result.get("processing_time_ms", 0)),
            result_kind=kind_map.get(result.get("result_kind"), ai_pb2.RESULT_KIND_UNSPECIFIED),
            input_index=input_index,
            output_index=int(result.get("output_index", fallback_output_index)),
        )
        if not self._servicer._null_qualitative_results:
            result_message.person_status = str(result.get("person_status", "") or "")
            result_message.person_distance = str(result.get("person_distance", "") or "")
            result_message.action_level = str(result.get("action_level", "") or "")
        return result_message

    def _publish_model_results(
        self,
        job: _AsyncInferJob,
        results: list[dict],
        oss_download_ms: int,
        oss_download_stats: dict[str, int],
        engine_infer_ms: int,
        engine_inflight: str,
    ) -> None:
        try:
            result_image_ms, result_image_stats = self._servicer._prepare_result_images_for_response(results)
            event = ai_pb2.InferResultEvent(
                device_id=job.device_id,
                batch_id=job.batch_id,
                sequence_id=job.sequence_id,
                processing_time_ms=int((time.perf_counter() - job.accepted_at) * 1000),
            )
            for output_index, result in enumerate(results):
                event.results.append(self._result_message_from_dict(result, output_index))
            self.result_hub.publish(event)
            logging.info(
                (
                    "Async infer result published: device_id=%s batch_id=%s sequence_id=%d "
                    "inputs=%d results=%d oss_download_ms=%d oss_download_count=%d "
                    "engine_infer_ms=%d result_image_ms=%d result_image_count=%d "
                    "result_image_bytes=%d "
                    "processing_time_ms=%d engine_inflight=%s"
                ),
                job.device_id,
                job.batch_id,
                job.sequence_id,
                len(job.images),
                len(results),
                oss_download_ms,
                oss_download_stats.get("count", 0),
                engine_infer_ms,
                result_image_ms,
                result_image_stats.get("count", 0),
                result_image_stats.get("bytes", 0),
                event.processing_time_ms,
                engine_inflight,
            )
        except Exception as exc:
            logging.exception(
                "Async infer result publish failed: device_id=%s batch_id=%s sequence_id=%d",
                job.device_id,
                job.batch_id,
                job.sequence_id,
            )
            self._publish_error(job, str(exc))
        finally:
            self._mark_result_done(job)

    def _publish_error(self, job: _AsyncInferJob, error_message: str) -> None:
        event = ai_pb2.InferResultEvent(
            device_id=job.device_id,
            batch_id=job.batch_id,
            sequence_id=job.sequence_id,
            processing_time_ms=int((time.perf_counter() - job.accepted_at) * 1000),
            error_message=error_message,
        )
        self.result_hub.publish(event)

    def _publish_error_and_complete(self, job: _AsyncInferJob, error_message: str) -> None:
        self._publish_error(job, error_message)
        self._mark_result_done(job)


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
        ir_preprocess: bool = False,
        model_input_size: int = MODEL_INPUT_SIZE_320,
        person_fill_background: str | None = None,
        person_fill_background_blend: float = PERSON_FILL_BACKGROUND_BLEND,
        pose_status_thigh_torso_ratio_threshold: float = POSE_STATUS_THIGH_TORSO_RATIO_THRESHOLD,
        depth_distance_close_threshold: float = DEPTH_PERSON_DISTANCE_CLOSE_THRESHOLD,
        ir_distance_close_gap_ratio: float = PERSON_DISTANCE_CLOSE_GAP_RATIO,
        ir_distance_close_center_ratio: float = PERSON_DISTANCE_CLOSE_CENTER_RATIO,
        cpu_worker_mode: str = CPU_WORKER_MODE_THREAD,
        cpu_process_start_method: str = "auto",
        model_instances: int = 1,
        parallel_models: bool = True,
        warmup_models: bool = True,
        warmup_batch_size: int = 10,
        device_binding_ttl_sec: float = 120.0,
        dynamic_batching: bool = False,
        dynamic_max_batch_size: int = 80,
        dynamic_max_wait_ms: int = 300,
        dynamic_max_queue_size: int = 2000,
        async_infer: bool = False,
        async_device_window: int = 3,
        async_result_buffer_size: int = 1000,
        async_prepare_workers: int = 8,
        async_result_workers: int = 8,
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
        self._dynamic_batching = bool(dynamic_batching)
        self._dynamic_max_batch_size = max(1, int(dynamic_max_batch_size))
        self._dynamic_max_wait_ms = max(0, int(dynamic_max_wait_ms))
        self._dynamic_max_queue_size = max(1, int(dynamic_max_queue_size))
        self._dynamic_batchers: list[_DynamicInferBatcher] = []
        self._async_infer = bool(async_infer)
        self._async_device_window = max(1, int(async_device_window))
        self._async_result_buffer_size = max(1, int(async_result_buffer_size))
        self._async_prepare_workers = max(1, int(async_prepare_workers))
        self._async_result_workers = max(1, int(async_result_workers))
        self._async_manager: _AsyncInferenceManager | None = None
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
                    ir_preprocess=ir_preprocess,
                    model_input_size=model_input_size,
                    person_fill_background=person_fill_background,
                    person_fill_background_blend=person_fill_background_blend,
                    pose_status_thigh_torso_ratio_threshold=pose_status_thigh_torso_ratio_threshold,
                    depth_distance_close_threshold=depth_distance_close_threshold,
                    ir_distance_close_gap_ratio=ir_distance_close_gap_ratio,
                    ir_distance_close_center_ratio=ir_distance_close_center_ratio,
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
        if self._dynamic_batching:
            self._dynamic_batchers = [
                _DynamicInferBatcher(
                    engine=engine,
                    instance_name=f"model-{idx}",
                    max_batch_size=self._dynamic_max_batch_size,
                    max_wait_ms=self._dynamic_max_wait_ms,
                    max_queue_size=self._dynamic_max_queue_size,
                )
                for idx, engine in enumerate(self._engines)
            ]
            logging.info(
                (
                    "Dynamic batching enabled: model_instances=%d max_batch_size=%d "
                    "max_wait_ms=%d max_queue_size=%d"
                ),
                self._engine_count,
                self._dynamic_max_batch_size,
                self._dynamic_max_wait_ms,
                self._dynamic_max_queue_size,
            )
        if self._async_infer:
            if not self._dynamic_batching:
                logging.warning("Async inference enabled without dynamic batching; throughput benefit may be limited")
            self._async_manager = _AsyncInferenceManager(
                servicer=self,
                device_window=self._async_device_window,
                result_buffer_size=self._async_result_buffer_size,
                prepare_workers=self._async_prepare_workers,
                result_workers=self._async_result_workers,
            )
            logging.info(
                (
                    "Async inference enabled: device_window=%d result_buffer_size=%d "
                    "prepare_workers=%d result_workers=%d"
                ),
                self._async_device_window,
                self._async_result_buffer_size,
                self._async_prepare_workers,
                self._async_result_workers,
            )

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
                if 0 <= index < len(self._engines):
                    self._engines[index].drop_stream_state(key)
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

    def _acquire_engine(self, device_id: str, batch_id: str) -> tuple[int, RealtimePoseEngine, str, str]:
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

        return index, self._engines[index], key, ",".join(str(value) for value in inflight_snapshot)

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
            image_data = bytes(getattr(image, "image_data", b"") or b"")
            if image_data:
                return index, (frame_id, image_data)
            object_key = str(getattr(image, "object_key", "") or "").strip()
            if object_key:
                object_start = time.perf_counter()
                data = self._run_object_storage_call(lambda: self._oss_client.get_bytes(object_key))
                stage_stats.add(int((time.perf_counter() - object_start) * 1000), len(data))
                return index, (frame_id, data)
            raise ValueError(f"image {frame_id!r} missing image_data/object_key")

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

    def _prepare_result_images_for_response(self, results: list[dict]) -> tuple[int, dict[str, int]]:
        prepare_start = time.perf_counter()
        stage_stats = _ObjectStorageStageStats()
        for result in results:
            result["pseudo_color_image"] = b""
            result["pseudo_color_image_format"] = ""
            result["pseudo_color_object_key"] = ""
            result["skeleton_contour_object_key"] = ""
            image_data = bytes(result.get("skeleton_contour_image", b"") or b"")
            if not image_data:
                raise ValueError("empty skeleton_contour image in inference result")
            stage_stats.add(0, len(image_data))

        summary = stage_stats.summary()
        summary["wait_timeout_ms"] = 0
        summary["pending"] = 0
        return int((time.perf_counter() - prepare_start) * 1000), summary

    def SubmitFrames(self, request, context):
        device_id = getattr(request, "device_id", "")
        batch_id = getattr(request, "batch_id", "")
        sequence_id = int(getattr(request, "sequence_id", 0) or 0)
        if self._async_manager is None:
            context.set_details("async inference is disabled; start server with --async-infer")
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            return ai_pb2.SubmitFramesResponse(
                device_id=device_id,
                batch_id=batch_id,
                sequence_id=sequence_id,
                accepted=False,
                message="async_infer_disabled",
            )
        try:
            return self._async_manager.submit(request)
        except ValueError as exc:
            context.set_details(str(exc))
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return ai_pb2.SubmitFramesResponse(
                device_id=device_id,
                batch_id=batch_id,
                sequence_id=sequence_id,
                accepted=False,
                message=str(exc),
            )
        except Exception as exc:
            logging.exception("SubmitFrames failed: device_id=%s batch_id=%s sequence_id=%s", device_id, batch_id, sequence_id)
            context.set_details(str(exc))
            context.set_code(grpc.StatusCode.INTERNAL)
            return ai_pb2.SubmitFramesResponse(
                device_id=device_id,
                batch_id=batch_id,
                sequence_id=sequence_id,
                accepted=False,
                message=str(exc),
            )

    def SubscribeResults(self, request, context):
        if self._async_manager is None:
            context.set_details("async inference is disabled; start server with --async-infer")
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            return

        device_ids = {
            str(device_id or "").strip()
            for device_id in getattr(request, "device_ids", [])
            if str(device_id or "").strip()
        }
        consumer_id = str(getattr(request, "consumer_id", "") or "").strip() or "anonymous"
        logging.info(
            "Result subscriber connected: consumer_id=%s device_filter=%s",
            consumer_id,
            ",".join(sorted(device_ids)) if device_ids else "*",
        )
        try:
            for event in self._async_manager.result_hub.subscribe(device_ids=device_ids, context=context):
                yield event
        except Exception:
            logging.exception("Result subscriber failed: consumer_id=%s", consumer_id)
            raise
        finally:
            logging.info("Result subscriber disconnected: consumer_id=%s", consumer_id)

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

        engine_index, engine, stream_key, engine_inflight = self._acquire_engine(device_id, batch_id)
        engine_infer_ms = 0
        try:
            engine_start = time.perf_counter()
            if self._dynamic_batching:
                if not (0 <= engine_index < len(self._dynamic_batchers)):
                    raise RuntimeError(f"dynamic batcher missing for model-{engine_index}")
                results = self._dynamic_batchers[engine_index].submit(
                    stream_key=stream_key,
                    frames=frames,
                )
            else:
                results = engine.infer_batch(frames)
            engine_infer_ms = int((time.perf_counter() - engine_start) * 1000)
            if len(results) != len(images):
                raise RuntimeError(f"expected {len(images)} results for {len(images)} inputs, got {len(results)}")
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
            result_image_ms, result_image_stats = self._prepare_result_images_for_response(results)
        except Exception as e:
            logging.exception("Infer result image preparation failed: device_id=%s batch_id=%s instance=%d", device_id, batch_id, engine_index)
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
                pseudo_color_image_format=result.get('pseudo_color_image_format', ''),
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
                "result_image_ms=%d result_image_count=%d result_image_bytes=%d "
                "result_image_p50_ms=%d result_image_p95_ms=%d result_image_max_ms=%d "
                "result_image_pending_count=%d "
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
            result_image_ms,
            result_image_stats["count"],
            result_image_stats["bytes"],
            result_image_stats["p50_ms"],
            result_image_stats["p95_ms"],
            result_image_stats["max_ms"],
            result_image_stats.get("pending", 0),
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
    ir_preprocess: bool = False,
    model_input_size: int = MODEL_INPUT_SIZE_320,
    person_fill_background: str | None = None,
    person_fill_background_blend: float = PERSON_FILL_BACKGROUND_BLEND,
    pose_status_thigh_torso_ratio_threshold: float = POSE_STATUS_THIGH_TORSO_RATIO_THRESHOLD,
    depth_distance_close_threshold: float = DEPTH_PERSON_DISTANCE_CLOSE_THRESHOLD,
    ir_distance_close_gap_ratio: float = PERSON_DISTANCE_CLOSE_GAP_RATIO,
    ir_distance_close_center_ratio: float = PERSON_DISTANCE_CLOSE_CENTER_RATIO,
    cpu_worker_mode: str = CPU_WORKER_MODE_THREAD,
    cpu_process_start_method: str = "auto",
    model_instances: int = 1,
    parallel_models: bool = True,
    warmup_models: bool = True,
    warmup_batch_size: int = 10,
    device_binding_ttl_sec: float = 120.0,
    dynamic_batching: bool = False,
    dynamic_max_batch_size: int = 80,
    dynamic_max_wait_ms: int = 300,
    dynamic_max_queue_size: int = 2000,
    async_infer: bool = False,
    async_device_window: int = 3,
    async_result_buffer_size: int = 1000,
    async_prepare_workers: int = 8,
    async_result_workers: int = 8,
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
            ir_preprocess=ir_preprocess,
            model_input_size=model_input_size,
            person_fill_background=person_fill_background,
            person_fill_background_blend=person_fill_background_blend,
            pose_status_thigh_torso_ratio_threshold=pose_status_thigh_torso_ratio_threshold,
            depth_distance_close_threshold=depth_distance_close_threshold,
            ir_distance_close_gap_ratio=ir_distance_close_gap_ratio,
            ir_distance_close_center_ratio=ir_distance_close_center_ratio,
            cpu_worker_mode=cpu_worker_mode,
            cpu_process_start_method=cpu_process_start_method,
            model_instances=model_instances,
            parallel_models=parallel_models,
            warmup_models=warmup_models,
            warmup_batch_size=warmup_batch_size,
            device_binding_ttl_sec=device_binding_ttl_sec,
            dynamic_batching=dynamic_batching,
            dynamic_max_batch_size=dynamic_max_batch_size,
            dynamic_max_wait_ms=dynamic_max_wait_ms,
            dynamic_max_queue_size=dynamic_max_queue_size,
            async_infer=async_infer,
            async_device_window=async_device_window,
            async_result_buffer_size=async_result_buffer_size,
            async_prepare_workers=async_prepare_workers,
            async_result_workers=async_result_workers,
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
        (
            'Starting gRPC server on %s (max_msg_mb=%d, max_workers=%d, model_instances=%d, '
            'parallel_models=%s, cpu_worker_mode=%s, cpu_process_start_method=%s, input_modality=%s, '
            'ir_preprocess=%s, model_input_size=%d, person_fill_background=%s, person_fill_background_blend=%.3f, '
            'pose_status_thigh_torso_ratio_threshold=%.3f, depth_distance_close_threshold=%.3f, ir_distance_close_gap_ratio=%.3f, '
            'ir_distance_close_center_ratio=%.3f, dynamic_batching=%s, dynamic_max_batch_size=%d, '
            'dynamic_max_wait_ms=%d, dynamic_max_queue_size=%d, async_infer=%s, '
            'async_device_window=%d, async_result_buffer_size=%d, async_prepare_workers=%d, '
            'async_result_workers=%d)'
        ),
        bound_address,
        max_msg_mb,
        max_workers,
        model_instances,
        "true" if parallel_models else "false",
        cpu_worker_mode,
        cpu_process_start_method,
        input_modality,
        "true" if ir_preprocess else "false",
        model_input_size,
        person_fill_background or PERSON_FILL_BACKGROUND_DEFAULT,
        person_fill_background_blend,
        pose_status_thigh_torso_ratio_threshold,
        depth_distance_close_threshold,
        ir_distance_close_gap_ratio,
        ir_distance_close_center_ratio,
        "true" if dynamic_batching else "false",
        dynamic_max_batch_size,
        dynamic_max_wait_ms,
        dynamic_max_queue_size,
        "true" if async_infer else "false",
        async_device_window,
        async_result_buffer_size,
        async_prepare_workers,
        async_result_workers,
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
        help='input image modality: ir returns grayscale images by default; depth returns pseudo color images',
    )
    parser.add_argument(
        '--ir-preprocess',
        action='store_true',
        help='enable median filtering plus CLAHE for infrared grayscale inputs',
    )
    parser.add_argument(
        '--model-input-size',
        default=MODEL_INPUT_SIZE_320,
        type=int,
        choices=MODEL_INPUT_SIZES,
        help='model inference input size; 160 scales the raw image directly before inference',
    )
    parser.add_argument(
        '--person-fill-background',
        default=PERSON_FILL_BACKGROUND_DEFAULT,
        help=(
            'background image used to fill detected person masks; pass an asset name '
            f'({", ".join(PERSON_FILL_BACKGROUND_NAMES)}) or an image file path'
        ),
    )
    parser.add_argument(
        '--person-fill-background-blend',
        default=PERSON_FILL_BACKGROUND_BLEND,
        type=float,
        help='background gray blend ratio for filled person masks, 0 keeps original person gray and 1 uses background gray',
    )
    parser.add_argument(
        '--pose-status-thigh-torso-ratio-threshold',
        default=POSE_STATUS_THIGH_TORSO_RATIO_THRESHOLD,
        type=float,
        help='person_status threshold: thigh_y / torso_y <= this value is sitting, otherwise standing',
    )
    parser.add_argument(
        '--depth-distance-close-threshold',
        default=DEPTH_PERSON_DISTANCE_CLOSE_THRESHOLD,
        type=float,
        help='depth-mode close/far threshold for pairwise person distance',
    )
    parser.add_argument(
        '--ir-distance-close-gap-ratio',
        default=PERSON_DISTANCE_CLOSE_GAP_RATIO,
        type=float,
        help='IR-mode close/far threshold as image-width gap ratio',
    )
    parser.add_argument(
        '--ir-distance-close-center-ratio',
        default=PERSON_DISTANCE_CLOSE_CENTER_RATIO,
        type=float,
        help='IR-mode close/far threshold as average-person-extent center-distance ratio',
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
        '--dynamic-batching',
        action='store_true',
        help='queue concurrent Infer calls per model instance and run one larger model batch',
    )
    parser.add_argument(
        '--dynamic-max-batch-size',
        default=80,
        type=int,
        help='maximum frames in one dynamic model batch before immediate inference',
    )
    parser.add_argument(
        '--dynamic-max-wait-ms',
        default=300,
        type=int,
        help='maximum milliseconds the first queued frame waits for more frames before inference',
    )
    parser.add_argument(
        '--dynamic-max-queue-size',
        default=2000,
        type=int,
        help='maximum queued frames per model instance before rejecting new requests',
    )
    parser.add_argument(
        '--async-infer',
        action='store_true',
        help='enable SubmitFrames/SubscribeResults asynchronous inference RPCs',
    )
    parser.add_argument(
        '--async-device-window',
        default=3,
        type=int,
        help='maximum accepted but unfinished batches per device_id for async inference',
    )
    parser.add_argument(
        '--async-result-buffer-size',
        default=1000,
        type=int,
        help='maximum recent async result events retained for active subscribers',
    )
    parser.add_argument(
        '--async-prepare-workers',
        default=8,
        type=int,
        help='background workers for async input download and model dispatch',
    )
    parser.add_argument(
        '--async-result-workers',
        default=8,
        type=int,
        help='background workers for async output upload and result publication',
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
        ir_preprocess=args.ir_preprocess,
        model_input_size=args.model_input_size,
        person_fill_background=args.person_fill_background,
        person_fill_background_blend=args.person_fill_background_blend,
        pose_status_thigh_torso_ratio_threshold=args.pose_status_thigh_torso_ratio_threshold,
        depth_distance_close_threshold=args.depth_distance_close_threshold,
        ir_distance_close_gap_ratio=args.ir_distance_close_gap_ratio,
        ir_distance_close_center_ratio=args.ir_distance_close_center_ratio,
        cpu_worker_mode=args.cpu_worker_mode,
        cpu_process_start_method=args.cpu_process_start_method,
        model_instances=args.model_instances,
        parallel_models=not args.serial_models,
        warmup_models=not args.no_warmup,
        warmup_batch_size=args.warmup_batch_size,
        device_binding_ttl_sec=args.device_binding_ttl_sec,
        dynamic_batching=args.dynamic_batching,
        dynamic_max_batch_size=args.dynamic_max_batch_size,
        dynamic_max_wait_ms=args.dynamic_max_wait_ms,
        dynamic_max_queue_size=args.dynamic_max_queue_size,
        async_infer=args.async_infer,
        async_device_window=args.async_device_window,
        async_result_buffer_size=args.async_result_buffer_size,
        async_prepare_workers=args.async_prepare_workers,
        async_result_workers=args.async_result_workers,
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
