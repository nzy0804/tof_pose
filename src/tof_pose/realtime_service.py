from __future__ import annotations

import atexit
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging
import multiprocessing
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from tof_pose.paths import DEFAULT_MODEL_PATH, DEFAULT_POSE_MODEL_PATH
from tof_pose.person_distance import estimate_person_distance_from_mask
from tof_pose.pose_drawing import draw_stick_figure


LOGGER = logging.getLogger(__name__)

CONF_THRESHOLD = 0.2
TRACKER_CONFIG = "botsort.yaml"
SEG_INFER_IMGSZ = 320
POSE_INFER_IMGSZ = 320
POSE_INFER_INTERVAL = 2
MASK_BLEND_ALPHA = 0.25
DISPLAY_SIZE = (320, 320)
DISPLAY_SCALE = 3

MEDIAN_BLUR_K = 5
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

DISPLAY_MODE_BOTH = "both"
DISPLAY_MODE_CONTOUR_ONLY = "contour"
DISPLAY_MODE_SKELETON_ONLY = "skeleton"
DISPLAY_MODE_RAW_ONLY = "raw"

CPU_WORKER_MODE_THREAD = "thread"
CPU_WORKER_MODE_PROCESS = "process"

TRACK_VOTE_WINDOW = 5
TRACK_VOTE_MIN_POS = 2
TRACK_TTL_FRAMES = 5
TRACK_STATE_STALE_AFTER = 30
CONTOUR_NEW_TRACK_CONF_THRESHOLD = 0.35
CONTOUR_EXISTING_TRACK_CONF_THRESHOLD = 0.20
CONTOUR_TRACK_STALE_AFTER = 30
CONTOUR_EXISTING_MAX_CENTER_JUMP_PX = 80.0
CONTOUR_EXISTING_MAX_AREA_CHANGE_RATIO = 2.5
MASK_MIN_AREA_RATIO = 0.003
MASK_MAX_AREA_RATIO = 0.5
MASK_AREA_JUMP_RATIO = 1.5
MASK_JUMP_HOLD_FRAMES = 2

TRACK_COLORS = [
    (40, 210, 255),
    (120, 220, 80),
    (255, 180, 50),
    (255, 110, 170),
    (180, 130, 255),
    (100, 245, 210),
    (255, 120, 80),
    (90, 170, 255),
]

_CPU_PROCESS_POOLS: dict[tuple[int, str], ProcessPoolExecutor] = {}
_CPU_PROCESS_POOLS_LOCK = threading.Lock()
_WORKER_CLAHE = None


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _rounded_or_none(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _format_number_list(values: list[float | int | None]) -> str:
    formatted: list[str] = []
    for value in values:
        if value is None:
            formatted.append("-")
        elif isinstance(value, float):
            formatted.append(f"{value:.3f}")
        else:
            formatted.append(str(value))
    return "[" + ",".join(formatted) + "]"


def _format_text_list(values: list[str]) -> str:
    return "[" + ",".join(str(value).replace(" ", "_") for value in values) + "]"


def _result_box_conf_values(result) -> list[float]:
    boxes = getattr(result, "boxes", None)
    conf = getattr(boxes, "conf", None) if boxes is not None else None
    if conf is None:
        return []
    try:
        return [float(value) for value in conf.cpu().numpy().reshape(-1).tolist()]
    except Exception:
        return []


def _result_box_count(result) -> int:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return 0
    try:
        return int(len(boxes))
    except Exception:
        return 0


def _result_kpt_gate_points_max(result, threshold: float) -> int:
    keypoints = getattr(result, "keypoints", None)
    conf = getattr(keypoints, "conf", None) if keypoints is not None else None
    if conf is None:
        return 0
    try:
        conf_np = conf.cpu().numpy()
    except Exception:
        return 0
    if conf_np.size == 0:
        return 0
    if conf_np.ndim == 1:
        return int(np.sum(conf_np >= threshold))
    return int(max((int(np.sum(row >= threshold)) for row in conf_np), default=0))


def _mean_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _cpu_worker_init() -> None:
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass


def _cpu_worker_ping(_value=None) -> bool:
    return True


def _shutdown_cpu_process_pools() -> None:
    with _CPU_PROCESS_POOLS_LOCK:
        pools = list(_CPU_PROCESS_POOLS.values())
        _CPU_PROCESS_POOLS.clear()
    for pool in pools:
        pool.shutdown(wait=False, cancel_futures=True)


atexit.register(_shutdown_cpu_process_pools)


def _resolve_cpu_process_start_method(method: str | None) -> str:
    requested = str(method or "auto").strip().lower()
    available = set(multiprocessing.get_all_start_methods())
    if requested == "auto":
        if "fork" in available:
            return "fork"
        return "spawn"
    if requested not in available:
        raise ValueError(f"multiprocessing start method {requested!r} is not available")
    return requested


def _get_cpu_process_pool(worker_count: int, start_method: str | None) -> ProcessPoolExecutor:
    workers = max(1, int(worker_count))
    resolved_method = _resolve_cpu_process_start_method(start_method)
    key = (workers, resolved_method)
    with _CPU_PROCESS_POOLS_LOCK:
        pool = _CPU_PROCESS_POOLS.get(key)
        if pool is None:
            context = multiprocessing.get_context(resolved_method)
            pool = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=_cpu_worker_init,
            )
            _CPU_PROCESS_POOLS[key] = pool
        return pool


def _warm_cpu_process_pool(worker_count: int, start_method: str | None) -> None:
    pool = _get_cpu_process_pool(worker_count, start_method)
    warmup_count = max(1, int(worker_count))
    list(pool.map(_cpu_worker_ping, range(warmup_count)))


def _get_worker_clahe():
    global _WORKER_CLAHE
    if _WORKER_CLAHE is None:
        _WORKER_CLAHE = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    return _WORKER_CLAHE


def _ensure_uint8_gray_cpu(gray: np.ndarray) -> np.ndarray:
    if gray.ndim != 2:
        raise ValueError("expected single-channel grayscale image")
    if gray.dtype == np.uint8:
        return gray

    gray_float = gray.astype(np.float32, copy=False)
    min_val = float(np.min(gray_float)) if gray_float.size else 0.0
    max_val = float(np.max(gray_float)) if gray_float.size else 0.0
    if not gray_float.size or max_val <= min_val:
        return np.zeros_like(gray_float, dtype=np.uint8)

    normalized = cv2.normalize(gray_float, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def _decode_image_bytes_cpu(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("cannot decode image")
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _decode_frame_cpu(frame: tuple[str, bytes]) -> tuple[str, np.ndarray]:
    frame_id, image_bytes = frame
    return frame_id, _decode_image_bytes_cpu(image_bytes)


def _prepare_depth_views_cpu(depth_gray: np.ndarray) -> dict:
    width, height = DISPLAY_SIZE
    depth_u8 = _ensure_uint8_gray_cpu(depth_gray)
    depth_raw = cv2.resize(depth_u8, DISPLAY_SIZE, interpolation=cv2.INTER_LINEAR)

    enhanced = depth_u8
    if MEDIAN_BLUR_K and MEDIAN_BLUR_K >= 3:
        enhanced = cv2.medianBlur(enhanced, MEDIAN_BLUR_K)
    enhanced = _get_worker_clahe().apply(enhanced)

    depth_up = cv2.resize(enhanced, DISPLAY_SIZE, interpolation=cv2.INTER_LINEAR)
    color_img = cv2.applyColorMap(depth_up, cv2.COLORMAP_MAGMA)
    return {
        "depth_up": depth_up,
        "depth_raw": depth_raw,
        "color_img": color_img,
        "width": width,
        "height": height,
    }


def _prepare_color_image_cpu(depth_gray: np.ndarray) -> np.ndarray:
    return _prepare_depth_views_cpu(depth_gray)["color_img"]


def _track_color(track_id: int) -> tuple[int, int, int]:
    return TRACK_COLORS[abs(int(track_id)) % len(TRACK_COLORS)]


def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in box_b[:4]]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0
    return inter_area / denom


def _box_center_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
    center_a = np.array([(box_a[0] + box_a[2]) * 0.5, (box_a[1] + box_a[3]) * 0.5], dtype=np.float32)
    center_b = np.array([(box_b[0] + box_b[2]) * 0.5, (box_b[1] + box_b[3]) * 0.5], dtype=np.float32)
    return float(np.linalg.norm(center_a - center_b))


def _match_pose_to_seg_tracks(
    seg_boxes: list[np.ndarray],
    seg_track_ids: list[int],
    pose_boxes: list[np.ndarray],
) -> dict[int, int]:
    if not seg_boxes or not seg_track_ids or not pose_boxes:
        return {}

    candidates: list[tuple[float, int, int]] = []
    for pose_idx, pose_box in enumerate(pose_boxes):
        for seg_idx, seg_box in enumerate(seg_boxes):
            iou = _box_iou(pose_box, seg_box)
            center_distance = _box_center_distance(pose_box, seg_box)
            if iou < 0.05 and center_distance > 90.0:
                continue
            score = iou - center_distance / 320.0
            candidates.append((score, pose_idx, seg_idx))

    candidates.sort(key=lambda item: item[0], reverse=True)

    assigned_pose: set[int] = set()
    assigned_seg: set[int] = set()
    matched: dict[int, int] = {}
    for score, pose_idx, seg_idx in candidates:
        if score <= -0.25:
            continue
        if pose_idx in assigned_pose or seg_idx in assigned_seg:
            continue
        matched[pose_idx] = seg_track_ids[seg_idx]
        assigned_pose.add(pose_idx)
        assigned_seg.add(seg_idx)
    return matched


def _compute_pairwise_distances(
    records: list[tuple[int, np.ndarray, float | None]],
    frame_width: int,
) -> tuple[str, list[tuple[int, int, float]]]:
    pairs: list[tuple[int, int, float]] = []
    for idx in range(len(records)):
        track_id_a, box_a, distance_a = records[idx]
        if distance_a is None:
            continue
        center_x_a = float((box_a[0] + box_a[2]) * 0.5)
        normalized_x_a = (center_x_a - frame_width * 0.5) / max(frame_width * 0.5, 1.0)
        lateral_x_a = float(distance_a) * float(np.tan(np.deg2rad(35.0) * normalized_x_a))

        for jdx in range(idx + 1, len(records)):
            track_id_b, box_b, distance_b = records[jdx]
            if distance_b is None:
                continue
            center_x_b = float((box_b[0] + box_b[2]) * 0.5)
            normalized_x_b = (center_x_b - frame_width * 0.5) / max(frame_width * 0.5, 1.0)
            lateral_x_b = float(distance_b) * float(np.tan(np.deg2rad(35.0) * normalized_x_b))
            spacing = float(np.hypot(lateral_x_a - lateral_x_b, float(distance_a) - float(distance_b)))
            if 0.0 < spacing <= 400.0:
                pairs.append((track_id_a, track_id_b, spacing))

    if not pairs:
        return "Pair Dist: N/A", []

    nearest = min(pairs, key=lambda item: item[2])
    return f"Pair Dist: {nearest[0]}-{nearest[1]} ~{nearest[2]:.1f}", pairs


def _render_skeleton_contour_cpu(analysis: dict) -> np.ndarray:
    records = analysis["records"]
    kpt_xy_np = analysis["kpt_xy_np"]
    kpt_conf_np = analysis["kpt_conf_np"]
    pose_to_track = analysis["pose_to_track"]
    validated_track_ids = analysis["validated_track_ids"]
    pose_only = bool(analysis.get("pose_only", False))
    pose_draw_indices = analysis.get("pose_draw_indices") or []
    pose_fallback_indices = analysis.get("pose_fallback_indices") or []
    display = analysis["color_img"].copy()

    for record in records:
        contour = record["contour"]
        if contour is None:
            continue

        box = record["box"]
        anchor = record.get("anchor")
        if anchor is not None and len(anchor) >= 2:
            offset_x, offset_y = int(anchor[0]), int(anchor[1])
        else:
            offset_x, offset_y = int(round(box[0])), int(round(box[1]))

        shifted_contour = contour + np.array([[[offset_x, offset_y]]])
        cv2.drawContours(display, [shifted_contour], -1, record["track_color"], 2, cv2.LINE_AA)

    if kpt_xy_np is not None and kpt_conf_np is not None:
        if pose_only:
            for idx in pose_draw_indices:
                if idx < 0 or idx >= len(kpt_xy_np) or idx >= len(kpt_conf_np):
                    continue
                draw_stick_figure(display, kpt_xy_np[idx], kpt_conf_np[idx], color_override=_track_color(idx + 1))
        else:
            drawn_pose_indices: set[int] = set()
            for idx in range(min(len(kpt_xy_np), len(kpt_conf_np))):
                matched_track = pose_to_track.get(idx)
                if matched_track is None or matched_track not in validated_track_ids:
                    continue
                draw_stick_figure(display, kpt_xy_np[idx], kpt_conf_np[idx], color_override=_track_color(matched_track))
                drawn_pose_indices.add(int(idx))
            for idx in pose_fallback_indices:
                if idx < 0 or idx >= len(kpt_xy_np) or idx >= len(kpt_conf_np) or int(idx) in drawn_pose_indices:
                    continue
                draw_stick_figure(display, kpt_xy_np[idx], kpt_conf_np[idx], color_override=_track_color(idx + 1))

    return display


def _render_analyzed_result_cpu(analyzed: dict) -> dict:
    analysis = analyzed["analysis"]
    return {
        "frame_id": analyzed["frame_id"],
        "pseudo_color_image": analysis["color_img"],
        "skeleton_contour_image": _render_skeleton_contour_cpu(analysis),
        "person_count": int(analyzed["person_count"]),
        "processing_time_ms": int(analyzed["processing_time_ms"]),
    }


def _encode_png_cpu(img: np.ndarray, png_compression: int) -> bytes:
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, min(9, max(0, int(png_compression)))])
    if not ok:
        raise RuntimeError("failed to encode PNG")
    return buf.tobytes()


def _encode_jpeg_cpu(img: np.ndarray, jpeg_quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, min(100, max(1, int(jpeg_quality)))])
    if not ok:
        raise RuntimeError("failed to encode JPEG")
    return buf.tobytes()


def _encode_output_image_cpu(
    img: np.ndarray,
    output_format: str,
    png_compression: int,
    jpeg_quality: int,
) -> tuple[bytes, str]:
    if output_format == "jpeg":
        return _encode_jpeg_cpu(img, jpeg_quality), "jpeg"
    return _encode_png_cpu(img, png_compression), "png"


def _encode_rendered_result_cpu(payload: tuple[dict, str, int, int]) -> dict:
    rendered, output_format, png_compression, jpeg_quality = payload
    person_count = int(rendered["person_count"])
    pseudo_color_image, pseudo_color_format = _encode_output_image_cpu(
        rendered["pseudo_color_image"],
        output_format,
        png_compression,
        jpeg_quality,
    )
    if person_count <= 0:
        skeleton_contour_image = pseudo_color_image
        skeleton_contour_format = pseudo_color_format
    else:
        skeleton_contour_image, skeleton_contour_format = _encode_output_image_cpu(
            rendered["skeleton_contour_image"],
            output_format,
            png_compression,
            jpeg_quality,
        )
    return {
        "frame_id": rendered["frame_id"],
        "pseudo_color_image": pseudo_color_image,
        "skeleton_contour_image": skeleton_contour_image,
        "pseudo_color_image_format": pseudo_color_format,
        "skeleton_contour_image_format": skeleton_contour_format,
        "person_count": person_count,
        "processing_time_ms": int(rendered["processing_time_ms"]),
    }


def _render_and_encode_analyzed_result_cpu(payload: tuple[dict, str, int, int]) -> dict:
    analyzed, output_format, png_compression, jpeg_quality = payload
    analysis = analyzed["analysis"]
    person_count = int(analyzed["person_count"])

    render_start = time.perf_counter()
    pseudo_color_image = analysis["color_img"]
    skeleton_contour_image = None
    if person_count > 0:
        skeleton_contour_image = _render_skeleton_contour_cpu(analysis)
    render_ms = _elapsed_ms(render_start)

    encode_start = time.perf_counter()
    pseudo_color_bytes, pseudo_color_format = _encode_output_image_cpu(
        pseudo_color_image,
        output_format,
        png_compression,
        jpeg_quality,
    )
    if person_count <= 0:
        skeleton_contour_bytes = pseudo_color_bytes
        skeleton_contour_format = pseudo_color_format
    else:
        skeleton_contour_bytes, skeleton_contour_format = _encode_output_image_cpu(
            skeleton_contour_image,
            output_format,
            png_compression,
            jpeg_quality,
        )
    encode_ms = _elapsed_ms(encode_start)

    return {
        "frame_id": analyzed["frame_id"],
        "pseudo_color_image": pseudo_color_bytes,
        "skeleton_contour_image": skeleton_contour_bytes,
        "pseudo_color_image_format": pseudo_color_format,
        "skeleton_contour_image_format": skeleton_contour_format,
        "person_count": person_count,
        "processing_time_ms": int(analyzed["processing_time_ms"]),
        "_render_ms": render_ms,
        "_encode_ms": encode_ms,
    }


@dataclass
class _FrameViewSet:
    pseudo_color_image: bytes
    skeleton_contour_image: bytes
    person_count: int


@dataclass
class _TrackGateState:
    votes: deque
    ttl_remaining: int
    confirmed: bool
    last_seen_frame: int


@dataclass
class _MaskJumpState:
    mask: np.ndarray
    last_seen_frame: int
    rejected_frames: int


@dataclass
class _ContourTrackState:
    box: np.ndarray
    last_seen_frame: int


class RealtimePoseEngine:
    def __init__(
        self,
        model_path: Path | None = None,
        pose_model_path: Path | None = None,
        *,
        stateless: bool = False,
        persist_tracks: bool | None = None,
        pose_only: bool = False,
        pose_validate_seg: bool = True,
        pose_fallback: bool = True,
        seg_conf_threshold: float | None = None,
        pose_conf_threshold: float | None = None,
        pose_kpt_conf_threshold: float | None = None,
        pose_gate_kpt_conf_threshold: float | None = None,
        pose_kpt_min_points: int = 4,
        mask_threshold: float = 0.5,
        contour_new_track_conf_threshold: float | None = None,
        contour_existing_track_conf_threshold: float | None = None,
        device: str | None = None,
        render_workers: int = 1,
        decode_workers: int = 1,
        png_compression: int = 1,
        output_format: str = "png",
        jpeg_quality: int = 80,
        cpu_worker_mode: str = CPU_WORKER_MODE_THREAD,
        cpu_process_start_method: str = "auto",
        instance_name: str | None = None,
    ) -> None:
        model_file = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        pose_model_file = Path(pose_model_path) if pose_model_path else DEFAULT_POSE_MODEL_PATH

        self.seg_model = YOLO(str(model_file))
        self.pose_model = YOLO(str(pose_model_file))
        self._device = str(device).strip() if device else None
        self._render_workers = max(1, int(render_workers))
        self._decode_workers = max(1, int(decode_workers))
        normalized_cpu_worker_mode = str(cpu_worker_mode or CPU_WORKER_MODE_THREAD).strip().lower()
        if normalized_cpu_worker_mode not in {CPU_WORKER_MODE_THREAD, CPU_WORKER_MODE_PROCESS}:
            raise ValueError("cpu_worker_mode must be thread or process")
        self._cpu_worker_mode = normalized_cpu_worker_mode
        self._cpu_process_start_method = _resolve_cpu_process_start_method(cpu_process_start_method)
        self._decode_executor: ThreadPoolExecutor | None = None
        self._render_executor: ThreadPoolExecutor | None = None
        if self._cpu_worker_mode == CPU_WORKER_MODE_THREAD:
            if self._decode_workers > 1:
                self._decode_executor = ThreadPoolExecutor(max_workers=self._decode_workers)
            if self._render_workers > 1:
                self._render_executor = ThreadPoolExecutor(max_workers=self._render_workers)
        else:
            _warm_cpu_process_pool(max(self._decode_workers, self._render_workers), self._cpu_process_start_method)
        self._png_compression = min(9, max(0, int(png_compression)))
        normalized_output_format = str(output_format or "png").strip().lower()
        if normalized_output_format == "jpg":
            normalized_output_format = "jpeg"
        if normalized_output_format not in {"png", "jpeg"}:
            raise ValueError("output_format must be png or jpeg")
        self._output_format = normalized_output_format
        self._jpeg_quality = min(100, max(1, int(jpeg_quality)))
        self._instance_name = str(instance_name).strip() if instance_name else "model-0"
        self._lock = threading.Lock()
        self._stateless = bool(stateless)
        self._persist_tracks = (not self._stateless) if persist_tracks is None else bool(persist_tracks)
        self._pose_only = bool(pose_only)
        self._pose_validate_seg = bool(pose_validate_seg)
        self._pose_fallback = bool(pose_fallback)
        self._seg_conf_threshold = (
            float(seg_conf_threshold) if seg_conf_threshold is not None else float(CONF_THRESHOLD)
        )
        if pose_conf_threshold is None and self._pose_only:
            self._pose_conf_threshold = 0.15
        else:
            self._pose_conf_threshold = float(pose_conf_threshold) if pose_conf_threshold is not None else float(CONF_THRESHOLD)

        # Pose-only person counting heuristics.
        self._pose_kpt_conf_threshold = float(pose_kpt_conf_threshold) if pose_kpt_conf_threshold is not None else 0.20
        self._pose_gate_kpt_conf_threshold = (
            float(pose_gate_kpt_conf_threshold)
            if pose_gate_kpt_conf_threshold is not None
            else 0.35
        )
        self._pose_kpt_min_points = int(pose_kpt_min_points)
        self._mask_threshold = min(1.0, max(0.0, float(mask_threshold)))
        self._contour_new_track_conf_threshold = (
            float(contour_new_track_conf_threshold)
            if contour_new_track_conf_threshold is not None
            else float(CONTOUR_NEW_TRACK_CONF_THRESHOLD)
        )
        self._contour_existing_track_conf_threshold = (
            float(contour_existing_track_conf_threshold)
            if contour_existing_track_conf_threshold is not None
            else float(CONTOUR_EXISTING_TRACK_CONF_THRESHOLD)
        )
        self._frame_idx = 0
        self._cached_pose_boxes: list[np.ndarray] = []
        self._cached_kpt_xy: np.ndarray | None = None
        self._cached_kpt_conf: np.ndarray | None = None
        self._warned_no_masks = False
        self._warned_no_keypoints = False
        self._warned_pose_gate_fallback = False
        self._last_source_depth: np.ndarray | None = None
        self._clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
        self._track_gate: dict[int, _TrackGateState] = {}
        self._mask_jump_state: dict[int, _MaskJumpState] = {}
        self._contour_track_state: dict[int, _ContourTrackState] = {}

    def reset(self) -> None:
        """Reset per-stream caches so next infer behaves like the first frame."""
        self._frame_idx = 0
        self._cached_pose_boxes = []
        self._cached_kpt_xy = None
        self._cached_kpt_conf = None
        self._last_source_depth = None
        self._track_gate.clear()
        self._mask_jump_state.clear()
        self._contour_track_state.clear()

    def warmup(self, batch_size: int = 1) -> None:
        """Run one synthetic model batch so CUDA kernels and model graphs are ready before serving traffic."""
        batch_size = max(1, int(batch_size))
        color_imgs = [
            np.zeros((DISPLAY_SIZE[1], DISPLAY_SIZE[0], 3), dtype=np.uint8)
            for _ in range(batch_size)
        ]

        with self._lock:
            warmup_start = time.perf_counter()
            seg_yolo_ms = 0

            if not self._pose_only:
                seg_start = time.perf_counter()
                self.seg_model.track(
                    color_imgs,
                    conf=self._seg_conf_threshold,
                    persist=False,
                    tracker=TRACKER_CONFIG,
                    classes=[0],
                    imgsz=SEG_INFER_IMGSZ,
                    device=self._device,
                    verbose=False,
                )
                seg_yolo_ms = _elapsed_ms(seg_start)

            pose_start = time.perf_counter()
            self.pose_model.predict(
                color_imgs,
                conf=self._pose_conf_threshold,
                classes=[0],
                imgsz=POSE_INFER_IMGSZ,
                device=self._device,
                verbose=False,
            )
            pose_yolo_ms = _elapsed_ms(pose_start)
            total_ms = _elapsed_ms(warmup_start)

            self.reset()

        LOGGER.info(
            "Warmup timing: instance=%s batch_size=%d seg_model_ms=%d pose_model_ms=%d total_ms=%d device=%s",
            self._instance_name,
            batch_size,
            seg_yolo_ms,
            pose_yolo_ms,
            total_ms,
            self._device or "auto",
        )

    def _guard_mask_jump_with_reason(self, track_id: int, mask: np.ndarray) -> tuple[np.ndarray | None, str | None]:
        """Reject short-lived, per-track mask area spikes without averaging masks."""
        if mask.ndim != 2:
            raise ValueError("mask must be single-channel")
        current = (mask > 0).astype(np.uint8)
        current_area = int(np.count_nonzero(current))
        max_area = int(current.size * float(MASK_MAX_AREA_RATIO))
        state = self._mask_jump_state.get(int(track_id))

        if state is None:
            if current_area > max_area:
                return None, "mask_area_large"
            self._mask_jump_state[int(track_id)] = _MaskJumpState(
                mask=current.copy(),
                last_seen_frame=int(self._frame_idx),
                rejected_frames=0,
            )
            return current, None

        previous = state.mask
        if previous.shape != current.shape:
            previous = cv2.resize(previous, (current.shape[1], current.shape[0]), interpolation=cv2.INTER_NEAREST)
        previous_area = int(np.count_nonzero(previous))

        if current_area > max_area:
            if previous_area > 0 and previous_area <= max_area and int(state.rejected_frames) < int(MASK_JUMP_HOLD_FRAMES):
                state.mask = previous.copy()
                state.last_seen_frame = int(self._frame_idx)
                state.rejected_frames = int(state.rejected_frames) + 1
                return previous.copy(), None
            return None, "mask_area_large"

        area_jump = False
        if previous_area > 0 and current_area > 0:
            area_ratio = float(max(current_area / previous_area, previous_area / current_area))
            area_jump = area_ratio > float(MASK_AREA_JUMP_RATIO)
        elif previous_area > 0 and current_area == 0:
            area_jump = True

        if area_jump and int(state.rejected_frames) < int(MASK_JUMP_HOLD_FRAMES):
            state.mask = previous.copy()
            state.last_seen_frame = int(self._frame_idx)
            state.rejected_frames = int(state.rejected_frames) + 1
            return previous.copy(), None

        state.mask = current.copy()
        state.last_seen_frame = int(self._frame_idx)
        state.rejected_frames = 0
        return current, None

    def _guard_mask_jump(self, track_id: int, mask: np.ndarray) -> np.ndarray | None:
        guarded_mask, _reason = self._guard_mask_jump_with_reason(track_id, mask)
        return guarded_mask

    def _contour_shape_reject_reason(
        self,
        box: np.ndarray,
        mask: np.ndarray,
        person_conf: float,
        width: int,
        height: int,
    ) -> str | None:
        if mask.ndim != 2:
            return "invalid_mask"

        mask_area = int(np.count_nonzero(mask > 0))
        image_area = max(1, int(width) * int(height))
        if mask_area < int(image_area * float(MASK_MIN_AREA_RATIO)):
            return "mask_area_small"
        if mask_area > int(image_area * float(MASK_MAX_AREA_RATIO)):
            return "mask_area_large"

        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        x1 = max(0.0, min(x1, float(width - 1)))
        y1 = max(0.0, min(y1, float(height - 1)))
        x2 = max(0.0, min(x2, float(width)))
        y2 = max(0.0, min(y2, float(height)))
        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        if box_w < 2.0 or box_h < 2.0:
            return "box_too_small"

        return None

    def _passes_contour_shape_rules(
        self,
        box: np.ndarray,
        mask: np.ndarray,
        person_conf: float,
        width: int,
        height: int,
    ) -> bool:
        """Validate segmentation shape without requiring pose support."""
        return self._contour_shape_reject_reason(box, mask, person_conf, width, height) is None

    def _get_active_contour_track(self, track_id: int) -> _ContourTrackState | None:
        state = self._contour_track_state.get(int(track_id))
        if state is None:
            return None
        if (self._frame_idx - int(state.last_seen_frame)) > int(CONTOUR_TRACK_STALE_AFTER):
            return None
        return state

    def _passes_existing_contour_position_rules(self, state: _ContourTrackState, box: np.ndarray) -> bool:
        return self._existing_contour_position_reject_reason(state, box) is None

    def _existing_contour_position_reject_reason(self, state: _ContourTrackState, box: np.ndarray) -> str | None:
        previous_box = state.box
        center_distance = _box_center_distance(previous_box, box)
        if center_distance > float(CONTOUR_EXISTING_MAX_CENTER_JUMP_PX):
            return "track_center_jump"

        prev_area = max(1.0, float(max(0.0, previous_box[2] - previous_box[0]) * max(0.0, previous_box[3] - previous_box[1])))
        curr_area = max(1.0, float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])))
        area_ratio = max(curr_area / prev_area, prev_area / curr_area)
        if area_ratio > float(CONTOUR_EXISTING_MAX_AREA_CHANGE_RATIO):
            return "track_area_jump"
        return None

    def _passes_contour_conf_rules(self, track_id: int, box: np.ndarray, person_conf: float) -> bool:
        return self._contour_conf_reject_reason(track_id, box, person_conf) is None

    def _contour_conf_reject_reason(self, track_id: int, box: np.ndarray, person_conf: float) -> str | None:
        if float(person_conf) >= float(self._contour_new_track_conf_threshold):
            return None
        if float(person_conf) < float(self._contour_existing_track_conf_threshold):
            return "conf_low"

        state = self._get_active_contour_track(track_id)
        if state is None:
            return "track_missing"
        return self._existing_contour_position_reject_reason(state, box)

    def _remember_contour_track(self, track_id: int, box: np.ndarray) -> None:
        self._contour_track_state[int(track_id)] = _ContourTrackState(
            box=box.astype(np.float32, copy=True),
            last_seen_frame=int(self._frame_idx),
        )

    def _prune_contour_track_state(self) -> None:
        stale: list[int] = []
        for tid, state in self._contour_track_state.items():
            if (self._frame_idx - int(state.last_seen_frame)) > int(CONTOUR_TRACK_STALE_AFTER):
                stale.append(int(tid))
        for tid in stale:
            self._contour_track_state.pop(int(tid), None)

    def _decode_image(self, data: bytes) -> np.ndarray:
        return _decode_image_bytes_cpu(data)

    def _ensure_uint8_gray(self, gray: np.ndarray) -> np.ndarray:
        """Ensure input is a single-channel uint8 image.

        The gRPC pipeline assumes an 8-bit depth-like grayscale image.
        If upstream data is 16-bit or float, normalize to 0..255.
        """
        if gray.ndim != 2:
            raise ValueError("expected single-channel grayscale image")
        if gray.dtype == np.uint8:
            return gray

        gray_float = gray.astype(np.float32, copy=False)
        min_val = float(np.min(gray_float)) if gray_float.size else 0.0
        max_val = float(np.max(gray_float)) if gray_float.size else 0.0
        if not gray_float.size or max_val <= min_val:
            return np.zeros_like(gray_float, dtype=np.uint8)

        normalized = cv2.normalize(gray_float, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)

    def _prepare_depth_views(self, depth_gray: np.ndarray) -> dict:
        width, height = DISPLAY_SIZE
        depth_u8 = self._ensure_uint8_gray(depth_gray)

        # Keep a non-equalized copy (resized only) for distance estimation.
        depth_raw = cv2.resize(depth_u8, DISPLAY_SIZE, interpolation=cv2.INTER_LINEAR)

        # Match the previous offline video inference pipeline:
        # grayscale -> median blur -> CLAHE -> (then) resize to 320x320.
        enhanced = depth_u8
        if MEDIAN_BLUR_K and MEDIAN_BLUR_K >= 3:
            enhanced = cv2.medianBlur(enhanced, MEDIAN_BLUR_K)
        enhanced = self._clahe.apply(enhanced)

        depth_up = cv2.resize(enhanced, DISPLAY_SIZE, interpolation=cv2.INTER_LINEAR)
        color_img = cv2.applyColorMap(depth_up, cv2.COLORMAP_MAGMA)
        return {
            "depth_up": depth_up,
            "depth_raw": depth_raw,
            "color_img": color_img,
            "width": width,
            "height": height,
        }

    def _analyze_frame(
        self,
        depth_gray: np.ndarray,
        *,
        seg_result=None,
        pose_result=None,
    ) -> dict:
        self._frame_idx += 1
        prepared = self._prepare_depth_views(depth_gray)
        depth_up = prepared["depth_up"]
        depth_raw = prepared["depth_raw"]
        color_img = prepared["color_img"]
        width = prepared["width"]
        height = prepared["height"]

        # Pose-only mode: do not rely on segmentation model outputs.
        if self._pose_only:
            if pose_result is None:
                pose_results = self.pose_model.predict(
                    color_img,
                    conf=self._pose_conf_threshold,
                    classes=[0],
                    imgsz=POSE_INFER_IMGSZ,
                    device=self._device,
                    verbose=False,
                )
                pose_result = pose_results[0] if pose_results else None
            kpt_xy_np = None
            kpt_conf_np = None
            if pose_result is not None and pose_result.keypoints is not None:
                keypoints_xy = pose_result.keypoints.xy
                keypoints_conf = pose_result.keypoints.conf
                if keypoints_xy is not None and keypoints_conf is not None:
                    kpt_xy_np = keypoints_xy.cpu().numpy()
                    kpt_conf_np = keypoints_conf.cpu().numpy()

            pose_draw_indices: list[int] = []
            if kpt_conf_np is not None:
                for idx in range(len(kpt_conf_np)):
                    confident_points = int(np.sum(kpt_conf_np[idx] >= self._pose_kpt_conf_threshold))
                    if confident_points >= self._pose_kpt_min_points:
                        pose_draw_indices.append(idx)

            return {
                "depth_up": depth_up,
                "color_img": color_img,
                "records": [],
                "person_count": int(len(pose_draw_indices)),
                "pair_text": "Pair Dist: N/A",
                "pair_stats": [],
                "tracked_labels": [],
                "kpt_xy_np": kpt_xy_np,
                "kpt_conf_np": kpt_conf_np,
                "pose_to_track": {},
                "validated_track_ids": set(),
                "pose_only": True,
                "pose_draw_indices": pose_draw_indices,
                "pose_fallback_indices": [],
                "contour_reject_reasons": ["pose_only"],
                "width": width,
                "height": height,
            }

        if seg_result is None:
            results = self.seg_model.track(
                color_img,
                conf=self._seg_conf_threshold,
                persist=self._persist_tracks,
                tracker=TRACKER_CONFIG,
                classes=[0],
                imgsz=SEG_INFER_IMGSZ,
                device=self._device,
                verbose=False,
            )
            result = results[0]
        else:
            result = seg_result

        run_pose_now = pose_result is not None or (self._frame_idx % POSE_INFER_INTERVAL == 0) or (self._cached_kpt_xy is None)
        if run_pose_now:
            if pose_result is None:
                pose_results = self.pose_model.predict(
                    color_img,
                    conf=self._pose_conf_threshold,
                    classes=[0],
                    imgsz=POSE_INFER_IMGSZ,
                    device=self._device,
                    verbose=False,
                )
                pose_result = pose_results[0] if pose_results else None
            if pose_result is not None:
                if pose_result.keypoints is not None:
                    keypoints_xy = pose_result.keypoints.xy
                    keypoints_conf = pose_result.keypoints.conf
                    if keypoints_xy is not None and keypoints_conf is not None:
                        self._cached_kpt_xy = keypoints_xy.cpu().numpy()
                        self._cached_kpt_conf = keypoints_conf.cpu().numpy()
                elif not self._warned_no_keypoints:
                    print(
                        "[tof_pose] 当前姿态模型没有输出 keypoints，请改用 pose 权重。",
                        flush=True,
                    )
                    self._warned_no_keypoints = True

                if pose_result.boxes is not None and len(pose_result.boxes) > 0:
                    self._cached_pose_boxes = [box.copy() for box in pose_result.boxes.xyxy.cpu().numpy()]

        pose_boxes_np = self._cached_pose_boxes
        kpt_xy_np = self._cached_kpt_xy
        kpt_conf_np = self._cached_kpt_conf

        if result.boxes is not None and len(result.boxes) > 0 and result.masks is None and not self._warned_no_masks:
            print(
                "[tof_pose] 当前模型没有输出 segmentation masks，请改用 segment 权重。",
                flush=True,
            )
            self._warned_no_masks = True

        records: list[dict] = []
        person_count = 0
        tracked_labels: list[str] = []
        pair_records: list[tuple[int, np.ndarray, float | None]] = []
        pose_to_track: dict[int, int] = {}
        validated_track_ids: set[int] = set()
        contour_reject_reasons: list[str] = []

        if result.boxes is not None and len(result.boxes) > 0 and result.masks is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            track_ids = (
                result.boxes.id.int().cpu().tolist()
                if result.boxes.id is not None
                else list(range(1, len(boxes_xyxy) + 1))
            )
            conf_scores = (
                result.boxes.conf.cpu().numpy()
                if getattr(result.boxes, "conf", None) is not None
                else np.ones(len(boxes_xyxy), dtype=np.float32)
            )
            masks_data = result.masks.data.cpu().numpy()

            match_count = min(len(boxes_xyxy), len(masks_data), len(track_ids), len(conf_scores))
            if match_count <= 0:
                contour_reject_reasons.append("candidate_mismatch")
            seg_boxes_for_match = [boxes_xyxy[idx].copy() for idx in range(match_count)]
            seg_track_ids_for_match = [int(track_ids[idx]) for idx in range(match_count)]

            # Pose-based validation candidates for this frame.
            pose_validated_track_ids: set[int] = set()

            if pose_boxes_np:
                pose_to_track = _match_pose_to_seg_tracks(seg_boxes_for_match, seg_track_ids_for_match, pose_boxes_np)
                if kpt_conf_np is not None:
                    for pose_idx, track_id in pose_to_track.items():
                        if pose_idx >= len(kpt_conf_np):
                            continue
                        confident_points = int(np.sum(kpt_conf_np[pose_idx] >= self._pose_gate_kpt_conf_threshold))
                        if confident_points >= 4:
                            pose_validated_track_ids.add(int(track_id))

            if not self._pose_validate_seg:
                # Disable pose-based skeleton gating: draw any pose matched to a seg track.
                validated_track_ids = set(int(tid) for tid in seg_track_ids_for_match)
            else:
                # Track-level temporal gating:
                # - Sliding window votes to confirm a pose-supported skeleton.
                # - TTL keep-alive to avoid brief pose dropouts causing skeleton gaps.
                pose_available = kpt_conf_np is not None
                if not pose_available:
                    # If pose model produces no keypoints at all, gating cannot work.
                    # Contours still follow seg+shape rules; skeleton output is skipped.
                    validated_track_ids = set(int(tid) for tid in seg_track_ids_for_match)
                    if not self._warned_pose_gate_fallback:
                        print(
                            "[tof_pose] pose 门控不可用(当前无 keypoints 输出)，已回退为 seg-only 以避免输出断档。",
                            flush=True,
                        )
                        self._warned_pose_gate_fallback = True
                else:
                    for track_id in seg_track_ids_for_match:
                        state = self._track_gate.get(int(track_id))
                        if state is None:
                            state = _TrackGateState(
                                votes=deque(maxlen=TRACK_VOTE_WINDOW),
                                ttl_remaining=0,
                                confirmed=False,
                                last_seen_frame=self._frame_idx,
                            )
                            self._track_gate[int(track_id)] = state

                        supported = int(track_id) in pose_validated_track_ids
                        state.votes.append(bool(supported))
                        state.last_seen_frame = self._frame_idx
                        if supported:
                            state.ttl_remaining = int(TRACK_TTL_FRAMES)
                        else:
                            state.ttl_remaining = max(0, int(state.ttl_remaining) - 1)

                        # Sliding-window confirmation: require >=K positives within last N frames.
                        vote_pos = int(sum(1 for v in state.votes if v))
                        if (not state.confirmed) and (vote_pos >= TRACK_VOTE_MIN_POS):
                            state.confirmed = True

                    # Drop stale states to avoid unbounded growth.
                    stale: list[int] = []
                    for tid, state in self._track_gate.items():
                        if (self._frame_idx - int(state.last_seen_frame)) > int(TRACK_STATE_STALE_AFTER):
                            stale.append(int(tid))
                    for tid in stale:
                        self._track_gate.pop(int(tid), None)

                    validated_track_ids = set()
                    for track_id in seg_track_ids_for_match:
                        state = self._track_gate.get(int(track_id))
                        if state is None:
                            continue

                        vote_pos = int(sum(1 for v in state.votes if v))
                        passes_vote = vote_pos >= TRACK_VOTE_MIN_POS
                        passes_ttl = int(state.ttl_remaining) > 0
                        passes_instant = int(track_id) in pose_validated_track_ids

                        if passes_instant or passes_vote or passes_ttl:
                            validated_track_ids.add(int(track_id))

            for idx in range(match_count):
                box = boxes_xyxy[idx]
                track_id = int(track_ids[idx])
                track_color = _track_color(track_id)
                person_conf = float(conf_scores[idx])
                conf_reject_reason = self._contour_conf_reject_reason(track_id, box, person_conf)
                if conf_reject_reason is not None:
                    contour_reject_reasons.append(conf_reject_reason)
                    continue
                mask = (masks_data[idx] > self._mask_threshold).astype(np.uint8)
                if mask.shape[:2] != (height, width):
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                mask, mask_reject_reason = self._guard_mask_jump_with_reason(track_id, mask)
                if mask is None:
                    contour_reject_reasons.append(mask_reject_reason or "mask_rejected")
                    continue
                shape_reject_reason = self._contour_shape_reject_reason(box, mask, person_conf, width, height)
                if shape_reject_reason is not None:
                    contour_reject_reasons.append(shape_reject_reason)
                    continue
                # Use non-equalized depth values for distance estimation.
                estimate = estimate_person_distance_from_mask(depth_raw, box, mask)
                person_count += 1

                if estimate.distance is None:
                    tracked_labels.append(f"{track_id}:N/A")
                    label = f"ID {track_id} P={person_conf * 100:.0f}% Dist=N/A"
                else:
                    tracked_labels.append(f"{track_id}:{estimate.distance:.1f}")
                    label = f"ID {track_id} P={person_conf * 100:.0f}% Dist~{estimate.distance:.1f}"

                pair_records.append((track_id, box.copy(), estimate.distance))
                records.append(
                    {
                        "track_id": track_id,
                        "box": box.copy(),
                        "mask": mask,
                        "contour": estimate.contour,
                        "anchor": getattr(estimate, "anchor", None),
                        "distance": estimate.distance,
                        "person_conf": person_conf,
                        "label": label,
                        "track_color": track_color,
                    }
                )
                self._remember_contour_track(track_id, box)

            stale_mask_states: list[int] = []
            for tid, state in self._mask_jump_state.items():
                if (self._frame_idx - int(state.last_seen_frame)) > int(TRACK_STATE_STALE_AFTER):
                    stale_mask_states.append(int(tid))
            for tid in stale_mask_states:
                self._mask_jump_state.pop(int(tid), None)
        elif result.boxes is not None and len(result.boxes) > 0 and result.masks is None:
            contour_reject_reasons.append("no_mask")

        self._prune_contour_track_state()

        pose_fallback_indices: list[int] = []
        if kpt_xy_np is not None and kpt_conf_np is not None:
            for idx in range(min(len(kpt_xy_np), len(kpt_conf_np))):
                matched_track = pose_to_track.get(idx)
                confident_points = int(np.sum(kpt_conf_np[idx] >= self._pose_gate_kpt_conf_threshold))
                if (
                    self._pose_fallback
                    and person_count <= 0
                    and confident_points >= self._pose_kpt_min_points
                ):
                    pose_fallback_indices.append(int(idx))
                if matched_track is None or matched_track not in validated_track_ids:
                    continue

        if person_count <= 0 and pose_fallback_indices:
            person_count = int(len(pose_fallback_indices))

        pair_text, pair_stats = _compute_pairwise_distances(pair_records, width)
        return {
            "depth_up": depth_up,
            "depth_raw": depth_raw,
            "color_img": color_img,
            "records": records,
            "person_count": person_count,
            "pair_text": pair_text,
            "pair_stats": pair_stats,
            "tracked_labels": tracked_labels,
            "kpt_xy_np": kpt_xy_np,
            "kpt_conf_np": kpt_conf_np,
            "pose_to_track": pose_to_track,
            "validated_track_ids": validated_track_ids,
            "pose_only": False,
            "pose_draw_indices": [],
            "pose_fallback_indices": pose_fallback_indices,
            "contour_reject_reasons": contour_reject_reasons,
            "width": width,
            "height": height,
        }

    def _render_display(self, analysis: dict, display_mode: str) -> np.ndarray:
        display = analysis["color_img"].copy()
        records = analysis["records"]
        kpt_xy_np = analysis["kpt_xy_np"]
        kpt_conf_np = analysis["kpt_conf_np"]
        pose_to_track = analysis["pose_to_track"]
        validated_track_ids = analysis["validated_track_ids"]
        pose_only = bool(analysis.get("pose_only", False))
        pose_draw_indices = analysis.get("pose_draw_indices") or []
        pose_fallback_indices = analysis.get("pose_fallback_indices") or []
        width = analysis["width"]
        height = analysis["height"]

        if display_mode in (DISPLAY_MODE_BOTH, DISPLAY_MODE_CONTOUR_ONLY):
            for record in records:
                box = record["box"]
                track_color = record["track_color"]
                contour = record["contour"]
                mask = record["mask"]
                label = record["label"]

                if contour is not None and display_mode != DISPLAY_MODE_SKELETON_ONLY:
                    anchor = record.get("anchor")
                    if anchor is not None and len(anchor) >= 2:
                        offset_x, offset_y = int(anchor[0]), int(anchor[1])
                    else:
                        offset_x, offset_y = int(round(box[0])), int(round(box[1]))
                    shifted_contour = contour + np.array([[[offset_x, offset_y]]])
                    cv2.drawContours(display, [shifted_contour], -1, track_color, 2, cv2.LINE_AA)

                mask_overlay = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                selected = mask_overlay > 0
                if np.any(selected):
                    blended = (
                        display[selected].astype(np.float32) * (1.0 - MASK_BLEND_ALPHA)
                        + np.array(track_color, dtype=np.float32) * MASK_BLEND_ALPHA
                    )
                    display[selected] = blended.astype(np.uint8)

                x1 = max(0, int(round(box[0])))
                y1 = max(18, int(round(box[1])) - 8)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.42
                thickness = 1
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                label_x = min(x1, max(0, width - text_w - 6))
                label_y = min(max(y1, text_h + 4), height - baseline - 2)
                bg_top = max(0, label_y - text_h - baseline - 4)
                bg_bottom = min(height - 1, label_y + baseline + 2)
                bg_right = min(width - 1, label_x + text_w + 5)
                cv2.rectangle(display, (label_x, bg_top), (bg_right, bg_bottom), (0, 0, 0), -1)
                cv2.putText(
                    display,
                    label,
                    (label_x + 2, label_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

        if display_mode in (DISPLAY_MODE_BOTH, DISPLAY_MODE_SKELETON_ONLY) and kpt_xy_np is not None and kpt_conf_np is not None:
            if pose_only:
                for idx in pose_draw_indices:
                    if idx < 0 or idx >= len(kpt_xy_np) or idx >= len(kpt_conf_np):
                        continue
                    color_override = _track_color(idx + 1)
                    draw_stick_figure(display, kpt_xy_np[idx], kpt_conf_np[idx], color_override=color_override)
            else:
                drawn_pose_indices: set[int] = set()
                for idx in range(min(len(kpt_xy_np), len(kpt_conf_np))):
                    matched_track = pose_to_track.get(idx)
                    if matched_track is None or matched_track not in validated_track_ids:
                        continue
                    color_override = _track_color(matched_track)
                    draw_stick_figure(display, kpt_xy_np[idx], kpt_conf_np[idx], color_override=color_override)
                    drawn_pose_indices.add(int(idx))
                for idx in pose_fallback_indices:
                    if idx < 0 or idx >= len(kpt_xy_np) or idx >= len(kpt_conf_np) or int(idx) in drawn_pose_indices:
                        continue
                    draw_stick_figure(display, kpt_xy_np[idx], kpt_conf_np[idx], color_override=_track_color(idx + 1))

        return display

    def _render_skeleton_contour(self, analysis: dict) -> np.ndarray:
        """Return pseudo color image with only contour and skeleton overlays."""
        return _render_skeleton_contour_cpu(analysis)

    def _render_gray_depth(self, analysis: dict) -> np.ndarray:
        """Return 320x320 uint8 grayscale depth image."""
        depth_up = analysis.get("depth_up")
        if depth_up is None:
            raise RuntimeError("missing depth_up")
        if not isinstance(depth_up, np.ndarray):
            raise RuntimeError("invalid depth_up")
        if depth_up.dtype != np.uint8:
            depth_up = depth_up.astype(np.uint8, copy=False)
        return depth_up

    def _render_color_depth(self, analysis: dict) -> np.ndarray:
        """Return 320x320 BGR colormap depth image."""
        color_img = analysis.get("color_img")
        if color_img is None:
            raise RuntimeError("missing color_img")
        return color_img

    def _encode_png(self, img: np.ndarray) -> bytes:
        return _encode_png_cpu(img, self._png_compression)

    def _encode_jpeg(self, img: np.ndarray) -> bytes:
        return _encode_jpeg_cpu(img, self._jpeg_quality)

    def _encode_output_image(self, img: np.ndarray) -> tuple[bytes, str]:
        if self._output_format == "jpeg":
            return self._encode_jpeg(img), "jpeg"
        return self._encode_png(img), "png"

    def _map_thread_stage(self, executor: ThreadPoolExecutor | None, worker_count: int, func, items: list):
        if worker_count <= 1 or len(items) <= 1:
            return [func(item) for item in items]

        if executor is not None:
            return list(executor.map(func, items))

        with ThreadPoolExecutor(max_workers=min(worker_count, len(items))) as temporary_executor:
            return list(temporary_executor.map(func, items))

    def _map_process_stage(self, func, items: list, worker_count: int):
        if worker_count <= 1 or len(items) <= 1:
            return [func(item) for item in items]

        pool = _get_cpu_process_pool(worker_count, self._cpu_process_start_method)
        return list(pool.map(func, items))

    def _map_decode_stage(self, func, items: list):
        if self._cpu_worker_mode == CPU_WORKER_MODE_PROCESS:
            return self._map_process_stage(func, items, self._decode_workers)
        return self._map_thread_stage(self._decode_executor, self._decode_workers, func, items)

    def _map_render_stage(self, func, items: list):
        if self._cpu_worker_mode == CPU_WORKER_MODE_PROCESS:
            return self._map_process_stage(func, items, self._render_workers)
        return self._map_thread_stage(self._render_executor, self._render_workers, func, items)

    def _interpolate_depth(self, previous_depth: np.ndarray | None, current_depth: np.ndarray) -> np.ndarray:
        current_u8 = self._ensure_uint8_gray(current_depth)
        if previous_depth is None:
            return current_u8.copy()

        previous_u8 = self._ensure_uint8_gray(previous_depth)
        if previous_u8.shape != current_u8.shape:
            previous_u8 = cv2.resize(
                previous_u8,
                (current_u8.shape[1], current_u8.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        midpoint = (
            previous_u8.astype(np.float32) * 0.5
            + current_u8.astype(np.float32) * 0.5
        )
        return np.clip(midpoint, 0, 255).astype(np.uint8)

    def _analyze_predecoded_unlocked(
        self,
        frame_id: str,
        depth: np.ndarray,
        *,
        seg_result=None,
        pose_result=None,
    ) -> dict:
        if self._stateless:
            self.reset()

        start = time.time()
        analysis = self._analyze_frame(depth, seg_result=seg_result, pose_result=pose_result)
        elapsed = int((time.time() - start) * 1000)
        return {
            "frame_id": frame_id,
            "analysis": analysis,
            "person_count": int(analysis["person_count"]),
            "processing_time_ms": elapsed,
        }

    def _reuse_analyzed_result_with_depth(self, analyzed: dict, frame_id: str, depth: np.ndarray) -> dict:
        prepared = self._prepare_depth_views(depth)
        analysis = dict(analyzed["analysis"])
        analysis["depth_up"] = prepared["depth_up"]
        analysis["depth_raw"] = prepared["depth_raw"]
        analysis["color_img"] = prepared["color_img"]
        analysis["width"] = prepared["width"]
        analysis["height"] = prepared["height"]
        return {
            "frame_id": frame_id,
            "analysis": analysis,
            "person_count": int(analyzed["person_count"]),
            "processing_time_ms": 0,
        }

    def _render_analyzed_result(self, analyzed: dict) -> dict:
        analysis = analyzed["analysis"]
        return {
            "frame_id": analyzed["frame_id"],
            "pseudo_color_image": self._render_color_depth(analysis),
            "skeleton_contour_image": self._render_skeleton_contour(analysis),
            "person_count": int(analyzed["person_count"]),
            "processing_time_ms": int(analyzed["processing_time_ms"]),
        }

    def _encode_rendered_result(self, rendered: dict) -> dict:
        person_count = int(rendered["person_count"])
        pseudo_color_image, pseudo_color_format = self._encode_output_image(rendered["pseudo_color_image"])
        if person_count <= 0:
            skeleton_contour_image = pseudo_color_image
            skeleton_contour_format = pseudo_color_format
        else:
            skeleton_contour_image, skeleton_contour_format = self._encode_output_image(rendered["skeleton_contour_image"])
        return {
            "frame_id": rendered["frame_id"],
            "pseudo_color_image": pseudo_color_image,
            "skeleton_contour_image": skeleton_contour_image,
            "pseudo_color_image_format": pseudo_color_format,
            "skeleton_contour_image_format": skeleton_contour_format,
            "person_count": person_count,
            "processing_time_ms": int(rendered["processing_time_ms"]),
        }

    def _encode_analyzed_result(self, analyzed: dict) -> dict:
        result = self._render_and_encode_analyzed_result(analyzed)
        result.pop("_render_ms", None)
        result.pop("_encode_ms", None)
        return result

    def _render_and_encode_analyzed_result(self, analyzed: dict) -> dict:
        return _render_and_encode_analyzed_result_cpu(
            (analyzed, self._output_format, self._png_compression, self._jpeg_quality)
        )

    @staticmethod
    def _split_render_encode_stage_ms(combined_ms: int, render_cpu_ms: int, encode_cpu_ms: int) -> tuple[int, int]:
        total_cpu_ms = max(0, int(render_cpu_ms)) + max(0, int(encode_cpu_ms))
        if combined_ms <= 0:
            return 0, 0
        if total_cpu_ms <= 0:
            return 0, combined_ms

        render_ms = int(round(combined_ms * max(0, int(render_cpu_ms)) / total_cpu_ms))
        render_ms = min(max(render_ms, 0), combined_ms)
        return render_ms, combined_ms - render_ms

    def _encode_analyzed_results(self, analyzed_results: list[dict], timings: dict[str, int] | None = None) -> list[dict]:
        render_encode_start = time.perf_counter()
        if self._cpu_worker_mode == CPU_WORKER_MODE_PROCESS:
            payloads = [
                (analyzed, self._output_format, self._png_compression, self._jpeg_quality)
                for analyzed in analyzed_results
            ]
            encoded_results = self._map_render_stage(_render_and_encode_analyzed_result_cpu, payloads)
        else:
            encoded_results = self._map_render_stage(self._render_and_encode_analyzed_result, analyzed_results)
        render_encode_ms = _elapsed_ms(render_encode_start)

        render_cpu_ms = 0
        encode_cpu_ms = 0
        for result in encoded_results:
            render_cpu_ms += int(result.pop("_render_ms", 0))
            encode_cpu_ms += int(result.pop("_encode_ms", 0))
        render_ms, png_encode_ms = self._split_render_encode_stage_ms(
            render_encode_ms,
            render_cpu_ms,
            encode_cpu_ms,
        )

        if timings is not None:
            timings["render_ms"] = render_ms
            timings["png_encode_ms"] = png_encode_ms

        return encoded_results

    def _decode_frames(self, frames: list[tuple[str, bytes]]) -> list[tuple[str, np.ndarray]]:
        return self._map_decode_stage(_decode_frame_cpu, frames)

    def _prepare_model_color_images(self, source_frames: list[dict]) -> list[np.ndarray]:
        depths = [item["depth"] for item in source_frames]
        if self._cpu_worker_mode == CPU_WORKER_MODE_PROCESS:
            return self._map_decode_stage(_prepare_color_image_cpu, depths)
        return [self._prepare_depth_views(depth)["color_img"] for depth in depths]

    def _infer_one_unlocked(self, frame_id: str, image_bytes: bytes) -> dict:
        analyzed = self._analyze_predecoded_unlocked(frame_id, self._decode_image(image_bytes))
        return self._encode_analyzed_result(analyzed)

    def infer(self, frame_id: str, image_bytes: bytes) -> dict:
        self._lock.acquire()
        try:
            return self._infer_one_unlocked(frame_id, image_bytes)
        finally:
            self._lock.release()

    def _log_batch_timings(self, input_count: int, output_count: int, timings: dict[str, int]) -> None:
        LOGGER.info(
            (
                "Infer batch timing: inputs=%d outputs=%d model_inputs=%d "
                "postprocess_inputs=%d instance=%s queue_wait_ms=%d decode_ms=%d model_prepare_ms=%d "
                "model_infer_ms=%d seg_model_ms=%d pose_model_ms=%d "
                "postprocess_ms=%d interpolate_ms=%d render_ms=%d png_encode_ms=%d total_ms=%d "
                "decode_workers=%d render_workers=%d cpu_worker_mode=%s cpu_process_start_method=%s "
                "png_compression=%d output_format=%s jpeg_quality=%d device=%s"
            ),
            input_count,
            output_count,
            int(timings.get("model_input_count", input_count)),
            int(timings.get("postprocess_input_count", output_count)),
            self._instance_name,
            int(timings.get("queue_wait_ms", 0)),
            int(timings.get("decode_ms", 0)),
            int(timings.get("model_prepare_ms", 0)),
            int(timings.get("model_infer_ms", 0)),
            int(timings.get("seg_model_ms", 0)),
            int(timings.get("pose_model_ms", 0)),
            int(timings.get("postprocess_ms", 0)),
            int(timings.get("interpolate_ms", 0)),
            int(timings.get("render_ms", 0)),
            int(timings.get("png_encode_ms", 0)),
            int(timings.get("total_ms", 0)),
            self._decode_workers,
            self._render_workers,
            self._cpu_worker_mode,
            self._cpu_process_start_method,
            self._png_compression,
            self._output_format,
            self._jpeg_quality,
            self._device or "auto",
        )

    def _log_model_confidences(
        self,
        input_count: int,
        current_analyzed_by_input: dict[int, dict],
        *,
        contour_results=None,
        pose_results=None,
    ) -> None:
        contour_counts: list[int] = []
        contour_conf_max: list[float | None] = []
        contour_conf_values: list[float] = []
        contour_raw_hints: list[str] = []
        if contour_results is not None:
            for result in contour_results:
                conf_values = _result_box_conf_values(result)
                contour_count = _result_box_count(result)
                contour_counts.append(contour_count)
                contour_conf_values.extend(conf_values)
                contour_conf_max.append(_rounded_or_none(max(conf_values), 3) if conf_values else None)
                if contour_count <= 0:
                    contour_raw_hints.append("no_candidate")
                    continue
                masks = getattr(result, "masks", None)
                if masks is None:
                    contour_raw_hints.append("no_mask")
                    continue
                masks_data = getattr(masks, "data", None)
                if masks_data is None:
                    contour_raw_hints.append("no_mask_data")
                    continue
                try:
                    mask_count = int(len(masks_data))
                except Exception:
                    mask_count = 0
                if mask_count <= 0:
                    contour_raw_hints.append("mask_count_zero")
                elif mask_count < contour_count:
                    contour_raw_hints.append("mask_count_mismatch")
                else:
                    contour_raw_hints.append("")

        pose_counts: list[int] = []
        pose_conf_max: list[float | None] = []
        pose_conf_values: list[float] = []
        pose_kpt_gate_points_max: list[int] = []
        if pose_results is not None:
            for result in pose_results:
                conf_values = _result_box_conf_values(result)
                pose_counts.append(_result_box_count(result))
                pose_conf_values.extend(conf_values)
                pose_conf_max.append(_rounded_or_none(max(conf_values), 3) if conf_values else None)
                pose_kpt_gate_points_max.append(
                    _result_kpt_gate_points_max(result, self._pose_gate_kpt_conf_threshold)
                )

        final_person_counts: list[int] = []
        post_contour_counts: list[int] = []
        post_contour_reject_reasons: list[str] = []
        pose_fallback_counts: list[int] = []
        for input_index in range(input_count):
            analyzed = current_analyzed_by_input.get(input_index) or {}
            post_contour_count = len(analyzed.get("records") or [])
            reject_reasons = [str(reason) for reason in (analyzed.get("contour_reject_reasons") or [])]
            raw_hint = contour_raw_hints[input_index] if input_index < len(contour_raw_hints) else ""
            post_contour_counts.append(post_contour_count)
            if post_contour_count > 0 and reject_reasons:
                post_contour_reject_reasons.append("kept+" + "|".join(reject_reasons))
            elif post_contour_count > 0:
                post_contour_reject_reasons.append("kept")
            elif reject_reasons:
                post_contour_reject_reasons.append("|".join(reject_reasons))
            elif raw_hint:
                post_contour_reject_reasons.append(raw_hint)
            elif contour_results is not None and input_index < len(contour_counts) and contour_counts[input_index] > 0:
                post_contour_reject_reasons.append("raw_candidate_unhandled")
            else:
                post_contour_reject_reasons.append("no_candidate")
            final_person_counts.append(int(analyzed.get("person_count", 0)))
            pose_fallback_counts.append(len(analyzed.get("pose_fallback_indices") or []))

        LOGGER.info(
            (
                "Infer model confidence: instance=%s inputs=%d "
                "contour_threshold=%.3f contour_counts=%s contour_conf_max=%s "
                "contour_conf_avg=%.3f contour_conf_peak=%.3f "
                "pose_threshold=%.3f pose_counts=%s pose_conf_max=%s "
                "pose_conf_avg=%.3f pose_conf_peak=%.3f "
                "pose_gate_kpt_threshold=%.3f pose_kpt_gate_points_max=%s "
                "post_contour_counts=%s post_contour_reject_reasons=%s "
                "final_person_counts=%s pose_fallback_counts=%s"
            ),
            self._instance_name,
            input_count,
            self._seg_conf_threshold,
            _format_number_list(contour_counts),
            _format_number_list(contour_conf_max),
            _mean_or_zero(contour_conf_values),
            max(contour_conf_values) if contour_conf_values else 0.0,
            self._pose_conf_threshold,
            _format_number_list(pose_counts),
            _format_number_list(pose_conf_max),
            _mean_or_zero(pose_conf_values),
            max(pose_conf_values) if pose_conf_values else 0.0,
            self._pose_gate_kpt_conf_threshold,
            _format_number_list(pose_kpt_gate_points_max),
            _format_number_list(post_contour_counts),
            _format_text_list(post_contour_reject_reasons),
            _format_number_list(final_person_counts),
            _format_number_list(pose_fallback_counts),
        )

    def infer_batch(self, frames: list[tuple[str, bytes]]) -> list[dict]:
        lock_start = time.perf_counter()
        self._lock.acquire()
        try:
            if not frames:
                return []

            timings: dict[str, int] = {
                "queue_wait_ms": _elapsed_ms(lock_start),
            }
            total_start = time.perf_counter()

            decode_start = time.perf_counter()
            decoded_frames = self._decode_frames(frames)
            timings["decode_ms"] = _elapsed_ms(decode_start)

            source_frames: list[dict] = []
            for input_index, (frame_id, current_depth) in enumerate(decoded_frames):
                source_frames.append(
                    {
                        "frame_id": frame_id,
                        "input_index": input_index,
                        "depth": current_depth,
                    }
                )

            timings["model_input_count"] = len(source_frames)
            timings["postprocess_input_count"] = len(source_frames)

            def build_output_results(current_analyzed_by_input: dict[int, dict]) -> tuple[list[dict], list[dict]]:
                interpolate_start = time.perf_counter()
                output_frames: list[dict] = []
                analyzed_results: list[dict] = []
                previous_source = None if self._stateless else self._last_source_depth
                for input_index, (frame_id, current_depth) in enumerate(decoded_frames):
                    current_analyzed = current_analyzed_by_input[input_index]
                    interpolated_depth = self._interpolate_depth(previous_source, current_depth)
                    interpolated_frame = {
                        "frame_id": f"{frame_id}_interpolated",
                        "source_frame_id": frame_id,
                        "input_index": input_index,
                        "result_kind": "interpolated",
                    }
                    current_frame = {
                        "frame_id": f"{frame_id}_current",
                        "source_frame_id": frame_id,
                        "input_index": input_index,
                        "result_kind": "current",
                    }
                    output_frames.append(interpolated_frame)
                    analyzed_results.append(
                        self._reuse_analyzed_result_with_depth(
                            current_analyzed,
                            interpolated_frame["frame_id"],
                            interpolated_depth,
                        )
                    )
                    output_frames.append(current_frame)
                    analyzed_results.append(current_analyzed)
                    previous_source = current_depth.copy()

                self._last_source_depth = previous_source.copy() if previous_source is not None and not self._stateless else None
                timings["interpolate_ms"] = _elapsed_ms(interpolate_start)
                return output_frames, analyzed_results

            model_prepare_start = time.perf_counter()
            color_imgs = self._prepare_model_color_images(source_frames)
            timings["model_prepare_ms"] = _elapsed_ms(model_prepare_start)

            if self._pose_only:
                yolo_start = time.perf_counter()
                pose_results = self.pose_model.predict(
                    color_imgs,
                    conf=self._pose_conf_threshold,
                    classes=[0],
                    imgsz=POSE_INFER_IMGSZ,
                    device=self._device,
                    verbose=False,
                )
                timings["pose_model_ms"] = _elapsed_ms(yolo_start)
                timings["seg_model_ms"] = 0
                timings["model_infer_ms"] = int(timings["pose_model_ms"])
                if len(pose_results) != len(source_frames):
                    raise RuntimeError(f"pose batch returned {len(pose_results)} results for {len(source_frames)} source frames")

                postprocess_start = time.perf_counter()
                current_analyzed_by_input: dict[int, dict] = {}
                for item in source_frames:
                    input_index = int(item["input_index"])
                    current_analyzed_by_input[input_index] = self._analyze_predecoded_unlocked(
                        f"{item['frame_id']}_current",
                        item["depth"],
                        pose_result=pose_results[input_index],
                    )
                timings["postprocess_ms"] = _elapsed_ms(postprocess_start)
                self._log_model_confidences(
                    len(source_frames),
                    current_analyzed_by_input,
                    pose_results=pose_results,
                )
                output_frames, analyzed_results = build_output_results(current_analyzed_by_input)

                results = self._encode_analyzed_results(analyzed_results, timings)
                for output_index, (result, item) in enumerate(zip(results, output_frames)):
                    result["source_frame_id"] = item["source_frame_id"]
                    result["input_index"] = int(item["input_index"])
                    result["output_index"] = int(output_index)
                    result["result_kind"] = item["result_kind"]
                timings["total_ms"] = _elapsed_ms(total_start)
                self._log_batch_timings(len(frames), len(output_frames), timings)
                return results

            yolo_start = time.perf_counter()
            seg_yolo_start = time.perf_counter()
            seg_results = self.seg_model.track(
                color_imgs,
                conf=self._seg_conf_threshold,
                persist=self._persist_tracks,
                tracker=TRACKER_CONFIG,
                classes=[0],
                imgsz=SEG_INFER_IMGSZ,
                device=self._device,
                verbose=False,
            )
            timings["seg_model_ms"] = _elapsed_ms(seg_yolo_start)
            if len(seg_results) != len(source_frames):
                raise RuntimeError(f"seg batch returned {len(seg_results)} results for {len(source_frames)} source frames")
            pose_yolo_start = time.perf_counter()
            pose_results = self.pose_model.predict(
                color_imgs,
                conf=self._pose_conf_threshold,
                classes=[0],
                imgsz=POSE_INFER_IMGSZ,
                device=self._device,
                verbose=False,
            )
            timings["pose_model_ms"] = _elapsed_ms(pose_yolo_start)
            timings["model_infer_ms"] = _elapsed_ms(yolo_start)
            if len(pose_results) != len(source_frames):
                raise RuntimeError(f"pose batch returned {len(pose_results)} results for {len(source_frames)} source frames")

            postprocess_start = time.perf_counter()
            current_analyzed_by_input: dict[int, dict] = {}
            for item in source_frames:
                input_index = int(item["input_index"])
                current_analyzed_by_input[input_index] = self._analyze_predecoded_unlocked(
                    f"{item['frame_id']}_current",
                    item["depth"],
                    seg_result=seg_results[input_index],
                    pose_result=pose_results[input_index],
                )
            timings["postprocess_ms"] = _elapsed_ms(postprocess_start)
            self._log_model_confidences(
                len(source_frames),
                current_analyzed_by_input,
                contour_results=seg_results,
                pose_results=pose_results,
            )
            output_frames, analyzed_results = build_output_results(current_analyzed_by_input)

            results = self._encode_analyzed_results(analyzed_results, timings)
            for output_index, (result, item) in enumerate(zip(results, output_frames)):
                result["source_frame_id"] = item["source_frame_id"]
                result["input_index"] = int(item["input_index"])
                result["output_index"] = int(output_index)
                result["result_kind"] = item["result_kind"]
            timings["total_ms"] = _elapsed_ms(total_start)
            self._log_batch_timings(len(frames), len(output_frames), timings)
            return results
        finally:
            self._lock.release()
