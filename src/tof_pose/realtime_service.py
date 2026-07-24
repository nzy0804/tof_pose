from __future__ import annotations

import atexit
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
import logging
import multiprocessing
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import ultralytics as _ultralytics

MODEL_CLS = getattr(_ultralytics, ''.join(chr(code) for code in (89, 79, 76, 79)))

from tof_pose.paths import DEFAULT_MODEL_PATH, DEFAULT_POSE_MODEL_PATH
from tof_pose.input_image import (
    decode_received_grayscale_views,
)
from tof_pose.person_distance import estimate_person_distance_from_mask, extract_draw_contour_from_mask
from tof_pose.pose_drawing import draw_stick_figure
from tof_pose.tracking import Detection, PersonTracker


LOGGER = logging.getLogger(__name__)

CONF_THRESHOLD = 0.2
TRACKER_CONFIG = "botsort.yaml"
SEG_INFER_IMGSZ = 320
POSE_INFER_IMGSZ = 320
MODEL_INPUT_SIZE_320 = 320
MODEL_INPUT_SIZE_160 = 160
MODEL_INPUT_SIZES = (MODEL_INPUT_SIZE_320, MODEL_INPUT_SIZE_160)
POSE_INFER_INTERVAL = 1
DISPLAY_SIZE = (320, 320)
SKELETON_CONTOUR_OUTPUT_SIZE = (100, 100)
DISPLAY_SCALE = 3

MEDIAN_BLUR_K = 5
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

INPUT_MODALITY_DEPTH = "depth"
INPUT_MODALITY_IR = "ir"
INPUT_MODALITIES = (INPUT_MODALITY_DEPTH, INPUT_MODALITY_IR)

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
CONTOUR_EXISTING_MAX_CENTER_JUMP_PX = 100.0
CONTOUR_EXISTING_MAX_AREA_CHANGE_RATIO = 2.5
MASK_MIN_AREA_RATIO = 0.003
MASK_MAX_AREA_RATIO = 0.9
MASK_AREA_JUMP_RATIO = 1.5
MASK_JUMP_HOLD_FRAMES = 2
DEPTH_PERSON_DISTANCE_CLOSE_THRESHOLD = 100.0
PERSON_DISTANCE_CLOSE_GAP_RATIO = 0.06
PERSON_DISTANCE_CLOSE_CENTER_RATIO = 0.30
QUALITATIVE_KPT_CONF_THRESHOLD = 0.35
POSE_STATUS_MIN_TORSO_Y_PX = 20.0
POSE_STATUS_MIN_THIGH_Y_PX = 12.0
POSE_STATUS_THIGH_TORSO_RATIO_THRESHOLD = 0.65
POSE_BATCH_REUSE_MIN_VALID_RATIO = 0.45
POSE_BATCH_REUSE_MAX_GAP = 1
ACTION_KEYPOINT_SHIFT_HIGH_RATIO = 0.025
ACTION_KEYPOINT_SHIFT_PEAK_HIGH_RATIO = 0.045
ACTION_BOX_CENTER_SHIFT_HIGH_RATIO = 0.025
ACTION_BOX_AREA_SHIFT_HIGH_RATIO = 0.025
ACTION_MOTION_WINDOW_FRAMES = 8
ACTION_MOTION_WINDOW_HIGH_RATIO = 0.25
ACTION_HIGH_CONFIRM_FRAMES = 2
QUALITATIVE_STABILITY_WINDOW_FRAMES = 8
QUALITATIVE_STABILITY_MIN_CONFIRM_FRAMES = 3
QUALITATIVE_STABILITY_SWITCH_RATIO = 0.60
POSE_FALLBACK_KPT_CONF_THRESHOLD = 0.45
POSE_FALLBACK_MIN_POINTS = 7
PERSON_DUPLICATE_KPT_CLOSE_COUNT = 4
PERSON_DUPLICATE_KPT_CLOSE_RATIO = 0.03
PERSON_DUPLICATE_KPT_CLOSE_MIN_PX = 3.0
PERSON_DUPLICATE_MASK_CONTAIN_RATIO = 0.85
PERSON_DUPLICATE_MASK_MIN_AREA_RATIO = 0.15
PERSON_DUPLICATE_MASK_CENTER_RATIO = 0.25
PERSON_DUPLICATE_MASK_CENTER_MIN_PX = 8.0
QUALITATIVE_RESULT_KEYS = ("person_status", "person_distance", "action_level")
PERSON_FILL_BACKGROUND_DEFAULT = "bg_08_dark_frost_reference.png"
PERSON_FILL_BACKGROUND_NAMES = (
    "bg_01_dense_white_fog.png",
    "bg_02_soft_frosted_gray.png",
    "bg_03_milky_glass.png",
    "bg_04_fogged_concrete.png",
    "bg_05_silver_mist.png",
    "bg_06_low_contrast_frost.png",
    "bg_07_white_smoke_patch.png",
    "bg_08_dark_frost_reference.png",
)
PERSON_FILL_FALLBACK_GRAY = 232
PERSON_FILL_BACKGROUND_BLEND = 0.5
DISPLAY_CONTOUR_HEAD_TOP_RATIO = 0.20
DISPLAY_CONTOUR_HEAD_MAX_SCALE = 1.10
DISPLAY_CONTOUR_HEAD_MIN_HEIGHT = 12

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
_PERSON_FILL_BACKGROUND_CACHE: dict[str, np.ndarray] = {}
_PERSON_FILL_BACKGROUND_GRAY_CACHE: dict[tuple[str, int, int], np.ndarray] = {}


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


def _analysis_payload(analyzed: dict) -> dict:
    analysis = analyzed.get("analysis") if isinstance(analyzed, dict) else None
    if isinstance(analysis, dict):
        return analysis
    return analyzed if isinstance(analyzed, dict) else {}


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


def _decode_image_views_cpu(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    received, decrypted = decode_received_grayscale_views(data)
    return decrypted, received


def _decode_frame_cpu(
    frame: tuple[str, bytes],
) -> tuple[str, np.ndarray, np.ndarray]:
    frame_id, image_bytes = frame
    decrypted, received = _decode_image_views_cpu(image_bytes)
    return frame_id, decrypted, received


def _normalize_input_modality(input_modality: str | None) -> str:
    normalized = str(input_modality or INPUT_MODALITY_DEPTH).strip().lower()
    if normalized not in INPUT_MODALITIES:
        raise ValueError(f"input_modality must be one of: {', '.join(INPUT_MODALITIES)}")
    return normalized


def _apply_depth_display_colormap(depth_u8: np.ndarray) -> np.ndarray:
    valid_mask = depth_u8 > 0
    inverted = (255 - depth_u8).astype(np.uint8, copy=False)
    if np.any(~valid_mask):
        inverted = inverted.copy()
        inverted[~valid_mask] = 0
    return cv2.applyColorMap(inverted, cv2.COLORMAP_VIRIDIS)


def _apply_display_colormap(depth_u8: np.ndarray, input_modality: str) -> np.ndarray:
    if input_modality == INPUT_MODALITY_DEPTH:
        return _apply_depth_display_colormap(depth_u8)
    return _gray_to_bgr(depth_u8)


def _gray_to_bgr(gray_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)


def normalize_model_input_size(value: int | str | None) -> int:
    if value is None:
        return MODEL_INPUT_SIZE_320
    try:
        size = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"model_input_size must be one of {MODEL_INPUT_SIZES}") from exc
    if size not in MODEL_INPUT_SIZES:
        raise ValueError(f"model_input_size must be one of {MODEL_INPUT_SIZES}")
    return size


def _scale_boxes_to_display(boxes: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    scaled = np.asarray(boxes, dtype=np.float32).copy()
    if scaled.size:
        scaled[..., [0, 2]] *= float(scale_x)
        scaled[..., [1, 3]] *= float(scale_y)
    return scaled


def _scale_keypoints_to_display(keypoints: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    scaled = np.asarray(keypoints, dtype=np.float32).copy()
    if scaled.size:
        scaled[..., 0] *= float(scale_x)
        scaled[..., 1] *= float(scale_y)
    return scaled


def _scale_contour_to_size(contour: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    if abs(float(scale_x) - 1.0) < 1e-6 and abs(float(scale_y) - 1.0) < 1e-6:
        return np.asarray(contour, dtype=np.int32)
    scaled = np.asarray(contour, dtype=np.float32).copy()
    scaled[..., 0] *= float(scale_x)
    scaled[..., 1] *= float(scale_y)
    return np.rint(scaled).astype(np.int32)


def _prepare_depth_views_cpu(
    depth_gray: np.ndarray,
    input_modality: str = INPUT_MODALITY_DEPTH,
    *,
    received_gray: np.ndarray | None = None,
    ir_preprocess: bool = False,
    model_input_size: int = MODEL_INPUT_SIZE_320,
) -> dict:
    model_input_size = normalize_model_input_size(model_input_size)
    width, height = DISPLAY_SIZE
    modality = _normalize_input_modality(input_modality)
    depth_u8 = _ensure_uint8_gray_cpu(depth_gray)
    input_source = depth_u8
    received_source = (
        input_source
        if received_gray is None
        else _ensure_uint8_gray_cpu(received_gray)
    )
    render_source_img = _gray_to_bgr(received_source)
    depth_raw = cv2.resize(input_source, DISPLAY_SIZE, interpolation=cv2.INTER_LINEAR)

    enhanced = input_source
    if modality == INPUT_MODALITY_IR and ir_preprocess:
        if MEDIAN_BLUR_K and MEDIAN_BLUR_K >= 3:
            enhanced = cv2.medianBlur(enhanced, MEDIAN_BLUR_K)
        enhanced = _get_worker_clahe().apply(enhanced)

    depth_up = cv2.resize(enhanced, DISPLAY_SIZE, interpolation=cv2.INTER_LINEAR)
    model_depth = cv2.resize(enhanced, (model_input_size, model_input_size), interpolation=cv2.INTER_LINEAR)
    if modality == INPUT_MODALITY_IR:
        color_img = _gray_to_bgr(depth_up)
        model_color_img = _gray_to_bgr(model_depth)
    else:
        model_color_img = cv2.applyColorMap(model_depth, cv2.COLORMAP_MAGMA)
        color_img = _apply_display_colormap(depth_up, modality)
    return {
        "depth_up": depth_up,
        "depth_raw": depth_raw,
        "color_img": color_img,
        "render_source_img": render_source_img,
        "model_color_img": model_color_img,
        "model_input_size": model_input_size,
        "model_width": int(model_input_size),
        "model_height": int(model_input_size),
        "model_to_display_scale_x": float(width) / max(float(model_input_size), 1.0),
        "model_to_display_scale_y": float(height) / max(float(model_input_size), 1.0),
        "width": width,
        "height": height,
    }


def _prepare_color_image_cpu(payload) -> np.ndarray:
    return _prepare_depth_views_payload_cpu(payload)["model_color_img"]


def _prepare_depth_views_payload_cpu(payload) -> dict:
    if isinstance(payload, tuple):
        if len(payload) >= 5:
            depth_gray, input_modality, ir_preprocess, model_input_size, received_gray = payload[:5]
        elif len(payload) >= 4:
            depth_gray, input_modality, ir_preprocess, model_input_size = payload[:4]
            received_gray = None
        elif len(payload) >= 3:
            depth_gray, input_modality, ir_preprocess = payload[:3]
            model_input_size = MODEL_INPUT_SIZE_320
            received_gray = None
        else:
            depth_gray, input_modality = payload
            ir_preprocess = False
            model_input_size = MODEL_INPUT_SIZE_320
            received_gray = None
    else:
        depth_gray = payload
        input_modality = INPUT_MODALITY_DEPTH
        ir_preprocess = False
        model_input_size = MODEL_INPUT_SIZE_320
        received_gray = None
    return _prepare_depth_views_cpu(
        depth_gray,
        input_modality=input_modality,
        received_gray=received_gray,
        ir_preprocess=bool(ir_preprocess),
        model_input_size=model_input_size,
    )


def _estimate_distance_candidate_cpu(payload: dict) -> dict:
    estimate = estimate_person_distance_from_mask(
        payload["depth_raw"],
        payload["box"],
        payload["mask"],
        mask_is_binary=True,
        include_draw_contour=False,
    )
    contour_area = 0.0
    if estimate.contour is not None:
        contour_area = float(cv2.contourArea(estimate.contour))
    return {
        "task_id": int(payload["task_id"]),
        "estimate": estimate,
        "contour_area": contour_area,
    }


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


def _empty_keypoint_arrays(
    kpt_xy_np: np.ndarray | None,
    kpt_conf_np: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if kpt_xy_np is not None and kpt_conf_np is not None and len(kpt_xy_np) > 0 and len(kpt_conf_np) > 0:
        return np.zeros_like(kpt_xy_np[0], dtype=np.float32), np.zeros_like(kpt_conf_np[0], dtype=np.float32)
    return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)


def _person_duplicate_kpt_threshold(width: int, height: int) -> float:
    base = max(1.0, float(min(int(width), int(height))))
    return max(PERSON_DUPLICATE_KPT_CLOSE_MIN_PX, base * PERSON_DUPLICATE_KPT_CLOSE_RATIO)


def _same_keypoint_close_count(
    kpt_xy_a: np.ndarray | None,
    kpt_conf_a: np.ndarray | None,
    kpt_xy_b: np.ndarray | None,
    kpt_conf_b: np.ndarray | None,
    *,
    threshold_px: float,
    conf_threshold: float,
    required_count: int = PERSON_DUPLICATE_KPT_CLOSE_COUNT,
) -> int:
    if kpt_xy_a is None or kpt_conf_a is None or kpt_xy_b is None or kpt_conf_b is None:
        return 0

    xy_a = np.asarray(kpt_xy_a, dtype=np.float32)
    xy_b = np.asarray(kpt_xy_b, dtype=np.float32)
    conf_a = np.asarray(kpt_conf_a, dtype=np.float32)
    conf_b = np.asarray(kpt_conf_b, dtype=np.float32)
    usable = min(len(xy_a), len(xy_b), len(conf_a), len(conf_b))
    if usable <= 0:
        return 0

    close_count = 0
    for idx in range(usable):
        if float(conf_a[idx]) < conf_threshold or float(conf_b[idx]) < conf_threshold:
            continue
        point_a = xy_a[idx]
        point_b = xy_b[idx]
        if point_a.shape[0] < 2 or point_b.shape[0] < 2:
            continue
        if not (np.isfinite(point_a[:2]).all() and np.isfinite(point_b[:2]).all()):
            continue
        if float(np.linalg.norm(point_a[:2] - point_b[:2])) <= threshold_px:
            close_count += 1
            if close_count >= required_count:
                return close_count
    return close_count


def _record_mask_area(record: dict) -> int:
    mask = record.get("mask")
    if isinstance(mask, np.ndarray):
        return int(np.count_nonzero(mask))
    return 0


def _prefer_duplicate_record(candidate: dict, incumbent: dict) -> bool:
    candidate_area = _record_mask_area(candidate)
    incumbent_area = _record_mask_area(incumbent)
    if candidate_area != incumbent_area:
        return candidate_area > incumbent_area
    return float(candidate.get("person_conf") or 0.0) > float(incumbent.get("person_conf") or 0.0)


def _mask_containment_duplicate_stats(record_a: dict, record_b: dict) -> tuple[bool, float, float]:
    mask_a = record_a.get("mask")
    mask_b = record_b.get("mask")
    if not isinstance(mask_a, np.ndarray) or not isinstance(mask_b, np.ndarray):
        return False, 0.0, 0.0
    if mask_a.shape[:2] != mask_b.shape[:2]:
        return False, 0.0, 0.0

    area_a = _record_mask_area(record_a)
    area_b = _record_mask_area(record_b)
    if area_a <= 0 or area_b <= 0:
        return False, 0.0, 0.0

    if area_a >= area_b:
        big_record, small_record = record_a, record_b
        big_area, small_area = area_a, area_b
    else:
        big_record, small_record = record_b, record_a
        big_area, small_area = area_b, area_a

    area_ratio = float(small_area) / max(float(big_area), 1.0)
    if area_ratio < PERSON_DUPLICATE_MASK_MIN_AREA_RATIO:
        return False, 0.0, area_ratio

    big_mask = np.asarray(big_record["mask"]) > 0
    small_mask = np.asarray(small_record["mask"]) > 0
    contained_ratio = float(np.count_nonzero(big_mask & small_mask)) / max(float(small_area), 1.0)
    if contained_ratio < PERSON_DUPLICATE_MASK_CONTAIN_RATIO:
        return False, contained_ratio, area_ratio

    big_box = np.asarray(big_record.get("box"), dtype=np.float32)
    small_box = np.asarray(small_record.get("box"), dtype=np.float32)
    if big_box.size < 4 or small_box.size < 4:
        return False, contained_ratio, area_ratio
    center_limit = max(
        PERSON_DUPLICATE_MASK_CENTER_MIN_PX,
        _box_extent(big_box) * PERSON_DUPLICATE_MASK_CENTER_RATIO,
    )
    if _box_center_distance(big_box, small_box) > center_limit:
        return False, contained_ratio, area_ratio

    return True, contained_ratio, area_ratio


def _dedupe_person_records(
    records: list[dict],
    *,
    width: int,
    height: int,
    conf_threshold: float,
) -> tuple[list[dict], list[str], list[str]]:
    if len(records) <= 1:
        return records, [], []

    threshold_px = _person_duplicate_kpt_threshold(width, height)
    kept: list[dict] = []
    reject_reasons: list[str] = []
    debug_summary: list[str] = []

    for record in records:
        duplicate_index: int | None = None
        duplicate_close_count = 0
        duplicate_reason = ""
        duplicate_contained_ratio = 0.0
        duplicate_area_ratio = 0.0
        for kept_index, kept_record in enumerate(kept):
            close_count = _same_keypoint_close_count(
                kept_record.get("keypoints"),
                kept_record.get("kpt_conf"),
                record.get("keypoints"),
                record.get("kpt_conf"),
                threshold_px=threshold_px,
                conf_threshold=conf_threshold,
            )
            if close_count >= PERSON_DUPLICATE_KPT_CLOSE_COUNT:
                duplicate_index = kept_index
                duplicate_close_count = close_count
                duplicate_reason = "duplicate_kpt"
                break

            contained, contained_ratio, area_ratio = _mask_containment_duplicate_stats(kept_record, record)
            if contained:
                duplicate_index = kept_index
                duplicate_reason = "duplicate_mask_contained"
                duplicate_contained_ratio = contained_ratio
                duplicate_area_ratio = area_ratio
                break

        if duplicate_index is None:
            kept.append(record)
            continue

        reject_reasons.append(duplicate_reason)
        incumbent = kept[duplicate_index]
        kept_track_id = int(incumbent.get("track_id", -1))
        dropped_track_id = int(record.get("track_id", -1))
        if _prefer_duplicate_record(record, incumbent):
            kept[duplicate_index] = record
            kept_track_id, dropped_track_id = dropped_track_id, kept_track_id
        if len(debug_summary) < 8:
            if duplicate_reason == "duplicate_kpt":
                debug_summary.append(
                    f"duplicate_kpt:keep{kept_track_id}:drop{dropped_track_id}:n{duplicate_close_count}:thr{threshold_px:.1f}"
                )
            else:
                debug_summary.append(
                    f"duplicate_mask:keep{kept_track_id}:drop{dropped_track_id}:contain{duplicate_contained_ratio:.2f}:area{duplicate_area_ratio:.2f}"
                )

    return kept, reject_reasons, debug_summary


def _rebuild_person_record_summaries(records: list[dict]) -> tuple[list[str], list[tuple[int, np.ndarray, float | None]]]:
    tracked_labels: list[str] = []
    pair_records: list[tuple[int, np.ndarray, float | None]] = []
    for record in records:
        track_id = int(record["track_id"])
        box = np.asarray(record["box"], dtype=np.float32).copy()
        distance = record.get("distance")
        if distance is None:
            tracked_labels.append(f"{track_id}:N/A")
        else:
            tracked_labels.append(f"{track_id}:{float(distance):.1f}")
        pair_records.append((track_id, box, distance))
    return tracked_labels, pair_records


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


def _mean_keypoint_y(
    kpt_xy: np.ndarray,
    kpt_conf: np.ndarray,
    indices: tuple[int, ...],
    threshold: float,
) -> float | None:
    values: list[float] = []
    for idx in indices:
        if idx >= len(kpt_xy) or idx >= len(kpt_conf):
            continue
        if float(kpt_conf[idx]) < threshold:
            continue
        values.append(float(kpt_xy[idx][1]))
    if not values:
        return None
    return float(np.mean(values))


def _normalize_pose_status_thigh_torso_ratio(value: float | None) -> float:
    if value is None:
        return POSE_STATUS_THIGH_TORSO_RATIO_THRESHOLD
    ratio = float(value)
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("pose_status_thigh_torso_ratio_threshold must be > 0")
    return ratio


def _classify_person_pose_status(
    kpt_xy: np.ndarray | None,
    kpt_conf: np.ndarray | None,
    threshold: float = QUALITATIVE_KPT_CONF_THRESHOLD,
    thigh_torso_ratio_threshold: float = POSE_STATUS_THIGH_TORSO_RATIO_THRESHOLD,
) -> str | None:
    if kpt_xy is None or kpt_conf is None or len(kpt_xy) <= 0 or len(kpt_conf) <= 0:
        return None

    shoulder_y = _mean_keypoint_y(kpt_xy, kpt_conf, (5, 6), threshold)
    hip_y = _mean_keypoint_y(kpt_xy, kpt_conf, (11, 12), threshold)
    knee_y = _mean_keypoint_y(kpt_xy, kpt_conf, (13, 14), threshold)

    if shoulder_y is not None and hip_y is not None and knee_y is not None:
        torso = abs(hip_y - shoulder_y)
        thigh = abs(knee_y - hip_y)
        if torso < POSE_STATUS_MIN_TORSO_Y_PX:
            return None
        thigh_torso_ratio = thigh / max(torso, 1.0)
        if thigh < POSE_STATUS_MIN_THIGH_Y_PX:
            return "坐"
        if thigh_torso_ratio <= thigh_torso_ratio_threshold:
            return "坐"
        return "站"

    return None


def _count_confident_keypoints(kpt_conf: np.ndarray, indices: tuple[int, ...], threshold: float) -> int:
    count = 0
    for idx in indices:
        if idx < len(kpt_conf) and float(kpt_conf[idx]) >= threshold:
            count += 1
    return count


def _pose_fallback_is_human_candidate(
    kpt_xy: np.ndarray | None,
    kpt_conf: np.ndarray | None,
    *,
    threshold: float,
    min_points: int,
    width: int,
    height: int,
) -> bool:
    if kpt_xy is None or kpt_conf is None or len(kpt_xy) <= 0 or len(kpt_conf) <= 0:
        return False

    effective_min_points = max(int(min_points), int(POSE_FALLBACK_MIN_POINTS))
    valid = np.asarray(kpt_conf >= threshold, dtype=bool)
    if int(np.sum(valid)) < effective_min_points:
        return False

    shoulder_count = _count_confident_keypoints(kpt_conf, (5, 6), threshold)
    hip_count = _count_confident_keypoints(kpt_conf, (11, 12), threshold)
    lower_count = _count_confident_keypoints(kpt_conf, (13, 14, 15, 16), threshold)
    if shoulder_count <= 0 or hip_count <= 0 or lower_count <= 0:
        return False

    shoulder_y = _mean_keypoint_y(kpt_xy, kpt_conf, (5, 6), threshold)
    hip_y = _mean_keypoint_y(kpt_xy, kpt_conf, (11, 12), threshold)
    if shoulder_y is None or hip_y is None or (hip_y - shoulder_y) < 12.0:
        return False

    points = np.asarray(kpt_xy, dtype=np.float32)[valid]
    if points.size <= 0:
        return False
    x1, y1 = np.min(points, axis=0)
    x2, y2 = np.max(points, axis=0)
    box_w = max(1.0, float(x2 - x1))
    box_h = max(1.0, float(y2 - y1))
    frame_area = max(1.0, float(width) * float(height))
    if box_h < max(35.0, float(height) * 0.10):
        return False
    if (box_w * box_h) / frame_area < 0.004:
        return False
    if box_h < box_w * 0.45:
        return False
    return True


def _select_pose_indices_for_status(analysis: dict) -> list[int]:
    kpt_conf_np = analysis.get("kpt_conf_np")
    if kpt_conf_np is None:
        return []

    selected: list[int] = []

    def add_index(value: object) -> None:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            return
        if idx < 0 or idx >= len(kpt_conf_np) or idx in selected:
            return
        selected.append(idx)

    if bool(analysis.get("pose_only", False)):
        for idx in analysis.get("pose_draw_indices") or []:
            add_index(idx)
    else:
        validated_track_ids = {int(tid) for tid in (analysis.get("validated_track_ids") or set())}
        pose_to_track = analysis.get("pose_to_track") or {}
        for idx in sorted(pose_to_track):
            if int(pose_to_track[idx]) in validated_track_ids:
                add_index(idx)
        for idx in analysis.get("pose_fallback_indices") or []:
            add_index(idx)

    if not selected:
        for idx in range(len(kpt_conf_np)):
            confident_points = int(np.sum(kpt_conf_np[idx] >= QUALITATIVE_KPT_CONF_THRESHOLD))
            if confident_points >= 4:
                add_index(idx)
            if len(selected) >= 2:
                break

    return selected[:2]


def _format_person_status(statuses: list[str], person_count: int) -> str:
    if person_count <= 0:
        return ""
    normalized = [status for status in statuses[:2] if status in {"坐", "站"}]
    if not normalized:
        return ""
    has_sit = "坐" in normalized
    has_stand = "站" in normalized
    if has_sit and has_stand:
        return "1坐1站"
    if has_sit:
        return "1坐"
    return "1站"


def _compute_person_status(analysis: dict) -> str:
    person_count = int(analysis.get("person_count", 0) or 0)
    kpt_xy_np = analysis.get("kpt_xy_np")
    kpt_conf_np = analysis.get("kpt_conf_np")
    thigh_torso_ratio_threshold = _normalize_pose_status_thigh_torso_ratio(
        analysis.get("pose_status_thigh_torso_ratio_threshold")
    )
    statuses: list[str] = []
    if kpt_xy_np is not None and kpt_conf_np is not None:
        for idx in _select_pose_indices_for_status(analysis):
            if idx >= len(kpt_xy_np) or idx >= len(kpt_conf_np):
                continue
            status = _classify_person_pose_status(
                kpt_xy_np[idx],
                kpt_conf_np[idx],
                thigh_torso_ratio_threshold=thigh_torso_ratio_threshold,
            )
            if status is not None:
                statuses.append(status)
    return _format_person_status(statuses, person_count)


def _normalize_box(box: object, width: int, height: int) -> np.ndarray | None:
    if box is None:
        return None
    try:
        arr = np.asarray(box, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in arr[:4]]
    x1 = max(0.0, min(x1, float(width - 1)))
    y1 = max(0.0, min(y1, float(height - 1)))
    x2 = max(0.0, min(x2, float(width)))
    y2 = max(0.0, min(y2, float(height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _box_extent(box: np.ndarray) -> float:
    return max(1.0, float(max(0.0, box[2] - box[0])), float(max(0.0, box[3] - box[1])))


def _box_area(box: np.ndarray) -> float:
    return max(1.0, float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])))


def _append_unique_box(boxes: list[np.ndarray], box: np.ndarray) -> None:
    for existing in boxes:
        if _box_iou(existing, box) >= 0.80:
            return
    boxes.append(box)


def _extract_person_boxes(analysis: dict, max_people: int = 2) -> list[np.ndarray]:
    width = int(analysis.get("width", DISPLAY_SIZE[0]) or DISPLAY_SIZE[0])
    height = int(analysis.get("height", DISPLAY_SIZE[1]) or DISPLAY_SIZE[1])
    boxes: list[np.ndarray] = []

    for record in analysis.get("records") or []:
        box = _normalize_box(record.get("box"), width, height)
        if box is not None:
            _append_unique_box(boxes, box)
        if len(boxes) >= max_people:
            return boxes[:max_people]

    pose_boxes = analysis.get("pose_boxes") or []
    pose_indices = _select_pose_indices_for_status(analysis)
    for idx in pose_indices:
        if idx < 0 or idx >= len(pose_boxes):
            continue
        box = _normalize_box(pose_boxes[idx], width, height)
        if box is not None:
            _append_unique_box(boxes, box)
        if len(boxes) >= max_people:
            return boxes[:max_people]

    for pose_box in pose_boxes:
        box = _normalize_box(pose_box, width, height)
        if box is not None:
            _append_unique_box(boxes, box)
        if len(boxes) >= max_people:
            break
    return boxes[:max_people]


def _compute_person_distance_ir(
    analysis: dict,
    close_gap_ratio: float = PERSON_DISTANCE_CLOSE_GAP_RATIO,
    close_center_ratio: float = PERSON_DISTANCE_CLOSE_CENTER_RATIO,
) -> str:
    person_count = int(analysis.get("person_count", 0) or 0)
    if person_count < 2:
        return ""

    boxes = _extract_person_boxes(analysis, max_people=2)
    if len(boxes) < 2:
        return ""

    width = int(analysis.get("width", DISPLAY_SIZE[0]) or DISPLAY_SIZE[0])
    close_gap_ratio = _normalize_distance_threshold(close_gap_ratio, PERSON_DISTANCE_CLOSE_GAP_RATIO)
    close_center_ratio = _normalize_distance_threshold(close_center_ratio, PERSON_DISTANCE_CLOSE_CENTER_RATIO)
    for idx in range(len(boxes)):
        box_a = boxes[idx]
        for jdx in range(idx + 1, len(boxes)):
            box_b = boxes[jdx]
            horizontal_gap = max(0.0, max(float(box_a[0]), float(box_b[0])) - min(float(box_a[2]), float(box_b[2])))
            vertical_gap = max(0.0, max(float(box_a[1]), float(box_b[1])) - min(float(box_a[3]), float(box_b[3])))
            separated_gap = float(np.hypot(horizontal_gap, vertical_gap))
            center_distance = _box_center_distance(box_a, box_b)
            avg_extent = (_box_extent(box_a) + _box_extent(box_b)) * 0.5
            close_gap_px = max(float(width) * close_gap_ratio, avg_extent * 0.20)
            if separated_gap <= close_gap_px or center_distance <= avg_extent * close_center_ratio:
                return "close"
    return "far"


def _compute_person_distance_depth(
    analysis: dict,
    close_threshold: float = DEPTH_PERSON_DISTANCE_CLOSE_THRESHOLD,
) -> str:
    person_count = int(analysis.get("person_count", 0) or 0)
    if person_count < 2:
        return ""
    pairs = analysis.get("pair_stats") or []
    if not pairs:
        return ""
    spacings = [float(item[2]) for item in pairs if len(item) >= 3]
    if not spacings:
        return ""
    nearest = min(spacings)
    close_threshold = _normalize_distance_threshold(close_threshold, DEPTH_PERSON_DISTANCE_CLOSE_THRESHOLD)
    return "close" if nearest <= close_threshold else "far"


def _build_action_signature(analysis: dict) -> dict:
    width = int(analysis.get("width", DISPLAY_SIZE[0]) or DISPLAY_SIZE[0])
    height = int(analysis.get("height", DISPLAY_SIZE[1]) or DISPLAY_SIZE[1])
    person_count = int(analysis.get("person_count", 0) or 0)
    if person_count <= 0:
        return {
            "person_count": 0,
            "boxes": [],
            "poses": [],
            "width": width,
            "height": height,
        }

    kpt_xy_np = analysis.get("kpt_xy_np")
    kpt_conf_np = analysis.get("kpt_conf_np")
    poses: list[dict] = []
    if kpt_xy_np is not None and kpt_conf_np is not None:
        for idx in _select_pose_indices_for_status(analysis):
            if idx < 0 or idx >= len(kpt_xy_np) or idx >= len(kpt_conf_np):
                continue
            valid = np.asarray(kpt_conf_np[idx] >= QUALITATIVE_KPT_CONF_THRESHOLD, dtype=bool)
            if int(np.sum(valid)) < 3:
                continue
            poses.append(
                {
                    "points": np.asarray(kpt_xy_np[idx], dtype=np.float32).copy(),
                    "valid": valid.copy(),
                }
            )

    return {
        "person_count": person_count,
        "boxes": [box.copy() for box in _extract_person_boxes(analysis, max_people=2)],
        "poses": poses[:2],
        "width": width,
        "height": height,
    }


def _action_frame_extent(signature: dict | None) -> float:
    if not signature:
        return float(max(DISPLAY_SIZE))
    width = int(signature.get("width", DISPLAY_SIZE[0]) or DISPLAY_SIZE[0])
    height = int(signature.get("height", DISPLAY_SIZE[1]) or DISPLAY_SIZE[1])
    return max(1.0, float(max(width, height)))


def _action_frame_area(signature: dict | None) -> float:
    if not signature:
        return float(DISPLAY_SIZE[0] * DISPLAY_SIZE[1])
    width = int(signature.get("width", DISPLAY_SIZE[0]) or DISPLAY_SIZE[0])
    height = int(signature.get("height", DISPLAY_SIZE[1]) or DISPLAY_SIZE[1])
    return max(1.0, float(width) * float(height))


def _action_signature_motion_state(previous: dict | None, current: dict | None) -> tuple[bool, bool]:
    if not previous or not current:
        return False, False

    previous_poses = previous.get("poses") or []
    current_poses = current.get("poses") or []
    has_pose_motion_source = False
    frame_extent = _action_frame_extent(current)
    for idx in range(min(len(previous_poses), len(current_poses))):
        prev_pose = previous_poses[idx]
        curr_pose = current_poses[idx]
        prev_valid = np.asarray(prev_pose.get("valid"), dtype=bool)
        curr_valid = np.asarray(curr_pose.get("valid"), dtype=bool)
        valid = prev_valid & curr_valid
        if int(np.sum(valid)) < 3:
            continue
        has_pose_motion_source = True
        prev_points = np.asarray(prev_pose.get("points"), dtype=np.float32)
        curr_points = np.asarray(curr_pose.get("points"), dtype=np.float32)
        shift_ratios = np.linalg.norm(curr_points[valid] - prev_points[valid], axis=1) / frame_extent
        if float(np.mean(shift_ratios)) >= ACTION_KEYPOINT_SHIFT_HIGH_RATIO:
            return True, True
        if float(np.percentile(shift_ratios, 75)) >= ACTION_KEYPOINT_SHIFT_PEAK_HIGH_RATIO:
            return True, True
    if has_pose_motion_source:
        return True, False

    previous_boxes = previous.get("boxes") or []
    current_boxes = current.get("boxes") or []
    has_box_motion_source = False
    frame_area = _action_frame_area(current)
    for idx in range(min(len(previous_boxes), len(current_boxes))):
        prev_box = previous_boxes[idx]
        curr_box = current_boxes[idx]
        has_box_motion_source = True
        center_shift_ratio = _box_center_distance(prev_box, curr_box) / frame_extent
        area_shift_ratio = abs(_box_area(curr_box) - _box_area(prev_box)) / frame_area
        if center_shift_ratio >= ACTION_BOX_CENTER_SHIFT_HIGH_RATIO:
            return True, True
        if area_shift_ratio >= ACTION_BOX_AREA_SHIFT_HIGH_RATIO:
            return True, True
    return has_box_motion_source, False


def _copy_qualitative_result_fields(source: dict) -> dict:
    return {key: str(source.get(key, "") or "") for key in QUALITATIVE_RESULT_KEYS}


def _clamp_float(value: float | None, default: float, min_value: float, max_value: float | None = None) -> float:
    try:
        numeric = float(default if value is None else value)
    except (TypeError, ValueError):
        numeric = float(default)
    if not np.isfinite(numeric):
        numeric = float(default)
    numeric = max(float(min_value), numeric)
    if max_value is not None:
        numeric = min(float(max_value), numeric)
    return float(numeric)


def _normalize_person_fill_background_blend(value: float | None) -> float:
    return _clamp_float(value, PERSON_FILL_BACKGROUND_BLEND, 0.0, 1.0)


def _normalize_distance_threshold(value: float | None, default: float) -> float:
    return _clamp_float(value, default, 0.0)


def _normalize_person_fill_background(background: str | None) -> str:
    value = str(background or "").strip()
    if not value:
        return PERSON_FILL_BACKGROUND_DEFAULT
    path = Path(value)
    if len(path.parts) <= 1 and path.suffix == "":
        value = f"{value}.png"
    return value


def _resolve_person_fill_background_path(background: str | None) -> Path:
    normalized = _normalize_person_fill_background(background)
    path = Path(normalized)
    if path.is_absolute() or len(path.parts) > 1:
        return path
    return Path(__file__).resolve().parent / "assets" / path.name


def _get_person_fill_background_gray(width: int, height: int, background: str | None = None) -> np.ndarray:
    normalized = _normalize_person_fill_background(background)
    cache_key = str(_resolve_person_fill_background_path(normalized))
    if cache_key not in _PERSON_FILL_BACKGROUND_CACHE:
        asset_path = Path(cache_key)
        background = cv2.imread(str(asset_path), cv2.IMREAD_COLOR)
        if background is None:
            background = np.full((DISPLAY_SIZE[1], DISPLAY_SIZE[0], 3), PERSON_FILL_FALLBACK_GRAY, dtype=np.uint8)
        _PERSON_FILL_BACKGROUND_CACHE[cache_key] = background

    gray_key = (cache_key, int(width), int(height))
    if gray_key not in _PERSON_FILL_BACKGROUND_GRAY_CACHE:
        background = _PERSON_FILL_BACKGROUND_CACHE[cache_key]
        if background.shape[:2] != (height, width):
            background = cv2.resize(background, (width, height), interpolation=cv2.INTER_LINEAR)
        _PERSON_FILL_BACKGROUND_GRAY_CACHE[gray_key] = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    return _PERSON_FILL_BACKGROUND_GRAY_CACHE[gray_key]


def _fill_person_mask_region(
    display: np.ndarray,
    mask: np.ndarray,
    background: str | None = None,
    background_blend: float | None = None,
) -> bool:
    if display.ndim != 3 or display.shape[2] != 3:
        return False
    if mask.shape[:2] != display.shape[:2]:
        mask = cv2.resize(mask, (display.shape[1], display.shape[0]), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.nonzero(mask > 0)
    if ys.size <= 0 or xs.size <= 0:
        return False
    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1
    return _fill_person_mask_region_roi(
        display,
        mask[y1:y2, x1:x2],
        x1,
        y1,
        background,
        background_blend,
    )


def _fill_person_mask_region_roi(
    display: np.ndarray,
    mask_roi: np.ndarray,
    x: int,
    y: int,
    background: str | None = None,
    background_blend: float | None = None,
) -> bool:
    if display.ndim != 3 or display.shape[2] != 3:
        return False
    height, width = display.shape[:2]
    x1 = max(0, min(int(x), width))
    y1 = max(0, min(int(y), height))
    x2 = max(x1, min(x1 + int(mask_roi.shape[1]), width))
    y2 = max(y1, min(y1 + int(mask_roi.shape[0]), height))
    if x2 <= x1 or y2 <= y1:
        return False
    if mask_roi.shape[:2] != (y2 - y1, x2 - x1):
        mask_roi = cv2.resize(mask_roi, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    selected = mask_roi > 0
    if not np.any(selected):
        return False

    display_roi = display[y1:y2, x1:x2]
    original_gray = cv2.cvtColor(display_roi, cv2.COLOR_BGR2GRAY)
    background_gray = _get_person_fill_background_gray(width, height, background)[y1:y2, x1:x2]
    background_weight = _normalize_person_fill_background_blend(background_blend)
    blended_gray = cv2.addWeighted(
        background_gray,
        background_weight,
        original_gray,
        1.0 - background_weight,
        0,
    )
    display_roi[selected] = blended_gray[selected][:, np.newaxis]
    return True


def _lift_display_contour_head(contour: np.ndarray, width: int, height: int) -> np.ndarray:
    points = contour.reshape(-1, 2).astype(np.float32)
    if len(points) < 3:
        return contour.astype(np.int32)

    y_min = float(np.min(points[:, 1]))
    y_max = float(np.max(points[:, 1]))
    contour_height = y_max - y_min + 1.0
    if contour_height < DISPLAY_CONTOUR_HEAD_MIN_HEIGHT:
        return contour.astype(np.int32)

    head_bottom = y_min + contour_height * DISPLAY_CONTOUR_HEAD_TOP_RATIO
    upper_mask = points[:, 1] <= head_bottom
    if not np.any(upper_mask):
        return contour.astype(np.int32)

    upper_points = points[upper_mask]
    upper_rows = np.rint(upper_points[:, 1]).astype(np.int32)
    _unique_rows, row_inverse = np.unique(upper_rows, return_inverse=True)
    row_mins = np.full(len(_unique_rows), np.inf, dtype=np.float32)
    row_maxs = np.full(len(_unique_rows), -np.inf, dtype=np.float32)
    np.minimum.at(row_mins, row_inverse, upper_points[:, 0])
    np.maximum.at(row_maxs, row_inverse, upper_points[:, 0])

    centers = (row_mins[row_inverse] + row_maxs[row_inverse]) * 0.5
    half_widths = np.maximum(1.0, (row_maxs[row_inverse] - row_mins[row_inverse]) * 0.5)
    lift_ratio = max(0.0, DISPLAY_CONTOUR_HEAD_MAX_SCALE - 1.0)
    adjusted = points.copy()

    upper_indices = np.flatnonzero(upper_mask)
    upper_x = upper_points[:, 0]
    upper_y = upper_points[:, 1]
    distance_ratio = np.minimum(1.0, np.abs(upper_x - centers) / half_widths)
    lateral_weight = np.maximum(0.0, 1.0 - distance_ratio * distance_ratio)
    vertical_distance = np.maximum(0.0, head_bottom - upper_y)
    adjusted[upper_indices, 1] = upper_y - vertical_distance * lift_ratio * lateral_weight

    adjusted[:, 0] = np.clip(np.rint(adjusted[:, 0]), 0, max(0, width - 1))
    adjusted[:, 1] = np.clip(np.rint(adjusted[:, 1]), 0, max(0, height - 1))
    return adjusted.astype(np.int32).reshape(-1, 1, 2)


def _record_display_contour(
    record: dict,
    width: int,
    height: int,
    *,
    target_width: int | None = None,
    target_height: int | None = None,
) -> np.ndarray | None:
    contour = record.get("draw_contour")
    if contour is None:
        mask = record.get("mask")
        if mask is not None:
            contour = extract_draw_contour_from_mask(mask, width, height, mask_is_binary=True)
            if contour is not None:
                record["draw_contour"] = contour
    if contour is None:
        contour = record.get("contour")
    if contour is None:
        return None

    if record.get("draw_contour") is not None:
        shifted_contour = contour
    else:
        box = record["box"]
        anchor = record.get("anchor")
        if anchor is not None and len(anchor) >= 2:
            offset_x, offset_y = int(anchor[0]), int(anchor[1])
        else:
            offset_x, offset_y = int(round(box[0])), int(round(box[1]))
        shifted_contour = contour + np.array([[[offset_x, offset_y]]])

    if target_width is not None and target_height is not None:
        target_width = int(target_width)
        target_height = int(target_height)
        if target_width <= 0 or target_height <= 0:
            raise ValueError("target contour size must be positive")
        if target_width != width or target_height != height:
            shifted_contour = _scale_contour_to_size(
                shifted_contour,
                float(target_width) / float(width),
                float(target_height) / float(height),
            )
            return _lift_display_contour_head(shifted_contour, target_width, target_height)

    return _lift_display_contour_head(shifted_contour, width, height)


def _fill_person_contour_region(
    display: np.ndarray,
    contour: np.ndarray,
    background: str | None = None,
    background_blend: float | None = None,
) -> bool:
    if display.ndim != 3 or display.shape[2] != 3:
        return False
    contour = np.asarray(contour, dtype=np.int32)
    if contour.size <= 0:
        return False
    height, width = display.shape[:2]
    x, y, w, h = cv2.boundingRect(contour)
    x1 = max(0, min(int(x), width))
    y1 = max(0, min(int(y), height))
    x2 = max(x1, min(int(x + w), width))
    y2 = max(y1, min(int(y + h), height))
    if x2 <= x1 or y2 <= y1:
        return False

    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    shifted_contour = contour - np.array([[[x1, y1]]], dtype=np.int32)
    cv2.drawContours(mask, [shifted_contour], -1, 255, thickness=cv2.FILLED)
    return _fill_person_mask_region_roi(display, mask, x1, y1, background, background_blend)


def _render_base_image_cpu(
    analysis: dict,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    source = analysis.get("render_source_img")
    if not isinstance(source, np.ndarray) or source.ndim not in (2, 3):
        source = analysis["color_img"]
    if source.ndim == 2:
        source = _gray_to_bgr(_ensure_uint8_gray_cpu(source))
    if source.shape[:2] == (target_height, target_width):
        return source.copy()

    source_height, source_width = source.shape[:2]
    interpolation = (
        cv2.INTER_AREA
        if target_width < source_width or target_height < source_height
        else cv2.INTER_LINEAR
    )
    return cv2.resize(source, (target_width, target_height), interpolation=interpolation)


def _render_skeleton_contour_cpu(
    analysis: dict,
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    records = analysis["records"]
    kpt_xy_np = analysis["kpt_xy_np"]
    kpt_conf_np = analysis["kpt_conf_np"]
    pose_to_track = analysis["pose_to_track"]
    validated_track_ids = analysis["validated_track_ids"]
    pose_only = bool(analysis.get("pose_only", False))
    pose_draw_indices = analysis.get("pose_draw_indices") or []
    pose_fallback_indices = analysis.get("pose_fallback_indices") or []
    person_fill_background = analysis.get("person_fill_background")
    person_fill_background_blend = analysis.get("person_fill_background_blend", PERSON_FILL_BACKGROUND_BLEND)
    source = analysis["color_img"]
    source_height, source_width = source.shape[:2]
    if output_size is None:
        target_width, target_height = source_width, source_height
    else:
        target_width, target_height = int(output_size[0]), int(output_size[1])
    if target_width <= 0 or target_height <= 0:
        raise ValueError("output_size must be positive")

    display = _render_base_image_cpu(analysis, target_width, target_height)

    scale_x = float(target_width) / float(source_width)
    scale_y = float(target_height) / float(source_height)
    contour_thickness = max(1, int(round(2 * min(scale_x, scale_y))))
    skeleton_thickness = max(1, int(round(2 * min(scale_x, scale_y))))
    keypoint_radius = max(1, int(round(5 * min(scale_x, scale_y))))

    for record in records:
        shifted_contour = _record_display_contour(
            record,
            source_width,
            source_height,
            target_width=target_width,
            target_height=target_height,
        )
        if shifted_contour is None:
            continue
        _fill_person_contour_region(
            display,
            shifted_contour,
            person_fill_background,
            person_fill_background_blend,
        )
        cv2.drawContours(display, [shifted_contour], -1, record["track_color"], contour_thickness, cv2.LINE_AA)

    def draw_pose(index: int, color_override: tuple[int, int, int]) -> None:
        keypoints = kpt_xy_np[index]
        if scale_x != 1.0 or scale_y != 1.0:
            keypoints = _scale_keypoints_to_display(keypoints, scale_x, scale_y)
        draw_stick_figure(
            display,
            keypoints,
            kpt_conf_np[index],
            color_override=color_override,
            line_thickness=skeleton_thickness,
            circle_radius=keypoint_radius,
        )

    if kpt_xy_np is not None and kpt_conf_np is not None:
        if pose_only:
            for idx in pose_draw_indices:
                if idx < 0 or idx >= len(kpt_xy_np) or idx >= len(kpt_conf_np):
                    continue
                draw_pose(idx, _track_color(idx + 1))
        else:
            drawn_pose_indices: set[int] = set()
            for idx in range(min(len(kpt_xy_np), len(kpt_conf_np))):
                matched_track = pose_to_track.get(idx)
                if matched_track is None or matched_track not in validated_track_ids:
                    continue
                draw_pose(idx, _track_color(matched_track))
                drawn_pose_indices.add(int(idx))
            for idx in pose_fallback_indices:
                if idx < 0 or idx >= len(kpt_xy_np) or idx >= len(kpt_conf_np) or int(idx) in drawn_pose_indices:
                    continue
                draw_pose(idx, _track_color(idx + 1))

    return display


def _build_scene_observation(analysis: dict) -> dict:
    source = analysis.get("color_img")
    if isinstance(source, np.ndarray) and source.ndim >= 2:
        source_height, source_width = source.shape[:2]
    else:
        source_height, source_width = 1, 1

    tracks: list[dict] = []
    for record in analysis.get("records") or []:
        try:
            track_id = int(record.get("track_id", -1))
        except (TypeError, ValueError):
            continue
        if track_id < 0:
            continue

        mask = record.get("mask")
        mask_area_ratio = 0.0
        mask_center = (0.0, 0.0)
        if isinstance(mask, np.ndarray) and mask.ndim >= 2 and mask.size > 0:
            binary_mask = np.asarray(mask > 0, dtype=np.uint8)
            mask_height, mask_width = binary_mask.shape[:2]
            nonzero_count = int(cv2.countNonZero(binary_mask))
            mask_area_ratio = float(nonzero_count) / float(binary_mask.size)
            if nonzero_count > 0:
                moments = cv2.moments(binary_mask, binaryImage=True)
                if moments["m00"] > 0:
                    mask_center = (
                        float(moments["m10"] / moments["m00"]) / max(1, mask_width),
                        float(moments["m01"] / moments["m00"]) / max(1, mask_height),
                    )

        keypoints = record.get("keypoints")
        confidences = record.get("kpt_conf")
        normalized_keypoints: list[tuple[float, float, float]] = []
        if keypoints is not None and confidences is not None:
            keypoint_array = np.asarray(keypoints)
            confidence_array = np.asarray(confidences).reshape(-1)
            if keypoint_array.ndim >= 2 and keypoint_array.shape[-1] >= 2:
                for point, confidence in zip(keypoint_array, confidence_array):
                    normalized_keypoints.append(
                        (
                            float(point[0]) / max(1, source_width),
                            float(point[1]) / max(1, source_height),
                            float(confidence),
                        )
                    )

        tracks.append(
            {
                "track_id": track_id,
                "confidence": float(record.get("person_conf", 0.0) or 0.0),
                "mask_area_ratio": mask_area_ratio,
                "mask_center": mask_center,
                "keypoints": normalized_keypoints,
            }
        )

    return {
        "person_count": int(analysis.get("person_count", len(tracks)) or 0),
        "tracks": tracks,
    }


def _render_analyzed_result_cpu(analyzed: dict) -> dict:
    analysis = analyzed["analysis"]
    return {
        "frame_id": analyzed["frame_id"],
        "pseudo_color_image": None,
        "skeleton_contour_image": _render_skeleton_contour_cpu(analysis, SKELETON_CONTOUR_OUTPUT_SIZE),
        "raw_person_count": int(analyzed.get("raw_person_count", analyzed["person_count"])),
        "person_count": int(analyzed["person_count"]),
        "processing_time_ms": int(analyzed["processing_time_ms"]),
        **_copy_qualitative_result_fields(analyzed),
        "_scene_observation": _build_scene_observation(analysis),
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
    skeleton_contour_image, skeleton_contour_format = _encode_output_image_cpu(
        rendered["skeleton_contour_image"],
        output_format,
        png_compression,
        jpeg_quality,
    )
    return {
        "frame_id": rendered["frame_id"],
        "pseudo_color_image": b"",
        "skeleton_contour_image": skeleton_contour_image,
        "pseudo_color_image_format": "",
        "skeleton_contour_image_format": skeleton_contour_format,
        "person_count": person_count,
        "processing_time_ms": int(rendered["processing_time_ms"]),
        **_copy_qualitative_result_fields(rendered),
        "_scene_observation": rendered.get("_scene_observation"),
    }


def _render_and_encode_analyzed_result_cpu(payload: tuple[dict, str, int, int]) -> dict:
    analyzed, output_format, png_compression, jpeg_quality = payload
    analysis = analyzed["analysis"]
    person_count = int(analyzed["person_count"])

    render_start = time.perf_counter()
    skeleton_contour_image = _render_skeleton_contour_cpu(analysis, SKELETON_CONTOUR_OUTPUT_SIZE)
    render_ms = _elapsed_ms(render_start)

    encode_start = time.perf_counter()
    skeleton_contour_bytes, skeleton_contour_format = _encode_output_image_cpu(
        skeleton_contour_image,
        output_format,
        png_compression,
        jpeg_quality,
    )
    encode_ms = _elapsed_ms(encode_start)

    return {
        "frame_id": analyzed["frame_id"],
        "pseudo_color_image": b"",
        "skeleton_contour_image": skeleton_contour_bytes,
        "pseudo_color_image_format": "",
        "skeleton_contour_image_format": skeleton_contour_format,
        "person_count": person_count,
        "processing_time_ms": int(analyzed["processing_time_ms"]),
        **_copy_qualitative_result_fields(analyzed),
        "_scene_observation": _build_scene_observation(analysis),
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


@dataclass
class _StableValueState:
    history: deque[object] = field(
        default_factory=lambda: deque(maxlen=QUALITATIVE_STABILITY_WINDOW_FRAMES)
    )
    stable_value: object | None = None
    initialized: bool = False


@dataclass
class _RealtimeStreamState:
    frame_idx: int = 0
    person_tracker: PersonTracker = field(default_factory=PersonTracker)
    cached_pose_boxes: list[np.ndarray] = field(default_factory=list)
    cached_kpt_xy: np.ndarray | None = None
    cached_kpt_conf: np.ndarray | None = None
    last_source_depth: np.ndarray | None = None
    last_action_signature: dict | None = None
    action_motion_window: deque[bool] = field(
        default_factory=lambda: deque(maxlen=ACTION_MOTION_WINDOW_FRAMES)
    )
    stable_person_count: _StableValueState = field(default_factory=_StableValueState)
    stable_person_status: _StableValueState = field(default_factory=_StableValueState)
    stable_action_level: _StableValueState = field(default_factory=_StableValueState)
    track_gate: dict[int, _TrackGateState] = field(default_factory=dict)
    mask_jump_state: dict[int, _MaskJumpState] = field(default_factory=dict)
    contour_track_state: dict[int, _ContourTrackState] = field(default_factory=dict)


@dataclass
class _MultiStreamModelBatch:
    input_count: int
    source_frames: list[dict]
    stream_groups: dict[str, list[dict]]
    stream_order: list[str]
    seg_results: list | None
    pose_results: list | None
    timings: dict[str, int]
    confidence_log_kwargs: dict
    total_start: float


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
        mask_min_area_ratio: float | None = None,
        mask_max_area_ratio: float | None = None,
        contour_new_track_conf_threshold: float | None = None,
        contour_existing_track_conf_threshold: float | None = None,
        device: str | None = None,
        render_workers: int = 1,
        decode_workers: int = 8,
        png_compression: int = 1,
        output_format: str = "png",
        jpeg_quality: int = 80,
        cpu_worker_mode: str = CPU_WORKER_MODE_THREAD,
        cpu_process_start_method: str = "auto",
        instance_name: str | None = None,
        parallel_models: bool = True,
        input_modality: str = INPUT_MODALITY_DEPTH,
        ir_preprocess: bool = False,
        model_input_size: int = MODEL_INPUT_SIZE_320,
        person_fill_background: str | None = PERSON_FILL_BACKGROUND_DEFAULT,
        person_fill_background_blend: float = PERSON_FILL_BACKGROUND_BLEND,
        pose_status_thigh_torso_ratio_threshold: float = POSE_STATUS_THIGH_TORSO_RATIO_THRESHOLD,
        depth_distance_close_threshold: float = DEPTH_PERSON_DISTANCE_CLOSE_THRESHOLD,
        ir_distance_close_gap_ratio: float = PERSON_DISTANCE_CLOSE_GAP_RATIO,
        ir_distance_close_center_ratio: float = PERSON_DISTANCE_CLOSE_CENTER_RATIO,
    ) -> None:
        model_file = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        pose_model_file = Path(pose_model_path) if pose_model_path else DEFAULT_POSE_MODEL_PATH

        self.seg_model = MODEL_CLS(str(model_file), task="segment")
        self.pose_model = MODEL_CLS(str(pose_model_file), task="pose")
        self._device = str(device).strip() if device else None
        self._instance_name = str(instance_name).strip() if instance_name else "model-0"
        self._render_workers = max(1, int(render_workers))
        self._decode_workers = max(1, int(decode_workers))
        normalized_cpu_worker_mode = str(cpu_worker_mode or CPU_WORKER_MODE_THREAD).strip().lower()
        if normalized_cpu_worker_mode not in {CPU_WORKER_MODE_THREAD, CPU_WORKER_MODE_PROCESS}:
            raise ValueError("cpu_worker_mode must be thread or process")
        self._cpu_worker_mode = normalized_cpu_worker_mode
        self._cpu_process_start_method = _resolve_cpu_process_start_method(cpu_process_start_method)
        self._decode_executor: ThreadPoolExecutor | None = None
        self._render_executor: ThreadPoolExecutor | None = None
        self._postprocess_workers = max(1, min(self._render_workers, self._decode_workers))
        self._postprocess_executor: ThreadPoolExecutor | None = None
        if self._cpu_worker_mode == CPU_WORKER_MODE_THREAD:
            if self._decode_workers > 1:
                self._decode_executor = ThreadPoolExecutor(max_workers=self._decode_workers)
        else:
            _warm_cpu_process_pool(self._decode_workers, self._cpu_process_start_method)
        # These stages run after model inference and often carry full-frame numpy arrays.
        # Keep them in-process so out-of-lock work does not queue behind decode/model prep.
        if self._render_workers > 1:
            self._render_executor = ThreadPoolExecutor(max_workers=self._render_workers)
        if self._postprocess_workers > 1:
            self._postprocess_executor = ThreadPoolExecutor(max_workers=self._postprocess_workers)
        self._pipeline_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"{self._instance_name}-postrender",
        )
        self._png_compression = min(9, max(0, int(png_compression)))
        normalized_output_format = str(output_format or "png").strip().lower()
        if normalized_output_format == "jpg":
            normalized_output_format = "jpeg"
        if normalized_output_format not in {"png", "jpeg"}:
            raise ValueError("output_format must be png or jpeg")
        self._output_format = normalized_output_format
        self._jpeg_quality = min(100, max(1, int(jpeg_quality)))
        self._input_modality = _normalize_input_modality(input_modality)
        self._ir_preprocess = bool(ir_preprocess)
        self._model_input_size = normalize_model_input_size(model_input_size)
        self._seg_infer_imgsz = int(self._model_input_size)
        self._pose_infer_imgsz = int(self._model_input_size)
        self._person_fill_background = _normalize_person_fill_background(person_fill_background)
        self._person_fill_background_blend = _normalize_person_fill_background_blend(person_fill_background_blend)
        self._pose_status_thigh_torso_ratio_threshold = _normalize_pose_status_thigh_torso_ratio(
            pose_status_thigh_torso_ratio_threshold
        )
        self._depth_distance_close_threshold = _normalize_distance_threshold(
            depth_distance_close_threshold,
            DEPTH_PERSON_DISTANCE_CLOSE_THRESHOLD,
        )
        self._ir_distance_close_gap_ratio = _normalize_distance_threshold(
            ir_distance_close_gap_ratio,
            PERSON_DISTANCE_CLOSE_GAP_RATIO,
        )
        self._ir_distance_close_center_ratio = _normalize_distance_threshold(
            ir_distance_close_center_ratio,
            PERSON_DISTANCE_CLOSE_CENTER_RATIO,
        )
        self._lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._stateless = bool(stateless)
        self._persist_tracks = (not self._stateless) if persist_tracks is None else bool(persist_tracks)
        self._pose_only = bool(pose_only)
        self._parallel_models = bool(parallel_models) and not self._pose_only
        self._model_executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"{self._instance_name}-model")
            if self._parallel_models
            else None
        )
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
        self._mask_min_area_ratio = min(
            1.0,
            max(
                0.0,
                float(mask_min_area_ratio) if mask_min_area_ratio is not None else float(MASK_MIN_AREA_RATIO),
            ),
        )
        self._mask_max_area_ratio = min(
            1.0,
            max(
                self._mask_min_area_ratio,
                float(mask_max_area_ratio) if mask_max_area_ratio is not None else float(MASK_MAX_AREA_RATIO),
            ),
        )
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
        self._person_tracker = PersonTracker()
        self._cached_pose_boxes: list[np.ndarray] = []
        self._cached_kpt_xy: np.ndarray | None = None
        self._cached_kpt_conf: np.ndarray | None = None
        self._warned_no_masks = False
        self._warned_no_keypoints = False
        self._warned_pose_gate_fallback = False
        self._last_source_depth: np.ndarray | None = None
        self._last_action_signature: dict | None = None
        self._action_motion_window: deque[bool] = deque(maxlen=ACTION_MOTION_WINDOW_FRAMES)
        self._stable_person_count = _StableValueState()
        self._stable_person_status = _StableValueState()
        self._stable_action_level = _StableValueState()
        self._track_gate: dict[int, _TrackGateState] = {}
        self._mask_jump_state: dict[int, _MaskJumpState] = {}
        self._contour_track_state: dict[int, _ContourTrackState] = {}
        self._stream_states: dict[str, _RealtimeStreamState] = {}

    def reset(self) -> None:
        """Reset per-stream caches so next infer behaves like the first frame."""
        self._frame_idx = 0
        self._person_tracker = PersonTracker()
        self._cached_pose_boxes = []
        self._cached_kpt_xy = None
        self._cached_kpt_conf = None
        self._last_source_depth = None
        self._last_action_signature = None
        self._action_motion_window.clear()
        self._stable_person_count = _StableValueState()
        self._stable_person_status = _StableValueState()
        self._stable_action_level = _StableValueState()
        self._track_gate.clear()
        self._mask_jump_state.clear()
        self._contour_track_state.clear()
        self._stream_states.clear()

    @staticmethod
    def _copy_track_gate_state(state: _TrackGateState) -> _TrackGateState:
        return _TrackGateState(
            votes=deque(state.votes, maxlen=state.votes.maxlen),
            ttl_remaining=int(state.ttl_remaining),
            confirmed=bool(state.confirmed),
            last_seen_frame=int(state.last_seen_frame),
        )

    @staticmethod
    def _copy_mask_jump_state(state: _MaskJumpState) -> _MaskJumpState:
        return _MaskJumpState(
            mask=state.mask.copy(),
            last_seen_frame=int(state.last_seen_frame),
            rejected_frames=int(state.rejected_frames),
        )

    @staticmethod
    def _copy_contour_track_state(state: _ContourTrackState) -> _ContourTrackState:
        return _ContourTrackState(
            box=state.box.copy(),
            last_seen_frame=int(state.last_seen_frame),
        )

    @staticmethod
    def _copy_stable_value_state(state: _StableValueState) -> _StableValueState:
        return _StableValueState(
            history=deque(state.history, maxlen=QUALITATIVE_STABILITY_WINDOW_FRAMES),
            stable_value=state.stable_value,
            initialized=bool(state.initialized),
        )

    @staticmethod
    def _initialized_stable_value_state(value: object) -> _StableValueState:
        state = _StableValueState()
        state.history.append(value)
        state.stable_value = value
        state.initialized = True
        return state

    def _capture_stream_state(self) -> _RealtimeStreamState:
        return _RealtimeStreamState(
            frame_idx=int(self._frame_idx),
            person_tracker=self._person_tracker.copy(),
            cached_pose_boxes=[box.copy() for box in self._cached_pose_boxes],
            cached_kpt_xy=None if self._cached_kpt_xy is None else self._cached_kpt_xy.copy(),
            cached_kpt_conf=None if self._cached_kpt_conf is None else self._cached_kpt_conf.copy(),
            last_source_depth=None if self._last_source_depth is None else self._last_source_depth.copy(),
            last_action_signature=None if self._last_action_signature is None else dict(self._last_action_signature),
            action_motion_window=deque(self._action_motion_window, maxlen=ACTION_MOTION_WINDOW_FRAMES),
            stable_person_count=self._copy_stable_value_state(self._stable_person_count),
            stable_person_status=self._copy_stable_value_state(self._stable_person_status),
            stable_action_level=self._copy_stable_value_state(self._stable_action_level),
            track_gate={int(tid): self._copy_track_gate_state(state) for tid, state in self._track_gate.items()},
            mask_jump_state={int(tid): self._copy_mask_jump_state(state) for tid, state in self._mask_jump_state.items()},
            contour_track_state={
                int(tid): self._copy_contour_track_state(state)
                for tid, state in self._contour_track_state.items()
            },
        )

    def _restore_stream_state(self, state: _RealtimeStreamState | None) -> None:
        state = state or _RealtimeStreamState()
        self._frame_idx = int(state.frame_idx)
        self._person_tracker = state.person_tracker.copy()
        self._cached_pose_boxes = [box.copy() for box in state.cached_pose_boxes]
        self._cached_kpt_xy = None if state.cached_kpt_xy is None else state.cached_kpt_xy.copy()
        self._cached_kpt_conf = None if state.cached_kpt_conf is None else state.cached_kpt_conf.copy()
        self._last_source_depth = None if state.last_source_depth is None else state.last_source_depth.copy()
        self._last_action_signature = None if state.last_action_signature is None else dict(state.last_action_signature)
        self._action_motion_window = deque(state.action_motion_window, maxlen=ACTION_MOTION_WINDOW_FRAMES)
        self._stable_person_count = self._copy_stable_value_state(state.stable_person_count)
        self._stable_person_status = self._copy_stable_value_state(state.stable_person_status)
        self._stable_action_level = self._copy_stable_value_state(state.stable_action_level)
        self._track_gate = {
            int(tid): self._copy_track_gate_state(track_state)
            for tid, track_state in state.track_gate.items()
        }
        self._mask_jump_state = {
            int(tid): self._copy_mask_jump_state(mask_state)
            for tid, mask_state in state.mask_jump_state.items()
        }
        self._contour_track_state = {
            int(tid): self._copy_contour_track_state(contour_state)
            for tid, contour_state in state.contour_track_state.items()
        }

    @staticmethod
    def _normalize_stream_key(stream_key: str | None) -> str:
        return str(stream_key or "default").strip() or "default"

    def drop_stream_state(self, stream_key: str | None) -> None:
        key = self._normalize_stream_key(stream_key)
        with self._lock:
            self._stream_states.pop(key, None)

    def warmup(self, batch_size: int = 1) -> None:
        """Run one synthetic model batch so CUDA kernels and model graphs are ready before serving traffic."""
        batch_size = max(1, int(batch_size))
        color_imgs = [
            np.zeros((self._model_input_size, self._model_input_size, 3), dtype=np.uint8)
            for _ in range(batch_size)
        ]

        with self._lock:
            warmup_start = time.perf_counter()
            seg_model_ms = 0

            if not self._pose_only:
                seg_start = time.perf_counter()
                self.seg_model.track(
                    color_imgs,
                    conf=self._seg_conf_threshold,
                    persist=False,
                    tracker=TRACKER_CONFIG,
                    classes=[0],
                    imgsz=self._seg_infer_imgsz,
                    device=self._device,
                    verbose=False,
                )
                seg_model_ms = _elapsed_ms(seg_start)

            pose_start = time.perf_counter()
            self.pose_model.predict(
                color_imgs,
                conf=self._pose_conf_threshold,
                classes=[0],
                imgsz=self._pose_infer_imgsz,
                device=self._device,
                verbose=False,
            )
            pose_model_ms = _elapsed_ms(pose_start)
            total_ms = _elapsed_ms(warmup_start)

            self.reset()

        LOGGER.info(
            "Warmup timing: instance=%s batch_size=%d seg_model_ms=%d pose_model_ms=%d total_ms=%d device=%s",
            self._instance_name,
            batch_size,
            seg_model_ms,
            pose_model_ms,
            total_ms,
            self._device or "auto",
        )

    def _guard_mask_jump_with_reason(
        self,
        track_id: int,
        mask: np.ndarray,
        current_area: int | None = None,
    ) -> tuple[np.ndarray | None, str | None, int]:
        """Reject short-lived, per-track mask area spikes without averaging masks."""
        if mask.ndim != 2:
            raise ValueError("mask must be single-channel")
        current = mask.astype(np.uint8, copy=False)
        if current.size and int(current.max()) > 1:
            current = (current > 0).astype(np.uint8)
        current_area = int(np.count_nonzero(current)) if current_area is None else int(current_area)
        max_area = int(current.size * float(self._mask_max_area_ratio))
        state = self._mask_jump_state.get(int(track_id))

        if state is None:
            if current_area > max_area:
                return None, "mask_area_large", 0
            self._mask_jump_state[int(track_id)] = _MaskJumpState(
                mask=current.copy(),
                last_seen_frame=int(self._frame_idx),
                rejected_frames=0,
            )
            return current, None, current_area

        previous = state.mask
        if previous.shape != current.shape:
            previous = cv2.resize(previous, (current.shape[1], current.shape[0]), interpolation=cv2.INTER_NEAREST)
        previous_area = int(np.count_nonzero(previous))

        if current_area > max_area:
            if previous_area > 0 and previous_area <= max_area and int(state.rejected_frames) < int(MASK_JUMP_HOLD_FRAMES):
                state.mask = previous.copy()
                state.last_seen_frame = int(self._frame_idx)
                state.rejected_frames = int(state.rejected_frames) + 1
                return previous.copy(), None, previous_area
            return None, "mask_area_large", 0

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
            return previous.copy(), None, previous_area

        state.mask = current.copy()
        state.last_seen_frame = int(self._frame_idx)
        state.rejected_frames = 0
        return current, None, current_area

    def _guard_mask_jump(self, track_id: int, mask: np.ndarray) -> np.ndarray | None:
        guarded_mask, _reason, _area = self._guard_mask_jump_with_reason(track_id, mask)
        return guarded_mask

    def _contour_shape_reject_reason(
        self,
        box: np.ndarray,
        mask: np.ndarray,
        person_conf: float,
        width: int,
        height: int,
        mask_area: int | None = None,
    ) -> str | None:
        if mask.ndim != 2:
            return "invalid_mask"

        mask_area = int(np.count_nonzero(mask > 0)) if mask_area is None else int(mask_area)
        image_area = max(1, int(width) * int(height))
        if mask_area < int(image_area * float(self._mask_min_area_ratio)):
            return "mask_area_small"
        if mask_area > int(image_area * float(self._mask_max_area_ratio)):
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

    def _decode_image(
        self,
        data: bytes,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _decode_image_views_cpu(data)

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

    def _prepare_depth_views(
        self,
        depth_gray: np.ndarray,
        received_gray: np.ndarray | None = None,
    ) -> dict:
        return _prepare_depth_views_cpu(
            depth_gray,
            input_modality=self._input_modality,
            received_gray=received_gray,
            ir_preprocess=self._ir_preprocess,
            model_input_size=self._model_input_size,
        )

    def _analyze_frame(
        self,
        depth_gray: np.ndarray,
        *,
        seg_result=None,
        pose_result=None,
        prepared: dict | None = None,
        distance_tasks: list[dict] | None = None,
    ) -> dict:
        self._frame_idx += 1
        if prepared is None:
            prepared = self._prepare_depth_views(depth_gray)
        depth_up = prepared["depth_up"]
        depth_raw = prepared["depth_raw"]
        color_img = prepared["color_img"]
        render_source_img = prepared["render_source_img"]
        model_color_img = prepared.get("model_color_img", color_img)
        width = prepared["width"]
        height = prepared["height"]
        model_to_display_scale_x = float(prepared.get("model_to_display_scale_x", 1.0) or 1.0)
        model_to_display_scale_y = float(prepared.get("model_to_display_scale_y", 1.0) or 1.0)

        # Pose-only mode: do not rely on segmentation model outputs.
        if self._pose_only:
            if pose_result is None:
                pose_results = self.pose_model.predict(
                    model_color_img,
                    conf=self._pose_conf_threshold,
                    classes=[0],
                    imgsz=self._pose_infer_imgsz,
                    device=self._device,
                    verbose=False,
                )
                pose_result = pose_results[0] if pose_results else None
            kpt_xy_np = None
            kpt_conf_np = None
            pose_boxes_np: list[np.ndarray] = []
            if pose_result is not None:
                if pose_result.boxes is not None and len(pose_result.boxes) > 0:
                    scaled_pose_boxes = _scale_boxes_to_display(
                        pose_result.boxes.xyxy.cpu().numpy(),
                        model_to_display_scale_x,
                        model_to_display_scale_y,
                    )
                    pose_boxes_np = [box.copy() for box in scaled_pose_boxes]
                if pose_result.keypoints is not None:
                    keypoints_xy = pose_result.keypoints.xy
                    keypoints_conf = pose_result.keypoints.conf
                    if keypoints_xy is not None and keypoints_conf is not None:
                        kpt_xy_np = _scale_keypoints_to_display(
                            keypoints_xy.cpu().numpy(),
                            model_to_display_scale_x,
                            model_to_display_scale_y,
                        )
                        kpt_conf_np = keypoints_conf.cpu().numpy()

            pose_draw_indices: list[int] = []
            if kpt_conf_np is not None:
                for idx in range(len(kpt_conf_np)):
                    confident_points = int(np.sum(kpt_conf_np[idx] >= self._pose_kpt_conf_threshold))
                    if confident_points >= self._pose_kpt_min_points:
                        pose_draw_indices.append(idx)

            return {
                "depth_up": depth_up,
                "depth_raw": depth_raw,
                "color_img": color_img,
                "render_source_img": render_source_img,
                "records": [],
                "person_count": int(len(pose_draw_indices)),
                "pair_text": "Pair Dist: N/A",
                "pair_stats": [],
                "tracked_labels": [],
                "kpt_xy_np": kpt_xy_np,
                "kpt_conf_np": kpt_conf_np,
                "pose_boxes": pose_boxes_np,
                "pose_to_track": {},
                "validated_track_ids": set(),
                "pose_only": True,
                "pose_draw_indices": pose_draw_indices,
                "pose_fallback_indices": [],
                "contour_reject_reasons": ["pose_only"],
                "person_fill_background": self._person_fill_background,
                "person_fill_background_blend": self._person_fill_background_blend,
                "pose_status_thigh_torso_ratio_threshold": self._pose_status_thigh_torso_ratio_threshold,
                "input_modality": self._input_modality,
                "width": width,
                "height": height,
            }

        if seg_result is None:
            results = self.seg_model.track(
                model_color_img,
                conf=self._seg_conf_threshold,
                persist=self._persist_tracks,
                tracker=TRACKER_CONFIG,
                classes=[0],
                imgsz=self._seg_infer_imgsz,
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
                    model_color_img,
                    conf=self._pose_conf_threshold,
                    classes=[0],
                    imgsz=self._pose_infer_imgsz,
                    device=self._device,
                    verbose=False,
                )
                pose_result = pose_results[0] if pose_results else None
            if pose_result is not None:
                if pose_result.keypoints is not None:
                    keypoints_xy = pose_result.keypoints.xy
                    keypoints_conf = pose_result.keypoints.conf
                    if keypoints_xy is not None and keypoints_conf is not None:
                        self._cached_kpt_xy = _scale_keypoints_to_display(
                            keypoints_xy.cpu().numpy(),
                            model_to_display_scale_x,
                            model_to_display_scale_y,
                        )
                        self._cached_kpt_conf = keypoints_conf.cpu().numpy()
                elif not self._warned_no_keypoints:
                    print(
                        "[tof_pose] 当前姿态模型没有输出 keypoints，请改用 pose 权重。",
                        flush=True,
                    )
                    self._warned_no_keypoints = True

                if pose_result.boxes is not None and len(pose_result.boxes) > 0:
                    scaled_pose_boxes = _scale_boxes_to_display(
                        pose_result.boxes.xyxy.cpu().numpy(),
                        model_to_display_scale_x,
                        model_to_display_scale_y,
                    )
                    self._cached_pose_boxes = [box.copy() for box in scaled_pose_boxes]

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
        contour_debug_summary: list[str] = []
        pending_distance_records: list[dict] = []
        tracker_updated = False

        def add_contour_debug(*parts: object) -> None:
            if len(contour_debug_summary) >= 8:
                return
            tokens = []
            for part in parts:
                if part is None:
                    continue
                text = str(part).strip().replace(" ", "_")
                if text:
                    tokens.append(text)
            if tokens:
                contour_debug_summary.append("/".join(tokens))

        if result.boxes is not None and len(result.boxes) > 0 and result.masks is not None:
            boxes_xyxy = _scale_boxes_to_display(
                result.boxes.xyxy.cpu().numpy(),
                model_to_display_scale_x,
                model_to_display_scale_y,
            )
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
            add_contour_debug(
                f"counts:b{len(boxes_xyxy)}",
                f"m{len(masks_data)}",
                f"t{len(track_ids)}",
                f"c{len(conf_scores)}",
                f"match{match_count}",
            )
            if match_count <= 0:
                contour_reject_reasons.append("candidate_mismatch")
            seg_boxes_for_match = [boxes_xyxy[idx].copy() for idx in range(match_count)]
            pose_to_seg_index = (
                _match_pose_to_seg_tracks(seg_boxes_for_match, list(range(match_count)), pose_boxes_np)
                if pose_boxes_np
                else {}
            )
            seg_index_to_pose = {int(seg_idx): int(pose_idx) for pose_idx, seg_idx in pose_to_seg_index.items()}

            if match_count > 0:
                detections: list[Detection] = []
                empty_kpt_xy, empty_kpt_conf = _empty_keypoint_arrays(kpt_xy_np, kpt_conf_np)
                for idx in range(match_count):
                    pose_idx = seg_index_to_pose.get(int(idx))
                    if (
                        pose_idx is not None
                        and kpt_xy_np is not None
                        and kpt_conf_np is not None
                        and 0 <= pose_idx < len(kpt_xy_np)
                        and pose_idx < len(kpt_conf_np)
                    ):
                        keypoints = kpt_xy_np[pose_idx].astype(np.float32, copy=True)
                        keypoint_conf = kpt_conf_np[pose_idx].astype(np.float32, copy=True)
                    else:
                        keypoints = empty_kpt_xy.copy()
                        keypoint_conf = empty_kpt_conf.copy()
                    detections.append(
                        Detection(
                            box=seg_boxes_for_match[idx].astype(np.float32, copy=True),
                            keypoints=keypoints,
                            kpt_conf=keypoint_conf,
                            distance=None,
                        )
                    )
                stable_track_ids = self._person_tracker.update(detections, timestamp=float(self._frame_idx))
                tracker_updated = True
                for idx, stable_track_id in enumerate(stable_track_ids):
                    if 0 <= idx < len(track_ids) and int(stable_track_id) > 0:
                        track_ids[idx] = int(stable_track_id)
            else:
                self._person_tracker.update([], timestamp=float(self._frame_idx))
                tracker_updated = True

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
                candidate_debug = [f"i{idx}", f"tid{track_id}", f"conf{person_conf:.3f}"]
                conf_reject_reason = self._contour_conf_reject_reason(track_id, box, person_conf)
                if conf_reject_reason is not None:
                    contour_reject_reasons.append(conf_reject_reason)
                    add_contour_debug(*candidate_debug, f"reject:{conf_reject_reason}")
                    continue
                mask = (masks_data[idx] > self._mask_threshold).astype(np.uint8)
                if mask.shape[:2] != (height, width):
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                mask_area = int(np.count_nonzero(mask))
                candidate_debug.append(f"mask{mask_area}")
                mask, mask_reject_reason, guarded_area = self._guard_mask_jump_with_reason(track_id, mask, mask_area)
                if mask is None:
                    contour_reject_reasons.append(mask_reject_reason or "mask_rejected")
                    add_contour_debug(*candidate_debug, f"guard:{mask_reject_reason or 'mask_rejected'}")
                    continue
                if guarded_area != mask_area:
                    candidate_debug.append(f"guarded{guarded_area}")
                shape_reject_reason = self._contour_shape_reject_reason(
                    box,
                    mask,
                    person_conf,
                    width,
                    height,
                    mask_area=guarded_area,
                )
                if shape_reject_reason is not None:
                    contour_reject_reasons.append(shape_reject_reason)
                    add_contour_debug(*candidate_debug, f"shape:{shape_reject_reason}")
                    continue
                pose_idx_for_record = seg_index_to_pose.get(int(idx))
                if (
                    pose_idx_for_record is not None
                    and kpt_xy_np is not None
                    and kpt_conf_np is not None
                    and 0 <= pose_idx_for_record < len(kpt_xy_np)
                    and pose_idx_for_record < len(kpt_conf_np)
                ):
                    record_keypoints = kpt_xy_np[pose_idx_for_record].astype(np.float32, copy=True)
                    record_kpt_conf = kpt_conf_np[pose_idx_for_record].astype(np.float32, copy=True)
                else:
                    record_keypoints = empty_kpt_xy.copy()
                    record_kpt_conf = empty_kpt_conf.copy()
                if distance_tasks is not None:
                    task_id = len(distance_tasks)
                    distance_tasks.append(
                        {
                            "task_id": task_id,
                            "depth_raw": depth_raw,
                            "box": box.copy(),
                            "mask": mask.copy(),
                        }
                    )
                    pending_distance_records.append(
                        {
                            "task_id": task_id,
                            "track_id": track_id,
                            "box": box.copy(),
                            "mask": mask,
                            "person_conf": person_conf,
                            "track_color": track_color,
                            "pose_idx": pose_idx_for_record,
                            "keypoints": record_keypoints,
                            "kpt_conf": record_kpt_conf,
                            "candidate_debug": list(candidate_debug),
                        }
                    )
                    person_count += 1
                    self._remember_contour_track(track_id, box)
                    continue

                # Use non-equalized depth values for distance estimation.
                estimate = estimate_person_distance_from_mask(
                    depth_raw,
                    box,
                    mask,
                    mask_is_binary=True,
                    include_draw_contour=False,
                )
                contour_area = 0.0
                if estimate.contour is not None:
                    contour_area = float(cv2.contourArea(estimate.contour))
                estimate_reason = getattr(estimate, "reason", None) or "ok"
                add_contour_debug(
                    *candidate_debug,
                    f"estimate:{estimate_reason}",
                    f"contour{contour_area:.1f}",
                    f"depthpx{int(estimate.valid_pixels)}",
                    "kept",
                )
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
                        "draw_contour": getattr(estimate, "draw_contour", None),
                        "anchor": getattr(estimate, "anchor", None),
                        "distance": estimate.distance,
                        "person_conf": person_conf,
                        "label": label,
                        "track_color": track_color,
                        "pose_idx": pose_idx_for_record,
                        "keypoints": record_keypoints,
                        "kpt_conf": record_kpt_conf,
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
            add_contour_debug(f"counts:b{len(result.boxes)}", "m0", "no_mask")

        if not tracker_updated:
            self._person_tracker.update([], timestamp=float(self._frame_idx))

        self._prune_contour_track_state()

        if records:
            records, duplicate_rejects, duplicate_debug = _dedupe_person_records(
                records,
                width=width,
                height=height,
                conf_threshold=max(float(self._pose_gate_kpt_conf_threshold), float(QUALITATIVE_KPT_CONF_THRESHOLD)),
            )
            if duplicate_rejects:
                contour_reject_reasons.extend(duplicate_rejects)
                for debug_item in duplicate_debug:
                    add_contour_debug(debug_item)
            person_count = len(records)
            tracked_labels, pair_records = _rebuild_person_record_summaries(records)
            kept_track_ids = {int(record["track_id"]) for record in records}
            validated_track_ids = {int(track_id) for track_id in validated_track_ids if int(track_id) in kept_track_ids}

        pose_fallback_indices: list[int] = []
        if kpt_xy_np is not None and kpt_conf_np is not None:
            fallback_kpt_threshold = max(
                float(self._pose_gate_kpt_conf_threshold),
                float(POSE_FALLBACK_KPT_CONF_THRESHOLD),
            )
            for idx in range(min(len(kpt_xy_np), len(kpt_conf_np))):
                matched_track = pose_to_track.get(idx)
                if (
                    self._pose_fallback
                    and person_count <= 0
                    and _pose_fallback_is_human_candidate(
                        kpt_xy_np[idx],
                        kpt_conf_np[idx],
                        threshold=fallback_kpt_threshold,
                        min_points=self._pose_kpt_min_points,
                        width=width,
                        height=height,
                    )
                ):
                    pose_fallback_indices.append(int(idx))
                if matched_track is None or matched_track not in validated_track_ids:
                    continue

        if person_count <= 0 and pose_fallback_indices:
            person_count = int(len(pose_fallback_indices))

        pair_text, pair_stats = _compute_pairwise_distances(pair_records, width)
        analysis = {
            "depth_up": depth_up,
            "depth_raw": depth_raw,
            "color_img": color_img,
            "render_source_img": render_source_img,
            "records": records,
            "person_count": person_count,
            "pair_text": pair_text,
            "pair_stats": pair_stats,
            "tracked_labels": tracked_labels,
            "kpt_xy_np": kpt_xy_np,
            "kpt_conf_np": kpt_conf_np,
            "pose_boxes": pose_boxes_np,
            "pose_to_track": pose_to_track,
            "validated_track_ids": validated_track_ids,
            "pose_only": False,
            "pose_draw_indices": [],
            "pose_fallback_indices": pose_fallback_indices,
            "contour_reject_reasons": contour_reject_reasons,
            "contour_debug_summary": contour_debug_summary,
            "person_fill_background": self._person_fill_background,
            "person_fill_background_blend": self._person_fill_background_blend,
            "pose_status_thigh_torso_ratio_threshold": self._pose_status_thigh_torso_ratio_threshold,
            "input_modality": self._input_modality,
            "width": width,
            "height": height,
        }
        if pending_distance_records:
            analysis["_pending_distance_records"] = pending_distance_records
        return analysis

    def _render_display(self, analysis: dict, display_mode: str) -> np.ndarray:
        display = _render_base_image_cpu(
            analysis,
            int(analysis["width"]),
            int(analysis["height"]),
        )
        records = analysis["records"]
        kpt_xy_np = analysis["kpt_xy_np"]
        kpt_conf_np = analysis["kpt_conf_np"]
        pose_to_track = analysis["pose_to_track"]
        validated_track_ids = analysis["validated_track_ids"]
        pose_only = bool(analysis.get("pose_only", False))
        pose_draw_indices = analysis.get("pose_draw_indices") or []
        pose_fallback_indices = analysis.get("pose_fallback_indices") or []
        person_fill_background = analysis.get("person_fill_background")
        person_fill_background_blend = analysis.get("person_fill_background_blend", PERSON_FILL_BACKGROUND_BLEND)
        width = analysis["width"]
        height = analysis["height"]

        if display_mode in (DISPLAY_MODE_BOTH, DISPLAY_MODE_CONTOUR_ONLY):
            for record in records:
                box = record["box"]
                track_color = record["track_color"]
                mask = record["mask"]
                label = record["label"]
                shifted_contour = _record_display_contour(record, width, height)

                if shifted_contour is not None:
                    _fill_person_contour_region(
                        display,
                        shifted_contour,
                        person_fill_background,
                        person_fill_background_blend,
                    )
                elif mask is not None:
                    mask_overlay = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    _fill_person_mask_region(
                        display,
                        mask_overlay,
                        person_fill_background,
                        person_fill_background_blend,
                    )

                if shifted_contour is not None:
                    cv2.drawContours(display, [shifted_contour], -1, track_color, 2, cv2.LINE_AA)

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

    def _render_skeleton_contour(
        self,
        analysis: dict,
        output_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Return pseudo color image with only contour and skeleton overlays."""
        return _render_skeleton_contour_cpu(analysis, output_size)

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
        return self._map_thread_stage(self._render_executor, self._render_workers, func, items)

    def _map_postprocess_stage(self, func, items: list):
        return self._map_thread_stage(self._postprocess_executor, self._postprocess_workers, func, items)

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

    def _classify_action_level_from_ir(self, analysis: dict) -> str:
        current_signature = _build_action_signature(analysis)
        has_motion_source, raw_high_motion = _action_signature_motion_state(
            self._last_action_signature,
            current_signature,
        )
        self._last_action_signature = current_signature
        return self._update_action_motion_window(bool(raw_high_motion and has_motion_source))

    def _classify_action_level_from_depth(self, analysis: dict) -> str:
        current_signature = _build_action_signature(analysis)
        has_motion_source, raw_high_motion = _action_signature_motion_state(
            self._last_action_signature,
            current_signature,
        )
        self._last_action_signature = current_signature
        return self._update_action_motion_window(bool(raw_high_motion and has_motion_source))

    def _update_action_motion_window(self, raw_high_motion: bool) -> str:
        self._action_motion_window.append(bool(raw_high_motion))
        window_size = len(self._action_motion_window)
        if window_size < ACTION_HIGH_CONFIRM_FRAMES:
            return "low"
        high_count = sum(1 for item in self._action_motion_window if item)
        required_high = max(
            ACTION_HIGH_CONFIRM_FRAMES,
            int(np.ceil(float(window_size) * float(ACTION_MOTION_WINDOW_HIGH_RATIO))),
        )
        return "high" if high_count >= required_high else "low"

    @staticmethod
    def _stability_required_count(window_size: int) -> int:
        return max(
            int(QUALITATIVE_STABILITY_MIN_CONFIRM_FRAMES),
            int(np.ceil(float(window_size) * float(QUALITATIVE_STABILITY_SWITCH_RATIO))),
        )

    @staticmethod
    def _trailing_value_count(history: deque[object], value: object) -> int:
        count = 0
        for item in reversed(history):
            if item != value:
                break
            count += 1
        return count

    def _stabilize_value(self, state: _StableValueState, raw_value: object) -> object:
        value = "" if raw_value is None else raw_value
        state.history.append(value)
        if not state.initialized:
            state.stable_value = value
            state.initialized = True
            return value

        if value == state.stable_value:
            return state.stable_value

        window_size = len(state.history)
        candidate_count = sum(1 for item in state.history if item == value)
        trailing_count = self._trailing_value_count(state.history, value)
        required_count = self._stability_required_count(window_size)
        if (
            trailing_count >= int(QUALITATIVE_STABILITY_MIN_CONFIRM_FRAMES)
            or candidate_count >= required_count
        ):
            state.stable_value = value
        return state.stable_value

    def _finalize_analyzed_qualitative_fields(self, analyzed: dict) -> dict:
        if analyzed.get("_qualitative_finalized"):
            return analyzed

        analysis = analyzed["analysis"]
        raw_person_count = int(analysis.get("person_count", 0) or 0)
        raw_person_status = _compute_person_status(analysis)
        raw_person_distance = self._compute_person_distance(analysis)
        raw_action_level = self._classify_action_level(analysis)

        stable_person_count = int(self._stabilize_value(self._stable_person_count, raw_person_count))
        if raw_person_count > 0 and not raw_person_status:
            stable_person_status = str(self._stable_person_status.stable_value or "")
        else:
            stable_person_status = str(self._stabilize_value(self._stable_person_status, raw_person_status))
        stable_action_level = str(self._stabilize_value(self._stable_action_level, raw_action_level))

        if stable_person_count <= 0:
            stable_person_status = ""
            stable_action_level = "low"
            self._stable_person_status = _StableValueState()
            self._stable_action_level = self._initialized_stable_value_state("low")

        analyzed["raw_person_count"] = raw_person_count
        analyzed["raw_person_status"] = raw_person_status
        analyzed["raw_action_level"] = raw_action_level
        analyzed["person_count"] = stable_person_count
        analyzed["person_status"] = stable_person_status
        analyzed["person_distance"] = raw_person_distance
        analyzed["action_level"] = stable_action_level
        analyzed["_qualitative_finalized"] = True
        return analyzed

    def _compute_person_distance(self, analysis: dict) -> str:
        if self._input_modality == INPUT_MODALITY_IR:
            return _compute_person_distance_ir(
                analysis,
                self._ir_distance_close_gap_ratio,
                self._ir_distance_close_center_ratio,
            )
        return _compute_person_distance_depth(analysis, self._depth_distance_close_threshold)

    def _classify_action_level(self, analysis: dict) -> str:
        if self._input_modality == INPUT_MODALITY_IR:
            return self._classify_action_level_from_ir(analysis)
        return self._classify_action_level_from_depth(analysis)

    def _analyze_predecoded_unlocked(
        self,
        frame_id: str,
        depth: np.ndarray,
        *,
        seg_result=None,
        pose_result=None,
        prepared: dict | None = None,
        distance_tasks: list[dict] | None = None,
        finalize_qualitative: bool = True,
    ) -> dict:
        if self._stateless:
            self.reset()

        start = time.time()
        analysis = self._analyze_frame(
            depth,
            seg_result=seg_result,
            pose_result=pose_result,
            prepared=prepared,
            distance_tasks=distance_tasks,
        )
        elapsed = int((time.time() - start) * 1000)
        analyzed = {
            "frame_id": frame_id,
            "analysis": analysis,
            "raw_person_count": int(analysis["person_count"]),
            "person_count": int(analysis["person_count"]),
            "processing_time_ms": elapsed,
        }
        if distance_tasks is not None or not finalize_qualitative:
            return analyzed
        return self._finalize_analyzed_qualitative_fields(analyzed)

    def _reuse_analyzed_result_with_depth(
        self,
        analyzed: dict,
        frame_id: str,
        depth: np.ndarray,
        received_depth: np.ndarray | None = None,
    ) -> dict:
        prepared = self._prepare_depth_views(depth, received_depth)
        analysis = dict(analyzed["analysis"])
        analysis["depth_up"] = prepared["depth_up"]
        analysis["depth_raw"] = prepared["depth_raw"]
        analysis["color_img"] = prepared["color_img"]
        analysis["render_source_img"] = prepared["render_source_img"]
        analysis["width"] = prepared["width"]
        analysis["height"] = prepared["height"]
        analysis["person_fill_background"] = self._person_fill_background
        analysis["person_fill_background_blend"] = self._person_fill_background_blend
        analysis["pose_status_thigh_torso_ratio_threshold"] = self._pose_status_thigh_torso_ratio_threshold
        analysis["input_modality"] = self._input_modality
        return {
            "frame_id": frame_id,
            "analysis": analysis,
            "raw_person_count": int(analyzed.get("raw_person_count", analyzed["person_count"])),
            "person_count": int(analyzed["person_count"]),
            "processing_time_ms": 0,
            **_copy_qualitative_result_fields(analyzed),
        }

    def _render_analyzed_result(self, analyzed: dict) -> dict:
        analysis = analyzed["analysis"]
        return {
            "frame_id": analyzed["frame_id"],
            "pseudo_color_image": None,
            "skeleton_contour_image": self._render_skeleton_contour(analysis, SKELETON_CONTOUR_OUTPUT_SIZE),
            "raw_person_count": int(analyzed.get("raw_person_count", analyzed["person_count"])),
            "person_count": int(analyzed["person_count"]),
            "processing_time_ms": int(analyzed["processing_time_ms"]),
            **_copy_qualitative_result_fields(analyzed),
            "_scene_observation": _build_scene_observation(analysis),
        }

    def _encode_rendered_result(self, rendered: dict) -> dict:
        person_count = int(rendered["person_count"])
        skeleton_contour_image, skeleton_contour_format = self._encode_output_image(rendered["skeleton_contour_image"])
        return {
            "frame_id": rendered["frame_id"],
            "pseudo_color_image": b"",
            "skeleton_contour_image": skeleton_contour_image,
            "pseudo_color_image_format": "",
            "skeleton_contour_image_format": skeleton_contour_format,
            "person_count": person_count,
            "processing_time_ms": int(rendered["processing_time_ms"]),
            **_copy_qualitative_result_fields(rendered),
            "_scene_observation": rendered.get("_scene_observation"),
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

    def _decode_frames(
        self,
        frames: list[tuple[str, bytes]],
    ) -> list[tuple[str, np.ndarray, np.ndarray]]:
        return self._map_decode_stage(_decode_frame_cpu, frames)

    def _prepare_source_views(self, source_frames: list[dict]) -> list[dict]:
        if self._cpu_worker_mode == CPU_WORKER_MODE_PROCESS:
            payloads = [
                (
                    item["depth"],
                    self._input_modality,
                    self._ir_preprocess,
                    self._model_input_size,
                    item.get("received_depth"),
                )
                for item in source_frames
            ]
            return self._map_decode_stage(_prepare_depth_views_payload_cpu, payloads)
        return [
            self._prepare_depth_views(
                item["depth"],
                item.get("received_depth"),
            )
            for item in source_frames
        ]

    def _finalize_deferred_distance_results(
        self,
        current_analyzed_by_input: dict[int, dict],
        distance_tasks: list[dict],
    ) -> None:
        if distance_tasks:
            distance_results = self._map_postprocess_stage(_estimate_distance_candidate_cpu, distance_tasks)
            distance_by_task = {int(result["task_id"]): result for result in distance_results}
        else:
            distance_by_task = {}

        def add_contour_debug(summary: list[str], *parts: object) -> None:
            if len(summary) >= 8:
                return
            tokens = []
            for part in parts:
                if part is None:
                    continue
                text = str(part).strip().replace(" ", "_")
                if text:
                    tokens.append(text)
            if tokens:
                summary.append("/".join(tokens))

        for input_index in sorted(current_analyzed_by_input):
            analyzed = current_analyzed_by_input[input_index]
            analysis = analyzed["analysis"]
            pending_records = list(analysis.pop("_pending_distance_records", []) or [])
            records = list(analysis.get("records") or [])
            tracked_labels = list(analysis.get("tracked_labels") or [])
            pair_records = [
                (
                    int(record["track_id"]),
                    np.asarray(record["box"], dtype=np.float32).copy(),
                    record.get("distance"),
                )
                for record in records
            ]
            contour_debug_summary = list(analysis.get("contour_debug_summary") or [])
            contour_reject_reasons = list(analysis.get("contour_reject_reasons") or [])

            for pending in pending_records:
                task_id = int(pending["task_id"])
                distance_result = distance_by_task.get(task_id)
                if distance_result is None:
                    raise RuntimeError(f"missing deferred distance result for task_id={task_id}")
                estimate = distance_result["estimate"]
                contour_area = float(distance_result.get("contour_area", 0.0) or 0.0)
                estimate_reason = getattr(estimate, "reason", None) or "ok"
                add_contour_debug(
                    contour_debug_summary,
                    *(pending.get("candidate_debug") or []),
                    f"estimate:{estimate_reason}",
                    f"contour{contour_area:.1f}",
                    f"depthpx{int(estimate.valid_pixels)}",
                    "kept",
                )

                track_id = int(pending["track_id"])
                person_conf = float(pending["person_conf"])
                if estimate.distance is None:
                    tracked_labels.append(f"{track_id}:N/A")
                    label = f"ID {track_id} P={person_conf * 100:.0f}% Dist=N/A"
                else:
                    tracked_labels.append(f"{track_id}:{estimate.distance:.1f}")
                    label = f"ID {track_id} P={person_conf * 100:.0f}% Dist~{estimate.distance:.1f}"

                box = np.asarray(pending["box"], dtype=np.float32).copy()
                pair_records.append((track_id, box.copy(), estimate.distance))
                records.append(
                    {
                        "track_id": track_id,
                        "box": box,
                        "mask": pending["mask"],
                        "contour": estimate.contour,
                        "draw_contour": getattr(estimate, "draw_contour", None),
                        "anchor": getattr(estimate, "anchor", None),
                        "distance": estimate.distance,
                        "person_conf": person_conf,
                        "label": label,
                        "track_color": pending["track_color"],
                        "pose_idx": pending.get("pose_idx"),
                        "keypoints": pending.get("keypoints"),
                        "kpt_conf": pending.get("kpt_conf"),
                    }
                )

            if records:
                records, duplicate_rejects, duplicate_debug = _dedupe_person_records(
                    records,
                    width=int(analysis.get("width", DISPLAY_SIZE[0]) or DISPLAY_SIZE[0]),
                    height=int(analysis.get("height", DISPLAY_SIZE[1]) or DISPLAY_SIZE[1]),
                    conf_threshold=max(float(self._pose_gate_kpt_conf_threshold), float(QUALITATIVE_KPT_CONF_THRESHOLD)),
                )
                if duplicate_rejects:
                    contour_reject_reasons.extend(duplicate_rejects)
                    for debug_item in duplicate_debug:
                        add_contour_debug(contour_debug_summary, debug_item)
                analysis["person_count"] = int(len(records))
                kept_track_ids = {int(record["track_id"]) for record in records}
                analysis["validated_track_ids"] = {
                    int(track_id)
                    for track_id in (analysis.get("validated_track_ids") or set())
                    if int(track_id) in kept_track_ids
                }
                tracked_labels, pair_records = _rebuild_person_record_summaries(records)

            pair_text, pair_stats = _compute_pairwise_distances(
                pair_records,
                int(analysis.get("width", DISPLAY_SIZE[0]) or DISPLAY_SIZE[0]),
            )
            analysis["records"] = records
            analysis["tracked_labels"] = tracked_labels
            analysis["pair_text"] = pair_text
            analysis["pair_stats"] = pair_stats
            analysis["contour_debug_summary"] = contour_debug_summary
            analysis["contour_reject_reasons"] = contour_reject_reasons
            analyzed["raw_person_count"] = int(analysis["person_count"])
            analyzed["person_count"] = int(analysis["person_count"])
            analyzed.pop("_qualitative_finalized", None)

    def _apply_batch_pose_semantic_reuse(self, current_analyzed_by_input: dict[int, dict]) -> None:
        if len(current_analyzed_by_input) <= 2:
            return

        ordered_indices = sorted(current_analyzed_by_input)
        semantic_indices_by_input: dict[int, list[int]] = {}
        valid_inputs: list[int] = []
        for input_index in ordered_indices:
            analysis = current_analyzed_by_input[input_index]["analysis"]
            semantic_indices = _select_pose_indices_for_status(analysis)
            semantic_indices_by_input[input_index] = semantic_indices
            if semantic_indices:
                valid_inputs.append(input_index)

        min_valid = max(2, int(np.ceil(len(ordered_indices) * POSE_BATCH_REUSE_MIN_VALID_RATIO)))
        if len(valid_inputs) < min_valid:
            return

        valid_set = set(valid_inputs)
        position_by_input = {input_index: pos for pos, input_index in enumerate(ordered_indices)}
        pos = 0
        while pos < len(ordered_indices):
            input_index = ordered_indices[pos]
            if input_index in valid_set:
                pos += 1
                continue

            run_start = pos
            while pos < len(ordered_indices) and ordered_indices[pos] not in valid_set:
                pos += 1
            run_end = pos - 1
            run_len = run_end - run_start + 1
            if run_len > POSE_BATCH_REUSE_MAX_GAP:
                continue

            prev_input = ordered_indices[run_start - 1] if run_start > 0 else None
            next_input = ordered_indices[pos] if pos < len(ordered_indices) else None
            source_input = None
            if prev_input in valid_set and next_input in valid_set:
                target_pos = run_start
                prev_distance = target_pos - position_by_input[prev_input]
                next_distance = position_by_input[next_input] - target_pos
                source_input = prev_input if prev_distance <= next_distance else next_input
            elif prev_input in valid_set:
                source_input = prev_input
            elif next_input in valid_set:
                source_input = next_input
            if source_input is None:
                continue

            source_analysis = current_analyzed_by_input[source_input]["analysis"]
            source_semantic_indices = semantic_indices_by_input.get(source_input) or []
            if not source_semantic_indices:
                continue

            for target_pos in range(run_start, run_end + 1):
                target_input = ordered_indices[target_pos]
                target = current_analyzed_by_input[target_input]
                target_analysis = target["analysis"]
                target_analysis["kpt_xy_np"] = source_analysis.get("kpt_xy_np")
                target_analysis["kpt_conf_np"] = source_analysis.get("kpt_conf_np")
                target_analysis["pose_boxes"] = source_analysis.get("pose_boxes") or []
                target_analysis["pose_to_track"] = {}
                target_analysis["validated_track_ids"] = set()
                target_analysis["pose_draw_indices"] = []
                target_analysis["pose_fallback_indices"] = list(source_semantic_indices[:2])
                target_analysis["pose_semantic_reused"] = True
                target_analysis["person_count"] = max(
                    int(target_analysis.get("person_count", 0) or 0),
                    len(source_semantic_indices[:2]),
                )
                target["raw_person_count"] = int(target_analysis["person_count"])
                target["person_count"] = int(target_analysis["person_count"])
                target.pop("_qualitative_finalized", None)

    def _infer_one_unlocked(self, frame_id: str, image_bytes: bytes) -> dict:
        depth, received_depth = self._decode_image(image_bytes)
        prepared = self._prepare_depth_views(depth, received_depth)
        analyzed = self._analyze_predecoded_unlocked(
            frame_id,
            depth,
            prepared=prepared,
        )
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
                "model_infer_ms=%d seg_model_ms=%d pose_model_ms=%d parallel_models=%s "
                "postprocess_ms=%d postprocess_queue_wait_ms=%d interpolate_ms=%d "
                "render_ms=%d png_encode_ms=%d total_ms=%d "
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
            "true" if timings.get("parallel_models", 0) else "false",
            int(timings.get("postprocess_ms", 0)),
            int(timings.get("postprocess_queue_wait_ms", 0)),
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
        if not LOGGER.isEnabledFor(logging.DEBUG):
            return

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
            analyzed_wrapper = current_analyzed_by_input.get(input_index) or {}
            analyzed = _analysis_payload(analyzed_wrapper)
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
            final_person_counts.append(int(analyzed_wrapper.get("person_count", analyzed.get("person_count", 0))))
            pose_fallback_counts.append(len(analyzed.get("pose_fallback_indices") or []))

        LOGGER.debug(
            (
                "Infer model confidence: instance=%s inputs=%d "
                "contour_threshold=%.3f mask_threshold=%.3f mask_area_ratio=[%.3f,%.3f] "
                "contour_counts=%s contour_conf_max=%s "
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
            self._mask_threshold,
            self._mask_min_area_ratio,
            self._mask_max_area_ratio,
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

    def infer_multi_stream_batch_async(self, items: list[dict]) -> Future:
        result_future: Future = Future()
        if not items:
            result_future.set_result([])
            return result_future
        try:
            batch = self._prepare_multi_stream_model_batch(items)
        except Exception as exc:
            result_future.set_exception(exc)
            return result_future
        return self._pipeline_executor.submit(self._finish_multi_stream_model_batch, batch)

    def _prepare_multi_stream_model_batch(self, items: list[dict]) -> _MultiStreamModelBatch:
        normalized_items: list[dict] = []
        for global_index, item in enumerate(items):
            frame_id = str(item.get("frame_id", "") or "")
            image_data = bytes(item.get("image_data", b"") or b"")
            if not image_data:
                raise ValueError(f"empty image_data for dynamic batch item {global_index}")
            try:
                request_id = int(item.get("request_id", 0) or 0)
            except (TypeError, ValueError):
                request_id = 0
            try:
                input_index = int(item.get("input_index", global_index))
            except (TypeError, ValueError):
                input_index = global_index
            normalized_items.append(
                {
                    **item,
                    "global_index": global_index,
                    "request_id": request_id,
                    "stream_key": self._normalize_stream_key(item.get("stream_key")),
                    "frame_id": frame_id,
                    "image_data": image_data,
                    "input_index": input_index,
                }
            )

        frames = [(item["frame_id"], item["image_data"]) for item in normalized_items]
        timings: dict[str, int] = {"queue_wait_ms": 0}
        total_start = time.perf_counter()

        decode_start = time.perf_counter()
        decoded_frames = self._decode_frames(frames)
        if len(decoded_frames) != len(normalized_items):
            raise RuntimeError(f"decoded {len(decoded_frames)} frames for {len(normalized_items)} dynamic batch items")
        timings["decode_ms"] = _elapsed_ms(decode_start)

        source_frames: list[dict] = []
        stream_groups: dict[str, list[dict]] = {}
        stream_order: list[str] = []
        for item, (frame_id, current_depth, received_depth) in zip(normalized_items, decoded_frames):
            source_item = {
                **item,
                "frame_id": frame_id,
                "depth": current_depth,
                "received_depth": received_depth,
            }
            source_frames.append(source_item)
            stream_key = str(source_item["stream_key"])
            if stream_key not in stream_groups:
                stream_groups[stream_key] = []
                stream_order.append(stream_key)
            stream_groups[stream_key].append(source_item)

        timings["model_input_count"] = len(source_frames)
        timings["postprocess_input_count"] = len(source_frames)

        model_prepare_start = time.perf_counter()
        prepared_views = self._prepare_source_views(source_frames)
        if len(prepared_views) != len(source_frames):
            raise RuntimeError(f"prepared {len(prepared_views)} views for {len(source_frames)} source frames")
        for item, prepared in zip(source_frames, prepared_views):
            item["prepared"] = prepared
        color_imgs = [prepared.get("model_color_img", prepared["color_img"]) for prepared in prepared_views]
        timings["model_prepare_ms"] = _elapsed_ms(model_prepare_start)

        model_lock_start = time.perf_counter()
        self._model_lock.acquire()
        try:
            timings["queue_wait_ms"] = _elapsed_ms(model_lock_start)
            if self._pose_only:
                model_start = time.perf_counter()
                pose_results = self.pose_model.predict(
                    color_imgs,
                    conf=self._pose_conf_threshold,
                    classes=[0],
                    imgsz=self._pose_infer_imgsz,
                    device=self._device,
                    verbose=False,
                )
                timings["pose_model_ms"] = _elapsed_ms(model_start)
                timings["seg_model_ms"] = 0
                timings["model_infer_ms"] = int(timings["pose_model_ms"])
                timings["parallel_models"] = 0
                if len(pose_results) != len(source_frames):
                    raise RuntimeError(f"pose batch returned {len(pose_results)} results for {len(source_frames)} source frames")
                seg_results = None
                confidence_log_kwargs = {"pose_results": pose_results}
            else:
                model_start = time.perf_counter()
                seg_persist = bool(self._persist_tracks and len(stream_groups) == 1)

                def run_seg_model():
                    seg_model_start = time.perf_counter()
                    results = self.seg_model.track(
                        color_imgs,
                        conf=self._seg_conf_threshold,
                        persist=seg_persist,
                        tracker=TRACKER_CONFIG,
                        classes=[0],
                        imgsz=self._seg_infer_imgsz,
                        device=self._device,
                        verbose=False,
                    )
                    return results, _elapsed_ms(seg_model_start)

                def run_pose_model():
                    pose_model_start = time.perf_counter()
                    results = self.pose_model.predict(
                        color_imgs,
                        conf=self._pose_conf_threshold,
                        classes=[0],
                        imgsz=self._pose_infer_imgsz,
                        device=self._device,
                        verbose=False,
                    )
                    return results, _elapsed_ms(pose_model_start)

                if self._parallel_models and self._model_executor is not None:
                    seg_future = self._model_executor.submit(run_seg_model)
                    pose_future = self._model_executor.submit(run_pose_model)
                    try:
                        seg_results, timings["seg_model_ms"] = seg_future.result()
                        pose_results, timings["pose_model_ms"] = pose_future.result()
                    except Exception:
                        for future in (seg_future, pose_future):
                            future.cancel()
                        for future in (seg_future, pose_future):
                            if future.cancelled():
                                continue
                            try:
                                future.result()
                            except Exception:
                                pass
                        raise
                    timings["parallel_models"] = 1
                else:
                    seg_results, timings["seg_model_ms"] = run_seg_model()
                    pose_results, timings["pose_model_ms"] = run_pose_model()
                    timings["parallel_models"] = 0

                timings["model_infer_ms"] = _elapsed_ms(model_start)
                if len(seg_results) != len(source_frames):
                    raise RuntimeError(f"seg batch returned {len(seg_results)} results for {len(source_frames)} source frames")
                if len(pose_results) != len(source_frames):
                    raise RuntimeError(f"pose batch returned {len(pose_results)} results for {len(source_frames)} source frames")
                confidence_log_kwargs = {
                    "contour_results": seg_results,
                    "pose_results": pose_results,
                }
        finally:
            self._model_lock.release()

        return _MultiStreamModelBatch(
            input_count=len(items),
            source_frames=source_frames,
            stream_groups=stream_groups,
            stream_order=stream_order,
            seg_results=seg_results,
            pose_results=pose_results,
            timings=timings,
            confidence_log_kwargs=confidence_log_kwargs,
            total_start=total_start,
        )

    def _finish_multi_stream_model_batch(self, batch: _MultiStreamModelBatch) -> list[dict]:
        source_frames = batch.source_frames
        stream_groups = batch.stream_groups
        stream_order = batch.stream_order
        timings = batch.timings
        pose_results = batch.pose_results or []
        seg_results = batch.seg_results

        state_lock_start = time.perf_counter()
        self._lock.acquire()
        original_state = self._capture_stream_state()
        try:
            timings["postprocess_queue_wait_ms"] = _elapsed_ms(state_lock_start)
            if self._pose_only:
                postprocess_start = time.perf_counter()
                current_analyzed_by_input: dict[int, dict] = {}
                output_frames_by_global: dict[int, dict] = {}
                for stream_key in stream_order:
                    group = stream_groups[stream_key]
                    self._restore_stream_state(self._stream_states.get(stream_key))
                    stream_analyzed: dict[int, dict] = {}
                    for item in group:
                        global_index = int(item["global_index"])
                        stream_analyzed[global_index] = self._analyze_predecoded_unlocked(
                            f"{item['frame_id']}_current",
                            item["depth"],
                            pose_result=pose_results[global_index],
                            prepared=item.get("prepared"),
                            finalize_qualitative=False,
                        )
                    self._apply_batch_pose_semantic_reuse(stream_analyzed)
                    for item in group:
                        self._finalize_analyzed_qualitative_fields(
                            stream_analyzed[int(item["global_index"])]
                        )
                    if not self._stateless:
                        self._stream_states[stream_key] = self._capture_stream_state()
                    else:
                        self._stream_states.pop(stream_key, None)
                    for item in group:
                        global_index = int(item["global_index"])
                        current_analyzed_by_input[global_index] = stream_analyzed[global_index]
                        output_frames_by_global[global_index] = {
                            "frame_id": f"{item['frame_id']}_current",
                            "source_frame_id": item["frame_id"],
                            "global_index": global_index,
                            "request_id": int(item["request_id"]),
                            "input_index": int(item["input_index"]),
                            "stream_key": stream_key,
                            "result_kind": "current",
                        }
                timings["postprocess_ms"] = _elapsed_ms(postprocess_start)
            else:
                if seg_results is None:
                    raise RuntimeError("missing contour results for multi-stream batch")
                postprocess_start = time.perf_counter()
                current_analyzed_by_input = {}
                output_frames_by_global = {}
                for stream_key in stream_order:
                    group = stream_groups[stream_key]
                    self._restore_stream_state(self._stream_states.get(stream_key))
                    stream_analyzed: dict[int, dict] = {}
                    distance_tasks: list[dict] = []
                    for item in group:
                        global_index = int(item["global_index"])
                        stream_analyzed[global_index] = self._analyze_predecoded_unlocked(
                            f"{item['frame_id']}_current",
                            item["depth"],
                            seg_result=seg_results[global_index],
                            pose_result=pose_results[global_index],
                            prepared=item.get("prepared"),
                            distance_tasks=distance_tasks,
                            finalize_qualitative=False,
                        )
                    self._finalize_deferred_distance_results(stream_analyzed, distance_tasks)
                    self._apply_batch_pose_semantic_reuse(stream_analyzed)
                    for item in group:
                        self._finalize_analyzed_qualitative_fields(
                            stream_analyzed[int(item["global_index"])]
                        )
                    if group and not self._stateless:
                        self._last_source_depth = group[-1]["depth"].copy()
                    if not self._stateless:
                        self._stream_states[stream_key] = self._capture_stream_state()
                    else:
                        self._stream_states.pop(stream_key, None)
                    for item in group:
                        global_index = int(item["global_index"])
                        current_analyzed_by_input[global_index] = stream_analyzed[global_index]
                        output_frames_by_global[global_index] = {
                            "frame_id": f"{item['frame_id']}_current",
                            "source_frame_id": item["frame_id"],
                            "global_index": global_index,
                            "request_id": int(item["request_id"]),
                            "input_index": int(item["input_index"]),
                            "stream_key": stream_key,
                            "result_kind": "current",
                        }
                timings["postprocess_ms"] = _elapsed_ms(postprocess_start)

            interpolate_start = time.perf_counter()
            output_frames = [output_frames_by_global[index] for index in range(len(source_frames))]
            analyzed_results = [current_analyzed_by_input[index] for index in range(len(source_frames))]
            timings["interpolate_ms"] = _elapsed_ms(interpolate_start)
        finally:
            self._restore_stream_state(original_state)
            self._lock.release()

        self._log_model_confidences(
            len(source_frames),
            current_analyzed_by_input,
            **batch.confidence_log_kwargs,
        )
        results = self._encode_analyzed_results(analyzed_results, timings)
        for result, item in zip(results, output_frames):
            result["source_frame_id"] = item["source_frame_id"]
            result["input_index"] = int(item["input_index"])
            result["output_index"] = int(item["input_index"])
            result["request_id"] = int(item["request_id"])
            result["stream_key"] = item["stream_key"]
            result["result_kind"] = item["result_kind"]
        timings["total_ms"] = _elapsed_ms(batch.total_start)
        self._log_batch_timings(batch.input_count, len(output_frames), timings)
        return results

    def infer_multi_stream_batch(self, items: list[dict]) -> list[dict]:
        """Run one model batch while keeping postprocess caches isolated by stream_key."""
        if not items:
            return []

        normalized_items: list[dict] = []
        for global_index, item in enumerate(items):
            frame_id = str(item.get("frame_id", "") or "")
            image_data = bytes(item.get("image_data", b"") or b"")
            if not image_data:
                raise ValueError(f"empty image_data for dynamic batch item {global_index}")
            try:
                request_id = int(item.get("request_id", 0) or 0)
            except (TypeError, ValueError):
                request_id = 0
            try:
                input_index = int(item.get("input_index", global_index) or 0)
            except (TypeError, ValueError):
                input_index = global_index
            normalized_items.append(
                {
                    "global_index": global_index,
                    "request_id": request_id,
                    "stream_key": self._normalize_stream_key(item.get("stream_key")),
                    "frame_id": frame_id,
                    "input_index": input_index,
                    "image_data": image_data,
                }
            )

        frames = [(item["frame_id"], item["image_data"]) for item in normalized_items]
        lock_start = time.perf_counter()
        self._lock.acquire()
        original_state = self._capture_stream_state()
        try:
            timings: dict[str, int] = {
                "queue_wait_ms": _elapsed_ms(lock_start),
            }
            total_start = time.perf_counter()

            decode_start = time.perf_counter()
            decoded_frames = self._decode_frames(frames)
            if len(decoded_frames) != len(normalized_items):
                raise RuntimeError(f"decoded {len(decoded_frames)} frames for {len(normalized_items)} dynamic batch items")
            timings["decode_ms"] = _elapsed_ms(decode_start)

            source_frames: list[dict] = []
            stream_groups: dict[str, list[dict]] = {}
            stream_order: list[str] = []
            for item, (frame_id, current_depth, received_depth) in zip(normalized_items, decoded_frames):
                source_item = {
                    **item,
                    "frame_id": frame_id,
                    "depth": current_depth,
                    "received_depth": received_depth,
                }
                source_frames.append(source_item)
                stream_key = str(source_item["stream_key"])
                if stream_key not in stream_groups:
                    stream_groups[stream_key] = []
                    stream_order.append(stream_key)
                stream_groups[stream_key].append(source_item)

            timings["model_input_count"] = len(source_frames)
            timings["postprocess_input_count"] = len(source_frames)

            model_prepare_start = time.perf_counter()
            prepared_views = self._prepare_source_views(source_frames)
            if len(prepared_views) != len(source_frames):
                raise RuntimeError(f"prepared {len(prepared_views)} views for {len(source_frames)} source frames")
            for item, prepared in zip(source_frames, prepared_views):
                item["prepared"] = prepared
            color_imgs = [prepared.get("model_color_img", prepared["color_img"]) for prepared in prepared_views]
            timings["model_prepare_ms"] = _elapsed_ms(model_prepare_start)

            if self._pose_only:
                model_start = time.perf_counter()
                pose_results = self.pose_model.predict(
                    color_imgs,
                    conf=self._pose_conf_threshold,
                    classes=[0],
                    imgsz=self._pose_infer_imgsz,
                    device=self._device,
                    verbose=False,
                )
                timings["pose_model_ms"] = _elapsed_ms(model_start)
                timings["seg_model_ms"] = 0
                timings["model_infer_ms"] = int(timings["pose_model_ms"])
                timings["parallel_models"] = 0
                if len(pose_results) != len(source_frames):
                    raise RuntimeError(f"pose batch returned {len(pose_results)} results for {len(source_frames)} source frames")

                postprocess_start = time.perf_counter()
                current_analyzed_by_input: dict[int, dict] = {}
                output_frames_by_global: dict[int, dict] = {}
                for stream_key in stream_order:
                    group = stream_groups[stream_key]
                    self._restore_stream_state(self._stream_states.get(stream_key))
                    stream_analyzed: dict[int, dict] = {}
                    for item in group:
                        global_index = int(item["global_index"])
                        stream_analyzed[global_index] = self._analyze_predecoded_unlocked(
                            f"{item['frame_id']}_current",
                            item["depth"],
                            pose_result=pose_results[global_index],
                            prepared=item.get("prepared"),
                            finalize_qualitative=False,
                        )
                    self._apply_batch_pose_semantic_reuse(stream_analyzed)
                    for item in group:
                        self._finalize_analyzed_qualitative_fields(
                            stream_analyzed[int(item["global_index"])]
                        )
                    if not self._stateless:
                        self._stream_states[stream_key] = self._capture_stream_state()
                    else:
                        self._stream_states.pop(stream_key, None)
                    for item in group:
                        global_index = int(item["global_index"])
                        current_analyzed_by_input[global_index] = stream_analyzed[global_index]
                        output_frames_by_global[global_index] = {
                            "frame_id": f"{item['frame_id']}_current",
                            "source_frame_id": item["frame_id"],
                            "global_index": global_index,
                            "request_id": int(item["request_id"]),
                            "input_index": int(item["input_index"]),
                            "stream_key": stream_key,
                            "result_kind": "current",
                        }
                timings["postprocess_ms"] = _elapsed_ms(postprocess_start)
                confidence_log_kwargs = {"pose_results": pose_results}
            else:
                model_start = time.perf_counter()
                seg_persist = bool(self._persist_tracks and len(stream_groups) == 1)

                def run_seg_model():
                    seg_model_start = time.perf_counter()
                    results = self.seg_model.track(
                        color_imgs,
                        conf=self._seg_conf_threshold,
                        persist=seg_persist,
                        tracker=TRACKER_CONFIG,
                        classes=[0],
                        imgsz=self._seg_infer_imgsz,
                        device=self._device,
                        verbose=False,
                    )
                    return results, _elapsed_ms(seg_model_start)

                def run_pose_model():
                    pose_model_start = time.perf_counter()
                    results = self.pose_model.predict(
                        color_imgs,
                        conf=self._pose_conf_threshold,
                        classes=[0],
                        imgsz=self._pose_infer_imgsz,
                        device=self._device,
                        verbose=False,
                    )
                    return results, _elapsed_ms(pose_model_start)

                if self._parallel_models and self._model_executor is not None:
                    seg_future = self._model_executor.submit(run_seg_model)
                    pose_future = self._model_executor.submit(run_pose_model)
                    try:
                        seg_results, timings["seg_model_ms"] = seg_future.result()
                        pose_results, timings["pose_model_ms"] = pose_future.result()
                    except Exception:
                        for future in (seg_future, pose_future):
                            future.cancel()
                        for future in (seg_future, pose_future):
                            if future.cancelled():
                                continue
                            try:
                                future.result()
                            except Exception:
                                pass
                        raise
                    timings["parallel_models"] = 1
                else:
                    seg_results, timings["seg_model_ms"] = run_seg_model()
                    pose_results, timings["pose_model_ms"] = run_pose_model()
                    timings["parallel_models"] = 0

                timings["model_infer_ms"] = _elapsed_ms(model_start)
                if len(seg_results) != len(source_frames):
                    raise RuntimeError(f"seg batch returned {len(seg_results)} results for {len(source_frames)} source frames")
                if len(pose_results) != len(source_frames):
                    raise RuntimeError(f"pose batch returned {len(pose_results)} results for {len(source_frames)} source frames")

                postprocess_start = time.perf_counter()
                current_analyzed_by_input = {}
                output_frames_by_global = {}
                for stream_key in stream_order:
                    group = stream_groups[stream_key]
                    self._restore_stream_state(self._stream_states.get(stream_key))
                    stream_analyzed: dict[int, dict] = {}
                    distance_tasks: list[dict] = []
                    for item in group:
                        global_index = int(item["global_index"])
                        stream_analyzed[global_index] = self._analyze_predecoded_unlocked(
                            f"{item['frame_id']}_current",
                            item["depth"],
                            seg_result=seg_results[global_index],
                            pose_result=pose_results[global_index],
                            prepared=item.get("prepared"),
                            distance_tasks=distance_tasks,
                            finalize_qualitative=False,
                        )
                    self._finalize_deferred_distance_results(stream_analyzed, distance_tasks)
                    self._apply_batch_pose_semantic_reuse(stream_analyzed)
                    for item in group:
                        self._finalize_analyzed_qualitative_fields(
                            stream_analyzed[int(item["global_index"])]
                        )
                    if group and not self._stateless:
                        self._last_source_depth = group[-1]["depth"].copy()
                    if not self._stateless:
                        self._stream_states[stream_key] = self._capture_stream_state()
                    else:
                        self._stream_states.pop(stream_key, None)
                    for item in group:
                        global_index = int(item["global_index"])
                        current_analyzed_by_input[global_index] = stream_analyzed[global_index]
                        output_frames_by_global[global_index] = {
                            "frame_id": f"{item['frame_id']}_current",
                            "source_frame_id": item["frame_id"],
                            "global_index": global_index,
                            "request_id": int(item["request_id"]),
                            "input_index": int(item["input_index"]),
                            "stream_key": stream_key,
                            "result_kind": "current",
                        }
                timings["postprocess_ms"] = _elapsed_ms(postprocess_start)
                confidence_log_kwargs = {
                    "contour_results": seg_results,
                    "pose_results": pose_results,
                }

            interpolate_start = time.perf_counter()
            output_frames = [output_frames_by_global[index] for index in range(len(source_frames))]
            analyzed_results = [current_analyzed_by_input[index] for index in range(len(source_frames))]
            timings["interpolate_ms"] = _elapsed_ms(interpolate_start)
        finally:
            self._restore_stream_state(original_state)
            self._lock.release()

        self._log_model_confidences(
            len(source_frames),
            current_analyzed_by_input,
            **confidence_log_kwargs,
        )
        results = self._encode_analyzed_results(analyzed_results, timings)
        for result, item in zip(results, output_frames):
            result["source_frame_id"] = item["source_frame_id"]
            result["input_index"] = int(item["input_index"])
            result["output_index"] = int(item["input_index"])
            result["request_id"] = int(item["request_id"])
            result["stream_key"] = item["stream_key"]
            result["result_kind"] = item["result_kind"]
        timings["total_ms"] = _elapsed_ms(total_start)
        self._log_batch_timings(len(items), len(output_frames), timings)
        return results

    def infer_batch(self, frames: list[tuple[str, bytes]]) -> list[dict]:
        if not frames:
            return []

        lock_start = time.perf_counter()
        self._lock.acquire()
        try:
            timings: dict[str, int] = {
                "queue_wait_ms": _elapsed_ms(lock_start),
            }
            total_start = time.perf_counter()

            decode_start = time.perf_counter()
            decoded_frames = self._decode_frames(frames)
            timings["decode_ms"] = _elapsed_ms(decode_start)

            source_frames: list[dict] = []
            for input_index, (frame_id, current_depth, received_depth) in enumerate(decoded_frames):
                source_frames.append(
                    {
                        "frame_id": frame_id,
                        "input_index": input_index,
                        "depth": current_depth,
                        "received_depth": received_depth,
                    }
                )

            timings["model_input_count"] = len(source_frames)
            timings["postprocess_input_count"] = len(source_frames)

            def build_output_results(current_analyzed_by_input: dict[int, dict]) -> tuple[list[dict], list[dict]]:
                interpolate_start = time.perf_counter()
                output_frames: list[dict] = []
                analyzed_results: list[dict] = []
                for input_index, (frame_id, current_depth, _received_depth) in enumerate(decoded_frames):
                    current_analyzed = current_analyzed_by_input[input_index]
                    current_frame = {
                        "frame_id": f"{frame_id}_current",
                        "source_frame_id": frame_id,
                        "input_index": input_index,
                        "result_kind": "current",
                    }
                    output_frames.append(current_frame)
                    analyzed_results.append(current_analyzed)

                if decoded_frames and not self._stateless:
                    self._last_source_depth = decoded_frames[-1][1].copy()
                timings["interpolate_ms"] = _elapsed_ms(interpolate_start)
                return output_frames, analyzed_results

            model_prepare_start = time.perf_counter()
            prepared_views = self._prepare_source_views(source_frames)
            if len(prepared_views) != len(source_frames):
                raise RuntimeError(f"prepared {len(prepared_views)} views for {len(source_frames)} source frames")
            for item, prepared in zip(source_frames, prepared_views):
                item["prepared"] = prepared
            color_imgs = [prepared.get("model_color_img", prepared["color_img"]) for prepared in prepared_views]
            timings["model_prepare_ms"] = _elapsed_ms(model_prepare_start)

            if self._pose_only:
                model_start = time.perf_counter()
                pose_results = self.pose_model.predict(
                    color_imgs,
                    conf=self._pose_conf_threshold,
                    classes=[0],
                    imgsz=self._pose_infer_imgsz,
                    device=self._device,
                    verbose=False,
                )
                timings["pose_model_ms"] = _elapsed_ms(model_start)
                timings["seg_model_ms"] = 0
                timings["model_infer_ms"] = int(timings["pose_model_ms"])
                timings["parallel_models"] = 0
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
                        prepared=item.get("prepared"),
                        finalize_qualitative=False,
                    )
                self._apply_batch_pose_semantic_reuse(current_analyzed_by_input)
                for item in source_frames:
                    self._finalize_analyzed_qualitative_fields(
                        current_analyzed_by_input[int(item["input_index"])]
                    )
                timings["postprocess_ms"] = _elapsed_ms(postprocess_start)
                confidence_log_kwargs = {"pose_results": pose_results}
                output_frames, analyzed_results = build_output_results(current_analyzed_by_input)
            else:
                model_start = time.perf_counter()

                def run_seg_model():
                    seg_model_start = time.perf_counter()
                    results = self.seg_model.track(
                        color_imgs,
                        conf=self._seg_conf_threshold,
                        persist=self._persist_tracks,
                        tracker=TRACKER_CONFIG,
                        classes=[0],
                        imgsz=self._seg_infer_imgsz,
                        device=self._device,
                        verbose=False,
                    )
                    return results, _elapsed_ms(seg_model_start)

                def run_pose_model():
                    pose_model_start = time.perf_counter()
                    results = self.pose_model.predict(
                        color_imgs,
                        conf=self._pose_conf_threshold,
                        classes=[0],
                        imgsz=self._pose_infer_imgsz,
                        device=self._device,
                        verbose=False,
                    )
                    return results, _elapsed_ms(pose_model_start)

                if self._parallel_models and self._model_executor is not None:
                    seg_future = self._model_executor.submit(run_seg_model)
                    pose_future = self._model_executor.submit(run_pose_model)
                    try:
                        seg_results, timings["seg_model_ms"] = seg_future.result()
                        pose_results, timings["pose_model_ms"] = pose_future.result()
                    except Exception:
                        for future in (seg_future, pose_future):
                            future.cancel()
                        for future in (seg_future, pose_future):
                            if future.cancelled():
                                continue
                            try:
                                future.result()
                            except Exception:
                                pass
                        raise
                    timings["parallel_models"] = 1
                else:
                    seg_results, timings["seg_model_ms"] = run_seg_model()
                    pose_results, timings["pose_model_ms"] = run_pose_model()
                    timings["parallel_models"] = 0

                timings["model_infer_ms"] = _elapsed_ms(model_start)
                if len(seg_results) != len(source_frames):
                    raise RuntimeError(f"seg batch returned {len(seg_results)} results for {len(source_frames)} source frames")
                if len(pose_results) != len(source_frames):
                    raise RuntimeError(f"pose batch returned {len(pose_results)} results for {len(source_frames)} source frames")

                postprocess_start = time.perf_counter()
                current_analyzed_by_input = {}
                distance_tasks: list[dict] = []
                for item in source_frames:
                    input_index = int(item["input_index"])
                    current_analyzed_by_input[input_index] = self._analyze_predecoded_unlocked(
                        f"{item['frame_id']}_current",
                        item["depth"],
                        seg_result=seg_results[input_index],
                        pose_result=pose_results[input_index],
                        prepared=item.get("prepared"),
                        distance_tasks=distance_tasks,
                        finalize_qualitative=False,
                    )
                self._finalize_deferred_distance_results(current_analyzed_by_input, distance_tasks)
                self._apply_batch_pose_semantic_reuse(current_analyzed_by_input)
                for item in source_frames:
                    self._finalize_analyzed_qualitative_fields(
                        current_analyzed_by_input[int(item["input_index"])]
                    )
                timings["postprocess_ms"] = _elapsed_ms(postprocess_start)
                confidence_log_kwargs = {
                    "contour_results": seg_results,
                    "pose_results": pose_results,
                }
                output_frames, analyzed_results = build_output_results(current_analyzed_by_input)
        finally:
            self._lock.release()

        self._log_model_confidences(
            len(source_frames),
            current_analyzed_by_input,
            **confidence_log_kwargs,
        )
        results = self._encode_analyzed_results(analyzed_results, timings)
        for output_index, (result, item) in enumerate(zip(results, output_frames)):
            result["source_frame_id"] = item["source_frame_id"]
            result["input_index"] = int(item["input_index"])
            result["output_index"] = int(output_index)
            result["result_kind"] = item["result_kind"]
        timings["total_ms"] = _elapsed_ms(total_start)
        self._log_batch_timings(len(frames), len(output_frames), timings)
        return results
