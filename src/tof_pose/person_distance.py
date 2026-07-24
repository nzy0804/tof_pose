from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


MIN_VALID_PIXELS = 24
DEPTH_OFFSET = 35.0
FOREGROUND_KEEP_RATIO = 0.30
MASK_ERODE_KERNEL = 3
DRAW_CONTOUR_CLOSE_KERNEL = 3
DRAW_CONTOUR_SMOOTH_WINDOW = 5
MASK_ERODE_KERNEL_ARRAY = (
    np.ones((MASK_ERODE_KERNEL, MASK_ERODE_KERNEL), np.uint8)
    if MASK_ERODE_KERNEL > 1
    else None
)
DRAW_CONTOUR_CLOSE_KERNEL_ARRAY = (
    np.ones((DRAW_CONTOUR_CLOSE_KERNEL, DRAW_CONTOUR_CLOSE_KERNEL), np.uint8)
    if DRAW_CONTOUR_CLOSE_KERNEL > 1
    else None
)
MASK_CLOSE_KERNEL_ARRAY = np.ones((5, 5), np.uint8)


@dataclass
class PersonDistanceEstimate:
    distance: float | None
    anchor: tuple[int, int]
    contour: np.ndarray | None
    valid_pixels: int
    reason: str | None = None
    draw_contour: np.ndarray | None = None


def _clip_box(
    box: np.ndarray,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """将检测框裁剪到图像边界内。"""
    x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def _normalize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """将输入 mask 统一到与深度图一致的 0/255 单通道格式。"""
    mask_uint8 = mask.astype(np.uint8)
    if mask_uint8.shape[:2] != (height, width):
        mask_uint8 = cv2.resize(mask_uint8, (width, height), interpolation=cv2.INTER_NEAREST)
    if mask_uint8.max() <= 1:
        mask_uint8 = mask_uint8 * 255
    return mask_uint8


def _smooth_closed_contour(contour: np.ndarray, width: int, height: int) -> np.ndarray:
    window = max(1, int(DRAW_CONTOUR_SMOOTH_WINDOW))
    if window <= 1 or window % 2 == 0:
        return contour.astype(np.int32)
    points = contour.reshape(-1, 2).astype(np.float32)
    if len(points) < window * 2:
        return contour.astype(np.int32)

    pad = window // 2
    padded = np.concatenate([points[-pad:], points, points[:pad]], axis=0)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed_x = np.convolve(padded[:, 0], kernel, mode="valid")
    smoothed_y = np.convolve(padded[:, 1], kernel, mode="valid")
    smoothed = np.stack([smoothed_x, smoothed_y], axis=1)
    smoothed = np.rint(smoothed).astype(np.int32)
    smoothed[:, 0] = np.clip(smoothed[:, 0], 0, max(0, width - 1))
    smoothed[:, 1] = np.clip(smoothed[:, 1], 0, max(0, height - 1))
    return smoothed.reshape(-1, 1, 2)


def _extract_draw_contour(mask_uint8: np.ndarray, width: int, height: int) -> np.ndarray | None:
    draw_mask = np.where(mask_uint8 > 0, 255, 0).astype(np.uint8)
    if DRAW_CONTOUR_CLOSE_KERNEL_ARRAY is not None:
        draw_mask = cv2.morphologyEx(draw_mask, cv2.MORPH_CLOSE, DRAW_CONTOUR_CLOSE_KERNEL_ARRAY, iterations=1)

    contours, _ = cv2.findContours(draw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    best_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best_contour) <= 0:
        return None
    return _smooth_closed_contour(best_contour, width, height)


def extract_draw_contour_from_mask(
    mask: np.ndarray,
    width: int,
    height: int,
    *,
    mask_is_binary: bool = False,
) -> np.ndarray | None:
    if mask_is_binary:
        mask_uint8 = mask.astype(np.uint8, copy=False)
        if mask_uint8.shape[:2] != (height, width):
            mask_uint8 = cv2.resize(mask_uint8, (width, height), interpolation=cv2.INTER_NEAREST)
        mask_uint8 = mask_uint8 * 255
    else:
        mask_uint8 = _normalize_mask(mask, width, height)
    return _extract_draw_contour(mask_uint8, width, height)


def estimate_person_distance_from_mask(
    depth_map: np.ndarray,
    box: np.ndarray,
    mask: np.ndarray,
    *,
    mask_is_binary: bool = False,
    include_draw_contour: bool = True,
) -> PersonDistanceEstimate:
    """根据人体分割 mask 估计人体距离并提取轮廓。"""
    height, width = depth_map.shape[:2]
    clipped = _clip_box(box, width, height)
    if clipped is None:
        return PersonDistanceEstimate(None, (0, 0), None, 0, "box_invalid")

    x1, y1, x2, y2 = clipped
    roi = depth_map[y1:y2, x1:x2]
    if roi.size == 0:
        return PersonDistanceEstimate(None, (x1, y1), None, 0, "roi_empty")

    if mask_is_binary:
        normalized_mask = mask.astype(np.uint8, copy=False)
        if normalized_mask.shape[:2] != (height, width):
            normalized_mask = cv2.resize(normalized_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        normalized_mask = normalized_mask * 255
    else:
        normalized_mask = _normalize_mask(mask, width, height)
    draw_contour = _extract_draw_contour(normalized_mask, width, height) if include_draw_contour else None
    mask_roi = normalized_mask[y1:y2, x1:x2]
    if mask_roi.size == 0:
        return PersonDistanceEstimate(None, (x1, y1), None, 0, "mask_roi_empty", draw_contour)
    if int(np.count_nonzero(mask_roi)) <= 0:
        return PersonDistanceEstimate(None, (x1, y1), None, 0, "mask_roi_empty", draw_contour)

    if MASK_ERODE_KERNEL_ARRAY is not None:
        mask_roi = cv2.erode(mask_roi, MASK_ERODE_KERNEL_ARRAY, iterations=1)

    mask_roi = cv2.morphologyEx(
        mask_roi,
        cv2.MORPH_CLOSE,
        MASK_CLOSE_KERNEL_ARRAY,
        iterations=1,
    )
    if int(np.count_nonzero(mask_roi)) <= 0:
        return PersonDistanceEstimate(None, (x1, y1), None, 0, "mask_empty_after_morph", draw_contour)

    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return PersonDistanceEstimate(None, (x1, y1), None, 0, "contour_missing", draw_contour)

    best_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best_contour) <= 0:
        return PersonDistanceEstimate(None, (x1, y1), None, 0, "contour_area_zero", draw_contour)

    contour_mask = np.zeros_like(mask_roi)
    cv2.drawContours(contour_mask, [best_contour], -1, 255, thickness=cv2.FILLED)
    valid_depths = roi[contour_mask > 0]
    valid_depths = valid_depths[valid_depths > 0]
    valid_pixels = int(valid_depths.size)
    if valid_pixels < MIN_VALID_PIXELS:
        return PersonDistanceEstimate(None, (x1, y1), best_contour, valid_pixels, "depth_valid_pixels_low", draw_contour)

    valid_depths_float = valid_depths.astype(np.float32, copy=False)
    keep_count = max(MIN_VALID_PIXELS, int(round(valid_depths_float.size * FOREGROUND_KEEP_RATIO)))
    keep_count = min(keep_count, valid_depths_float.size)
    if keep_count < valid_depths_float.size:
        foreground_depths = np.partition(valid_depths_float, keep_count - 1)[:keep_count]
    else:
        foreground_depths = valid_depths_float
    distance = float(np.median(foreground_depths)) - DEPTH_OFFSET
    return PersonDistanceEstimate(distance, (x1, y1), best_contour, valid_pixels, None, draw_contour)
