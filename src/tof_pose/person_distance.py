from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


MIN_VALID_PIXELS = 24
DEPTH_OFFSET = 35.0
FOREGROUND_KEEP_RATIO = 0.30
MASK_ERODE_KERNEL = 3


@dataclass
class PersonDistanceEstimate:
    distance: float | None
    anchor: tuple[int, int]
    contour: np.ndarray | None
    valid_pixels: int
    reason: str | None = None


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


def estimate_person_distance_from_mask(
    depth_map: np.ndarray,
    box: np.ndarray,
    mask: np.ndarray,
) -> PersonDistanceEstimate:
    """根据 YOLO segmentation mask 估计人体距离并提取轮廓。"""
    height, width = depth_map.shape[:2]
    clipped = _clip_box(box, width, height)
    if clipped is None:
        return PersonDistanceEstimate(None, (0, 0), None, 0, "box_invalid")

    x1, y1, x2, y2 = clipped
    roi = depth_map[y1:y2, x1:x2]
    if roi.size == 0:
        return PersonDistanceEstimate(None, (x1, y1), None, 0, "roi_empty")

    normalized_mask = _normalize_mask(mask, width, height)
    mask_roi = normalized_mask[y1:y2, x1:x2]
    if mask_roi.size == 0:
        return PersonDistanceEstimate(None, (x1, y1), None, 0, "mask_roi_empty")
    if int(np.count_nonzero(mask_roi)) <= 0:
        return PersonDistanceEstimate(None, (x1, y1), None, 0, "mask_roi_empty")

    if MASK_ERODE_KERNEL > 1:
        kernel = np.ones((MASK_ERODE_KERNEL, MASK_ERODE_KERNEL), np.uint8)
        mask_roi = cv2.erode(mask_roi, kernel, iterations=1)

    mask_roi = cv2.morphologyEx(
        mask_roi,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
        iterations=1,
    )
    if int(np.count_nonzero(mask_roi)) <= 0:
        return PersonDistanceEstimate(None, (x1, y1), None, 0, "mask_empty_after_morph")

    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return PersonDistanceEstimate(None, (x1, y1), None, 0, "contour_missing")

    best_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best_contour) <= 0:
        return PersonDistanceEstimate(None, (x1, y1), None, 0, "contour_area_zero")

    contour_mask = np.zeros_like(mask_roi)
    cv2.drawContours(contour_mask, [best_contour], -1, 255, thickness=cv2.FILLED)
    valid_depths = roi[contour_mask > 0]
    valid_depths = valid_depths[valid_depths > 0]
    valid_pixels = int(valid_depths.size)
    if valid_pixels < MIN_VALID_PIXELS:
        return PersonDistanceEstimate(None, (x1, y1), best_contour, valid_pixels, "depth_valid_pixels_low")

    sorted_depths = np.sort(valid_depths.astype(np.float32), axis=None)
    keep_count = max(MIN_VALID_PIXELS, int(round(sorted_depths.size * FOREGROUND_KEEP_RATIO)))
    foreground_depths = sorted_depths[: min(keep_count, sorted_depths.size)]
    distance = float(np.median(foreground_depths)) - DEPTH_OFFSET
    return PersonDistanceEstimate(distance, (x1, y1), best_contour, valid_pixels, None)
