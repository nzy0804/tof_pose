from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


MIN_VALID_PIXELS = 24
TORSO_INDICES = (5, 6, 11, 12)
HEAD_INDICES = (0, 1, 2, 3, 4)
STABLE_DISTANCE_INDICES = (5, 6, 11, 12)
DEPTH_WINDOW_RADIUS = 2
DEPTH_NEAREST_COUNT = 3
KEYPOINT_MASK_MARGIN_RATIO = 0.1
KEYPOINT_MASK_MIN_MARGIN = 6


@dataclass
class PersonDistanceEstimate:
    distance: float | None
    anchor: tuple[int, int]
    contour: np.ndarray | None
    valid_pixels: int


def _clip_box(
    box: np.ndarray,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """将检测框裁剪到图像边界内。"""
    x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def _collect_depth_samples(
    depth_map: np.ndarray,
    keypoints: np.ndarray,
    kpt_conf: np.ndarray,
    indices: tuple[int, ...],
    conf_threshold: float,
) -> list[float]:
    """按关键点索引收集单点深度样本。"""
    height, width = depth_map.shape[:2]
    samples: list[float] = []
    for idx in indices:
        if idx >= len(kpt_conf) or idx >= len(keypoints) or kpt_conf[idx] < conf_threshold:
            continue
        px = int(round(keypoints[idx][0]))
        py = int(round(keypoints[idx][1]))
        if 0 <= px < width and 0 <= py < height:
            value = float(depth_map[py, px])
            if value > 0:
                samples.append(value)
    return samples


def _collect_robust_keypoint_depths(
    depth_map: np.ndarray,
    keypoints: np.ndarray,
    kpt_conf: np.ndarray,
    indices: tuple[int, ...],
    conf_threshold: float,
    window_radius: int = DEPTH_WINDOW_RADIUS,
    nearest_count: int = DEPTH_NEAREST_COUNT,
) -> list[float]:
    """在关键点邻域内收集更稳的深度样本，优先保留较近的前景值。"""
    height, width = depth_map.shape[:2]
    samples: list[float] = []

    for idx in indices:
        if idx >= len(kpt_conf) or idx >= len(keypoints) or kpt_conf[idx] < conf_threshold:
            continue

        px = int(round(keypoints[idx][0]))
        py = int(round(keypoints[idx][1]))
        if not (0 <= px < width and 0 <= py < height):
            continue

        x1 = max(0, px - window_radius)
        y1 = max(0, py - window_radius)
        x2 = min(width, px + window_radius + 1)
        y2 = min(height, py + window_radius + 1)
        patch = depth_map[y1:y2, x1:x2]
        valid_patch = patch[patch > 0]
        if valid_patch.size == 0:
            continue
                    
        sorted_patch = np.sort(valid_patch.astype(np.float32), axis=None)
        keep = sorted_patch[: min(nearest_count, sorted_patch.size)]
        samples.append(float(np.median(keep)))

    return samples


def _build_keypoint_person_mask(
    roi_shape: tuple[int, int],
    keypoints: np.ndarray,
    kpt_conf: np.ndarray,
    conf_threshold: float,
    anchor: tuple[int, int],
) -> np.ndarray | None:
    """根据有效关键点生成一个限制人体区域的 mask。"""
    roi_h, roi_w = roi_shape
    anchor_x, anchor_y = anchor

    valid_points: list[list[int]] = []
    for idx, conf in enumerate(kpt_conf):
        if idx >= len(keypoints) or conf < conf_threshold:
            continue
        px = int(round(float(keypoints[idx][0]) - anchor_x))
        py = int(round(float(keypoints[idx][1]) - anchor_y))
        if 0 <= px < roi_w and 0 <= py < roi_h:
            valid_points.append([px, py])

    if len(valid_points) < 3:
        return None

    points = np.asarray(valid_points, dtype=np.int32)
    margin = max(
        KEYPOINT_MASK_MIN_MARGIN,
        int(round(max(roi_w, roi_h) * KEYPOINT_MASK_MARGIN_RATIO)),
    )

    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)

    if len(points) >= 5:
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)
    else:
        x, y, w, h = cv2.boundingRect(points)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(roi_w, x + w + margin)
        y2 = min(roi_h, y + h + margin)
        mask[y1:y2, x1:x2] = 255

    kernel_size = max(3, margin * 2 + 1)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def estimate_person_distance(
    depth_map: np.ndarray,
    box: np.ndarray,
    keypoints: np.ndarray,
    kpt_conf: np.ndarray,
    conf_threshold: float,
) -> PersonDistanceEstimate:
    """根据深度图、人体框和关键点估计单个人体到相机的距离。"""
    height, width = depth_map.shape[:2]
    clipped = _clip_box(box, width, height)
    if clipped is None:
        return PersonDistanceEstimate(None, (0, 0), None, 0)

    x1, y1, x2, y2 = clipped
    roi = depth_map[y1:y2, x1:x2]
    if roi.size == 0:
        return PersonDistanceEstimate(None, (x1, y1), None, 0)

    distance_samples = _collect_robust_keypoint_depths(
        depth_map,
        keypoints,
        kpt_conf,
        STABLE_DISTANCE_INDICES,
        conf_threshold,
    )
    distance = None
    if distance_samples:
        distance = float(np.median(np.asarray(distance_samples, dtype=np.float32))) - 35.0

    torso_samples = _collect_depth_samples(
        depth_map,
        keypoints,
        kpt_conf,
        TORSO_INDICES,
        conf_threshold,
    )
    head_samples = _collect_depth_samples(
        depth_map,
        keypoints,
        kpt_conf,
        HEAD_INDICES,
        conf_threshold,
    )

    valid_roi = roi[roi > 0]
    if valid_roi.size == 0:
        return PersonDistanceEstimate(distance, (x1, y1), None, len(distance_samples))

    if torso_samples:
        body_seed_depth = float(np.median(np.asarray(torso_samples, dtype=np.float32)))
        body_tolerance = max(12.0, body_seed_depth * 0.12)
    else:
        body_seed_depth = float(np.median(valid_roi))
        body_tolerance = max(16.0, body_seed_depth * 0.18)

    roi_float = roi.astype(np.float32)
    body_mask = (roi > 0) & (np.abs(roi_float - body_seed_depth) <= body_tolerance)
    depth_mask = body_mask

    if head_samples:
        head_seed_depth = float(np.median(np.asarray(head_samples, dtype=np.float32)))
        head_tolerance = max(18.0, head_seed_depth * 0.12)
        head_mask = (roi > 0) & (np.abs(roi_float - head_seed_depth) <= head_tolerance)
        depth_mask = body_mask | head_mask

    mask = (depth_mask.astype(np.uint8)) * 255

    # 用关键点构造一个人体区域约束，尽量过滤掉不在骨架附近的干扰区域。
    keypoint_mask = _build_keypoint_person_mask(
        roi.shape[:2],
        keypoints,
        kpt_conf,
        conf_threshold,
        (x1, y1),
    )
    if keypoint_mask is not None:
        mask = cv2.bitwise_and(mask, keypoint_mask)

    close_kernel = np.ones((7, 7), np.uint8)
    open_kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_contour = None
    best_score = -1.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 0:
            continue

        hits = 0
        for idx, conf in enumerate(kpt_conf):
            if conf < conf_threshold:
                continue
            px = float(keypoints[idx][0]) - x1
            py = float(keypoints[idx][1]) - y1
            if cv2.pointPolygonTest(contour, (px, py), False) >= 0:
                hits += 1

        # 关键点覆盖率比纯面积更重要，避免把旁边的大块干扰区域选进来。
        score = hits * 1000.0 + area
        if hits >= 2 and score > best_score:
            best_contour = contour
            best_score = score

    if best_contour is None:
        return PersonDistanceEstimate(distance, (x1, y1), None, len(distance_samples))

    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, [best_contour], -1, 255, thickness=cv2.FILLED)
    contour_pixels = int(np.count_nonzero(contour_mask))
    return PersonDistanceEstimate(distance, (x1, y1), best_contour, contour_pixels)
