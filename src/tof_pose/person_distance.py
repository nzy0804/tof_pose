from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


MIN_VALID_PIXELS = 24


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
    x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def estimate_person_distance(
    depth_map: np.ndarray,
    box: np.ndarray,
    keypoints: np.ndarray,
    kpt_conf: np.ndarray,
    conf_threshold: float,
) -> PersonDistanceEstimate:
    height, width = depth_map.shape[:2]
    clipped = _clip_box(box, width, height)
    if clipped is None:
        return PersonDistanceEstimate(None, (0, 0), None, 0)

    x1, y1, x2, y2 = clipped
    roi = depth_map[y1:y2, x1:x2]
    if roi.size == 0:
        return PersonDistanceEstimate(None, (x1, y1), None, 0)

    # 优先使用躯干关键点估计人物主体深度，减少背景像素干扰。
    torso_indices = (5, 6, 11, 12)
    torso_samples: list[float] = []
    for idx in torso_indices:
        if idx >= len(kpt_conf) or kpt_conf[idx] < conf_threshold:
            continue
        px = int(round(keypoints[idx][0]))
        py = int(round(keypoints[idx][1]))
        if 0 <= px < width and 0 <= py < height:
            value = float(depth_map[py, px])
            if value > 0:
                torso_samples.append(value)

    if torso_samples:
        seed_depth = float(np.median(np.asarray(torso_samples, dtype=np.float32)))
        tolerance = max(12.0, seed_depth * 0.12)
    else:
        valid_roi = roi[roi > 0]
        if valid_roi.size == 0:
            return PersonDistanceEstimate(None, (x1, y1), None, 0)
        seed_depth = float(np.median(valid_roi))
        tolerance = max(16.0, seed_depth * 0.18)

    depth_mask = (roi > 0) & (np.abs(roi.astype(np.float32) - seed_depth) <= tolerance)
    mask = (depth_mask.astype(np.uint8)) * 255

    # 通过形态学操作把人体主体区域连成稳定轮廓。
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_contour = None
    best_area = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= best_area:
            continue

        # 优先保留覆盖更多关键点的轮廓，避免把背景团块误认为人体。
        hits = 0
        for idx, conf in enumerate(kpt_conf):
            if conf < conf_threshold:
                continue
            px = float(keypoints[idx][0]) - x1
            py = float(keypoints[idx][1]) - y1
            if cv2.pointPolygonTest(contour, (px, py), False) >= 0:
                hits += 1
        if hits >= 2 or best_contour is None:
            best_contour = contour
            best_area = area

    if best_contour is None:
        valid_roi = roi[mask > 0]
        if valid_roi.size < MIN_VALID_PIXELS:
            return PersonDistanceEstimate(None, (x1, y1), None, int(valid_roi.size))
        distance = float(np.median(valid_roi))
        return PersonDistanceEstimate(distance, (x1, y1), None, int(valid_roi.size))

    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, [best_contour], -1, 255, thickness=cv2.FILLED)
    person_pixels = roi[contour_mask > 0]
    person_pixels = person_pixels[person_pixels > 0]
    if person_pixels.size < MIN_VALID_PIXELS:
        return PersonDistanceEstimate(None, (x1, y1), best_contour, int(person_pixels.size))

    distance = float(np.median(person_pixels))
    return PersonDistanceEstimate(distance, (x1, y1), best_contour, int(person_pixels.size))
