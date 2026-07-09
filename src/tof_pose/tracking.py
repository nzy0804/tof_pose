from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MAX_MISSED_FRAMES = 40
MIN_IOU = 0.05
MAX_CENTER_DISTANCE = 110.0
MAX_KEYPOINT_DISTANCE = 65.0
MAX_DISTANCE_GAP = 32.0
MAX_MATCH_COST = 0.85
TORSO_INDICES = (5, 6, 11, 12)

MIN_DT = 1e-3
DEFAULT_DT = 1.0 / 19.0
PROCESS_NOISE = 2.5
SIZE_PROCESS_NOISE = 1.0
DISTANCE_PROCESS_NOISE = 3.0
MEASUREMENT_NOISE = 12.0
SIZE_MEASUREMENT_NOISE = 6.0
DISTANCE_MEASUREMENT_NOISE = 10.0
KEYPOINT_SMOOTH_ALPHA = 0.65
MISSED_CONF_DECAY = 0.92


@dataclass
class Detection:
    """描述当前帧中单个人体检测的几何、关键点和深度特征。"""

    box: np.ndarray
    keypoints: np.ndarray
    kpt_conf: np.ndarray
    distance: float | None


class KalmanFilterBoxDistance:
    """跟踪人体框中心、框尺寸和深度的一阶匀速 Kalman 滤波器。"""

    def __init__(self, box: np.ndarray, distance: float | None) -> None:
        cx, cy, width, height = box_to_measurement(box)
        depth = float(distance) if distance is not None else 0.0

        self.state = np.array(
            [cx, cy, width, height, depth, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        self.covariance = np.eye(10, dtype=np.float32) * 50.0
        self.has_distance = distance is not None

    def predict(self, dt: float) -> None:
        """根据匀速模型外推下一时刻状态。"""
        dt = max(float(dt), MIN_DT)
        transition = np.eye(10, dtype=np.float32)
        for idx in range(5):
            transition[idx, idx + 5] = dt

        process = np.diag(
            [
                PROCESS_NOISE * dt * dt,
                PROCESS_NOISE * dt * dt,
                SIZE_PROCESS_NOISE * dt * dt,
                SIZE_PROCESS_NOISE * dt * dt,
                DISTANCE_PROCESS_NOISE * dt * dt,
                PROCESS_NOISE * dt,
                PROCESS_NOISE * dt,
                SIZE_PROCESS_NOISE * dt,
                SIZE_PROCESS_NOISE * dt,
                DISTANCE_PROCESS_NOISE * dt,
            ]
        ).astype(np.float32)

        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process
        self.state[2] = max(self.state[2], 2.0)
        self.state[3] = max(self.state[3], 2.0)

    def update(self, box: np.ndarray, distance: float | None) -> None:
        """使用当前观测对预测状态做校正。"""
        cx, cy, width, height = box_to_measurement(box)
        measured_depth = float(distance) if distance is not None else float(self.state[4])
        measurement = np.array([cx, cy, width, height, measured_depth], dtype=np.float32)

        observation = np.zeros((5, 10), dtype=np.float32)
        observation[0, 0] = 1.0
        observation[1, 1] = 1.0
        observation[2, 2] = 1.0
        observation[3, 3] = 1.0
        observation[4, 4] = 1.0

        depth_noise = DISTANCE_MEASUREMENT_NOISE * (6.0 if distance is None else 1.0)
        if distance is not None:
            self.has_distance = True

        measurement_cov = np.diag(
            [
                MEASUREMENT_NOISE,
                MEASUREMENT_NOISE,
                SIZE_MEASUREMENT_NOISE,
                SIZE_MEASUREMENT_NOISE,
                depth_noise,
            ]
        ).astype(np.float32)

        innovation = measurement - observation @ self.state
        innovation_cov = observation @ self.covariance @ observation.T + measurement_cov
        kalman_gain = self.covariance @ observation.T @ np.linalg.inv(innovation_cov)

        self.state = self.state + kalman_gain @ innovation
        identity = np.eye(10, dtype=np.float32)
        self.covariance = (identity - kalman_gain @ observation) @ self.covariance
        self.state[2] = max(self.state[2], 2.0)
        self.state[3] = max(self.state[3], 2.0)

    def predicted_box(self) -> np.ndarray:
        """读取当前滤波器估计框。"""
        return measurement_to_box(self.state[:4])

    def predicted_distance(self) -> float | None:
        """读取当前滤波器估计深度。"""
        if not self.has_distance:
            return None
        return float(self.state[4])

    def copy(self) -> "KalmanFilterBoxDistance":
        copied = object.__new__(KalmanFilterBoxDistance)
        copied.state = self.state.copy()
        copied.covariance = self.covariance.copy()
        copied.has_distance = bool(self.has_distance)
        return copied


@dataclass
class Track:
    """保存一条人体轨迹的滤波状态与最近观测。"""

    track_id: int
    kf: KalmanFilterBoxDistance
    box: np.ndarray
    keypoints: np.ndarray
    kpt_conf: np.ndarray
    smoothed_keypoints: np.ndarray
    smoothed_kpt_conf: np.ndarray
    distance: float | None
    predicted_box: np.ndarray
    predicted_distance: float | None
    missed_frames: int = 0
    last_timestamp: float | None = None
    hits: int = 1

    def copy(self) -> "Track":
        return Track(
            track_id=int(self.track_id),
            kf=self.kf.copy(),
            box=self.box.copy(),
            keypoints=self.keypoints.copy(),
            kpt_conf=self.kpt_conf.copy(),
            smoothed_keypoints=self.smoothed_keypoints.copy(),
            smoothed_kpt_conf=self.smoothed_kpt_conf.copy(),
            distance=self.distance,
            predicted_box=self.predicted_box.copy(),
            predicted_distance=self.predicted_distance,
            missed_frames=int(self.missed_frames),
            last_timestamp=self.last_timestamp,
            hits=int(self.hits),
        )


def box_to_measurement(box: np.ndarray) -> tuple[float, float, float, float]:
    """将检测框转换为 [cx, cy, width, height] 形式。"""
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
        max(2.0, x2 - x1),
        max(2.0, y2 - y1),
    )


def measurement_to_box(measurement: np.ndarray) -> np.ndarray:
    """将 [cx, cy, width, height] 转回 [x1, y1, x2, y2]。"""
    cx, cy, width, height = [float(v) for v in measurement[:4]]
    half_w = max(width, 2.0) / 2.0
    half_h = max(height, 2.0) / 2.0
    return np.array(
        [cx - half_w, cy - half_h, cx + half_w, cy + half_h],
        dtype=np.float32,
    )


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """计算两个检测框的 IoU。"""
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def compute_center_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """计算两个检测框中心点之间的欧氏距离。"""
    center_a = np.array([(box_a[0] + box_a[2]) / 2.0, (box_a[1] + box_a[3]) / 2.0])
    center_b = np.array([(box_b[0] + box_b[2]) / 2.0, (box_b[1] + box_b[3]) / 2.0])
    return float(np.linalg.norm(center_a - center_b))


def compute_keypoint_distance(
    keypoints_a: np.ndarray,
    conf_a: np.ndarray,
    keypoints_b: np.ndarray,
    conf_b: np.ndarray,
) -> float | None:
    """计算两个人体在躯干关键点上的平均位移差。"""
    distances: list[float] = []
    for idx in TORSO_INDICES:
        if idx >= len(conf_a) or idx >= len(conf_b):
            continue
        if conf_a[idx] < 0.1 or conf_b[idx] < 0.1:
            continue
        distances.append(float(np.linalg.norm(keypoints_a[idx] - keypoints_b[idx])))

    if len(distances) < 2:
        return None
    return float(np.mean(distances))


def compute_distance_gap(distance_a: float | None, distance_b: float | None) -> float | None:
    """计算两个人体距离估计值的差异。"""
    if distance_a is None or distance_b is None:
        return None
    return abs(distance_a - distance_b)


class PersonTracker:
    """基于 Kalman 滤波的多人追踪器。"""

    def __init__(
        self,
        max_missed_frames: int = MAX_MISSED_FRAMES,
        min_iou: float = MIN_IOU,
        max_center_distance: float = MAX_CENTER_DISTANCE,
        max_keypoint_distance: float = MAX_KEYPOINT_DISTANCE,
        max_distance_gap: float = MAX_DISTANCE_GAP,
        max_match_cost: float = MAX_MATCH_COST,
    ) -> None:
        self.max_missed_frames = max_missed_frames
        self.min_iou = min_iou
        self.max_center_distance = max_center_distance
        self.max_keypoint_distance = max_keypoint_distance
        self.max_distance_gap = max_distance_gap
        self.max_match_cost = max_match_cost
        self.next_track_id = 1
        self.tracks: list[Track] = []

    def copy(self) -> "PersonTracker":
        copied = PersonTracker(
            max_missed_frames=self.max_missed_frames,
            min_iou=self.min_iou,
            max_center_distance=self.max_center_distance,
            max_keypoint_distance=self.max_keypoint_distance,
            max_distance_gap=self.max_distance_gap,
            max_match_cost=self.max_match_cost,
        )
        copied.next_track_id = int(self.next_track_id)
        copied.tracks = [track.copy() for track in self.tracks]
        return copied

    def update(self, detections: list[Detection], timestamp: float | None = None) -> list[int]:
        """用当前帧人体观测更新追踪器状态，并返回稳定的人体 ID。"""
        current_time = timestamp
        self._predict_tracks(current_time)

        if len(detections) == 0:
            self._mark_all_missed(current_time)
            self._prune_tracks()
            return []

        assignments = [-1] * len(detections)
        unmatched_detection_indices = set(range(len(detections)))
        unmatched_track_indices = set(range(len(self.tracks)))
        candidates: list[tuple[float, int, int]] = []

        for track_idx, track in enumerate(self.tracks):
            for det_idx, detection in enumerate(detections):
                iou = compute_iou(track.predicted_box, detection.box)
                center_distance = compute_center_distance(track.predicted_box, detection.box)
                keypoint_distance = compute_keypoint_distance(
                    track.smoothed_keypoints,
                    track.smoothed_kpt_conf,
                    detection.keypoints,
                    detection.kpt_conf,
                )
                distance_gap = compute_distance_gap(track.predicted_distance, detection.distance)

                if iou < self.min_iou and center_distance > self.max_center_distance:
                    continue
                if keypoint_distance is not None and keypoint_distance > self.max_keypoint_distance:
                    continue
                if distance_gap is not None and distance_gap > self.max_distance_gap:
                    continue

                center_score = max(0.0, 1.0 - center_distance / max(self.max_center_distance, 1.0))
                keypoint_score = 0.5
                if keypoint_distance is not None:
                    keypoint_score = max(
                        0.0,
                        1.0 - keypoint_distance / max(self.max_keypoint_distance, 1.0),
                    )
                depth_score = 0.5
                if distance_gap is not None:
                    depth_score = max(0.0, 1.0 - distance_gap / max(self.max_distance_gap, 1.0))

                score = 0.50 * iou + 0.20 * center_score + 0.15 * keypoint_score + 0.15 * depth_score
                cost = 1.0 - score
                if cost <= self.max_match_cost:
                    candidates.append((cost, track_idx, det_idx))

        candidates.sort(key=lambda item: item[0])
        for _, track_idx, det_idx in candidates:
            if track_idx not in unmatched_track_indices or det_idx not in unmatched_detection_indices:
                continue

            detection = detections[det_idx]
            track = self.tracks[track_idx]
            track.kf.update(detection.box, detection.distance)
            track.box = detection.box.copy()
            track.keypoints = detection.keypoints.copy()
            track.kpt_conf = detection.kpt_conf.copy()
            track.smoothed_keypoints, track.smoothed_kpt_conf = smooth_keypoints(
                track.smoothed_keypoints,
                track.smoothed_kpt_conf,
                detection.keypoints,
                detection.kpt_conf,
            )
            track.distance = detection.distance
            track.predicted_box = track.kf.predicted_box()
            track.predicted_distance = track.kf.predicted_distance()
            track.missed_frames = 0
            track.last_timestamp = current_time
            track.hits += 1
            assignments[det_idx] = track.track_id
            unmatched_track_indices.remove(track_idx)
            unmatched_detection_indices.remove(det_idx)

        for track_idx in unmatched_track_indices:
            track = self.tracks[track_idx]
            track.missed_frames += 1
            track.box = track.predicted_box.copy()
            track.smoothed_kpt_conf = np.clip(
                track.smoothed_kpt_conf * MISSED_CONF_DECAY,
                0.0,
                1.0,
            ).astype(np.float32)
            track.distance = track.predicted_distance
            track.last_timestamp = current_time

        for det_idx in unmatched_detection_indices:
            detection = detections[det_idx]
            kf = KalmanFilterBoxDistance(detection.box, detection.distance)
            new_track = Track(
                track_id=self.next_track_id,
                kf=kf,
                box=detection.box.copy(),
                keypoints=detection.keypoints.copy(),
                kpt_conf=detection.kpt_conf.copy(),
                smoothed_keypoints=detection.keypoints.copy(),
                smoothed_kpt_conf=detection.kpt_conf.copy(),
                distance=detection.distance,
                predicted_box=kf.predicted_box(),
                predicted_distance=kf.predicted_distance(),
                last_timestamp=current_time,
            )
            self.next_track_id += 1
            self.tracks.append(new_track)
            assignments[det_idx] = new_track.track_id

        self._prune_tracks()
        return assignments

    def _predict_tracks(self, current_time: float | None) -> None:
        """在当前帧匹配前，先对所有轨迹做一次状态预测。"""
        for track in self.tracks:
            if current_time is None or track.last_timestamp is None:
                dt = DEFAULT_DT
            else:
                dt = max(current_time - track.last_timestamp, MIN_DT)
            track.kf.predict(dt)
            track.predicted_box = track.kf.predicted_box()
            track.predicted_distance = track.kf.predicted_distance()

    def _mark_all_missed(self, current_time: float | None) -> None:
        """在当前帧没有检测结果时，统一增加所有轨迹的丢失计数。"""
        for track in self.tracks:
            track.missed_frames += 1
            track.box = track.predicted_box.copy()
            track.smoothed_kpt_conf = np.clip(
                track.smoothed_kpt_conf * MISSED_CONF_DECAY,
                0.0,
                1.0,
            ).astype(np.float32)
            track.distance = track.predicted_distance
            track.last_timestamp = current_time

    def _prune_tracks(self) -> None:
        """删除长时间未匹配到检测框的轨迹。"""
        self.tracks = [
            track
            for track in self.tracks
            if track.missed_frames <= self.max_missed_frames
        ]

    def get_track(self, track_id: int) -> Track | None:
        """按 track id 读取轨迹。"""
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None

    def visible_tracks(
        self,
        max_missed_frames: int | None = None,
        min_hits: int = 1,
    ) -> list[Track]:
        """返回当前仍可用于显示与统计的轨迹。"""
        limit = self.max_missed_frames if max_missed_frames is None else max_missed_frames
        return [
            track
            for track in self.tracks
            if track.hits >= min_hits and track.missed_frames <= limit
        ]


def smooth_keypoints(
    previous_keypoints: np.ndarray,
    previous_conf: np.ndarray,
    current_keypoints: np.ndarray,
    current_conf: np.ndarray,
    alpha: float = KEYPOINT_SMOOTH_ALPHA,
) -> tuple[np.ndarray, np.ndarray]:
    """用当前观测与历史状态做关键点平滑。"""
    previous_keypoints = previous_keypoints.astype(np.float32, copy=False)
    previous_conf = previous_conf.astype(np.float32, copy=False)
    current_keypoints = current_keypoints.astype(np.float32, copy=False)
    current_conf = current_conf.astype(np.float32, copy=False)

    fused_keypoints = previous_keypoints.copy()
    fused_conf = previous_conf.copy()
    point_count = min(len(previous_keypoints), len(current_keypoints))
    for idx in range(point_count):
        prev_ok = idx < len(previous_conf) and previous_conf[idx] > 0.01
        curr_ok = idx < len(current_conf) and current_conf[idx] > 0.01

        if curr_ok and prev_ok:
            fused_keypoints[idx] = (
                alpha * current_keypoints[idx] + (1.0 - alpha) * previous_keypoints[idx]
            )
            fused_conf[idx] = max(float(current_conf[idx]), float(previous_conf[idx]) * 0.9)
        elif curr_ok:
            fused_keypoints[idx] = current_keypoints[idx]
            fused_conf[idx] = current_conf[idx]
        elif prev_ok:
            fused_keypoints[idx] = previous_keypoints[idx]
            fused_conf[idx] = previous_conf[idx] * MISSED_CONF_DECAY
        else:
            fused_conf[idx] = 0.0

    return fused_keypoints.astype(np.float32), np.clip(fused_conf, 0.0, 1.0).astype(np.float32)
