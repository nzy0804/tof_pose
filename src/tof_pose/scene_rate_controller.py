from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from tof_pose.input_image import decode_received_grayscale


LOGGER = logging.getLogger(__name__)

SCENE_MODE_OFF = "off"
SCENE_MODE_SHADOW = "shadow"
SCENE_MODE_ENFORCE = "enforce"
SCENE_MODES = (SCENE_MODE_OFF, SCENE_MODE_SHADOW, SCENE_MODE_ENFORCE)

SCENE_STATE_UNKNOWN = "unknown"
SCENE_STATE_EMPTY = "empty"
SCENE_STATE_STATIC = "static"
SCENE_STATE_ACTIVE = "active"

DECISION_FULL = "full"
DECISION_STATIC_REUSE = "static_reuse"
DECISION_EMPTY_SKIP = "empty_skip"


@dataclass(frozen=True)
class SceneRateConfig:
    mode: str = SCENE_MODE_OFF
    empty_hz: float = 1.0
    static_hz: float = 5.0
    active_hz: float = 10.0
    motion_pixel_threshold: int = 12
    motion_weak_ratio: float = 0.015
    motion_strong_ratio: float = 0.05
    motion_weak_frames: int = 2
    active_min_ms: int = 3000
    static_confirm_ms: int = 3000
    empty_confirm_count: int = 3
    empty_confirm_ms: int = 1000
    state_stale_ms: int = 6000
    state_ttl_sec: float = 300.0
    min_track_confidence: float = 0.35
    max_mask_area_change_ratio: float = 0.10
    max_mask_center_shift_ratio: float = 0.03
    max_keypoint_speed_ratio: float = 0.12
    keypoint_confidence: float = 0.35


@dataclass(frozen=True)
class SceneFrameMetric:
    input_index: int
    capture_timestamp_ms: int
    motion_ratio: float
    motion_energy: float
    motion_level: str


@dataclass(frozen=True)
class SceneBatchPlan:
    device_id: str
    sequence_id: int
    frame_ids: tuple[str, ...]
    capture_timestamps_ms: tuple[int, ...]
    model_indices: tuple[int, ...]
    scheduled_indices: tuple[int, ...]
    metrics: tuple[SceneFrameMetric, ...]
    mode: str
    state_before: str
    state_after_plan: str
    state_version: int

    @property
    def input_count(self) -> int:
        return len(self.frame_ids)

    @property
    def model_input_count(self) -> int:
        return len(self.model_indices)

    @property
    def scheduled_input_count(self) -> int:
        return len(self.scheduled_indices)


@dataclass
class _DeviceSceneState:
    scene_state: str = SCENE_STATE_UNKNOWN
    state_since_capture_ms: int = 0
    last_frame_capture_ms: int = 0
    last_model_capture_ms: int = 0
    last_planned_sequence_id: int | None = None
    last_result_sequence_id: int | None = None
    stable_since_capture_ms: int = 0
    empty_since_capture_ms: int = 0
    active_until_capture_ms: int = 0
    next_model_capture_ms: int = 0
    weak_motion_count: int = 0
    consecutive_empty_results: int = 0
    previous_gray: np.ndarray | None = field(default=None, repr=False)
    previous_observation: dict | None = field(default=None, repr=False)
    previous_observation_capture_ms: int = 0
    last_access_monotonic: float = field(default_factory=time.monotonic)
    version: int = 0


class SceneRateController:
    def __init__(self, config: SceneRateConfig | None = None) -> None:
        self.config = config or SceneRateConfig()
        mode = str(self.config.mode or SCENE_MODE_OFF).strip().lower()
        if mode not in SCENE_MODES:
            raise ValueError(f"scene rate mode must be one of: {', '.join(SCENE_MODES)}")
        self._mode = mode
        for name in ("empty_hz", "static_hz", "active_hz"):
            if float(getattr(self.config, name)) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if int(self.config.motion_pixel_threshold) < 0:
            raise ValueError("motion_pixel_threshold must be non-negative")
        weak_ratio = float(self.config.motion_weak_ratio)
        strong_ratio = float(self.config.motion_strong_ratio)
        if not 0.0 <= weak_ratio <= strong_ratio <= 1.0:
            raise ValueError(
                "motion ratios must satisfy 0 <= weak_ratio <= strong_ratio <= 1"
            )
        if int(self.config.motion_weak_frames) <= 0:
            raise ValueError("motion_weak_frames must be greater than zero")
        for name in (
            "active_min_ms",
            "static_confirm_ms",
            "empty_confirm_ms",
            "state_stale_ms",
        ):
            if int(getattr(self.config, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if int(self.config.empty_confirm_count) <= 0:
            raise ValueError("empty_confirm_count must be greater than zero")
        if float(self.config.state_ttl_sec) <= 0:
            raise ValueError("state_ttl_sec must be greater than zero")
        for name in ("min_track_confidence", "keypoint_confidence"):
            value = float(getattr(self.config, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        for name in (
            "max_mask_area_change_ratio",
            "max_mask_center_shift_ratio",
            "max_keypoint_speed_ratio",
        ):
            if float(getattr(self.config, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        self._states: dict[str, _DeviceSceneState] = {}
        self._states_lock = threading.Lock()
        self._device_locks = [threading.RLock() for _ in range(64)]
        self._next_cleanup_monotonic = 0.0

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def enabled(self) -> bool:
        return self._mode != SCENE_MODE_OFF

    def _device_lock(self, device_id: str) -> threading.RLock:
        return self._device_locks[hash(device_id) % len(self._device_locks)]

    def _get_state(self, device_id: str) -> _DeviceSceneState:
        with self._states_lock:
            state = self._states.get(device_id)
            if state is None:
                state = _DeviceSceneState()
                self._states[device_id] = state
            state.last_access_monotonic = time.monotonic()
            return state

    def _prune_stale_states(self, now: float) -> None:
        if now < self._next_cleanup_monotonic:
            return
        ttl_sec = max(1.0, float(self.config.state_ttl_sec))
        with self._states_lock:
            stale = [
                device_id
                for device_id, state in self._states.items()
                if now - float(state.last_access_monotonic) > ttl_sec
            ]
            for device_id in stale:
                self._states.pop(device_id, None)
            self._next_cleanup_monotonic = now + min(60.0, ttl_sec)
        if stale:
            LOGGER.info(
                "Adaptive scene state pruned: devices=%d ttl_sec=%.1f",
                len(stale),
                ttl_sec,
            )

    @staticmethod
    def _capture_timestamp_ms(value) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _decode_gray(image_data: bytes) -> np.ndarray:
        image = decode_received_grayscale(image_data)
        if image.dtype != np.uint8:
            image_float = image.astype(np.float32, copy=False)
            min_value = float(np.min(image_float)) if image_float.size else 0.0
            max_value = float(np.max(image_float)) if image_float.size else 0.0
            if max_value <= min_value:
                image = np.zeros(image.shape, dtype=np.uint8)
            else:
                image = cv2.normalize(
                    image_float,
                    None,
                    0,
                    255,
                    cv2.NORM_MINMAX,
                ).astype(np.uint8)
        if image.shape[:2] != (100, 100):
            image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(image)

    def _motion_metric(
        self,
        previous: np.ndarray | None,
        current: np.ndarray,
        input_index: int,
        capture_timestamp_ms: int,
    ) -> SceneFrameMetric:
        if previous is None or previous.shape != current.shape:
            return SceneFrameMetric(
                input_index=input_index,
                capture_timestamp_ms=capture_timestamp_ms,
                motion_ratio=0.0,
                motion_energy=0.0,
                motion_level="unknown",
            )

        delta = current.astype(np.int16) - previous.astype(np.int16)
        global_shift = int(np.median(delta))
        residual = np.abs(delta - global_shift)
        pixel_threshold = max(1, int(self.config.motion_pixel_threshold))
        motion_ratio = float(np.mean(residual >= pixel_threshold))
        motion_energy = float(np.mean(np.minimum(residual, 255))) / 255.0
        if motion_ratio >= max(0.0, float(self.config.motion_strong_ratio)):
            motion_level = "strong"
        elif motion_ratio >= max(0.0, float(self.config.motion_weak_ratio)):
            motion_level = "weak"
        else:
            motion_level = "low"
        return SceneFrameMetric(
            input_index=input_index,
            capture_timestamp_ms=capture_timestamp_ms,
            motion_ratio=motion_ratio,
            motion_energy=motion_energy,
            motion_level=motion_level,
        )

    def _transition(
        self,
        device_id: str,
        sequence_id: int,
        state: _DeviceSceneState,
        new_state: str,
        capture_timestamp_ms: int,
        reason: str,
    ) -> None:
        old_state = state.scene_state
        if old_state == new_state:
            return
        state.scene_state = new_state
        state.state_since_capture_ms = max(0, int(capture_timestamp_ms))
        state.version += 1
        LOGGER.info(
            (
                "Adaptive scene transition: device_id=%s sequence_id=%d "
                "from=%s to=%s capture_timestamp_ms=%d reason=%s version=%d"
            ),
            device_id,
            sequence_id,
            old_state,
            new_state,
            capture_timestamp_ms,
            reason,
            state.version,
        )

    def _activate(
        self,
        device_id: str,
        sequence_id: int,
        state: _DeviceSceneState,
        capture_timestamp_ms: int,
        reason: str,
    ) -> None:
        was_active = state.scene_state == SCENE_STATE_ACTIVE
        self._transition(
            device_id,
            sequence_id,
            state,
            SCENE_STATE_ACTIVE,
            capture_timestamp_ms,
            reason,
        )
        if was_active:
            state.version += 1
        state.active_until_capture_ms = max(
            state.active_until_capture_ms,
            capture_timestamp_ms + max(0, int(self.config.active_min_ms)),
        )
        state.stable_since_capture_ms = 0
        state.next_model_capture_ms = 0

    def _reset_stale_state(
        self,
        device_id: str,
        sequence_id: int,
        state: _DeviceSceneState,
        capture_timestamp_ms: int,
        reason: str,
    ) -> None:
        state.scene_state = SCENE_STATE_UNKNOWN
        state.state_since_capture_ms = capture_timestamp_ms
        state.stable_since_capture_ms = 0
        state.empty_since_capture_ms = 0
        state.active_until_capture_ms = 0
        state.next_model_capture_ms = 0
        state.weak_motion_count = 0
        state.consecutive_empty_results = 0
        state.previous_gray = None
        state.previous_observation = None
        state.previous_observation_capture_ms = 0
        state.version += 1
        LOGGER.info(
            (
                "Adaptive scene state reset: device_id=%s sequence_id=%d "
                "capture_timestamp_ms=%d reason=%s version=%d"
            ),
            device_id,
            sequence_id,
            capture_timestamp_ms,
            reason,
            state.version,
        )

    def _period_ms(self, state_name: str) -> int:
        if state_name == SCENE_STATE_EMPTY:
            hz = max(0.01, float(self.config.empty_hz))
        elif state_name == SCENE_STATE_STATIC:
            hz = max(0.01, float(self.config.static_hz))
        else:
            hz = max(0.01, float(self.config.active_hz))
        return max(1, int(round(1000.0 / hz)))

    @staticmethod
    def _advance_due_timestamp(_current_due_ms: int, timestamp_ms: int, period_ms: int) -> int:
        return timestamp_ms + period_ms

    def plan_batch(
        self,
        *,
        device_id: str,
        sequence_id: int,
        frames: list[tuple[str, bytes]],
        capture_timestamps_ms: list[int],
    ) -> SceneBatchPlan:
        frame_ids = tuple(str(frame_id or "") for frame_id, _ in frames)
        timestamps = tuple(
            self._capture_timestamp_ms(value) for value in capture_timestamps_ms
        )
        if len(frames) != len(timestamps):
            raise ValueError("adaptive scene frames and timestamps must have equal length")
        all_indices = tuple(range(len(frames)))
        if self._mode == SCENE_MODE_OFF or not frames:
            return SceneBatchPlan(
                device_id=device_id,
                sequence_id=sequence_id,
                frame_ids=frame_ids,
                capture_timestamps_ms=timestamps,
                model_indices=all_indices,
                scheduled_indices=all_indices,
                metrics=tuple(),
                mode=self._mode,
                state_before=SCENE_STATE_ACTIVE,
                state_after_plan=SCENE_STATE_ACTIVE,
                state_version=0,
            )

        now = time.monotonic()
        self._prune_stale_states(now)
        state = self._get_state(device_id)
        with self._device_lock(device_id):
            state.last_access_monotonic = now
            state_before = state.scene_state
            scheduled_indices: list[int] = []
            metrics: list[SceneFrameMetric] = []
            force_all = False

            if (
                state.last_planned_sequence_id is not None
                and sequence_id <= state.last_planned_sequence_id
            ):
                force_all = True
                self._activate(
                    device_id,
                    sequence_id,
                    state,
                    timestamps[0] if timestamps else 0,
                    "non_increasing_sequence",
                )

            for input_index, ((_, image_data), timestamp_ms) in enumerate(
                zip(frames, timestamps)
            ):
                if timestamp_ms <= 0:
                    force_all = True
                    self._activate(
                        device_id,
                        sequence_id,
                        state,
                        state.last_frame_capture_ms,
                        "invalid_capture_timestamp",
                    )
                elif state.last_frame_capture_ms > 0:
                    gap_ms = timestamp_ms - state.last_frame_capture_ms
                    if gap_ms <= 0:
                        force_all = True
                        self._reset_stale_state(
                            device_id,
                            sequence_id,
                            state,
                            timestamp_ms,
                            "non_increasing_capture_timestamp",
                        )
                    elif gap_ms > max(1, int(self.config.state_stale_ms)):
                        self._reset_stale_state(
                            device_id,
                            sequence_id,
                            state,
                            timestamp_ms,
                            "capture_gap",
                        )

                try:
                    current_gray = self._decode_gray(image_data)
                    metric = self._motion_metric(
                        state.previous_gray,
                        current_gray,
                        input_index,
                        timestamp_ms,
                    )
                    state.previous_gray = current_gray
                except Exception:
                    LOGGER.exception(
                        (
                            "Adaptive scene frame analysis failed: "
                            "device_id=%s sequence_id=%d input_index=%d"
                        ),
                        device_id,
                        sequence_id,
                        input_index,
                    )
                    force_all = True
                    metric = SceneFrameMetric(
                        input_index=input_index,
                        capture_timestamp_ms=timestamp_ms,
                        motion_ratio=1.0,
                        motion_energy=1.0,
                        motion_level="error",
                    )

                metrics.append(metric)
                if metric.motion_level == "strong":
                    state.weak_motion_count = 0
                    self._activate(
                        device_id,
                        sequence_id,
                        state,
                        timestamp_ms,
                        "strong_frame_motion",
                    )
                elif metric.motion_level == "weak":
                    state.weak_motion_count += 1
                    if state.weak_motion_count >= max(
                        1,
                        int(self.config.motion_weak_frames),
                    ):
                        self._activate(
                            device_id,
                            sequence_id,
                            state,
                            timestamp_ms,
                            "repeated_frame_motion",
                        )
                else:
                    state.weak_motion_count = 0

                if state.scene_state == SCENE_STATE_UNKNOWN:
                    self._activate(
                        device_id,
                        sequence_id,
                        state,
                        timestamp_ms,
                        "unknown_state",
                    )

                if force_all or state.scene_state == SCENE_STATE_ACTIVE:
                    scheduled_indices.append(input_index)
                else:
                    period_ms = self._period_ms(state.scene_state)
                    if (
                        state.next_model_capture_ms <= 0
                        or timestamp_ms >= state.next_model_capture_ms
                    ):
                        scheduled_indices.append(input_index)
                        state.next_model_capture_ms = self._advance_due_timestamp(
                            state.next_model_capture_ms,
                            timestamp_ms,
                            period_ms,
                        )

                if timestamp_ms > 0:
                    state.last_frame_capture_ms = timestamp_ms

            if not scheduled_indices and frames:
                scheduled_indices.append(len(frames) - 1)
            state.last_planned_sequence_id = sequence_id
            state_version = state.version
            scheduled = tuple(dict.fromkeys(scheduled_indices))
            model_indices = (
                all_indices if self._mode == SCENE_MODE_SHADOW else scheduled
            )
            return SceneBatchPlan(
                device_id=device_id,
                sequence_id=sequence_id,
                frame_ids=frame_ids,
                capture_timestamps_ms=timestamps,
                model_indices=model_indices,
                scheduled_indices=scheduled,
                metrics=tuple(metrics),
                mode=self._mode,
                state_before=state_before,
                state_after_plan=state.scene_state,
                state_version=state_version,
            )

    def _observations_stable(
        self,
        previous: dict | None,
        current: dict,
        elapsed_ms: int,
    ) -> tuple[bool, str]:
        if not previous:
            return False, "missing_previous_observation"
        previous_count = max(0, int(previous.get("person_count", 0) or 0))
        current_count = max(0, int(current.get("person_count", 0) or 0))
        if previous_count != current_count:
            return False, "person_count_changed"
        if current_count <= 0:
            return False, "no_person"

        def track_map(observation: dict) -> dict[int, dict]:
            tracks: dict[int, dict] = {}
            for item in observation.get("tracks") or []:
                try:
                    track_id = int(item.get("track_id", -1))
                except (AttributeError, TypeError, ValueError):
                    continue
                if track_id >= 0:
                    tracks[track_id] = item
            return tracks

        previous_tracks = track_map(previous)
        current_tracks = track_map(current)
        if (
            len(previous_tracks) != current_count
            or len(current_tracks) != current_count
            or set(previous_tracks) != set(current_tracks)
        ):
            return False, "track_ids_changed"

        elapsed_sec = max(0.001, float(elapsed_ms) / 1000.0)
        for track_id in sorted(current_tracks):
            previous_track = previous_tracks[track_id]
            current_track = current_tracks[track_id]
            confidence = float(current_track.get("confidence", 0.0) or 0.0)
            if confidence < float(self.config.min_track_confidence):
                return False, "track_confidence_low"

            previous_area = max(
                1e-6,
                float(previous_track.get("mask_area_ratio", 0.0) or 0.0),
            )
            current_area = max(
                0.0,
                float(current_track.get("mask_area_ratio", 0.0) or 0.0),
            )
            area_change = abs(current_area - previous_area) / previous_area
            if area_change > float(self.config.max_mask_area_change_ratio):
                return False, "mask_area_changed"

            previous_center = previous_track.get("mask_center") or (0.0, 0.0)
            current_center = current_track.get("mask_center") or (0.0, 0.0)
            center_shift = math.hypot(
                float(current_center[0]) - float(previous_center[0]),
                float(current_center[1]) - float(previous_center[1]),
            )
            if center_shift > float(self.config.max_mask_center_shift_ratio):
                return False, "mask_center_changed"

            previous_points = previous_track.get("keypoints") or []
            current_points = current_track.get("keypoints") or []
            point_speeds: list[float] = []
            for previous_point, current_point in zip(
                previous_points,
                current_points,
            ):
                if (
                    len(previous_point) < 3
                    or len(current_point) < 3
                    or float(previous_point[2]) < float(self.config.keypoint_confidence)
                    or float(current_point[2]) < float(self.config.keypoint_confidence)
                ):
                    continue
                distance = math.hypot(
                    float(current_point[0]) - float(previous_point[0]),
                    float(current_point[1]) - float(previous_point[1]),
                )
                point_speeds.append(distance / elapsed_sec)
            if point_speeds:
                point_speeds.sort()
                median_speed = point_speeds[len(point_speeds) // 2]
                if median_speed > float(self.config.max_keypoint_speed_ratio):
                    return False, "keypoint_speed_high"

        return True, "stable"

    def _update_state_from_results(
        self,
        plan: SceneBatchPlan,
        results_by_input: dict[int, dict],
    ) -> None:
        if self._mode == SCENE_MODE_OFF:
            return
        state = self._get_state(plan.device_id)
        metric_by_input = {metric.input_index: metric for metric in plan.metrics}
        with self._device_lock(plan.device_id):
            state.last_access_monotonic = time.monotonic()
            if (
                state.last_result_sequence_id is not None
                and plan.sequence_id <= state.last_result_sequence_id
            ):
                return
            for input_index in plan.scheduled_indices:
                result = results_by_input.get(input_index)
                if result is None:
                    continue
                timestamp_ms = plan.capture_timestamps_ms[input_index]
                observation = result.get("_scene_observation")
                if not isinstance(observation, dict):
                    self._activate(
                        plan.device_id,
                        plan.sequence_id,
                        state,
                        timestamp_ms,
                        "missing_scene_observation",
                    )
                    continue

                metric = metric_by_input.get(input_index)
                motion_low = metric is not None and metric.motion_level in {
                    "low",
                    "unknown",
                }
                person_count = max(
                    0,
                    int(observation.get("person_count", 0) or 0),
                )
                if person_count <= 0:
                    if state.consecutive_empty_results <= 0:
                        state.empty_since_capture_ms = timestamp_ms
                    state.consecutive_empty_results += 1
                    state.stable_since_capture_ms = 0
                    if (
                        state.version == plan.state_version
                        and motion_low
                        and state.consecutive_empty_results
                        >= max(1, int(self.config.empty_confirm_count))
                        and timestamp_ms - state.empty_since_capture_ms
                        >= max(0, int(self.config.empty_confirm_ms))
                    ):
                        self._transition(
                            plan.device_id,
                            plan.sequence_id,
                            state,
                            SCENE_STATE_EMPTY,
                            timestamp_ms,
                            "confirmed_empty",
                        )
                        state.next_model_capture_ms = (
                            timestamp_ms + self._period_ms(SCENE_STATE_EMPTY)
                        )
                else:
                    state.consecutive_empty_results = 0
                    state.empty_since_capture_ms = 0
                    elapsed_ms = (
                        timestamp_ms - state.previous_observation_capture_ms
                        if state.previous_observation_capture_ms > 0
                        else 0
                    )
                    stable, reason = self._observations_stable(
                        state.previous_observation,
                        observation,
                        elapsed_ms,
                    )
                    if state.scene_state == SCENE_STATE_EMPTY:
                        self._activate(
                            plan.device_id,
                            plan.sequence_id,
                            state,
                            timestamp_ms,
                            "person_detected",
                        )
                    elif not stable or not motion_low:
                        self._activate(
                            plan.device_id,
                            plan.sequence_id,
                            state,
                            timestamp_ms,
                            reason if not stable else "frame_motion",
                        )
                    else:
                        if state.stable_since_capture_ms <= 0:
                            state.stable_since_capture_ms = timestamp_ms
                        if (
                            state.version == plan.state_version
                            and state.scene_state == SCENE_STATE_ACTIVE
                            and timestamp_ms >= state.active_until_capture_ms
                            and timestamp_ms - state.stable_since_capture_ms
                            >= max(0, int(self.config.static_confirm_ms))
                        ):
                            self._transition(
                                plan.device_id,
                                plan.sequence_id,
                                state,
                                SCENE_STATE_STATIC,
                                timestamp_ms,
                                "confirmed_static",
                            )
                            state.next_model_capture_ms = (
                                timestamp_ms + self._period_ms(SCENE_STATE_STATIC)
                            )

                state.previous_observation = observation
                state.previous_observation_capture_ms = timestamp_ms
                state.last_model_capture_ms = max(
                    state.last_model_capture_ms,
                    timestamp_ms,
                )

            state.last_result_sequence_id = plan.sequence_id

    @staticmethod
    def _nearest_index(target: int, available: tuple[int, ...]) -> int:
        return min(available, key=lambda value: (abs(value - target), value))

    def finalize_batch(
        self,
        *,
        plan: SceneBatchPlan,
        model_results: list[dict],
    ) -> list[dict]:
        if len(model_results) != len(plan.model_indices):
            raise RuntimeError(
                (
                    f"adaptive scene expected {len(plan.model_indices)} model "
                    f"results, got {len(model_results)}"
                )
            )
        if not model_results:
            return []

        fresh_by_input: dict[int, dict] = {}
        for model_position, (input_index, result) in enumerate(
            zip(plan.model_indices, model_results)
        ):
            fresh = dict(result)
            fresh["input_index"] = int(input_index)
            fresh["output_index"] = int(input_index)
            fresh["source_frame_id"] = plan.frame_ids[input_index]
            fresh["capture_timestamp_ms"] = plan.capture_timestamps_ms[input_index]
            fresh["_scene_decision"] = DECISION_FULL
            fresh_by_input[int(input_index)] = fresh

        available = tuple(sorted(fresh_by_input))
        scheduled_set = set(plan.scheduled_indices)
        assembled: list[dict] = []
        for input_index in range(plan.input_count):
            if input_index in fresh_by_input:
                result = dict(fresh_by_input[input_index])
            else:
                source_index = self._nearest_index(input_index, available)
                result = dict(fresh_by_input[source_index])
                if plan.state_after_plan == SCENE_STATE_EMPTY:
                    result["_scene_decision"] = DECISION_EMPTY_SKIP
                else:
                    result["_scene_decision"] = DECISION_STATIC_REUSE
            result["frame_id"] = f"{plan.frame_ids[input_index]}_current"
            result["source_frame_id"] = plan.frame_ids[input_index]
            result["capture_timestamp_ms"] = plan.capture_timestamps_ms[input_index]
            result["input_index"] = input_index
            result["output_index"] = input_index
            result["result_kind"] = "current"
            assembled.append(result)

        update_results = {
            input_index: assembled[input_index]
            for input_index in scheduled_set
            if 0 <= input_index < len(assembled)
        }
        try:
            self._update_state_from_results(plan, update_results)
        except Exception:
            LOGGER.exception(
                (
                    "Adaptive scene result update failed: "
                    "device_id=%s sequence_id=%d"
                ),
                plan.device_id,
                plan.sequence_id,
            )
        return assembled

    def snapshot(self, device_id: str) -> dict:
        state = self._get_state(device_id)
        with self._device_lock(device_id):
            return {
                "scene_state": state.scene_state,
                "state_since_capture_ms": state.state_since_capture_ms,
                "last_frame_capture_ms": state.last_frame_capture_ms,
                "last_model_capture_ms": state.last_model_capture_ms,
                "last_planned_sequence_id": state.last_planned_sequence_id,
                "last_result_sequence_id": state.last_result_sequence_id,
                "stable_since_capture_ms": state.stable_since_capture_ms,
                "empty_since_capture_ms": state.empty_since_capture_ms,
                "active_until_capture_ms": state.active_until_capture_ms,
                "next_model_capture_ms": state.next_model_capture_ms,
                "weak_motion_count": state.weak_motion_count,
                "consecutive_empty_results": state.consecutive_empty_results,
                "version": state.version,
            }
