import unittest

import cv2
import numpy as np

from tof_pose.scene_rate_controller import (
    DECISION_STATIC_REUSE,
    SCENE_MODE_ENFORCE,
    SCENE_MODE_OFF,
    SCENE_MODE_SHADOW,
    SCENE_STATE_ACTIVE,
    SCENE_STATE_EMPTY,
    SCENE_STATE_STATIC,
    SceneRateConfig,
    SceneRateController,
)


def _encoded_frame(*, moving: bool = False) -> bytes:
    image = np.zeros((100, 100), dtype=np.uint8)
    if moving:
        image[25:75, 25:75] = 255
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("failed to encode test frame")
    return encoded.tobytes()


def _frames(
    count: int,
    *,
    start_timestamp_ms: int,
    step_ms: int = 100,
    moving_indices: set[int] | None = None,
) -> tuple[list[tuple[str, bytes]], list[int]]:
    moving_indices = moving_indices or set()
    frames = [
        (f"frame-{start_timestamp_ms}-{index}", _encoded_frame(moving=index in moving_indices))
        for index in range(count)
    ]
    timestamps = [
        start_timestamp_ms + index * step_ms
        for index in range(count)
    ]
    return frames, timestamps


def _person_observation() -> dict:
    return {
        "person_count": 1,
        "tracks": [
            {
                "track_id": 7,
                "confidence": 0.9,
                "mask_area_ratio": 0.2,
                "mask_center": (0.5, 0.5),
                "keypoints": [(0.4, 0.4, 0.9), (0.6, 0.6, 0.9)],
            }
        ],
    }


def _model_results(plan, observation: dict) -> list[dict]:
    return [
        {
            "frame_id": plan.frame_ids[input_index],
            "skeleton_contour_image": f"image-{input_index}".encode(),
            "person_count": int(observation["person_count"]),
            "processing_time_ms": 1,
            "_scene_observation": observation,
        }
        for input_index in plan.model_indices
    ]


class SceneRateControllerTests(unittest.TestCase):
    def _controller(self, **overrides) -> SceneRateController:
        values = {
            "mode": SCENE_MODE_ENFORCE,
            "active_min_ms": 0,
            "static_confirm_ms": 100,
            "empty_confirm_count": 2,
            "empty_confirm_ms": 100,
        }
        values.update(overrides)
        return SceneRateController(SceneRateConfig(**values))

    def _reach_static(self, controller: SceneRateController) -> None:
        frames, timestamps = _frames(2, start_timestamp_ms=1000)
        first = controller.plan_batch(
            device_id="device-a",
            sequence_id=1,
            frames=frames,
            capture_timestamps_ms=timestamps,
        )
        controller.finalize_batch(
            plan=first,
            model_results=_model_results(first, _person_observation()),
        )

        frames, timestamps = _frames(3, start_timestamp_ms=1200)
        second = controller.plan_batch(
            device_id="device-a",
            sequence_id=2,
            frames=frames,
            capture_timestamps_ms=timestamps,
        )
        controller.finalize_batch(
            plan=second,
            model_results=_model_results(second, _person_observation()),
        )
        self.assertEqual(
            controller.snapshot("device-a")["scene_state"],
            SCENE_STATE_STATIC,
        )

    def test_off_mode_sends_every_frame(self) -> None:
        controller = SceneRateController(SceneRateConfig(mode=SCENE_MODE_OFF))
        frames, timestamps = _frames(20, start_timestamp_ms=1000)
        plan = controller.plan_batch(
            device_id="device-a",
            sequence_id=1,
            frames=frames,
            capture_timestamps_ms=timestamps,
        )
        self.assertEqual(plan.model_indices, tuple(range(20)))

    def test_static_state_schedules_five_hz(self) -> None:
        controller = self._controller()
        self._reach_static(controller)

        frames, timestamps = _frames(20, start_timestamp_ms=1500)
        plan = controller.plan_batch(
            device_id="device-a",
            sequence_id=3,
            frames=frames,
            capture_timestamps_ms=timestamps,
        )

        self.assertEqual(plan.state_before, SCENE_STATE_STATIC)
        self.assertEqual(plan.model_indices, tuple(range(0, 20, 2)))

    def test_motion_wakes_static_state_from_current_frame(self) -> None:
        controller = self._controller()
        self._reach_static(controller)

        frames, timestamps = _frames(
            8,
            start_timestamp_ms=1500,
            moving_indices={1, 2, 3, 4, 5, 6, 7},
        )
        plan = controller.plan_batch(
            device_id="device-a",
            sequence_id=3,
            frames=frames,
            capture_timestamps_ms=timestamps,
        )

        self.assertEqual(plan.model_indices, tuple(range(8)))
        self.assertEqual(plan.state_after_plan, SCENE_STATE_ACTIVE)

    def test_confirmed_empty_state_schedules_one_hz(self) -> None:
        controller = self._controller()
        frames, timestamps = _frames(2, start_timestamp_ms=1000)
        plan = controller.plan_batch(
            device_id="device-a",
            sequence_id=1,
            frames=frames,
            capture_timestamps_ms=timestamps,
        )
        empty = {"person_count": 0, "tracks": []}
        controller.finalize_batch(
            plan=plan,
            model_results=_model_results(plan, empty),
        )
        self.assertEqual(
            controller.snapshot("device-a")["scene_state"],
            SCENE_STATE_EMPTY,
        )

        frames, timestamps = _frames(20, start_timestamp_ms=1200)
        next_plan = controller.plan_batch(
            device_id="device-a",
            sequence_id=2,
            frames=frames,
            capture_timestamps_ms=timestamps,
        )
        self.assertEqual(next_plan.model_indices, (9, 19))

    def test_stale_capture_gap_fails_open_to_active(self) -> None:
        controller = self._controller(state_stale_ms=500)
        self._reach_static(controller)

        frames, timestamps = _frames(2, start_timestamp_ms=3000)
        plan = controller.plan_batch(
            device_id="device-a",
            sequence_id=3,
            frames=frames,
            capture_timestamps_ms=timestamps,
        )

        self.assertEqual(plan.model_indices, (0, 1))
        self.assertEqual(plan.state_after_plan, SCENE_STATE_ACTIVE)

    def test_older_results_do_not_overwrite_newer_state(self) -> None:
        controller = self._controller()
        frames, timestamps = _frames(2, start_timestamp_ms=1000)
        older = controller.plan_batch(
            device_id="device-a",
            sequence_id=1,
            frames=frames,
            capture_timestamps_ms=timestamps,
        )
        frames, timestamps = _frames(2, start_timestamp_ms=1200)
        newer = controller.plan_batch(
            device_id="device-a",
            sequence_id=2,
            frames=frames,
            capture_timestamps_ms=timestamps,
        )

        empty = {"person_count": 0, "tracks": []}
        controller.finalize_batch(
            plan=newer,
            model_results=_model_results(newer, empty),
        )
        controller.finalize_batch(
            plan=older,
            model_results=_model_results(older, _person_observation()),
        )

        snapshot = controller.snapshot("device-a")
        self.assertEqual(snapshot["scene_state"], SCENE_STATE_EMPTY)
        self.assertEqual(snapshot["last_result_sequence_id"], 2)

    def test_finalize_restores_all_frame_results(self) -> None:
        controller = self._controller()
        self._reach_static(controller)
        frames, timestamps = _frames(20, start_timestamp_ms=1500)
        plan = controller.plan_batch(
            device_id="device-a",
            sequence_id=3,
            frames=frames,
            capture_timestamps_ms=timestamps,
        )

        results = controller.finalize_batch(
            plan=plan,
            model_results=_model_results(plan, _person_observation()),
        )

        self.assertEqual(len(results), 20)
        self.assertEqual(
            [result["input_index"] for result in results],
            list(range(20)),
        )
        self.assertEqual(
            [result["capture_timestamp_ms"] for result in results],
            timestamps,
        )
        self.assertEqual(results[1]["_scene_decision"], DECISION_STATIC_REUSE)

    def test_shadow_mode_observes_schedule_but_runs_all_frames(self) -> None:
        controller = SceneRateController(
            SceneRateConfig(
                mode=SCENE_MODE_SHADOW,
                active_min_ms=0,
                static_confirm_ms=100,
            )
        )
        self._reach_static(controller)
        frames, timestamps = _frames(20, start_timestamp_ms=1500)
        plan = controller.plan_batch(
            device_id="device-a",
            sequence_id=3,
            frames=frames,
            capture_timestamps_ms=timestamps,
        )

        self.assertEqual(plan.model_indices, tuple(range(20)))
        self.assertEqual(plan.scheduled_indices, tuple(range(0, 20, 2)))


if __name__ == "__main__":
    unittest.main()
