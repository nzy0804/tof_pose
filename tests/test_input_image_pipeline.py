import unittest

import cv2
import numpy as np

from tof_pose.input_image import (
    decode_received_grayscale,
    decode_received_grayscale_views,
    decrypt_2x2_diagonal_swap,
)
from tof_pose.realtime_service import (
    DISPLAY_SIZE,
    INPUT_MODALITY_IR,
    _prepare_depth_views_cpu,
    _render_skeleton_contour_cpu,
)


class InputImagePipelineTests(unittest.TestCase):
    def test_diagonal_swap_is_its_own_inverse(self) -> None:
        source = np.arange(6 * 8, dtype=np.uint8).reshape(6, 8)

        encrypted = decrypt_2x2_diagonal_swap(source)
        restored = decrypt_2x2_diagonal_swap(encrypted)

        np.testing.assert_array_equal(restored, source)
        np.testing.assert_array_equal(
            encrypted[:2, :2],
            np.array([[9, 8], [1, 0]], dtype=np.uint8),
        )

    def test_diagonal_swap_leaves_unpaired_edges_unchanged(self) -> None:
        source = np.arange(5 * 7, dtype=np.uint8).reshape(5, 7)

        encrypted = decrypt_2x2_diagonal_swap(source)

        np.testing.assert_array_equal(encrypted[-1, :], source[-1, :])
        np.testing.assert_array_equal(encrypted[:, -1], source[:, -1])
        np.testing.assert_array_equal(
            decrypt_2x2_diagonal_swap(encrypted),
            source,
        )

    def test_received_png_is_decrypted_before_use(self) -> None:
        source = np.arange(100 * 100, dtype=np.uint8).reshape(100, 100)
        encrypted = decrypt_2x2_diagonal_swap(source)
        ok, encoded = cv2.imencode(".png", encrypted)
        self.assertTrue(ok)

        received, decoded = decode_received_grayscale_views(encoded.tobytes())

        np.testing.assert_array_equal(received, encrypted)
        np.testing.assert_array_equal(decoded, source)
        np.testing.assert_array_equal(
            decode_received_grayscale(encoded.tobytes()),
            source,
        )

    def test_depth_views_do_not_apply_gamma(self) -> None:
        source = np.array(
            [
                [0, 16, 64],
                [96, 144, 192],
                [224, 240, 255],
            ],
            dtype=np.uint8,
        )

        prepared = _prepare_depth_views_cpu(
            source,
            input_modality=INPUT_MODALITY_IR,
            model_input_size=160,
        )

        expected_up = cv2.resize(source, DISPLAY_SIZE, interpolation=cv2.INTER_LINEAR)
        np.testing.assert_array_equal(prepared["depth_up"], expected_up)
        np.testing.assert_array_equal(
            prepared["color_img"],
            cv2.cvtColor(expected_up, cv2.COLOR_GRAY2BGR),
        )

    def test_result_background_uses_undecrypted_received_frame(self) -> None:
        decrypted = np.arange(100 * 100, dtype=np.uint8).reshape(100, 100)
        received = decrypt_2x2_diagonal_swap(decrypted)
        prepared = _prepare_depth_views_cpu(
            decrypted,
            input_modality=INPUT_MODALITY_IR,
            received_gray=received,
            model_input_size=160,
        )
        analysis = {
            **prepared,
            "records": [],
            "kpt_xy_np": None,
            "kpt_conf_np": None,
            "pose_to_track": {},
            "validated_track_ids": set(),
            "pose_only": False,
            "pose_draw_indices": [],
            "pose_fallback_indices": [],
        }

        rendered = _render_skeleton_contour_cpu(analysis, (100, 100))

        np.testing.assert_array_equal(
            rendered,
            cv2.cvtColor(received, cv2.COLOR_GRAY2BGR),
        )


if __name__ == "__main__":
    unittest.main()
