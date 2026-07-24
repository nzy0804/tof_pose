from __future__ import annotations

import cv2
import numpy as np


def decrypt_2x2_diagonal_swap(image: np.ndarray) -> np.ndarray:
    """Undo the per-block diagonal swap applied by the image sender."""
    source = np.asarray(image)
    if source.ndim < 2:
        raise ValueError("expected an image with at least two dimensions")

    height, width = source.shape[:2]
    block_height = height - (height % 2)
    block_width = width - (width % 2)
    restored = np.array(source, copy=True, order="C")
    if block_height == 0 or block_width == 0:
        return restored

    complete_blocks = source[:block_height, :block_width]
    restored[0:block_height:2, 0:block_width:2] = complete_blocks[1:block_height:2, 1:block_width:2]
    restored[1:block_height:2, 1:block_width:2] = complete_blocks[0:block_height:2, 0:block_width:2]
    restored[0:block_height:2, 1:block_width:2] = complete_blocks[1:block_height:2, 0:block_width:2]
    restored[1:block_height:2, 0:block_width:2] = complete_blocks[0:block_height:2, 1:block_width:2]
    return restored


def decode_received_grayscale_views(
    image_data: bytes,
) -> tuple[np.ndarray, np.ndarray]:
    encoded = np.frombuffer(image_data, dtype=np.uint8)
    received = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if received is None:
        raise ValueError("cannot decode image")

    if received.ndim == 3:
        received = cv2.cvtColor(received, cv2.COLOR_BGR2GRAY)
    received = np.ascontiguousarray(received)
    decrypted = decrypt_2x2_diagonal_swap(received)
    return received, decrypted


def decode_received_grayscale(image_data: bytes) -> np.ndarray:
    _, decrypted = decode_received_grayscale_views(image_data)
    return decrypted
