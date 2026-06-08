#!/usr/bin/env python3
"""Receive image frames over UDP, run local inference, and display results.

By default, each UDP datagram is treated as one complete encoded image
(PNG/JPEG/etc.) and is passed to RealtimePoseEngine.infer(), the same engine
used by scripts/grpc_server.py.

For larger frames, this receiver also supports a tiny chunked UDP protocol:

    magic     4 bytes  b"MSXU"
    frame_id  4 bytes  unsigned int, big endian
    chunk_id  2 bytes  unsigned short, big endian, starts at 0
    chunks    2 bytes  unsigned short, big endian
    payload   remaining bytes

All chunks for a frame are concatenated in chunk_id order before inference.

Usage:
    python scripts/udp_infer_viewer.py --host 0.0.0.0 --port 50060
    python scripts/udp_infer_viewer.py --port 50060 --view montage
    python scripts/udp_infer_viewer.py --port 50060 --raw-depth 100x100
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (str(ROOT), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

# Match grpc_server.py: keep Ultralytics settings in a writable location.
_cfg_env = "".join(["Y", "O", "L", "O", "_CONFIG_DIR"])
if not os.environ.get(_cfg_env):
    home_dir = os.path.expanduser("~")
    if home_dir and home_dir != "~":
        os.environ[_cfg_env] = os.path.join(home_dir, ".ultralytics")
        try:
            os.makedirs(os.environ[_cfg_env], exist_ok=True)
        except Exception:
            pass

from tof_pose.realtime_service import RealtimePoseEngine


CHUNK_MAGIC = b"MSXU"
CHUNK_HEADER = struct.Struct("!4sIHH")
IMAGE_MAGIC_PREFIXES = (
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"BM",
    b"GIF87a",
    b"GIF89a",
    b"II*\x00",
    b"MM\x00*",
    b"RIFF",
)


@dataclass
class UdpFrame:
    frame_id: str
    image_bytes: bytes
    source: tuple[str, int]
    received_at: float


@dataclass
class PartialFrame:
    chunk_count: int
    source: tuple[str, int]
    first_seen: float
    chunks: dict[int, bytes] = field(default_factory=dict)


def _parse_size(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    normalized = text.lower().replace("*", "x")
    try:
        width_text, height_text = normalized.split("x", 1)
        width = int(width_text)
        height = int(height_text)
    except Exception as exc:
        raise argparse.ArgumentTypeError("--raw-depth must look like WIDTHxHEIGHT, e.g. 100x100") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("--raw-depth width and height must be positive")
    return width, height


def _encode_depth_png(raw: bytes, size: tuple[int, int]) -> bytes:
    width, height = size
    depth = np.frombuffer(raw, dtype=np.uint8)
    expected = width * height
    if depth.size != expected:
        raise ValueError(f"raw depth frame has {depth.size} bytes, expected {expected}")
    depth = depth.reshape((height, width))
    ok, buf = cv2.imencode(".png", depth, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError("failed to encode raw depth frame as PNG")
    return buf.tobytes()


def _decode_png_result(image_bytes: bytes) -> np.ndarray | None:
    if not image_bytes:
        return None
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _put_latest(frame_queue: queue.Queue[UdpFrame], frame: UdpFrame) -> None:
    while True:
        try:
            frame_queue.put_nowait(frame)
            return
        except queue.Full:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass


def _is_encoded_image(data: bytes) -> bool:
    return any(data.startswith(prefix) for prefix in IMAGE_MAGIC_PREFIXES)


def _udp_receiver(
    sock: socket.socket,
    frame_queue: queue.Queue[UdpFrame],
    stop_event: threading.Event,
    raw_depth_size: tuple[int, int] | None,
    max_partial_age_s: float,
) -> None:
    partials: dict[tuple[tuple[str, int], int], PartialFrame] = {}
    datagram_seq = 0

    while not stop_event.is_set():
        try:
            data, addr = sock.recvfrom(65535)
        except socket.timeout:
            now = time.time()
            stale_keys = [
                key for key, partial in partials.items() if now - partial.first_seen > max_partial_age_s
            ]
            for key in stale_keys:
                partials.pop(key, None)
            continue
        except OSError:
            if not stop_event.is_set():
                logging.exception("UDP socket error")
            return

        if not data:
            continue

        if data.startswith(CHUNK_MAGIC):
            if len(data) < CHUNK_HEADER.size:
                logging.warning("Ignoring short chunk from %s:%s", addr[0], addr[1])
                continue
            _, numeric_frame_id, chunk_id, chunk_count = CHUNK_HEADER.unpack_from(data)
            if chunk_count <= 0 or chunk_id >= chunk_count:
                logging.warning("Ignoring invalid chunk frame=%s chunk=%s/%s", numeric_frame_id, chunk_id, chunk_count)
                continue

            payload = data[CHUNK_HEADER.size :]
            key = (addr, numeric_frame_id)
            partial = partials.get(key)
            if partial is None or partial.chunk_count != chunk_count:
                partial = PartialFrame(chunk_count=chunk_count, source=addr, first_seen=time.time())
                partials[key] = partial
            partial.chunks[int(chunk_id)] = payload

            if len(partial.chunks) != partial.chunk_count:
                continue

            data = b"".join(partial.chunks[idx] for idx in range(partial.chunk_count))
            partials.pop(key, None)
            frame_id = f"udp_{numeric_frame_id:08d}"
        else:
            datagram_seq += 1
            frame_id = f"udp_{datagram_seq:08d}"

        try:
            if raw_depth_size is not None:
                image_bytes = _encode_depth_png(data, raw_depth_size)
            else:
                if not _is_encoded_image(data):
                    logging.warning(
                        "Frame %s from %s:%s does not look like PNG/JPEG image bytes; "
                        "use --raw-depth WIDTHxHEIGHT for raw uint8 depth frames",
                        frame_id,
                        addr[0],
                        addr[1],
                    )
                image_bytes = data
        except Exception:
            logging.exception("Failed to prepare frame %s from %s:%s", frame_id, addr[0], addr[1])
            continue

        _put_latest(
            frame_queue,
            UdpFrame(frame_id=frame_id, image_bytes=image_bytes, source=addr, received_at=time.time()),
        )


def _make_montage(images: list[np.ndarray | None]) -> np.ndarray:
    prepared: list[np.ndarray] = []
    fallback_shape = (320, 320, 3)
    for img in images:
        if img is None:
            prepared.append(np.zeros(fallback_shape, dtype=np.uint8))
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        prepared.append(cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA))

    top = np.hstack(prepared[:2])
    bottom = np.hstack(prepared[2:])
    return np.vstack([top, bottom])


def _select_display_image(result: dict, view: str) -> np.ndarray | None:
    if view in ("gray", "color"):
        return _decode_png_result(result.get("pseudo_color_image", b""))
    if view in ("skeleton", "contour"):
        return _decode_png_result(result.get("skeleton_contour_image", b""))

    return _make_montage(
        [
            _decode_png_result(result.get("pseudo_color_image", b"")),
            _decode_png_result(result.get("skeleton_contour_image", b"")),
            None,
            None,
        ]
    )


def _draw_status(
    image: np.ndarray,
    frame: UdpFrame,
    person_count: int,
    processing_time_ms: int,
    display_fps: float,
) -> np.ndarray:
    out = image.copy()
    latency_ms = int((time.time() - frame.received_at) * 1000)
    text = (
        f"{frame.frame_id}  persons={person_count}  infer={processing_time_ms}ms  "
        f"latency={latency_ms}ms  fps={display_fps:.1f}  {frame.source[0]}:{frame.source[1]}"
    )
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def run(
    host: str,
    port: int,
    view: str,
    raw_depth_size: tuple[int, int] | None,
    window_name: str,
    max_partial_age_s: float,
) -> int:
    frame_queue: queue.Queue[UdpFrame] = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(0.2)
    sock.bind((host, port))

    receiver = threading.Thread(
        target=_udp_receiver,
        args=(sock, frame_queue, stop_event, raw_depth_size, max_partial_age_s),
        daemon=True,
    )
    receiver.start()

    logging.info("Listening for UDP image frames on %s:%d", host, port)
    logging.info("Loading inference engine; the first frame may take a moment")
    engine = RealtimePoseEngine()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    displayed = 0
    fps_window_start = time.time()
    display_fps = 0.0

    try:
        while True:
            try:
                frame = frame_queue.get(timeout=0.2)
            except queue.Empty:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                continue

            try:
                result = engine.infer(frame.frame_id, frame.image_bytes)
            except Exception:
                logging.exception("Inference failed for %s", frame.frame_id)
                continue

            display = _select_display_image(result, view)
            if display is None:
                logging.warning("No display image returned for %s", frame.frame_id)
                continue

            displayed += 1
            now = time.time()
            elapsed = now - fps_window_start
            if elapsed >= 1.0:
                display_fps = displayed / elapsed
                displayed = 0
                fps_window_start = now

            display = _draw_status(
                display,
                frame,
                int(result.get("person_count", 0)),
                int(result.get("processing_time_ms", 0)),
                display_fps,
            )
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    except KeyboardInterrupt:
        logging.info("Interrupted")
    finally:
        stop_event.set()
        try:
            sock.close()
        except Exception:
            pass
        cv2.destroyAllWindows()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="UDP image receiver -> local RealtimePoseEngine inference viewer")
    parser.add_argument("--host", default="0.0.0.0", help="UDP bind host")
    parser.add_argument("--port", type=int, default=50060, help="UDP bind port")
    parser.add_argument(
        "--view",
        choices=("gray", "color", "skeleton", "contour", "montage"),
        default="contour",
        help="which inference output to display",
    )
    parser.add_argument(
        "--raw-depth",
        type=_parse_size,
        default=None,
        help="interpret each UDP frame as raw uint8 depth bytes with WIDTHxHEIGHT instead of encoded image bytes",
    )
    parser.add_argument("--window-name", default="MaixSense UDP Inference")
    parser.add_argument("--max-partial-age-s", type=float, default=1.0)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    return run(
        host=args.host,
        port=int(args.port),
        view=args.view,
        raw_depth_size=args.raw_depth,
        window_name=args.window_name,
        max_partial_age_s=float(args.max_partial_age_s),
    )


if __name__ == "__main__":
    raise SystemExit(main())
