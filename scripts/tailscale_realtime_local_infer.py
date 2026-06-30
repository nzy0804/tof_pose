"""Receive PNG frames over TCP/Tailscale and run local RealtimePoseEngine inference.

Protocol:
    Repeated frames of:
      - 4-byte big-endian unsigned frame payload length
      - PNG payload bytes

Example:
    python scripts/tailscale_realtime_local_infer.py \
        --host 0.0.0.0 \
        --port 9000 \
        --batch-size 20 \
        --batch-timeout 1 \
        --device cuda:0 \
        --output-format jpeg \
        --jpeg-quality 60

To compose received input frames into a video:
    ffmpeg -framerate 10 -i /tmp/tof_frames/frame_%06d.png \
      -c:v libx264 -pix_fmt yuv420p /tmp/tof.mp4
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import multiprocessing
from pathlib import Path
import queue
import socket
import struct
import sys
import threading
import time
from typing import Iterable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tof_pose.realtime_service import (
    CPU_WORKER_MODE_PROCESS,
    CPU_WORKER_MODE_THREAD,
    DEPTH_PERSON_DISTANCE_CLOSE_THRESHOLD,
    INPUT_MODALITIES,
    INPUT_MODALITY_DEPTH,
    MODEL_INPUT_SIZES,
    MODEL_INPUT_SIZE_320,
    PERSON_FILL_BACKGROUND_DEFAULT,
    PERSON_FILL_BACKGROUND_BLEND,
    PERSON_FILL_BACKGROUND_NAMES,
    PERSON_DISTANCE_CLOSE_CENTER_RATIO,
    PERSON_DISTANCE_CLOSE_GAP_RATIO,
    RealtimePoseEngine,
)


@dataclass(frozen=True)
class ReceivedFrame:
    index: int
    frame_id: str
    timestamp_ms: int
    data: bytes
    saved_path: str


def recv_exact(conn: socket.socket, n: int) -> bytes | None:
    data = bytearray()
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def safe_put(frame_queue: queue.Queue, item: ReceivedFrame, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            frame_queue.put(item, timeout=0.2)
            return
        except queue.Full:
            continue


def receive_loop(
    *,
    host: str,
    port: int,
    input_dir: Path,
    frame_queue: queue.Queue,
    stop_event: threading.Event,
    save_inputs: bool,
    max_frames: int,
    socket_backlog: int,
) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    frame_index = 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, int(port)))
        server.listen(max(1, int(socket_backlog)))
        server.settimeout(1.0)
        print(json.dumps({"event": "listening", "host": host, "port": int(port), "input_dir": str(input_dir)}, ensure_ascii=False), flush=True)

        while not stop_event.is_set():
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue
            print(json.dumps({"event": "connected", "addr": str(addr)}, ensure_ascii=False), flush=True)
            with conn:
                conn.settimeout(30.0)
                while not stop_event.is_set():
                    header = recv_exact(conn, 4)
                    if header is None:
                        break
                    size = struct.unpack(">I", header)[0]
                    if size <= 0:
                        print(json.dumps({"event": "skip_empty_frame", "size": int(size)}, ensure_ascii=False), flush=True)
                        continue
                    png = recv_exact(conn, size)
                    if png is None:
                        break

                    frame_index += 1
                    timestamp_ms = int(time.time() * 1000)
                    frame_id = f"frame_{frame_index:06d}"
                    saved_path = ""
                    if save_inputs:
                        path = input_dir / f"{frame_id}.png"
                        path.write_bytes(png)
                        saved_path = str(path)

                    safe_put(
                        frame_queue,
                        ReceivedFrame(
                            index=frame_index,
                            frame_id=frame_id,
                            timestamp_ms=timestamp_ms,
                            data=png,
                            saved_path=saved_path,
                        ),
                        stop_event,
                    )

                    if frame_index % 10 == 0:
                        print(
                            json.dumps(
                                {
                                    "event": "received",
                                    "frames": frame_index,
                                    "queue_size": frame_queue.qsize(),
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                    if max_frames > 0 and frame_index >= max_frames:
                        stop_event.set()
                        break
            print(json.dumps({"event": "disconnected", "frames": frame_index}, ensure_ascii=False), flush=True)


def build_engine(args: argparse.Namespace) -> RealtimePoseEngine:
    engine = RealtimePoseEngine(
        model_path=Path(args.model_path) if args.model_path else None,
        pose_model_path=Path(args.pose_model_path) if args.pose_model_path else None,
        stateless=args.stateless,
        pose_only=args.pose_only,
        pose_validate_seg=not args.no_pose_validate,
        pose_fallback=not args.no_pose_fallback,
        seg_conf_threshold=args.seg_conf,
        pose_conf_threshold=args.pose_conf,
        pose_kpt_conf_threshold=args.pose_kpt_conf,
        pose_gate_kpt_conf_threshold=args.pose_gate_kpt_conf,
        pose_kpt_min_points=args.pose_kpt_min_points,
        mask_threshold=args.mask_threshold,
        mask_min_area_ratio=args.mask_min_area_ratio,
        mask_max_area_ratio=args.mask_max_area_ratio,
        contour_new_track_conf_threshold=args.contour_new_conf,
        contour_existing_track_conf_threshold=args.contour_existing_conf,
        device=args.device,
        render_workers=args.render_workers,
        decode_workers=args.decode_workers,
        png_compression=args.png_compression,
        output_format=args.output_format,
        jpeg_quality=args.jpeg_quality,
        input_modality=args.input_modality,
        ir_preprocess=args.ir_preprocess,
        model_input_size=args.model_input_size,
        person_fill_background=args.person_fill_background,
        person_fill_background_blend=args.person_fill_background_blend,
        depth_distance_close_threshold=args.depth_distance_close_threshold,
        ir_distance_close_gap_ratio=args.ir_distance_close_gap_ratio,
        ir_distance_close_center_ratio=args.ir_distance_close_center_ratio,
        cpu_worker_mode=args.cpu_worker_mode,
        cpu_process_start_method=args.cpu_process_start_method,
        instance_name="tailscale-local",
    )
    if not args.no_warmup:
        engine.warmup(batch_size=args.warmup_batch_size)
    return engine


def write_result_images(output_dir: Path, batch_index: int, results: list[dict]) -> None:
    batch_dir = output_dir / f"batch_{batch_index:06d}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    for idx, result in enumerate(results):
        frame_id = str(result.get("frame_id") or f"result_{idx:04d}").replace("/", "_")
        pseudo_format = str(result.get("pseudo_color_image_format") or "png")
        skeleton_format = str(result.get("skeleton_contour_image_format") or "png")
        (batch_dir / f"{idx:04d}_{frame_id}_pseudo.{pseudo_format}").write_bytes(result.get("pseudo_color_image", b""))
        (batch_dir / f"{idx:04d}_{frame_id}_skeleton_contour.{skeleton_format}").write_bytes(
            result.get("skeleton_contour_image", b"")
        )


def result_summary(results: list[dict]) -> dict:
    return {
        "result_count": len(results),
        "person_counts": [int(item.get("person_count", 0) or 0) for item in results],
        "person_status": [str(item.get("person_status", "") or "") for item in results],
        "person_distance": [str(item.get("person_distance", "") or "") for item in results],
        "action_level": [str(item.get("action_level", "") or "") for item in results],
    }


def decode_image_bytes(data: bytes, tile_size: int) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        img = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
    elif img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_AREA)


def normalize_status_for_display(value: str) -> str:
    value = str(value or "")
    if value == "1站":
        return "stand"
    if value == "1坐":
        return "sit"
    if value == "1坐1站":
        return "sit+stand"
    return value or "-"


def label_tile(img: np.ndarray, title: str, line2: str = "") -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(out, title, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if line2:
        cv2.putText(out, line2, (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 255, 180), 1, cv2.LINE_AA)
    return out


def choose_current_results(results: list[dict]) -> dict[int, dict]:
    by_input: dict[int, dict] = {}
    fallback: dict[int, dict] = {}
    for result in results:
        input_index = int(result.get("input_index", -1))
        if input_index < 0:
            continue
        fallback.setdefault(input_index, result)
        if result.get("result_kind") == "current":
            by_input[input_index] = result
    for input_index, result in fallback.items():
        by_input.setdefault(input_index, result)
    return by_input


def build_display_row(
    args: argparse.Namespace,
    frame: ReceivedFrame,
    result: dict | None,
) -> np.ndarray:
    tile_size = max(80, int(args.show_tile_size))
    input_img = decode_image_bytes(frame.data, tile_size)
    input_tile = label_tile(input_img, f"input {frame.frame_id}")

    if result is None:
        blank = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        pseudo_tile = label_tile(blank, "pseudo", "missing")
        skeleton_tile = label_tile(blank, "skeleton", "missing")
    else:
        pseudo_tile = decode_image_bytes(result.get("pseudo_color_image", b""), tile_size)
        skeleton_tile = decode_image_bytes(result.get("skeleton_contour_image", b""), tile_size)
        metrics = (
            f"P={int(result.get('person_count', 0) or 0)} "
            f"S={normalize_status_for_display(str(result.get('person_status', '') or ''))} "
            f"D={str(result.get('person_distance', '') or '-')} "
            f"A={str(result.get('action_level', '') or '-')}"
        )
        pseudo_tile = label_tile(pseudo_tile, "pseudo current", metrics)
        skeleton_tile = label_tile(skeleton_tile, "skeleton+contour", metrics)

    return np.hstack([input_tile, pseudo_tile, skeleton_tile])


def show_batch_window(args: argparse.Namespace, batch: list[ReceivedFrame], results: list[dict]) -> bool:
    current_by_input = choose_current_results(results)
    if args.show_mode == "grid":
        max_rows = max(1, int(args.show_max_rows))
        rows = [
            build_display_row(args, frame, current_by_input.get(input_index))
            for input_index, frame in enumerate(batch[:max_rows])
        ]
        if not rows:
            return True
        canvas = np.vstack(rows)
        cv2.imshow(args.show_window_name, canvas)
        key = cv2.waitKey(max(1, int(args.show_wait_ms))) & 0xFF
        return key not in (27, ord("q"), ord("Q"))

    wait_ms = max(1, int(round(1000.0 / max(1.0, float(args.show_fps)))))
    for input_index, frame in enumerate(batch):
        canvas = build_display_row(args, frame, current_by_input.get(input_index))
        cv2.imshow(args.show_window_name, canvas)
        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            return False
    return True

def get_batch(
    frame_queue: queue.Queue,
    *,
    batch_size: int,
    batch_timeout: float,
    stop_event: threading.Event,
) -> list[ReceivedFrame]:
    batch: list[ReceivedFrame] = []
    deadline: float | None = None

    while not stop_event.is_set() or not frame_queue.empty():
        timeout = 0.2
        if batch and batch_timeout > 0 and deadline is not None:
            timeout = max(0.0, min(0.2, deadline - time.perf_counter()))
            if timeout <= 0:
                return batch

        try:
            item = frame_queue.get(timeout=timeout)
        except queue.Empty:
            if batch and batch_timeout > 0 and deadline is not None and time.perf_counter() >= deadline:
                return batch
            continue

        batch.append(item)
        if len(batch) == 1 and batch_timeout > 0:
            deadline = time.perf_counter() + float(batch_timeout)
        if len(batch) >= batch_size:
            return batch

    return batch


def inference_loop(
    *,
    args: argparse.Namespace,
    frame_queue: queue.Queue,
    stop_event: threading.Event,
) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = build_engine(args)
    batch_index = 0

    while not stop_event.is_set() or not frame_queue.empty():
        if args.max_batches > 0 and batch_index >= args.max_batches:
            stop_event.set()
            break

        batch = get_batch(
            frame_queue,
            batch_size=max(1, int(args.batch_size)),
            batch_timeout=max(0.0, float(args.batch_timeout)),
            stop_event=stop_event,
        )
        if not batch:
            continue

        batch_index += 1
        infer_start = time.perf_counter()
        results = engine.infer_batch([(item.frame_id, item.data) for item in batch])
        infer_ms = int((time.perf_counter() - infer_start) * 1000)

        if args.save_results:
            write_result_images(output_dir, batch_index, results)

        summary = result_summary(results)
        summary.update(
            {
                "event": "batch_done",
                "batch_index": batch_index,
                "input_count": len(batch),
                "first_frame": batch[0].frame_id,
                "last_frame": batch[-1].frame_id,
                "queue_size": frame_queue.qsize(),
                "infer_ms": infer_ms,
                "saved_dir": str(output_dir / f"batch_{batch_index:06d}") if args.save_results else "",
            }
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        if args.show_window:
            try:
                if not show_batch_window(args, batch, results):
                    stop_event.set()
                    break
            except cv2.error as exc:
                print(json.dumps({"event": "show_window_failed", "error": str(exc)}, ensure_ascii=False), flush=True)
                args.show_window = False

    return 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Receive Tailscale TCP PNG frames and run local MaixSense inference.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=9000, type=int)
    parser.add_argument("--input-dir", default="/tmp/tof_frames")
    parser.add_argument("--output-dir", default="/tmp/tof_infer_results")
    parser.add_argument("--batch-size", default=20, type=int)
    parser.add_argument("--batch-timeout", default=1.0, type=float)
    parser.add_argument("--queue-max", default=200, type=int)
    parser.add_argument("--max-frames", default=0, type=int)
    parser.add_argument("--max-batches", default=0, type=int)
    parser.add_argument("--socket-backlog", default=1, type=int)
    parser.add_argument("--save-inputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-results", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-window", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-window-name", default="MaixSense Local Inference")
    parser.add_argument("--show-mode", default="video", choices=("video", "grid"))
    parser.add_argument("--show-fps", default=10.0, type=float)
    parser.add_argument("--show-max-rows", default=6, type=int)
    parser.add_argument("--show-tile-size", default=180, type=int)
    parser.add_argument("--show-wait-ms", default=1, type=int)

    parser.add_argument("--model-path", default=None)
    parser.add_argument("--pose-model-path", default=None)
    parser.add_argument("--seg-conf", default=None, type=float)
    parser.add_argument("--pose-conf", default=None, type=float)
    parser.add_argument("--pose-kpt-conf", default=None, type=float)
    parser.add_argument("--pose-gate-kpt-conf", default=None, type=float)
    parser.add_argument("--pose-kpt-min-points", default=4, type=int)
    parser.add_argument("--mask-threshold", default=0.5, type=float)
    parser.add_argument("--mask-min-area-ratio", default=None, type=float)
    parser.add_argument("--mask-max-area-ratio", default=None, type=float)
    parser.add_argument("--contour-new-conf", default=None, type=float)
    parser.add_argument("--contour-existing-conf", default=None, type=float)
    parser.add_argument("--device", default=None, help="local inference device, for example cuda:0 or cpu")
    parser.add_argument("--render-workers", default=1, type=int)
    parser.add_argument("--decode-workers", default=1, type=int)
    parser.add_argument("--cpu-worker-mode", default=CPU_WORKER_MODE_THREAD, choices=(CPU_WORKER_MODE_THREAD, CPU_WORKER_MODE_PROCESS))
    parser.add_argument("--cpu-process-start-method", default="auto", choices=("auto", "fork", "spawn", "forkserver"))
    parser.add_argument("--png-compression", default=1, type=int)
    parser.add_argument("--output-format", default="jpeg", choices=("png", "jpeg", "jpg"))
    parser.add_argument("--jpeg-quality", default=60, type=int)
    parser.add_argument("--input-modality", default=INPUT_MODALITY_DEPTH, choices=INPUT_MODALITIES)
    parser.add_argument("--ir-preprocess", action="store_true", help="enable median filtering plus CLAHE for infrared grayscale inputs")
    parser.add_argument(
        "--model-input-size",
        default=MODEL_INPUT_SIZE_320,
        type=int,
        choices=MODEL_INPUT_SIZES,
        help="model inference input size; 160 scales the raw image directly before inference",
    )
    parser.add_argument(
        "--person-fill-background",
        default=PERSON_FILL_BACKGROUND_DEFAULT,
        help=(
            "background image used to fill detected person masks; pass an asset name "
            f"({', '.join(PERSON_FILL_BACKGROUND_NAMES)}) or an image file path"
        ),
    )
    parser.add_argument("--person-fill-background-blend", default=PERSON_FILL_BACKGROUND_BLEND, type=float)
    parser.add_argument("--depth-distance-close-threshold", default=DEPTH_PERSON_DISTANCE_CLOSE_THRESHOLD, type=float)
    parser.add_argument("--ir-distance-close-gap-ratio", default=PERSON_DISTANCE_CLOSE_GAP_RATIO, type=float)
    parser.add_argument("--ir-distance-close-center-ratio", default=PERSON_DISTANCE_CLOSE_CENTER_RATIO, type=float)
    parser.add_argument("--stateless", action="store_true")
    parser.add_argument("--pose-only", action="store_true")
    parser.add_argument("--no-pose-validate", action="store_true")
    parser.add_argument("--no-pose-fallback", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--warmup-batch-size", default=20, type=int)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    stop_event = threading.Event()
    frame_queue: queue.Queue[ReceivedFrame] = queue.Queue(maxsize=max(1, int(args.queue_max)))

    receiver = threading.Thread(
        target=receive_loop,
        kwargs={
            "host": args.host,
            "port": int(args.port),
            "input_dir": Path(args.input_dir),
            "frame_queue": frame_queue,
            "stop_event": stop_event,
            "save_inputs": bool(args.save_inputs),
            "max_frames": max(0, int(args.max_frames)),
            "socket_backlog": max(1, int(args.socket_backlog)),
        },
        daemon=True,
    )
    receiver.start()

    try:
        return inference_loop(args=args, frame_queue=frame_queue, stop_event=stop_event)
    except KeyboardInterrupt:
        stop_event.set()
        print(json.dumps({"event": "interrupted"}, ensure_ascii=False), flush=True)
        return 130
    finally:
        stop_event.set()
        if args.show_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
