#!/usr/bin/env python3
"""Read ToF frames from serial, encode as PNG bytes, call gRPC ModelService.Infer.

This is a headless smoke-test for the full pipeline:
serial -> depth bytes -> PNG bytes -> gRPC -> server inference -> returned PNGs.

Example (local server)
  python scripts/serial_to_grpc_test.py --serial-port COM8 --host 127.0.0.1 --port 50052 --frames 10

Example (remote server)
  python scripts/serial_to_grpc_test.py --serial-port COM8 --host <公网IP> --port 50052 --frames 5 --timeout 180
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import cv2
import grpc
import numpy as np
import serial

# Ensure repository root and src/ are on sys.path so ai_pb2 and tof_pose can be imported.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import ai_pb2
import ai_pb2_grpc


BAUD_DEFAULT = 921600
READ_TIMEOUT_S = 0.05
ENDIAN = "<"
FRAME_HEAD = b"\x00\xFF"
ALLOWED_TAILS = (0xCC, 0xDD)


def _save_outputs(resp: ai_pb2.InferResponse, outdir: Path) -> dict[str, int]:
    outdir.mkdir(parents=True, exist_ok=True)
    mapping = [
        ("S11", resp.output_image_S11),
        ("S12", resp.output_image_S12),
        ("S13", resp.output_image_S13),
        ("S14", resp.output_image_S14),
        ("S21", resp.output_image_S21),
        ("S22", resp.output_image_S22),
        ("S23", resp.output_image_S23),
        ("S24", resp.output_image_S24),
    ]

    prefix = (
        f"{getattr(resp, 'device_id', '')}_{resp.frame_id}_{getattr(resp, 'capture_timestamp_ms', 0)}"
        if getattr(resp, 'device_id', '')
        else f"{resp.frame_id}"
    )

    sizes: dict[str, int] = {}
    for name, blob in mapping:
        path = outdir / f"{prefix}_{name}.png"
        if blob:
            path.write_bytes(blob)
            sizes[name] = len(blob)
        else:
            sizes[name] = 0
    return sizes


def _encode_depth_png(depth_u8: np.ndarray) -> bytes:
    if depth_u8.dtype != np.uint8:
        depth_u8 = depth_u8.astype(np.uint8, copy=False)
    ok, buf = cv2.imencode(".png", depth_u8, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError("failed to encode depth png")
    return buf.tobytes()


def _iter_serial_frames(ser: serial.Serial):
    """Yield (res_r, res_c, frameid, payload_bytes) from serial stream."""
    last_frameid: int | None = None
    buf = bytearray()

    while True:
        n = ser.in_waiting
        chunk = ser.read(min(4096, n) if n else 256)
        if not chunk:
            time.sleep(0.001)
            continue

        buf += chunk
        while True:
            idx = buf.find(FRAME_HEAD)
            if idx < 0:
                if len(buf) > 8192:
                    buf.clear()
                break
            if idx > 0:
                del buf[:idx]
            if len(buf) < 4:
                break

            try:
                data_len = struct.unpack(ENDIAN + "H", buf[2:4])[0]
            except struct.error:
                break

            frame_len = 2 + 2 + data_len + 2
            if len(buf) < frame_len:
                break

            frame = bytes(buf[:frame_len])
            del buf[:frame_len]

            # Tail byte
            if frame[-1] not in ALLOWED_TAILS:
                continue
            # Checksum byte (sum of all bytes except last 2)
            if frame[-2] != (sum(frame[:-2]) & 0xFF):
                continue

            try:
                res_r = int(frame[14])
                res_c = int(frame[15])
                frameid = int(struct.unpack(ENDIAN + "H", frame[16:18])[0])
            except (IndexError, struct.error, ValueError):
                continue

            if last_frameid is not None and frameid == last_frameid:
                continue
            last_frameid = frameid

            payload_len = int(data_len) - 16
            if payload_len <= 0:
                continue

            payload = frame[20 : 20 + payload_len]
            if len(payload) != payload_len:
                continue

            yield res_r, res_c, frameid, payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Serial -> PNG bytes -> gRPC Infer smoke test")
    parser.add_argument("--serial-port", default="COM8", help="serial port (e.g., COM8)")
    parser.add_argument("--baud", type=int, default=BAUD_DEFAULT)
    parser.add_argument("--device-id", default="device_0000")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50052)
    parser.add_argument(
        "--start-local-server",
        action="store_true",
        help="start a local grpc_server.py (using current python) before sending frames",
    )
    parser.add_argument(
        "--server-python",
        default=sys.executable,
        help="python executable used to start local server (default: current python)",
    )
    parser.add_argument(
        "--server-max-workers",
        type=int,
        default=1,
        help="grpc server max workers when --start-local-server is set",
    )
    parser.add_argument(
        "--keep-server",
        action="store_true",
        help="do not stop the local server after test",
    )
    parser.add_argument("--frames", type=int, default=10, help="number of frames to send")
    parser.add_argument("--timeout", type=float, default=120.0, help="grpc per-call timeout seconds")
    parser.add_argument("--max-msg-mb", type=int, default=50)
    parser.add_argument("--outdir", default=str(Path("outputs") / "serial_grpc_test"))
    args = parser.parse_args()

    outdir = Path(args.outdir)

    server_proc: subprocess.Popen | None = None
    server_log_handle = None

    if args.start_local_server:
        outdir.mkdir(parents=True, exist_ok=True)
        server_log_path = outdir / "grpc_server.log"
        server_log_handle = server_log_path.open("a", encoding="utf-8")
        env = os.environ.copy()
        env.setdefault("CUDA_VISIBLE_DEVICES", "-1")

        cmd = [
            str(args.server_python),
            str(ROOT / "scripts" / "grpc_server.py"),
            "--host",
            str(args.host),
            "--port",
            str(int(args.port)),
            "--max-workers",
            str(int(args.server_max_workers)),
        ]

        server_log_handle.write("[serial_to_grpc] starting local server: " + " ".join(cmd) + "\n")
        server_log_handle.flush()

        server_proc = subprocess.Popen(
            cmd,
            stdout=server_log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )

        # Give it a moment to bind; channel readiness check below will be the real gate.
        time.sleep(0.3)

    # gRPC channel
    opts = [
        ("grpc.max_send_message_length", int(args.max_msg_mb) * 1024 * 1024),
        ("grpc.max_receive_message_length", int(args.max_msg_mb) * 1024 * 1024),
    ]
    target = f"{args.host}:{args.port}"
    channel = grpc.insecure_channel(target, options=opts)

    try:
        grpc.channel_ready_future(channel).result(timeout=min(10.0, float(args.timeout)))
    except Exception as exc:
        if server_proc is not None:
            rc = server_proc.poll()
            raise SystemExit(
                f"gRPC channel not ready for {target}. local server rc={rc}. "
                f"Check {outdir / 'grpc_server.log'} for details. ({exc})"
            )
        raise SystemExit(
            f"gRPC channel not ready for {target}. "
            f"Check server process, firewall/security-group, and port binding. ({exc})"
        )

    stub = ai_pb2_grpc.ModelServiceStub(channel)

    # Serial
    ser = serial.Serial(args.serial_port, int(args.baud), timeout=READ_TIMEOUT_S)
    print(f"[serial_to_grpc] serial={args.serial_port} baud={args.baud} -> grpc={target}", flush=True)

    # Optional sensor configuration (safe to ignore if unsupported)
    try:
        ser.write(b"AT+FPS=19\r")
        time.sleep(0.1)
        ser.write(b"AT+DISP=2\r")
        time.sleep(0.1)
    except Exception:
        pass

    sent = 0
    t0 = time.time()

    try:
        for res_r, res_c, frameid, payload in _iter_serial_frames(ser):
            depth = np.frombuffer(payload, dtype=np.uint8)
            if depth.size != int(res_r) * int(res_c):
                continue
            depth = depth.reshape((int(res_r), int(res_c)))
            png_bytes = _encode_depth_png(depth)

            frame_id = f"serial_{frameid:06d}"
            req = ai_pb2.InferRequest(
                device_id=str(args.device_id),
                frame_id=frame_id,
                capture_timestamp_ms=int(time.time() * 1000),
                image_data=png_bytes,
            )

            call_start = time.time()
            resp = stub.Infer(req, timeout=float(args.timeout))
            call_end = time.time()

            sizes = _save_outputs(resp, outdir)
            sent += 1
            print(
                f"[{sent}/{args.frames}] device_id={getattr(resp, 'device_id', '')} {frame_id} src={res_c}x{res_r} "
                f"server_ms={resp.processing_time_ms} roundtrip_ms={int((call_end - call_start) * 1000)} "
                f"ts_ms={getattr(resp, 'capture_timestamp_ms', 0)} person_count={resp.person_count} sizes={sizes}",
                flush=True,
            )

            if sent >= int(args.frames):
                break

    except KeyboardInterrupt:
        print("\n[serial_to_grpc] interrupted", flush=True)
    finally:
        try:
            ser.close()
        except Exception:
            pass

        if server_proc is not None and not args.keep_server:
            try:
                server_proc.terminate()
            except Exception:
                pass

        if server_log_handle is not None:
            try:
                server_log_handle.flush()
                server_log_handle.close()
            except Exception:
                pass

    elapsed = max(time.time() - t0, 1e-6)
    fps = sent / elapsed
    print(f"[serial_to_grpc] done sent={sent} elapsed_s={elapsed:.2f} fps~{fps:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
