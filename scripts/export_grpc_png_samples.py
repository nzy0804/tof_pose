#!/usr/bin/env python3
"""Export protobuf/gRPC byte-stream samples for the batch Infer API."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ai_pb2
from scripts.infer_service import ModelService


OUT_DIR = ROOT / "outputs" / "grpc_png_byte_stream_samples"
INPUTS = [
    ROOT / "outputs" / "tmp_png_size_from_video_frame.png",
    ROOT / "outputs" / "png_size_samples" / "mid_frame_000826.png",
    ROOT / "outputs" / "png_size_samples" / "mid_frame_000954.png",
]
BATCH_SIZE = 10
INPUT_SIZE = (100, 100)
OUTPUT_SIZE = (320, 320)


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_hex(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hex_pairs = [f"{b:02x}" for b in data]
    lines = []
    for offset in range(0, len(hex_pairs), 16):
        chunk = hex_pairs[offset : offset + 16]
        lines.append(f"{offset:08x}  {' '.join(chunk)}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def grpc_message_bytes(protobuf_payload: bytes) -> bytes:
    return b"\x00" + len(protobuf_payload).to_bytes(4, byteorder="big") + protobuf_payload


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode_png(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("cannot decode PNG bytes")
    return img


def image_size(data: bytes) -> tuple[int, int]:
    img = decode_png(data)
    height, width = img.shape[:2]
    return width, height


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError("failed to encode PNG")
    return buf.tobytes()


def make_request_png(data: bytes) -> bytes:
    img = decode_png(data)
    resized = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    request_png = encode_png(resized)
    if image_size(request_png) != INPUT_SIZE:
        raise AssertionError(f"request image must be {INPUT_SIZE}, got {image_size(request_png)}")
    return request_png


def require_size(field: str, data: bytes, expected: tuple[int, int]) -> None:
    actual = image_size(data)
    if actual != expected:
        raise AssertionError(f"{field} must be {expected[0]}x{expected[1]}, got {actual[0]}x{actual[1]}")


def main() -> None:
    existing_inputs = [path for path in INPUTS if path.exists()]
    if not existing_inputs:
        raise FileNotFoundError("no sample input images found")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_dir = OUT_DIR / "batch_sample_000001"
    sample_dir.mkdir(parents=True, exist_ok=True)

    svc = ModelService()
    device_id = "device_sample"
    batch_id = "batch_sample_000001"
    req = ai_pb2.InferRequest(device_id=device_id, batch_id=batch_id)
    frames: list[tuple[str, bytes]] = []

    for idx in range(BATCH_SIZE):
        input_path = existing_inputs[idx % len(existing_inputs)]
        frame_id = f"frame_sample_{idx + 1:06d}"
        image_data = make_request_png(input_path.read_bytes())
        req.images.append(
            ai_pb2.InferImage(
                frame_id=frame_id,
                capture_timestamp_ms=1700000000000 + idx,
                image_data=image_data,
            )
        )
        frames.append((frame_id, image_data))
        write_bytes(sample_dir / f"{idx + 1:02d}_request_image.png.bin", image_data)
        write_hex(sample_dir / f"{idx + 1:02d}_request_image.png.hex", image_data)

    batch_start_results = svc.infer_batch(frames)
    resp = ai_pb2.InferResponse(device_id=device_id, batch_id=batch_id)
    for request_image, result in zip(req.images, batch_start_results):
        resp.results.append(
            ai_pb2.InferResult(
                frame_id=request_image.frame_id,
                capture_timestamp_ms=request_image.capture_timestamp_ms,
                pseudo_color_image=result["pseudo_color_image"],
                skeleton_contour_image=result["skeleton_contour_image"],
                person_count=int(result["person_count"]),
                processing_time_ms=int(result["processing_time_ms"]),
            )
        )
    resp.processing_time_ms = sum(item.processing_time_ms for item in resp.results)

    req_pb = req.SerializeToString()
    resp_pb = resp.SerializeToString()
    write_bytes(sample_dir / "request.pb.bin", req_pb)
    write_hex(sample_dir / "request.pb.hex", req_pb)
    write_bytes(sample_dir / "response.pb.bin", resp_pb)
    write_hex(sample_dir / "response.pb.hex", resp_pb)
    write_bytes(sample_dir / "request.grpc.bin", grpc_message_bytes(req_pb))
    write_hex(sample_dir / "request.grpc.hex", grpc_message_bytes(req_pb))
    write_bytes(sample_dir / "response.grpc.bin", grpc_message_bytes(resp_pb))
    write_hex(sample_dir / "response.grpc.hex", grpc_message_bytes(resp_pb))

    lines = [
        "# gRPC batch PNG byte-stream sample",
        "",
        f"- device_id: `{device_id}`",
        f"- batch_id: `{batch_id}`",
        f"- images: `{len(req.images)}`",
        f"- results: `{len(resp.results)}`",
        f"- request protobuf bytes: `{len(req_pb)}`, sha256 `{digest(req_pb)}`",
        f"- response protobuf bytes: `{len(resp_pb)}`, sha256 `{digest(resp_pb)}`",
        "",
        "| index | frame_id | request bytes | pseudo color bytes | skeleton contour bytes | person_count |",
        "|---:|---|---:|---:|---:|---:|",
    ]

    for idx, (request_image, result) in enumerate(zip(req.images, resp.results), start=1):
        require_size("pseudo_color_image", result.pseudo_color_image, OUTPUT_SIZE)
        require_size("skeleton_contour_image", result.skeleton_contour_image, OUTPUT_SIZE)
        write_bytes(sample_dir / f"{idx:02d}_pseudo_color.png.bin", result.pseudo_color_image)
        write_hex(sample_dir / f"{idx:02d}_pseudo_color.png.hex", result.pseudo_color_image)
        write_bytes(sample_dir / f"{idx:02d}_skeleton_contour.png.bin", result.skeleton_contour_image)
        write_hex(sample_dir / f"{idx:02d}_skeleton_contour.png.hex", result.skeleton_contour_image)
        lines.append(
            f"| {idx} | `{request_image.frame_id}` | {len(request_image.image_data)} | "
            f"{len(result.pseudo_color_image)} | {len(result.skeleton_contour_image)} | {result.person_count} |"
        )

    (sample_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {sample_dir / 'README.md'}")


if __name__ == "__main__":
    main()
