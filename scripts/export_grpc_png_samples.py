#!/usr/bin/env python3
"""Export PNG byte-stream samples for the gRPC IoT/AI integration document."""
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
WARMUP_INPUT = ROOT / "outputs" / "tmp_png_size_check.png"
OUTPUT_FIELDS = [
    "output_image_S11",
    "output_image_S12",
    "output_image_S13",
    "output_image_S14",
    "output_image_S21",
    "output_image_S22",
    "output_image_S23",
    "output_image_S24",
]
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
    """Return the complete gRPC message bytes for one uncompressed unary message."""
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


def png_summary(data: bytes) -> str:
    if not data:
        return "`<empty>`"
    head = data[:16].hex(" ")
    tail = data[-16:].hex(" ")
    return f"`{head} ... {tail}`"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svc = ModelService()

    if WARMUP_INPUT.exists():
        svc.infer("warmup_000000", make_request_png(WARMUP_INPUT.read_bytes()))

    lines = [
        "# gRPC PNG 字节流样例",
        "",
        "> 由 `scripts/export_grpc_png_samples.py` 根据 `ai.proto`、`scripts/infer_service.py`、"
        "`scripts/grpc_server.py`、`scripts/grpc_client_test.py` 生成。",
        "> 字段顺序严格对应 `iot平台与ai算法.md` 中的 `InferRequest` / `InferResponse`。",
        "",
        "## 字节流说明",
        "",
        "| 类型 | 文件 | 内容 |",
        "|------|------|------|",
        "| PNG 原始字节流 | `*.png.bin` | 可直接写为 `.png` 文件的 bytes 字段内容 |",
        "| PNG 十六进制流 | `*.png.hex` | 同一 PNG bytes 的可读 hex dump |",
        "| Protobuf 消息体 | `*.pb.bin` / `*.pb.hex` | `InferRequest` 或 `InferResponse` 序列化后的 protobuf payload |",
        "| gRPC 完整消息字节流 | `*.grpc.bin` / `*.grpc.hex` | `compressed_flag(1 byte) + message_length(4 bytes, big-endian) + protobuf payload` |",
        "",
    ]

    for idx, input_path in enumerate(INPUTS, start=1):
        if not input_path.exists():
            raise FileNotFoundError(input_path)

        frame_id = f"frame_sample_{idx:06d}"
        device_id = "device_sample"
        capture_timestamp_ms = int(1700000000000 + idx)  # stable sample timestamp
        sample_dir = OUT_DIR / frame_id
        image_data = make_request_png(input_path.read_bytes())
        req = ai_pb2.InferRequest(
            device_id=device_id,
            frame_id=frame_id,
            capture_timestamp_ms=capture_timestamp_ms,
            image_data=image_data,
        )
        res = svc.infer(frame_id, image_data)
        resp = ai_pb2.InferResponse(
            device_id=device_id,
            frame_id=res["frame_id"],
            capture_timestamp_ms=capture_timestamp_ms,
            output_image_S11=res["output_image_S11"],
            output_image_S12=res["output_image_S12"],
            output_image_S13=res["output_image_S13"],
            output_image_S14=res["output_image_S14"],
            output_image_S21=res["output_image_S21"],
            output_image_S22=res["output_image_S22"],
            output_image_S23=res["output_image_S23"],
            output_image_S24=res["output_image_S24"],
            person_count=int(res["person_count"]),
            processing_time_ms=int(res["processing_time_ms"]),
        )

        req_pb = req.SerializeToString()
        resp_pb = resp.SerializeToString()
        req_grpc = grpc_message_bytes(req_pb)
        resp_grpc = grpc_message_bytes(resp_pb)
        write_bytes(sample_dir / "request.pb.bin", req_pb)
        write_hex(sample_dir / "request.pb.hex", req_pb)
        write_bytes(sample_dir / "response.pb.bin", resp_pb)
        write_hex(sample_dir / "response.pb.hex", resp_pb)
        write_bytes(sample_dir / "request.grpc.bin", req_grpc)
        write_hex(sample_dir / "request.grpc.hex", req_grpc)
        write_bytes(sample_dir / "response.grpc.bin", resp_grpc)
        write_hex(sample_dir / "response.grpc.hex", resp_grpc)
        write_bytes(sample_dir / "request_image_data.png.bin", image_data)
        write_hex(sample_dir / "request_image_data.png.hex", image_data)

        lines.extend(
            [
                f"## 样例 {idx}: {frame_id}",
                "",
                "### InferRequest",
                "",
                "| 参数 | 类型 | 值 | 分辨率 | 字节数 | SHA-256 | PNG 字节流首尾 |",
                "|------|------|----|--------|--------|---------|----------------|",
                f"| device_id | string | `{device_id}` | - | {len(device_id.encode('utf-8'))} | - | - |",
                f"| frame_id | string | `{frame_id}` | - | {len(frame_id.encode('utf-8'))} | - | - |",
                f"| capture_timestamp_ms | int64 | `{capture_timestamp_ms}` | - | - | - | - |",
                f"| image_data | bytes | `{sample_dir.relative_to(ROOT) / 'request_image_data.png.bin'}` | "
                f"{INPUT_SIZE[0]}x{INPUT_SIZE[1]} | {len(image_data)} | `{digest(image_data)}` | {png_summary(image_data)} |",
                "",
                f"- `InferRequest` Protobuf: `{(sample_dir.relative_to(ROOT) / 'request.pb.bin')}` "
                f"({len(req_pb)} bytes, sha256 `{digest(req_pb)}`)",
                f"- `InferRequest` Hex: `{(sample_dir.relative_to(ROOT) / 'request.pb.hex')}`",
                f"- `InferRequest` gRPC 完整字节流: `{(sample_dir.relative_to(ROOT) / 'request.grpc.bin')}` "
                f"({len(req_grpc)} bytes, sha256 `{digest(req_grpc)}`)",
                f"- `InferRequest` gRPC 完整 Hex: `{(sample_dir.relative_to(ROOT) / 'request.grpc.hex')}`",
                "",
                "### InferResponse",
                "",
                "| 参数 | 类型 | 值 | 分辨率 | 字节数 | SHA-256 | PNG 字节流首尾 |",
                "|------|------|----|--------|--------|---------|----------------|",
                f"| device_id | string | `{resp.device_id}` | - | {len(resp.device_id.encode('utf-8'))} | - | - |",
                f"| frame_id | string | `{resp.frame_id}` | - | {len(resp.frame_id.encode('utf-8'))} | - | - |",
                f"| capture_timestamp_ms | int64 | `{resp.capture_timestamp_ms}` | - | - | - | - |",
            ]
        )

        for field in OUTPUT_FIELDS:
            data = getattr(resp, field)
            require_size(field, data, OUTPUT_SIZE)
            png_bin = sample_dir / f"{field}.png.bin"
            png_hex = sample_dir / f"{field}.png.hex"
            write_bytes(png_bin, data)
            write_hex(png_hex, data)
            lines.append(
                f"| {field} | bytes | `{png_bin.relative_to(ROOT)}` | {OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]} | {len(data)} | "
                f"`{digest(data)}` | {png_summary(data)} |"
            )

        lines.extend(
            [
                f"| person_count | int32 | `{resp.person_count}` | - | - | - | - |",
                f"| processing_time_ms | int32 | `{resp.processing_time_ms}` | - | - | - | - |",
                "",
                f"- `InferResponse` Protobuf: `{(sample_dir.relative_to(ROOT) / 'response.pb.bin')}` "
                f"({len(resp_pb)} bytes, sha256 `{digest(resp_pb)}`)",
                f"- `InferResponse` Hex: `{(sample_dir.relative_to(ROOT) / 'response.pb.hex')}`",
                f"- `InferResponse` gRPC 完整字节流: `{(sample_dir.relative_to(ROOT) / 'response.grpc.bin')}` "
                f"({len(resp_grpc)} bytes, sha256 `{digest(resp_grpc)}`)",
                f"- `InferResponse` gRPC 完整 Hex: `{(sample_dir.relative_to(ROOT) / 'response.grpc.hex')}`",
                "",
            ]
        )

    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'README.md'}")


if __name__ == "__main__":
    main()
