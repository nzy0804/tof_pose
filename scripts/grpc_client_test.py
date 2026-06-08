#!/usr/bin/env python3
"""Simple gRPC client to call ModelService.Infer with an image batch.

Usage:
    python scripts/grpc_client_test.py img01.png img02.png ... --host 127.0.0.1 --port 50052 --device-id dev01
"""
import argparse
import os
import time
from pathlib import Path

import grpc
import sys

# ensure repository root is on sys.path so ai_pb2 / ai_pb2_grpc can be imported
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ai_pb2
import ai_pb2_grpc


def _result_kind_name(kind: int) -> str:
    if kind == ai_pb2.RESULT_KIND_INTERPOLATED:
        return "interpolated"
    if kind == ai_pb2.RESULT_KIND_CURRENT:
        return "current"
    return "unknown"


def save_outputs(resp, outdir):
    os.makedirs(outdir, exist_ok=True)
    result = {}
    batch_prefix = f"{resp.device_id}_{resp.batch_id}" if getattr(resp, 'device_id', '') else f"{resp.batch_id}"
    for idx, item in enumerate(resp.results, start=1):
        kind = _result_kind_name(item.result_kind)
        frame_prefix = f"{batch_prefix}_{idx:02d}_{kind}_{item.frame_id}"
        mapping = [
            ('pseudo_color', item.pseudo_color_image),
            ('skeleton_contour', item.skeleton_contour_image),
        ]
        for name, b in mapping:
            path = os.path.join(outdir, f"{frame_prefix}_{name}.png")
            key = f"{idx:02d}_{kind}_{name}"
            if b:
                with open(path, 'wb') as f:
                    f.write(b)
                result[key] = len(b)
            else:
                result[key] = 0
    return result


def call_infer(host, port, device_id, batch_id, img_paths, timeout=10.0, max_msg_mb=50):
    if not img_paths:
        raise ValueError("expected at least one input image")

    opts = [
        ('grpc.max_send_message_length', max_msg_mb * 1024 * 1024),
        ('grpc.max_receive_message_length', max_msg_mb * 1024 * 1024),
    ]
    target = f"{host}:{port}"
    channel = grpc.insecure_channel(target, options=opts)

    # Wait for server to become ready (cold start can be slow).
    try:
        grpc.channel_ready_future(channel).result(timeout=min(30.0, float(timeout)))
    except Exception as exc:
        raise RuntimeError(
            f"gRPC channel not ready for {target}. "
            f"Check server process, security-group/firewall, and port binding. ({exc})"
        )

    stub = ai_pb2_grpc.ModelServiceStub(channel)

    if batch_id is None:
        batch_id = f"batch_{int(time.time() * 1000)}"

    req = ai_pb2.InferRequest(device_id=device_id, batch_id=batch_id, batch_size=len(img_paths))
    now_ms = int(time.time() * 1000)
    for idx, img_path in enumerate(img_paths, start=1):
        path = Path(img_path)
        req.images.append(
            ai_pb2.InferImage(
                frame_id=path.stem or f"frame_{idx:06d}",
                capture_timestamp_ms=now_ms + idx,
                image_data=path.read_bytes(),
            )
        )

    t0 = time.time()
    resp = stub.Infer(req, timeout=timeout)
    t1 = time.time()

    outdir = os.path.join('outputs', 'client_test')
    sizes = save_outputs(resp, outdir)
    print('device_id', getattr(resp, 'device_id', ''))
    print('batch_id', resp.batch_id)
    print('result_count:', len(resp.results))
    print('output_image_count:', len(resp.results) * 2)
    print('person_counts:', [item.person_count for item in resp.results])
    print('processing_time_ms (batch reported):', resp.processing_time_ms)
    print('roundtrip_ms:', int((t1 - t0) * 1000))
    print('saved sizes:', sizes)
    return resp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+', help='input image paths for one batch')
    parser.add_argument('--device-id', default='device_0000')
    parser.add_argument('--batch-id', default=None)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=50052, type=int)
    # First request can be slow due to model load; use a safer default.
    parser.add_argument('--timeout', default=120.0, type=float)
    parser.add_argument('--max-msg-mb', default=50, type=int)
    args = parser.parse_args()

    resp = call_infer(
        args.host,
        args.port,
        args.device_id,
        args.batch_id,
        args.inputs,
        timeout=args.timeout,
        max_msg_mb=args.max_msg_mb,
    )


if __name__ == '__main__':
    main()
