#!/usr/bin/env python3
"""Simple gRPC client to call ModelService.Infer and save returned PNGs.

Usage:
    python scripts/grpc_client_test.py input.png frame_000001 --host 127.0.0.1 --port 50052 --device-id dev01
"""
import argparse
import os
import time

import grpc
import sys

# ensure repository root is on sys.path so ai_pb2 / ai_pb2_grpc can be imported
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ai_pb2
import ai_pb2_grpc


def save_outputs(resp, outdir):
    os.makedirs(outdir, exist_ok=True)
    mapping = [
        ('S11', resp.output_image_S11),
        ('S12', resp.output_image_S12),
        ('S13', resp.output_image_S13),
        ('S14', resp.output_image_S14),
        ('S21', resp.output_image_S21),
        ('S22', resp.output_image_S22),
        ('S23', resp.output_image_S23),
        ('S24', resp.output_image_S24),
    ]
    result = {}
    prefix = f"{resp.device_id}_{resp.frame_id}_{resp.capture_timestamp_ms}" if getattr(resp, 'device_id', '') else f"{resp.frame_id}"
    for name, b in mapping:
        path = os.path.join(outdir, f"{prefix}_{name}.png")
        if b:
            with open(path, 'wb') as f:
                f.write(b)
            result[name] = len(b)
        else:
            result[name] = 0
    return result


def call_infer(host, port, device_id, frame_id, capture_timestamp_ms, img_path, timeout=10.0, max_msg_mb=50):
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

    with open(img_path, 'rb') as f:
        data = f.read()

    if capture_timestamp_ms is None:
        capture_timestamp_ms = int(time.time() * 1000)
    req = ai_pb2.InferRequest(
        device_id=device_id,
        frame_id=frame_id,
        capture_timestamp_ms=int(capture_timestamp_ms),
        image_data=data,
    )
    t0 = time.time()
    resp = stub.Infer(req, timeout=timeout)
    t1 = time.time()

    outdir = os.path.join('outputs', 'client_test')
    sizes = save_outputs(resp, outdir)
    print('device_id', getattr(resp, 'device_id', ''))
    print('frame_id', resp.frame_id)
    print('capture_timestamp_ms', getattr(resp, 'capture_timestamp_ms', 0))
    print('person_count:', resp.person_count)
    print('processing_time_ms (reported):', resp.processing_time_ms)
    print('roundtrip_ms:', int((t1 - t0) * 1000))
    print('saved sizes:', sizes)
    return resp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input image path')
    parser.add_argument('frame_id', help='frame id string')
    parser.add_argument('--device-id', default='device_0000')
    parser.add_argument(
        '--capture-timestamp-ms',
        default=None,
        type=int,
        help='capture timestamp in milliseconds; default: now()'
    )
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
        args.frame_id,
        args.capture_timestamp_ms,
        args.input,
        timeout=args.timeout,
        max_msg_mb=args.max_msg_mb,
    )


if __name__ == '__main__':
    main()
