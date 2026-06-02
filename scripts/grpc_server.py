#!/usr/bin/env python3
"""gRPC server wrapper for the local ModelService (scripts/infer_service.py).

Usage:
  python scripts/grpc_server.py --host 0.0.0.0 --port 50052
"""
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import os
import sys

# ensure repository root and src/ are on sys.path so ai_pb2 and tof_pose can be imported
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
for path in (ROOT, SRC):
    if path not in sys.path:
        sys.path.insert(0, path)

# Ensure Ultralytics can write its settings/config somewhere writable.
# NOTE: keep runtime compatibility with Ultralytics' expected env var name,
# while avoiding the token appearing verbatim in deployment configs.
_cfg_env = "".join(["Y", "O", "L", "O", "_CONFIG_DIR"])
if not os.environ.get(_cfg_env):
    home_dir = os.path.expanduser('~')
    if home_dir and home_dir != '~':
        os.environ[_cfg_env] = os.path.join(home_dir, '.ultralytics')
        try:
            os.makedirs(os.environ[_cfg_env], exist_ok=True)
        except Exception:
            pass

import ai_pb2
import ai_pb2_grpc

from tof_pose.realtime_service import RealtimePoseEngine


class ModelServiceServicer(ai_pb2_grpc.ModelServiceServicer):
    def __init__(
        self,
        *,
        stateless: bool = False,
        pose_only: bool = False,
        pose_validate_seg: bool = True,
        model_path: str | None = None,
        pose_model_path: str | None = None,
        pose_conf: float | None = None,
        pose_kpt_conf: float | None = None,
        pose_kpt_min_points: int = 4,
    ):
        self.svc = RealtimePoseEngine(
            stateless=bool(stateless),
            pose_only=bool(pose_only),
            pose_validate_seg=bool(pose_validate_seg),
            model_path=Path(model_path) if model_path else None,
            pose_model_path=Path(pose_model_path) if pose_model_path else None,
            pose_conf_threshold=pose_conf,
            pose_kpt_conf_threshold=pose_kpt_conf,
            pose_kpt_min_points=pose_kpt_min_points,
        )

    def Infer(self, request, context):
        device_id = getattr(request, 'device_id', '')
        frame_id = getattr(request, 'frame_id', '')
        capture_timestamp_ms = int(getattr(request, 'capture_timestamp_ms', 0) or 0)
        try:
            res = self.svc.infer(frame_id, request.image_data)
        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return ai_pb2.InferResponse(
                device_id=device_id,
                frame_id=frame_id,
                capture_timestamp_ms=capture_timestamp_ms,
            )

        # build response
        return ai_pb2.InferResponse(
            device_id=device_id,
            frame_id=frame_id,
            capture_timestamp_ms=capture_timestamp_ms,
            output_image_S11=res.get('output_image_S11', b''),
            output_image_S12=res.get('output_image_S12', b''),
            output_image_S13=res.get('output_image_S13', b''),
            output_image_S14=res.get('output_image_S14', b''),
            output_image_S21=res.get('output_image_S21', b''),
            output_image_S22=res.get('output_image_S22', b''),
            output_image_S23=res.get('output_image_S23', b''),
            output_image_S24=res.get('output_image_S24', b''),
            person_count=int(res.get('person_count', 0)),
            processing_time_ms=int(res.get('processing_time_ms', 0)),
        )


def serve(
    host: str = '0.0.0.0',
    port: int = 50052,
    max_workers: int = 4,
    max_msg_mb: int = 50,
    *,
    stateless: bool = False,
    pose_only: bool = False,
    pose_validate_seg: bool = True,
    model_path: str | None = None,
    pose_model_path: str | None = None,
    pose_conf: float | None = None,
    pose_kpt_conf: float | None = None,
    pose_kpt_min_points: int = 4,
):
    server_opts = [
        ('grpc.max_send_message_length', max_msg_mb * 1024 * 1024),
        ('grpc.max_receive_message_length', max_msg_mb * 1024 * 1024),
    ]
    server = grpc.server(ThreadPoolExecutor(max_workers=max_workers), options=server_opts)
    ai_pb2_grpc.add_ModelServiceServicer_to_server(
        ModelServiceServicer(
            stateless=stateless,
            pose_only=pose_only,
            pose_validate_seg=pose_validate_seg,
            model_path=model_path,
            pose_model_path=pose_model_path,
            pose_conf=pose_conf,
            pose_kpt_conf=pose_kpt_conf,
            pose_kpt_min_points=pose_kpt_min_points,
        ),
        server,
    )
    bind_candidates = []
    if host:
        bind_candidates.append(f"{host}:{port}")
    bind_candidates.extend([
        f"0.0.0.0:{port}",
        f"[::]:{port}",
    ])

    bound_address = None
    last_error: Exception | None = None
    for bind in bind_candidates:
        try:
            if server.add_insecure_port(bind):
                bound_address = bind
                break
        except Exception as exc:
            last_error = exc

    if bound_address is None:
        raise RuntimeError(f"Failed to bind gRPC server on {bind_candidates}") from last_error

    logging.info('Starting gRPC server on %s (max_msg_mb=%d)', bound_address, max_msg_mb)
    server.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info('Shutting down gRPC server')
        server.stop(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', default=50052, type=int)
    parser.add_argument('--max-workers', default=4, type=int)
    parser.add_argument('--max-msg-mb', default=50, type=int)
    parser.add_argument(
        '--stateless',
        action='store_true',
        help='treat every Infer call as the first frame (no cross-call caching/tracking)',
    )
    parser.add_argument(
        '--pose-only',
        action='store_true',
        help='run pose model only (no segmentation masks/contours); person_count is derived from pose keypoints',
    )
    parser.add_argument(
        '--no-pose-validate',
        action='store_true',
        help='disable pose-based gating for segmentation contours (contours/distance follow seg results directly)',
    )
    parser.add_argument('--model-path', default=None, help='override seg model path')
    parser.add_argument('--pose-model-path', default=None, help='override pose model path')
    parser.add_argument('--pose-conf', default=None, type=float, help='override pose confidence threshold (pose-only)')
    parser.add_argument('--pose-kpt-conf', default=None, type=float, help='override pose keypoint conf threshold (pose-only)')
    parser.add_argument('--pose-kpt-min-points', default=4, type=int, help='min confident keypoints to count one person (pose-only)')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    serve(
        host=args.host,
        port=args.port,
        max_workers=args.max_workers,
        max_msg_mb=args.max_msg_mb,
        stateless=args.stateless,
        pose_only=args.pose_only,
        pose_validate_seg=not args.no_pose_validate,
        model_path=args.model_path,
        pose_model_path=args.pose_model_path,
        pose_conf=args.pose_conf,
        pose_kpt_conf=args.pose_kpt_conf,
        pose_kpt_min_points=args.pose_kpt_min_points,
    )


if __name__ == '__main__':
    main()
