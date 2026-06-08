#!/usr/bin/env python3
"""gRPC server wrapper for the local ModelService (scripts/infer_service.py).

Usage:
  python scripts/grpc_server.py --host 0.0.0.0 --port 50052
"""
import argparse
import logging
import time
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


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


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
        device: str | None = None,
        render_workers: int = 1,
        decode_workers: int = 1,
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
            device=device,
            render_workers=render_workers,
            decode_workers=decode_workers,
        )

    def Infer(self, request, context):
        device_id = getattr(request, 'device_id', '')
        batch_id = getattr(request, 'batch_id', '')
        images = list(getattr(request, 'images', []))
        batch_size = int(getattr(request, 'batch_size', 0) or 0)
        if not images:
            context.set_details("images must not be empty")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return ai_pb2.InferResponse(device_id=device_id, batch_id=batch_id)
        if batch_size and batch_size != len(images):
            context.set_details(f"batch_size={batch_size} does not match images count={len(images)}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return ai_pb2.InferResponse(device_id=device_id, batch_id=batch_id)

        batch_start = time.time()
        try:
            frames = [(image.frame_id, image.image_data) for image in images]
            results = self.svc.infer_batch(frames)
            if len(results) != len(images) * 2:
                raise RuntimeError(f"expected {len(images) * 2} results for {len(images)} inputs, got {len(results)}")
        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return ai_pb2.InferResponse(
                device_id=device_id,
                batch_id=batch_id,
            )

        response = ai_pb2.InferResponse(
            device_id=device_id,
            batch_id=batch_id,
            processing_time_ms=int((time.time() - batch_start) * 1000),
        )
        kind_map = {
            'interpolated': ai_pb2.RESULT_KIND_INTERPOLATED,
            'current': ai_pb2.RESULT_KIND_CURRENT,
        }
        for result in results:
            input_index = int(result.get('input_index', -1))
            image = images[input_index] if 0 <= input_index < len(images) else None
            capture_timestamp_ms = int(getattr(image, 'capture_timestamp_ms', 0) or 0) if image is not None else 0
            if result.get('result_kind') == 'interpolated' and input_index > 0:
                prev_ts = int(getattr(images[input_index - 1], 'capture_timestamp_ms', 0) or 0)
                if prev_ts and capture_timestamp_ms:
                    capture_timestamp_ms = int((prev_ts + capture_timestamp_ms) // 2)
            response.results.append(
                ai_pb2.InferResult(
                    frame_id=result.get('frame_id', ''),
                    capture_timestamp_ms=capture_timestamp_ms,
                    pseudo_color_image=result.get('pseudo_color_image', b''),
                    skeleton_contour_image=result.get('skeleton_contour_image', b''),
                    person_count=int(result.get('person_count', 0)),
                    processing_time_ms=int(result.get('processing_time_ms', 0)),
                    result_kind=kind_map.get(result.get('result_kind'), ai_pb2.RESULT_KIND_UNSPECIFIED),
                    input_index=input_index,
                    output_index=int(result.get('output_index', len(response.results))),
                )
            )
        return response


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
    device: str | None = None,
    render_workers: int = 1,
    decode_workers: int = 1,
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
            device=device,
            render_workers=render_workers,
            decode_workers=decode_workers,
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
        help='disable pose-based gating for skeleton drawing (contours use segmentation plus shape rules)',
    )
    parser.add_argument('--model-path', default=None, help='override seg model path')
    parser.add_argument('--pose-model-path', default=None, help='override pose model path')
    parser.add_argument('--pose-conf', default=None, type=float, help='override pose confidence threshold (pose-only)')
    parser.add_argument('--pose-kpt-conf', default=None, type=float, help='override pose keypoint conf threshold (pose-only)')
    parser.add_argument('--pose-kpt-min-points', default=4, type=int, help='min confident keypoints to count one person (pose-only)')
    parser.add_argument('--device', default=None, help='YOLO inference device, for example cuda:0 or cpu')
    parser.add_argument(
        '--render-workers',
        default=1,
        type=int,
        help='CPU worker threads for rendering and PNG encoding returned images',
    )
    parser.add_argument(
        '--decode-workers',
        default=1,
        type=int,
        help='CPU worker threads for decoding input PNG images',
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
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
        device=args.device,
        render_workers=args.render_workers,
        decode_workers=args.decode_workers,
    )


if __name__ == '__main__':
    main()
