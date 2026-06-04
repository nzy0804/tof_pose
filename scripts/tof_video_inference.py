from pathlib import Path
import argparse
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tof_pose.video_inference import run


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline inference on a video using the same pipeline as the gRPC server.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input video path. If omitted, uses the default capture video path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path. If omitted, writes to the default output path.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Segmentation model weights path. If omitted, uses the default model path.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="(Deprecated) kept for compatibility; gRPC pipeline is fixed to 320x320.",
    )

    parser.add_argument(
        "--stateless",
        action="store_true",
        help="Treat every frame as the first frame (no cross-frame caching/tracking).",
    )
    parser.add_argument(
        "--pose-only",
        action="store_true",
        help="Use pose model only (no segmentation dependency); person_count is derived from keypoints.",
    )
    parser.add_argument(
        "--no-pose-validate",
        action="store_true",
        help="Disable pose-based gating for skeleton drawing (contours use segmentation plus shape rules).",
    )
    parser.add_argument(
        "--pose-model-path",
        type=Path,
        default=None,
        help="Pose model weights path. If omitted, uses the default pose model path.",
    )
    parser.add_argument(
        "--pose-conf",
        type=float,
        default=None,
        help="Override pose confidence threshold (pose-only).",
    )
    parser.add_argument(
        "--pose-kpt-conf",
        type=float,
        default=None,
        help="Override pose keypoint confidence threshold (pose-only).",
    )
    parser.add_argument(
        "--pose-kpt-min-points",
        type=int,
        default=4,
        help="Min confident keypoints to count one person (pose-only).",
    )
    parser.add_argument(
        "--view",
        choices=["gray", "color", "skeleton", "contour"],
        default="skeleton",
        help="Which view to write: gray/color use pseudo color; skeleton/contour use combined skeleton+contour.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV preview window.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        input_video=args.input,
        output_video=args.output,
        model_path=args.model,
        stateless=args.stateless,
        pose_only=args.pose_only,
        pose_validate_seg=not args.no_pose_validate,
        pose_model_path=args.pose_model_path,
        pose_conf=args.pose_conf,
        pose_kpt_conf=args.pose_kpt_conf,
        pose_kpt_min_points=args.pose_kpt_min_points,
        view=args.view,
        show_window=not args.no_display,
    )
