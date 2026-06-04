from pathlib import Path

import cv2
import numpy as np

from tof_pose.paths import DEFAULT_CAPTURE_VIDEO, DEFAULT_MODEL_PATH, DEFAULT_POSE_MODEL_PATH, DEFAULT_POSE_VIDEO
from tof_pose.realtime_service import RealtimePoseEngine


WINDOW_NAME = "tof_pose_video_inference"
LOG_PREFIX = "[tof_pose_video_inference]"


VIEW_GRAY = "gray"
VIEW_COLOR = "color"
VIEW_SKELETON = "skeleton"
VIEW_CONTOUR = "contour"


def run(
    input_video: Path | None = None,
    output_video: Path | None = None,
    model_path: Path | None = None,
    *,
    stateless: bool = False,
    pose_only: bool = False,
    pose_validate_seg: bool = True,
    pose_model_path: Path | None = None,
    pose_conf: float | None = None,
    pose_kpt_conf: float | None = None,
    pose_kpt_min_points: int = 4,
    view: str = VIEW_SKELETON,
    show_window: bool = True,
) -> None:
    source = Path(input_video) if input_video else DEFAULT_CAPTURE_VIDEO
    target = Path(output_video) if output_video else DEFAULT_POSE_VIDEO
    model_file = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    pose_model_file = Path(pose_model_path) if pose_model_path else DEFAULT_POSE_MODEL_PATH

    target.parent.mkdir(parents=True, exist_ok=True)

    print(f"{LOG_PREFIX} 正在加载引擎(与 gRPC 一致)")
    print(f"{LOG_PREFIX} seg_model={model_file}")
    print(f"{LOG_PREFIX} pose_model={pose_model_file} pose_only={bool(pose_only)} stateless={bool(stateless)}")
    engine = RealtimePoseEngine(
        model_path=model_file,
        pose_model_path=pose_model_file,
        stateless=bool(stateless),
        pose_only=bool(pose_only),
        pose_validate_seg=bool(pose_validate_seg),
        pose_conf_threshold=pose_conf,
        pose_kpt_conf_threshold=pose_kpt_conf,
        pose_kpt_min_points=int(pose_kpt_min_points),
    )

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        print(f"{LOG_PREFIX} 无法打开视频: {source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_width = 320
    out_height = 320

    out = cv2.VideoWriter(
        str(target),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_width, out_height),
    )

    print(f"{LOG_PREFIX} 开始处理: {source} total_frames={total_frames}")
    frame_idx = 0
    warned_empty_view = False

    view = str(view or VIEW_SKELETON).strip().lower()
    if view not in (VIEW_GRAY, VIEW_COLOR, VIEW_SKELETON, VIEW_CONTOUR):
        raise ValueError(f"invalid view: {view}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ok, buf = cv2.imencode(".png", gray, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            raise RuntimeError("failed to encode frame as PNG")

        res = engine.infer(str(frame_idx), buf.tobytes())
        if view in (VIEW_GRAY, VIEW_COLOR):
            view_bytes = res.get("pseudo_color_image_s2", b"")
        else:
            view_bytes = res.get("skeleton_contour_image_s2", b"")

        if not view_bytes:
            if not warned_empty_view:
                print(f"{LOG_PREFIX} 视图输出为空，可能是首帧语义或推理异常。")
                warned_empty_view = True
            continue

        img_arr = np.frombuffer(view_bytes, dtype=np.uint8)
        output_frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if output_frame is None:
            raise RuntimeError("failed to decode engine output PNG")
        if output_frame.shape[1] != out_width or output_frame.shape[0] != out_height:
            output_frame = cv2.resize(output_frame, (out_width, out_height), interpolation=cv2.INTER_NEAREST)

        out.write(output_frame)
        frame_idx += 1

        if frame_idx % 10 == 0 and total_frames:
            percent = (frame_idx / total_frames) * 100
            print(f"{LOG_PREFIX} {frame_idx}/{total_frames} ({percent:.1f}%)", end="\r")

        if show_window:
            cv2.imshow(WINDOW_NAME, output_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    if show_window:
        cv2.destroyAllWindows()
    print(f"\n{LOG_PREFIX} 处理完成: {target}")
