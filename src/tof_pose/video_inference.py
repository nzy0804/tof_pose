from pathlib import Path

import cv2
from ultralytics import YOLO

from tof_pose.paths import DEFAULT_CAPTURE_VIDEO, DEFAULT_MODEL_PATH, DEFAULT_POSE_VIDEO
from tof_pose.pose_drawing import KPT_CONF_THRESHOLD, draw_stick_figure


CONF_THRESHOLD = 0.25
WINDOW_NAME = "tof_pose_video_inference"
LOG_PREFIX = "[tof_pose_video_inference]"


def run(
    input_video: Path | None = None,
    output_video: Path | None = None,
    model_path: Path | None = None,
) -> None:
    source = Path(input_video) if input_video else DEFAULT_CAPTURE_VIDEO
    target = Path(output_video) if output_video else DEFAULT_POSE_VIDEO
    model_file = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    target.parent.mkdir(parents=True, exist_ok=True)

    print(f"{LOG_PREFIX} 正在加载模型: {model_file}")
    model = YOLO(str(model_file))

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        print(f"{LOG_PREFIX} 无法打开视频: {source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(
        str(target),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    print(f"{LOG_PREFIX} 开始处理: {source} total_frames={total_frames}")
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        display_frame = frame.copy()
        result = results[0]
        if result.keypoints is not None:
            kpts_xy = result.keypoints.xy.cpu().numpy()
            kpts_conf = result.keypoints.conf.cpu().numpy()
            for idx in range(len(kpts_xy)):
                draw_stick_figure(display_frame, kpts_xy[idx], kpts_conf[idx], KPT_CONF_THRESHOLD)

        # 每一帧都写回输出视频，保证结果视频与输入视频帧数对齐。
        out.write(display_frame)
        frame_idx += 1

        if frame_idx % 10 == 0 and total_frames:
            percent = (frame_idx / total_frames) * 100
            print(f"{LOG_PREFIX} {frame_idx}/{total_frames} ({percent:.1f}%)", end="\r")

        cv2.imshow(WINDOW_NAME, display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n{LOG_PREFIX} 处理完成: {target}")
