from pathlib import Path

import cv2
from ultralytics import YOLO

from maixsense.paths import DEFAULT_CAPTURE_VIDEO, DEFAULT_MODEL_PATH, DEFAULT_POSE_VIDEO
from maixsense.pose_drawing import KPT_CONF_THRESHOLD, draw_stick_figure


CONF_THRESHOLD = 0.25


def run(
    input_video: Path | None = None,
    output_video: Path | None = None,
    model_path: Path | None = None,
) -> None:
    source = Path(input_video) if input_video else DEFAULT_CAPTURE_VIDEO
    target = Path(output_video) if output_video else DEFAULT_POSE_VIDEO
    model_file = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    target.parent.mkdir(parents=True, exist_ok=True)

    print(f"[infer] loading model: {model_file}")
    model = YOLO(str(model_file))

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        print(f"[infer] cannot open video: {source}")
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

    print(f"[infer] processing {source} total_frames={total_frames}")
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
            print(f"[infer] {frame_idx}/{total_frames} ({percent:.1f}%)", end="\r")

        cv2.imshow("Video Inference", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n[infer] done: {target}")
