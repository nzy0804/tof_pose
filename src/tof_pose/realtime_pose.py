import queue
import struct
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import serial
from ultralytics import YOLO

from tof_pose.paths import DEFAULT_MODEL_PATH
from tof_pose.person_distance import estimate_person_distance
from tof_pose.pose_drawing import KPT_CONF_THRESHOLD, draw_stick_figure
from tof_pose.tracking import Detection, PersonTracker, Track


BAUD = 921600
TIMEOUT = 0.05
ENDIAN = "<"
FRAME_HEAD = b"\x00\xFF"
ALLOWED_TAILS = (0xCC, 0xDD)
RAW_QUEUE_MAXSIZE = 50
FRAME_QUEUE_MAXSIZE = 3
DISPLAY_SIZE = (320, 320)
DISPLAY_SCALE = 3
CONF_THRESHOLD = 0.2
WINDOW_NAME = "tof_pose"
LOG_PREFIX = "[tof_pose]"
CONTOUR_SMOOTH_ALPHA = 0.70
CONTOUR_HISTORY_MAX_MISSED = 12
TRACK_RENDER_MAX_MISSED = 8
TRACK_MIN_HITS = 2
PAIRWISE_HORIZONTAL_FOV_DEG = 70.0
PAIRWISE_MAX_DISTANCE = 400.0


def _clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray | None:
    clipped = box.astype(np.float32).copy()
    clipped[0] = np.clip(clipped[0], 0, width - 1)
    clipped[1] = np.clip(clipped[1], 0, height - 1)
    clipped[2] = np.clip(clipped[2], 1, width)
    clipped[3] = np.clip(clipped[3], 1, height)
    if clipped[2] - clipped[0] < 2 or clipped[3] - clipped[1] < 2:
        return None
    return clipped


def _blend_box_with_track(detection_box: np.ndarray, track: Track) -> np.ndarray:
    """用 Kalman 预测框平滑当前检测框，减少 YOLO 抖动。"""
    track_box = track.predicted_box.astype(np.float32)
    det_box = detection_box.astype(np.float32)
    history_weight = 0.35 if track.hits < 4 else 0.45
    return ((1.0 - history_weight) * det_box + history_weight * track_box).astype(np.float32)


def _project_track_position(track: Track, frame_width: int) -> tuple[float, float] | None:
    """将人体中心近似投影到相机前方平面，用于估计人间距离。"""
    distance = track.distance if track.distance is not None else track.predicted_distance
    if distance is None:
        return None

    box = track.box if track.missed_frames == 0 else track.predicted_box
    center_x = float((box[0] + box[2]) * 0.5)
    normalized_x = (center_x - frame_width * 0.5) / max(frame_width * 0.5, 1.0)
    half_fov_rad = np.deg2rad(PAIRWISE_HORIZONTAL_FOV_DEG * 0.5)
    angle = normalized_x * half_fov_rad
    lateral_x = float(distance) * float(np.tan(angle))
    return lateral_x, float(distance)


def _compute_pairwise_distances(
    tracks: list[Track],
    frame_width: int,
) -> tuple[str, list[tuple[int, int, float]]]:
    """统计当前可见人体之间的近似空间距离。"""
    pairs: list[tuple[int, int, float]] = []
    valid_tracks = [track for track in tracks if track.missed_frames == 0]
    for idx in range(len(valid_tracks)):
        pos_a = _project_track_position(valid_tracks[idx], frame_width)
        if pos_a is None:
            continue
        for jdx in range(idx + 1, len(valid_tracks)):
            pos_b = _project_track_position(valid_tracks[jdx], frame_width)
            if pos_b is None:
                continue
            distance = float(np.hypot(pos_a[0] - pos_b[0], pos_a[1] - pos_b[1]))
            if 0.0 < distance <= PAIRWISE_MAX_DISTANCE:
                pairs.append((valid_tracks[idx].track_id, valid_tracks[jdx].track_id, distance))

    if not pairs:
        return "Pair Dist: N/A", []

    nearest = min(pairs, key=lambda item: item[2])
    return f"Pair Dist: {nearest[0]}-{nearest[1]} ~{nearest[2]:.1f}", pairs


def run(port: str = "COM8", model_path: Path | None = None) -> None:
    """启动实时 ToF 姿态推理、距离估计和人体追踪流程。"""
    model_file = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    ser = serial.Serial(port, BAUD, timeout=TIMEOUT)
    raw_queue: queue.Queue[bytes] = queue.Queue(maxsize=RAW_QUEUE_MAXSIZE)
    frame_queue: queue.Queue[tuple[int, int, bytes]] = queue.Queue(
        maxsize=FRAME_QUEUE_MAXSIZE
    )
    stop_event = threading.Event()

    print(f"{LOG_PREFIX} port={port} model={model_file} size={DISPLAY_SIZE}")
    ser.write(b"AT+FPS=19\r")
    time.sleep(0.1)
    ser.write(b"AT+DISP=2\r")
    time.sleep(0.1)

    def reader_thread() -> None:
        while not stop_event.is_set():
            try:
                n = ser.in_waiting
                data = ser.read(min(4096, n) if n else 256)
            except Exception:
                break
            if not data:
                time.sleep(0.001)
                continue

            while raw_queue.full():
                try:
                    raw_queue.get_nowait()
                except queue.Empty:
                    break
            try:
                raw_queue.put_nowait(data)
            except queue.Full:
                time.sleep(0.001)

    def relay_thread() -> None:
        last_frameid = 0
        buf = bytearray()
        while not stop_event.is_set() or not raw_queue.empty():
            try:
                chunk = raw_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            buf += chunk
            while True:
                idx = buf.find(FRAME_HEAD)
                if idx < 0:
                    if len(buf) > 8192:
                        buf.clear()
                    break
                if idx > 0:
                    del buf[:idx]
                if len(buf) < 4:
                    break

                try:
                    data_len = struct.unpack(ENDIAN + "H", buf[2:4])[0]
                except struct.error:
                    break

                frame_len = 2 + 2 + data_len + 2
                if len(buf) < frame_len:
                    break

                frame = bytes(buf[:frame_len])
                del buf[:frame_len]

                if frame[-1] not in ALLOWED_TAILS:
                    continue
                if frame[-2] != (sum(frame[:-2]) & 0xFF):
                    continue

                try:
                    res_r = frame[14]
                    res_c = frame[15]
                    frameid = struct.unpack(ENDIAN + "H", frame[16:18])[0]
                except (IndexError, struct.error):
                    continue

                if frameid == last_frameid:
                    continue
                last_frameid = frameid

                payload_len = data_len - 16
                payload = frame[20 : 20 + payload_len]
                try:
                    frame_queue.put_nowait((res_r, res_c, payload))
                except queue.Full:
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        frame_queue.put_nowait((res_r, res_c, payload))
                    except queue.Full:
                        pass

    def processor_thread() -> None:
        print(f"{LOG_PREFIX} 正在加载模型...", flush=True)
        model = YOLO(str(model_file))
        print(f"{LOG_PREFIX} 模型加载完成，按 q 退出。", flush=True)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        width, height = DISPLAY_SIZE
        last_print = time.time()
        interval_count = 0
        tracker = PersonTracker()
        contour_history: dict[int, np.ndarray] = {}
        contour_history_missed: dict[int, int] = {}

        def smooth_contour_mask(track_id: int, contour: np.ndarray | None) -> np.ndarray | None:
            prev_mask = contour_history.get(track_id)
            if contour is None:
                if prev_mask is None:
                    return None
                contour_history_missed[track_id] = contour_history_missed.get(track_id, 0) + 1
                if contour_history_missed[track_id] > CONTOUR_HISTORY_MAX_MISSED:
                    contour_history.pop(track_id, None)
                    contour_history_missed.pop(track_id, None)
                    return None
                return prev_mask.copy()

            current_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(current_mask, [contour], -1, 255, thickness=cv2.FILLED)

            if prev_mask is None:
                smoothed_mask = current_mask
            else:
                blended = cv2.addWeighted(
                    current_mask,
                    CONTOUR_SMOOTH_ALPHA,
                    prev_mask,
                    1.0 - CONTOUR_SMOOTH_ALPHA,
                    0.0,
                )
                _, smoothed_mask = cv2.threshold(blended, 127, 255, cv2.THRESH_BINARY)

            contour_history[track_id] = smoothed_mask.copy()
            contour_history_missed[track_id] = 0
            return smoothed_mask

        while not stop_event.is_set() or not frame_queue.empty():
            try:
                res_r, res_c, payload = frame_queue.get(timeout=0.1)
            except queue.Empty:
                now = time.time()
                if now - last_print >= 5.0:
                    fps = interval_count / (now - last_print) if now > last_print else 0.0
                    print(f"[{time.strftime('%H:%M:%S')}] {LOG_PREFIX} fps={fps:.2f}", flush=True)
                    last_print = now
                    interval_count = 0
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                continue

            depth = np.frombuffer(payload, dtype=np.uint8)
            if depth.size != res_r * res_c:
                continue
            depth = depth.reshape((res_r, res_c))

            depth_up = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
            color_img = cv2.applyColorMap(depth_up, cv2.COLORMAP_MAGMA)

            try:
                results = model(color_img, conf=CONF_THRESHOLD, verbose=False)
            except Exception as exc:
                print(f"{LOG_PREFIX} 推理失败: {exc}", flush=True)
                cv2.imshow(WINDOW_NAME, color_img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break
                continue

            display = color_img.copy()
            result = results[0]
            num_persons = 0
            person_count = 0
            tracked_labels: list[str] = []
            now = time.time()
            active_track_ids: set[int] = set()
            if result.keypoints is not None and result.boxes is not None:
                kpts_xy = result.keypoints.xy.cpu().numpy()
                kpts_conf = result.keypoints.conf.cpu().numpy()
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                num_persons = min(len(kpts_xy), len(boxes_xyxy))

                detections: list[Detection] = []
                for idx in range(num_persons):
                    rough_estimate = estimate_person_distance(
                        depth_up,
                        boxes_xyxy[idx],
                        kpts_xy[idx],
                        kpts_conf[idx],
                        KPT_CONF_THRESHOLD,
                    )
                    detections.append(
                        Detection(
                            box=boxes_xyxy[idx].copy(),
                            keypoints=kpts_xy[idx].copy(),
                            kpt_conf=kpts_conf[idx].copy(),
                            distance=rough_estimate.distance,
                        )
                    )

                track_ids = tracker.update(detections, timestamp=now)
                track_map = {
                    track.track_id: track
                    for track in tracker.visible_tracks(
                        max_missed_frames=TRACK_RENDER_MAX_MISSED,
                        min_hits=1,
                    )
                }

                for idx in range(num_persons):
                    track_id = track_ids[idx] if idx < len(track_ids) else idx + 1
                    track = track_map.get(track_id)
                    if track is None:
                        continue
                    active_track_ids.add(track_id)

                    fused_box = _blend_box_with_track(boxes_xyxy[idx], track)
                    fused_box = _clip_box(fused_box, width, height)
                    if fused_box is None:
                        continue
                    person_count += 1
                    fused_keypoints = track.smoothed_keypoints
                    fused_kpt_conf = track.smoothed_kpt_conf
                    draw_stick_figure(display, fused_keypoints, fused_kpt_conf, KPT_CONF_THRESHOLD)

                    estimate = estimate_person_distance(
                        depth_up,
                        fused_box,
                        fused_keypoints,
                        fused_kpt_conf,
                        KPT_CONF_THRESHOLD,
                    )
                    if estimate.distance is not None:
                        track.distance = estimate.distance

                    shifted_contour = None
                    if estimate.contour is not None:
                        shifted_contour = estimate.contour + np.array([[[estimate.anchor[0], estimate.anchor[1]]]])

                    smoothed_mask = smooth_contour_mask(track_id, shifted_contour)
                    if smoothed_mask is not None:
                        smooth_contours, _ = cv2.findContours(
                            smoothed_mask,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )
                        if smooth_contours:
                            best_smooth = max(smooth_contours, key=cv2.contourArea)
                            cv2.drawContours(display, [best_smooth], -1, (0, 255, 0), 2, cv2.LINE_AA)

                    box = fused_box
                    x1 = max(0, int(round(box[0])))
                    y1 = max(18, int(round(box[1])) - 8)
                    if estimate.distance is None:
                        tracked_labels.append(f"{track_id}:N/A")
                        label = f"ID {track_id} Dist=N/A"
                    else:
                        tracked_labels.append(f"{track_id}:{estimate.distance:.1f}")
                        label = f"ID {track_id} Dist~{estimate.distance:.1f}"
                    cv2.putText(
                        display,
                        label,
                        (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        display,
                        label,
                        (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (20, 20, 20),
                        1,
                        cv2.LINE_AA,
                    )
            else:
                tracker.update([], timestamp=now)

            for track_id in list(contour_history.keys()):
                if track_id in active_track_ids:
                    continue
                smooth_contour_mask(track_id, None)

            interval_count += 1
            now = time.time()
            if now - last_print >= 5.0:
                fps = interval_count / (now - last_print) if now > last_print else 0.0
                labels_text = ", ".join(tracked_labels) if tracked_labels else "none"
                pair_text, _ = _compute_pairwise_distances(
                    tracker.visible_tracks(
                        max_missed_frames=0,
                        min_hits=TRACK_MIN_HITS,
                    ),
                    width,
                )
                print(
                    f"[{time.strftime('%H:%M:%S')}] {LOG_PREFIX} fps={fps:.2f} persons={person_count} tracks={labels_text} {pair_text}",
                    flush=True,
                )
                last_print = now
                interval_count = 0

            elapsed = max(now - last_print, 1e-6)
            fps_approx = interval_count / elapsed
            pair_text, pair_stats = _compute_pairwise_distances(
                tracker.visible_tracks(
                    max_missed_frames=0,
                    min_hits=TRACK_MIN_HITS,
                ),
                width,
            )
            cv2.putText(
                display,
                f"Persons: {person_count}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                f"Src: {res_c}x{res_r} FPS~{fps_approx:.1f}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                pair_text,
                (10, 74),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 255, 180) if pair_stats else (170, 170, 170),
                1,
                cv2.LINE_AA,
            )

            display_show = cv2.resize(
                display,
                (width * DISPLAY_SCALE, height * DISPLAY_SCALE),
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.imshow(WINDOW_NAME, display_show)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break

        cv2.destroyAllWindows()

    threads = [
        threading.Thread(target=reader_thread, name="serial-reader", daemon=True),
        threading.Thread(target=relay_thread, name="serial-relay", daemon=True),
        threading.Thread(target=processor_thread, name="frame-processor", daemon=True),
    ]

    for thread in threads:
        thread.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"\n{LOG_PREFIX} 已中断，正在退出...")
        stop_event.set()
    finally:
        stop_event.set()
        for thread in threads:
            thread.join(timeout=3.0)
        try:
            ser.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
