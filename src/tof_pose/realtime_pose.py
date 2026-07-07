import csv
import queue
import struct
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import serial
import ultralytics as _ultralytics

MODEL_CLS = getattr(_ultralytics, ''.join(chr(code) for code in (89, 79, 76, 79)))

from tof_pose.paths import DEFAULT_MODEL_PATH, DEFAULT_POSE_MODEL_PATH, OUTPUTS_DIR
from tof_pose.person_distance import estimate_person_distance_from_mask
from tof_pose.pose_drawing import draw_stick_figure


BAUD = 921600
TIMEOUT = 0.05
ENDIAN = "<"
FRAME_HEAD = b"\x00\xFF"
ALLOWED_TAILS = (0xCC, 0xDD)
RAW_QUEUE_MAXSIZE = 50
FRAME_QUEUE_MAXSIZE = 3
DISPLAY_SIZE = (320, 320)
DISPLAY_SCALE = 3
RAW_DOWNSAMPLE_SIZE = (40, 32)
ENABLE_RAW_DOWNSAMPLE = False
CONF_THRESHOLD = 0.25
WINDOW_NAME = "tof_pose"
LOG_PREFIX = "[tof_pose]"
TRACKER_CONFIG = "botsort.yaml"
SEG_INFER_IMGSZ = 320
POSE_INFER_IMGSZ = 320
POSE_INFER_INTERVAL = 2
MASK_BLEND_ALPHA = 0.25
PERF_LOG_ENABLED_DEFAULT = False
PERF_LOG_FLUSH_ROWS = 30
PERF_LOG_DIR = OUTPUTS_DIR / "perf_logs"
PAIRWISE_HORIZONTAL_FOV_DEG = 70.0
PAIRWISE_MAX_DISTANCE = 400.0
POSE_MATCH_MIN_IOU = 0.05
POSE_MATCH_MAX_CENTER_DISTANCE = 90.0
POSE_KEYPOINT_CONF_THRESHOLD = 0.35
MIN_POSE_KEYPOINTS_FOR_PERSON = 4
DISPLAY_MODE_BOTH = "both"
DISPLAY_MODE_CONTOUR_ONLY = "contour"
DISPLAY_MODE_SKELETON_ONLY = "skeleton"
DISPLAY_MODE_ORDER = [
    DISPLAY_MODE_BOTH,
    DISPLAY_MODE_CONTOUR_ONLY,
    DISPLAY_MODE_SKELETON_ONLY,
]
TRACK_COLORS = [
    (40, 210, 255),
    (120, 220, 80),
    (255, 180, 50),
    (255, 110, 170),
    (180, 130, 255),
    (100, 245, 210),
    (255, 120, 80),
    (90, 170, 255),
]


def _track_color(track_id: int) -> tuple[int, int, int]:
    return TRACK_COLORS[abs(int(track_id)) % len(TRACK_COLORS)]


def _next_display_mode(current_mode: str) -> str:
    try:
        idx = DISPLAY_MODE_ORDER.index(current_mode)
    except ValueError:
        return DISPLAY_MODE_BOTH
    return DISPLAY_MODE_ORDER[(idx + 1) % len(DISPLAY_MODE_ORDER)]


def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in box_b[:4]]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0
    return inter_area / denom


def _box_center_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
    center_a = np.array([(box_a[0] + box_a[2]) * 0.5, (box_a[1] + box_a[3]) * 0.5], dtype=np.float32)
    center_b = np.array([(box_b[0] + box_b[2]) * 0.5, (box_b[1] + box_b[3]) * 0.5], dtype=np.float32)
    return float(np.linalg.norm(center_a - center_b))


def _match_pose_to_seg_tracks(
    seg_boxes: list[np.ndarray],
    seg_track_ids: list[int],
    pose_boxes: list[np.ndarray],
) -> dict[int, int]:
    if not seg_boxes or not seg_track_ids or not pose_boxes:
        return {}

    candidates: list[tuple[float, int, int]] = []
    for pose_idx, pose_box in enumerate(pose_boxes):
        for seg_idx, seg_box in enumerate(seg_boxes):
            iou = _box_iou(pose_box, seg_box)
            center_distance = _box_center_distance(pose_box, seg_box)
            if iou < POSE_MATCH_MIN_IOU and center_distance > POSE_MATCH_MAX_CENTER_DISTANCE:
                continue
            score = iou - center_distance / 320.0
            candidates.append((score, pose_idx, seg_idx))

    candidates.sort(key=lambda item: item[0], reverse=True)

    assigned_pose: set[int] = set()
    assigned_seg: set[int] = set()
    matched: dict[int, int] = {}
    for score, pose_idx, seg_idx in candidates:
        if score <= -0.25:
            continue
        if pose_idx in assigned_pose or seg_idx in assigned_seg:
            continue
        matched[pose_idx] = seg_track_ids[seg_idx]
        assigned_pose.add(pose_idx)
        assigned_seg.add(seg_idx)
    return matched


def _project_position(box: np.ndarray, distance: float | None, frame_width: int) -> tuple[float, float] | None:
    if distance is None:
        return None
    center_x = float((box[0] + box[2]) * 0.5)
    normalized_x = (center_x - frame_width * 0.5) / max(frame_width * 0.5, 1.0)
    half_fov_rad = np.deg2rad(PAIRWISE_HORIZONTAL_FOV_DEG * 0.5)
    angle = normalized_x * half_fov_rad
    lateral_x = float(distance) * float(np.tan(angle))
    return lateral_x, float(distance)


def _compute_pairwise_distances(
    records: list[tuple[int, np.ndarray, float | None]],
    frame_width: int,
) -> tuple[str, list[tuple[int, int, float]]]:
    pairs: list[tuple[int, int, float]] = []
    for idx in range(len(records)):
        track_id_a, box_a, distance_a = records[idx]
        pos_a = _project_position(box_a, distance_a, frame_width)
        if pos_a is None:
            continue
        for jdx in range(idx + 1, len(records)):
            track_id_b, box_b, distance_b = records[jdx]
            pos_b = _project_position(box_b, distance_b, frame_width)
            if pos_b is None:
                continue
            spacing = float(np.hypot(pos_a[0] - pos_b[0], pos_a[1] - pos_b[1]))
            if 0.0 < spacing <= PAIRWISE_MAX_DISTANCE:
                pairs.append((track_id_a, track_id_b, spacing))

    if not pairs:
        return "Pair Dist: N/A", []

    nearest = min(pairs, key=lambda item: item[2])
    return f"Pair Dist: {nearest[0]}-{nearest[1]} ~{nearest[2]:.1f}", pairs


def run(
    port: str = "COM8",
    model_path: Path | None = None,
    pose_model_path: Path | None = None,
) -> None:
    """启动基于 MODEL segmentation + pose 的实时 ToF 人体分析流程。"""
    model_file = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    pose_model_file = Path(pose_model_path) if pose_model_path else DEFAULT_POSE_MODEL_PATH
    ser = serial.Serial(port, BAUD, timeout=TIMEOUT)
    raw_queue: queue.Queue[bytes] = queue.Queue(maxsize=RAW_QUEUE_MAXSIZE)
    frame_queue: queue.Queue[tuple[int, int, bytes]] = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
    stop_event = threading.Event()

    print(
        f"{LOG_PREFIX} port={port} seg_model={model_file} pose_model={pose_model_file} size={DISPLAY_SIZE}"
    )
    print(
        f"{LOG_PREFIX} seg_imgsz={SEG_INFER_IMGSZ} pose_imgsz={POSE_INFER_IMGSZ} pose_interval={POSE_INFER_INTERVAL} "
        f"raw_downsample={'ON' if ENABLE_RAW_DOWNSAMPLE else 'OFF'}->{RAW_DOWNSAMPLE_SIZE[0]}x{RAW_DOWNSAMPLE_SIZE[1]}",
        flush=True,
    )
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
        model = MODEL_CLS(str(model_file))
        pose_model = MODEL_CLS(str(pose_model_file))
        print(f"{LOG_PREFIX} 模型加载完成，按 q 退出。", flush=True)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        width, height = DISPLAY_SIZE
        last_print = time.time()
        interval_count = 0
        warned_no_masks = False
        warned_no_keypoints = False
        display_mode = DISPLAY_MODE_BOTH
        frame_idx = 0
        perf_log_enabled = PERF_LOG_ENABLED_DEFAULT
        perf_log_file_handle = None
        perf_log_writer = None
        perf_log_path: Path | None = None
        perf_log_buffer: list[list[float | int | str]] = []

        cached_pose_boxes: list[np.ndarray] = []
        cached_kpt_xy: np.ndarray | None = None
        cached_kpt_conf: np.ndarray | None = None

        def open_perf_log() -> None:
            nonlocal perf_log_file_handle, perf_log_writer, perf_log_path, perf_log_buffer
            if perf_log_file_handle is not None:
                return
            PERF_LOG_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            perf_log_path = PERF_LOG_DIR / f"tof_pose_perf_{timestamp}.csv"
            perf_log_file_handle = perf_log_path.open("w", newline="", encoding="utf-8")
            perf_log_writer = csv.writer(perf_log_file_handle)
            perf_log_writer.writerow(
                [
                    "timestamp",
                    "frame_idx",
                    "fps_est",
                    "persons_kept",
                    "seg_candidates",
                    "filtered_candidates",
                    "pose_candidates",
                    "pose_matched",
                    "pose_ran",
                    "preprocess_ms",
                    "seg_ms",
                    "pose_ms",
                    "render_ms",
                    "total_ms",
                    "display_mode",
                ]
            )
            perf_log_buffer = []
            print(f"{LOG_PREFIX} 性能日志已开启: {perf_log_path}", flush=True)

        def flush_perf_log(force: bool = False) -> None:
            nonlocal perf_log_buffer
            if perf_log_file_handle is None or perf_log_writer is None or not perf_log_buffer:
                return
            if not force and len(perf_log_buffer) < PERF_LOG_FLUSH_ROWS:
                return
            perf_log_writer.writerows(perf_log_buffer)
            perf_log_file_handle.flush()
            perf_log_buffer = []

        def close_perf_log() -> None:
            nonlocal perf_log_file_handle, perf_log_writer, perf_log_path, perf_log_buffer
            flush_perf_log(force=True)
            if perf_log_file_handle is not None:
                perf_log_file_handle.close()
            if perf_log_path is not None:
                print(f"{LOG_PREFIX} 性能日志已关闭: {perf_log_path}", flush=True)
            perf_log_file_handle = None
            perf_log_writer = None
            perf_log_path = None
            perf_log_buffer = []

        if perf_log_enabled:
            open_perf_log()

        def handle_keypress() -> bool:
            nonlocal display_mode, perf_log_enabled
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                stop_event.set()
                return True
            if key == ord("m"):
                display_mode = _next_display_mode(display_mode)
                print(f"{LOG_PREFIX} 显示模式切换为: {display_mode}", flush=True)
            if key == ord("l"):
                perf_log_enabled = not perf_log_enabled
                if perf_log_enabled:
                    open_perf_log()
                else:
                    close_perf_log()
            return False

        while not stop_event.is_set() or not frame_queue.empty():
            frame_idx += 1
            try:
                res_r, res_c, payload = frame_queue.get(timeout=0.1)
            except queue.Empty:
                now = time.time()
                if now - last_print >= 5.0:
                    fps = interval_count / (now - last_print) if now > last_print else 0.0
                    print(f"[{time.strftime('%H:%M:%S')}] {LOG_PREFIX} fps={fps:.2f}", flush=True)
                    last_print = now
                    interval_count = 0
                if handle_keypress():
                    continue
                continue

            frame_start = time.perf_counter()

            preprocess_start = time.perf_counter()
            depth = np.frombuffer(payload, dtype=np.uint8)
            if depth.size != res_r * res_c:
                continue
            depth = depth.reshape((res_r, res_c))

            if ENABLE_RAW_DOWNSAMPLE:
                depth_for_model = cv2.resize(depth, RAW_DOWNSAMPLE_SIZE, interpolation=cv2.INTER_AREA)
            else:
                depth_for_model = depth

            depth_up = cv2.resize(depth_for_model, (width, height), interpolation=cv2.INTER_LINEAR)
            color_img = cv2.applyColorMap(depth_up, cv2.COLORMAP_MAGMA)
            preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

            seg_start = time.perf_counter()
            try:
                results = model.track(
                    color_img,
                    conf=CONF_THRESHOLD,
                    persist=True,
                    tracker=TRACKER_CONFIG,
                    classes=[0],
                    imgsz=SEG_INFER_IMGSZ,
                    verbose=False,
                )
            except Exception as exc:
                print(f"{LOG_PREFIX} 推理失败: {exc}", flush=True)
                cv2.imshow(WINDOW_NAME, color_img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break
                continue
            seg_ms = (time.perf_counter() - seg_start) * 1000.0

            run_pose_now = (frame_idx % POSE_INFER_INTERVAL == 0) or (cached_kpt_xy is None)
            pose_result = None
            pose_ms = 0.0
            if run_pose_now:
                pose_start = time.perf_counter()
                try:
                    pose_results = pose_model.predict(
                        color_img,
                        conf=CONF_THRESHOLD,
                        classes=[0],
                        imgsz=POSE_INFER_IMGSZ,
                        verbose=False,
                    )
                    pose_result = pose_results[0] if pose_results else None
                except Exception as exc:
                    print(f"{LOG_PREFIX} 姿态推理失败: {exc}", flush=True)
                    pose_result = None
                pose_ms = (time.perf_counter() - pose_start) * 1000.0

            render_start = time.perf_counter()
            display = color_img.copy()
            result = results[0]
            person_count = 0
            tracked_labels: list[str] = []
            pair_records: list[tuple[int, np.ndarray, float | None]] = []
            seg_boxes_for_match: list[np.ndarray] = []
            seg_track_ids_for_match: list[int] = []

            pose_boxes_np: list[np.ndarray] = []
            kpt_xy_np: np.ndarray | None = None
            kpt_conf_np: np.ndarray | None = None
            pose_to_track: dict[int, int] = {}
            validated_track_ids: set[int] = set()
            seg_candidate_count = 0
            filtered_candidate_count = 0

            if pose_result is not None and pose_result.keypoints is not None:
                keypoints_xy = pose_result.keypoints.xy
                keypoints_conf = pose_result.keypoints.conf
                if keypoints_xy is not None and keypoints_conf is not None:
                    cached_kpt_xy = keypoints_xy.cpu().numpy()
                    cached_kpt_conf = keypoints_conf.cpu().numpy()
                if pose_result.boxes is not None and len(pose_result.boxes) > 0:
                    cached_pose_boxes = [box.copy() for box in pose_result.boxes.xyxy.cpu().numpy()]
            elif pose_result is not None and pose_result.keypoints is None and not warned_no_keypoints:
                print(
                    f"{LOG_PREFIX} 当前姿态模型没有输出 keypoints，请改用 MODEL pose 权重，例如 model11n-pose.pt。",
                    flush=True,
                )
                warned_no_keypoints = True

            pose_boxes_np = cached_pose_boxes
            kpt_xy_np = cached_kpt_xy
            kpt_conf_np = cached_kpt_conf

            if result.boxes is not None and len(result.boxes) > 0:
                if result.masks is None:
                    if not warned_no_masks:
                        print(
                            f"{LOG_PREFIX} 当前模型没有输出 segmentation masks，请改用 MODEL segment 权重，例如 model11n-seg.pt。",
                            flush=True,
                        )
                        warned_no_masks = True
                    cv2.putText(
                        display,
                        "Seg model required: no masks returned",
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                    track_ids = (
                        result.boxes.id.int().cpu().tolist()
                        if result.boxes.id is not None
                        else list(range(1, len(boxes_xyxy) + 1))
                    )
                    masks_data = result.masks.data.cpu().numpy()

                    match_count = min(len(boxes_xyxy), len(masks_data), len(track_ids))
                    seg_candidate_count = match_count
                    seg_boxes_for_match = [boxes_xyxy[idx].copy() for idx in range(match_count)]
                    seg_track_ids_for_match = [int(track_ids[idx]) for idx in range(match_count)]

                    if pose_boxes_np:
                        pose_to_track = _match_pose_to_seg_tracks(
                            seg_boxes_for_match,
                            seg_track_ids_for_match,
                            pose_boxes_np,
                        )
                        if kpt_conf_np is not None:
                            for pose_idx, track_id in pose_to_track.items():
                                if pose_idx >= len(kpt_conf_np):
                                    continue
                                confident_points = int(np.sum(kpt_conf_np[pose_idx] >= POSE_KEYPOINT_CONF_THRESHOLD))
                                if confident_points >= MIN_POSE_KEYPOINTS_FOR_PERSON:
                                    validated_track_ids.add(int(track_id))

                    for idx in range(match_count):
                        box = boxes_xyxy[idx]
                        track_id = int(track_ids[idx])
                        if track_id not in validated_track_ids: 
                            filtered_candidate_count += 1
                            continue
                        track_color = _track_color(track_id)
                        mask = (masks_data[idx] > 0.5).astype(np.uint8)
                        estimate = estimate_person_distance_from_mask(depth_up, box, mask)
                        person_count += 1

                        shifted_contour = None
                        if estimate.contour is not None and display_mode != DISPLAY_MODE_SKELETON_ONLY:
                            shifted_contour = estimate.contour + np.array(
                                [[[estimate.anchor[0], estimate.anchor[1]]]]
                            )
                            cv2.drawContours(display, [shifted_contour], -1, track_color, 2, cv2.LINE_AA)

                        if display_mode != DISPLAY_MODE_SKELETON_ONLY:
                            mask_overlay = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                            selected = mask_overlay > 0
                            if np.any(selected):
                                blended = (
                                    display[selected].astype(np.float32) * (1.0 - MASK_BLEND_ALPHA)
                                    + np.array(track_color, dtype=np.float32) * MASK_BLEND_ALPHA
                                )
                                display[selected] = blended.astype(np.uint8)

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
                            track_color,
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
                        pair_records.append((track_id, box.copy(), estimate.distance))

            if kpt_xy_np is not None and kpt_conf_np is not None:
                for idx in range(min(len(kpt_xy_np), len(kpt_conf_np))):
                    matched_track = pose_to_track.get(idx)
                    if matched_track is None or matched_track not in validated_track_ids:
                        continue
                    color_override = _track_color(matched_track) if matched_track is not None else None
                    if display_mode != DISPLAY_MODE_CONTOUR_ONLY:
                        draw_stick_figure(
                            display,
                            kpt_xy_np[idx],
                            kpt_conf_np[idx],
                            color_override=color_override,
                        )

            render_ms = (time.perf_counter() - render_start) * 1000.0

            interval_count += 1
            now = time.time()
            pair_text, pair_stats = _compute_pairwise_distances(pair_records, width)
            if now - last_print >= 5.0:
                fps = interval_count / (now - last_print) if now > last_print else 0.0
                labels_text = ", ".join(tracked_labels) if tracked_labels else "none"
                print(
                    f"[{time.strftime('%H:%M:%S')}] {LOG_PREFIX} fps={fps:.2f} persons={person_count} tracks={labels_text} {pair_text}",
                    flush=True,
                )
                last_print = now
                interval_count = 0

            elapsed = max(now - last_print, 1e-6)
            fps_approx = interval_count / elapsed
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
            if ENABLE_RAW_DOWNSAMPLE:
                cv2.putText(
                    display,
                    f"DS: {RAW_DOWNSAMPLE_SIZE[0]}x{RAW_DOWNSAMPLE_SIZE[1]} -> {width}x{height}",
                    (10, 74),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (180, 220, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.putText(
                display,
                pair_text,
                (10, 98),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 255, 180) if pair_stats else (170, 170, 170),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                f"Mode(m): {display_mode}",
                (10, 122),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                f"PerfLog(l): {'ON' if perf_log_enabled else 'OFF'}",
                (10, 146),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 230, 255) if perf_log_enabled else (130, 130, 130),
                1,
                cv2.LINE_AA,
            )

            display_show = cv2.resize(
                display,
                (width * DISPLAY_SCALE, height * DISPLAY_SCALE),
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.imshow(WINDOW_NAME, display_show)

            total_ms = (time.perf_counter() - frame_start) * 1000.0
            if perf_log_enabled and perf_log_writer is not None:
                pose_candidate_count = int(len(kpt_xy_np)) if kpt_xy_np is not None else 0
                row = [
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    int(frame_idx),
                    round(float(fps_approx), 3),
                    int(person_count),
                    int(seg_candidate_count),
                    int(filtered_candidate_count),
                    int(pose_candidate_count),
                    int(len(pose_to_track)),
                    int(run_pose_now),
                    round(float(preprocess_ms), 3),
                    round(float(seg_ms), 3),
                    round(float(pose_ms), 3),
                    round(float(render_ms), 3),
                    round(float(total_ms), 3),
                    str(display_mode),
                ]
                perf_log_buffer.append(row)
                flush_perf_log(force=False)

            if handle_keypress():
                break

        close_perf_log()
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
