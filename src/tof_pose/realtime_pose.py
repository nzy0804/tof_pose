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


BAUD = 921600
TIMEOUT = 0.05
ENDIAN = "<"
FRAME_HEAD = b"\x00\xFF"
ALLOWED_TAILS = (0xCC, 0xDD)
RAW_QUEUE_MAXSIZE = 50
FRAME_QUEUE_MAXSIZE = 3
DISPLAY_SIZE = (320, 320)
DISPLAY_SCALE = 3
CONF_THRESHOLD = 0.25
WINDOW_NAME = "tof_pose"
LOG_PREFIX = "[tof_pose]"


def run(port: str = "COM8", model_path: Path | None = None) -> None:
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

            # 只保留较新的串口数据块，优先保证实时显示不卡顿。
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

                # 串口协议格式为：帧头 + 长度 + 数据体 + 校验 + 帧尾。
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

                # 传感器卡顿时可能重复发出相同 frame id，这里直接跳过。
                if frameid == last_frameid:
                    continue
                last_frameid = frameid

                payload_len = data_len - 16
                payload = frame[20 : 20 + payload_len]
                try:
                    frame_queue.put_nowait((res_r, res_c, payload))
                except queue.Full:
                    # 推理跟不上时丢掉最旧帧，尽量处理最新画面。
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

            # 先把低分辨率深度图上采样，再转成伪彩图供 YOLO 推理。
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
            distance_labels: list[str] = []
            if result.keypoints is not None and result.boxes is not None:
                kpts_xy = result.keypoints.xy.cpu().numpy()
                kpts_conf = result.keypoints.conf.cpu().numpy()
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                num_persons = min(len(kpts_xy), len(boxes_xyxy))

                for idx in range(num_persons):
                    draw_stick_figure(display, kpts_xy[idx], kpts_conf[idx], KPT_CONF_THRESHOLD)

                    estimate = estimate_person_distance(
                        depth_up,
                        boxes_xyxy[idx],
                        kpts_xy[idx],
                        kpts_conf[idx],
                        KPT_CONF_THRESHOLD,
                    )

                    if estimate.contour is not None:
                        shifted_contour = estimate.contour + np.array([[[estimate.anchor[0], estimate.anchor[1]]]])
                        cv2.drawContours(display, [shifted_contour], -1, (255, 255, 255), 1, cv2.LINE_AA)

                    box = boxes_xyxy[idx]
                    x1 = max(0, int(round(box[0])))
                    y1 = max(0, int(round(box[1])) - 8)
                    if estimate.distance is None:
                        label = f"P{idx + 1} Dist=N/A"
                    else:
                        label = f"P{idx + 1} Dist~{estimate.distance:.1f}"
                        distance_labels.append(f"P{idx + 1}:{estimate.distance:.1f}")
                    cv2.putText(
                        display,
                        label,
                        (x1, max(18, y1)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        display,
                        label,
                        (x1, max(18, y1)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (20, 20, 20),
                        1,
                        cv2.LINE_AA,
                    )

            interval_count += 1
            now = time.time()
            if now - last_print >= 5.0:
                fps = interval_count / (now - last_print) if now > last_print else 0.0
                distance_text = " ".join(distance_labels) if distance_labels else "none"
                print(
                    f"[{time.strftime('%H:%M:%S')}] {LOG_PREFIX} fps={fps:.2f} detected={num_persons} dist={distance_text}",
                    flush=True,
                )
                last_print = now
                interval_count = 0

            elapsed = max(now - last_print, 1e-6)
            fps_approx = interval_count / elapsed
            cv2.putText(
                display,
                f"Persons: {num_persons}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
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
