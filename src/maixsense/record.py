import queue
import struct
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import serial

from maixsense.paths import DEFAULT_CAPTURE_VIDEO


BAUD = 921600
TIMEOUT = 0.05
ENDIAN = "<"
FRAME_HEAD = b"\x00\xFF"
ALLOWED_TAILS = (0xCC, 0xDD)
RAW_QUEUE_MAXSIZE = 100
FRAME_QUEUE_MAXSIZE = 10
DISPLAY_SIZE = (320, 320)
DISPLAY_SCALE = 2
VIDEO_FPS = 20.0
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")


def run(port: str = "COM8", output_file: Path | None = None) -> None:
    target_file = Path(output_file) if output_file else DEFAULT_CAPTURE_VIDEO
    target_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        ser = serial.Serial(port, BAUD, timeout=TIMEOUT)
        ser.reset_input_buffer()
        time.sleep(0.2)
        ser.write(b"AT+FPS=20\r")
        time.sleep(0.1)
        ser.write(b"AT+DISP=2\r")
        time.sleep(0.1)
        print(f"[record] serial ready: {port}")
    except Exception as exc:
        print(f"[record] serial open failed: {exc}")
        return

    raw_queue: queue.Queue[bytes] = queue.Queue(maxsize=RAW_QUEUE_MAXSIZE)
    frame_queue: queue.Queue[tuple[int, int, bytes]] = queue.Queue(
        maxsize=FRAME_QUEUE_MAXSIZE
    )
    stop_event = threading.Event()

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
            if not raw_queue.full():
                raw_queue.put_nowait(data)

    def relay_thread() -> None:
        last_frameid = 0
        buf = bytearray()
        while not stop_event.is_set():
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

                # 串口读取到的是连续字节流，需要重新拼成完整传感器帧。
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

                payload = frame[20 : 20 + (data_len - 16)]
                try:
                    frame_queue.put_nowait((res_r, res_c, payload))
                except queue.Full:
                    # 录制场景允许丢弃旧帧，以免消费速度跟不上采集速度。
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    frame_queue.put_nowait((res_r, res_c, payload))

    def processor() -> None:
        print(f"[record] writing to {target_file}")
        video_writer = cv2.VideoWriter(str(target_file), FOURCC, VIDEO_FPS, DISPLAY_SIZE)
        cv2.namedWindow("ToF Record", cv2.WINDOW_AUTOSIZE)
        frame_count = 0

        while not stop_event.is_set():
            try:
                res_r, res_c, payload = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            depth = np.frombuffer(payload, dtype=np.uint8)
            if depth.size != res_r * res_c:
                continue
            depth = depth.reshape((res_r, res_c))

            # 录制为伪彩视频，便于后续直接复用到离线姿态推理流程。
            depth_up = cv2.resize(depth, DISPLAY_SIZE, interpolation=cv2.INTER_LINEAR)
            color_img = cv2.applyColorMap(depth_up, cv2.COLORMAP_MAGMA)
            video_writer.write(color_img)
            frame_count += 1

            show_img = cv2.resize(
                color_img,
                None,
                fx=DISPLAY_SCALE,
                fy=DISPLAY_SCALE,
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.putText(
                show_img,
                f"REC: {frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("ToF Record", show_img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break

        video_writer.release()
        cv2.destroyAllWindows()
        print(f"[record] saved: {target_file} frames={frame_count}")

    threads = [
        threading.Thread(target=reader_thread, daemon=True),
        threading.Thread(target=relay_thread, daemon=True),
    ]
    for thread in threads:
        thread.start()

    try:
        processor()
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        try:
            ser.close()
        except Exception:
            pass
        print("[record] serial closed.")
