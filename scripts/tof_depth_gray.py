from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import queue
import struct
import threading
import time

import cv2
import numpy as np
import serial



BAUD = 921600
TIMEOUT = 0.05
ENDIAN = "<"
FRAME_HEAD = b"\x00\xFF"
ALLOWED_TAILS = (0xCC, 0xDD)
RAW_QUEUE_MAXSIZE = 50
FRAME_QUEUE_MAXSIZE = 3
DISPLAY_SIZE = (320, 320)
DISPLAY_SCALE = 3
WINDOW_NAME = "tof_depth_gray"
LOG_PREFIX = "[tof_depth_gray]"


def run(port: str = "COM8") -> None:
    ser = serial.Serial(port, BAUD, timeout=TIMEOUT)
    raw_queue: queue.Queue[bytes] = queue.Queue(maxsize=RAW_QUEUE_MAXSIZE)
    frame_queue: queue.Queue[tuple[int, int, bytes]] = queue.Queue(
        maxsize=FRAME_QUEUE_MAXSIZE
    )
    stop_event = threading.Event()

    print(f"{LOG_PREFIX} port={port} size={DISPLAY_SIZE}")
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

            gray_img = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
            display = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

            interval_count += 1
            now = time.time()
            elapsed = max(now - last_print, 1e-6)
            fps_approx = interval_count / elapsed
            cv2.putText(
                display,
                f"Src: {res_c}x{res_r} FPS~{fps_approx:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
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


if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else "COM8"
    run(port=port)
