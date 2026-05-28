import threading
import queue
import struct
import time
import numpy as np
import cv2
import serial

# 配置（按需修改）
# Configuration (modify as needed)
PORT = "/dev/ttyUSB0"
BAUD = 921600
TIMEOUT = 0.05
ENDIAN = "<"   # 如果设备是 big-endian 改为 ">"
FRAME_HEAD = b"\x00\xFF"
ALLOWED_TAILS = (0xCC, 0xDD)
RAW_QUEUE_MAXSIZE = 50   # 原始字节队列
FRAME_QUEUE_MAXSIZE = 10  # 已解析完整 payload 队列

# 打开串口（若已有 Serial 对象可改为复用）
# Open the serial port (if you already have a Serial object you can reuse it)
ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)

raw_queue = queue.Queue(maxsize=RAW_QUEUE_MAXSIZE)   # 生产者 -> 中继（原始 bytes）
frame_queue = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)  # 中继 -> 消费者（(resR,resC,payload)）
stop_event = threading.Event()

ser.write(b"AT+FPS=19\r")
time.sleep(0.1)
######################## PAY ATTENTION HERE Start ########################
ser.write(b"AT+DISP=2\r") # usb display on (FASTER IF YOU DO NOT NEED LCD)
# ser.write(b"AT+DISP=3\r") # lcd and usb display on (SLOWER)
######################## PAY ATTENTION HERE End  ########################
time.sleep(0.1)


def reader_thread():
    """
    只负责从串口读取原始字节，尽量不断读取以避免内核串口缓冲区堆积。
    将读取到的 bytes 块放入 raw_queue。若队列已满，丢弃最旧项以腾出空间。

    Read-only thread that reads raw bytes from the serial port as fast as possible to
    avoid the kernel serial buffer filling up. Puts received byte chunks into
    `raw_queue`. If the queue is full, the oldest entries are discarded to free space.
    """
    while not stop_event.is_set():
        try:
            n = ser.in_waiting
        except Exception:
            break
        if not n:
            # 仍然尝试读取一点数据以触发 timeout
            # Still try to read a small amount to trigger the timeout
            try:
                data = ser.read(256)
            except Exception:
                break
        else:
            try:
                data = ser.read(min(4096, n))
            except Exception:
                break

        if not data:
            time.sleep(0.001)
            continue

        # 确保能放入队列：若满则丢弃最旧项，保持流动
        # Ensure the data can be put into the queue: if it's full, drop the oldest
        # items to keep the flow moving.
        try:
            while raw_queue.full():
                try:
                    raw_queue.get_nowait()  # 丢弃最旧
                except queue.Empty:
                    break
            raw_queue.put_nowait(data)
        except Exception:
            # 若仍有异常，短暂休眠并继续
            # If there's still an exception, sleep briefly and continue
            time.sleep(0.001)
            continue


def relay_thread():
    """
    从 raw_queue 读取原始 bytes，维护一个缓冲区，解析出完整帧并校验。
    解析成功后把 (resR, resC, payload) 放入 frame_queue。
    中继层负责查找帧头、计算整帧长度、校验 checksum 和 tail。

    Relay thread: reads raw bytes from `raw_queue`, keeps an internal buffer,
    parses out complete frames and validates them. On successful parsing it
    puts tuples of (resR, resC, payload) into `frame_queue`. Responsibilities
    include finding the frame header, computing full frame length, and
    validating checksum and tail bytes.
    """
    last_frameid = 0
    buf = bytearray()
    while not stop_event.is_set() or not raw_queue.empty():
        try:
            chunk = raw_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        buf += chunk

        # 在 buf 中循环解析尽可能多的完整帧
        while True:
            idx = buf.find(FRAME_HEAD)
            if idx < 0:
                # 没有帧头，避免 buf 无限制增长
                if len(buf) > 8192:
                    buf.clear()
                break
            if idx > 0:
                del buf[:idx]  # 丢弃头之前的数据
                # Discard any bytes before the header

            # 需要至少 4 字节以读取 dataLen（head 2 + len 2）
            # Need at least 4 bytes to read dataLen (head 2 + len 2)
            if len(buf) < 4:
                break
            # 读取 dataLen（2字节）
            try:
                dataLen = struct.unpack(ENDIAN + "H", buf[2:4])[0]
            except struct.error:
                break

            # 计算整帧长度: head(2) + len(2) + dataLen + checksum(1) + tail(1)
            # Compute full frame length: head(2) + len(2) + dataLen + checksum(1) + tail(1)
            frameLen = 2 + 2 + dataLen + 2
            if len(buf) < frameLen:
                # 等待更多数据
                break

            frame = bytes(buf[:frameLen])
            # 消耗缓冲区
            del buf[:frameLen]

            # 校验尾和校验和
            # Validate tail byte and checksum
            frame_tail = frame[-1]
            checksum_byte = frame[-2]
            calc_sum = sum(frame[:-2]) & 0xFF

            if frame_tail not in ALLOWED_TAILS or checksum_byte != calc_sum:
                # 非法帧丢弃，继续尝试下一个帧头位置
                continue

            # 解析分辨率与 payload 偏移（保持原协议）
            # frame[14] = rows, frame[15] = cols
            # Parse resolution and payload offsets (protocol preserved)
            try:
                resR = frame[14]
                resC = frame[15]
                frameid = struct.unpack(ENDIAN + "H", frame[16:18])[0]
            except IndexError:
                continue

            # print(f"Received frame id: {frameid}, resolution: {resR}x{resC}", flush=True)

            # 根据原协议：dataLen 包含一些头部，payload 长度 = dataLen - 16（与原代码一致）
            # According to the original protocol: dataLen includes some header bytes,
            # so payload length = dataLen - 16 (keeps behavior consistent with the
            # original code).
            frameDataLen = dataLen - 16
            data_start = 20
            data_end = data_start + frameDataLen
            # payload 应位于 data_start:data_end，且在 checksum 前
            if data_end > len(frame) - 2:
                # 如果不满足则跳过
                continue

            if frameid == last_frameid:
                # 重复帧丢弃
                # Drop duplicate frames
                continue
            last_frameid = frameid

            payload = frame[data_start:data_end]

            # 放入 frame_queue（若队列满则丢帧）
            # Put into `frame_queue` (if full, drop the frame)
            try:
                frame_queue.put_nowait((resR, resC, payload))
            except queue.Full:
                # 丢帧以确保中继不会阻塞读取流程
                # Drop frames to ensure the relay doesn't block the read flow
                continue


def processor_thread():
    """
    从 frame_queue 读取 payload，解码为图像并显示（或进一步处理）。
    使用 OpenCV colormap 显示，按 'q' 退出。每隔5秒在命令行打印一次帧率。

    Processor thread: reads payloads from `frame_queue`, decodes them into
    images and displays them (or further processes them). Uses OpenCV colormap
    for visualization. Press 'q' to quit. Prints FPS to the console every 5
    seconds.
    """
    cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)

    last_print = time.time()
    interval_count = 0

    while not stop_event.is_set() or not frame_queue.empty():
        try:
            res = frame_queue.get(timeout=0.1)
        except queue.Empty:
            # 即使没有新帧，也检查是否到达打印间隔
            # Even when there's no new frame, check whether the print interval
            # has been reached
            now = time.time()
            if now - last_print >= 5.0:
                elapsed = now - last_print
                fps = interval_count / elapsed if elapsed > 0 else 0.0
                print(f"[{time.strftime('%H:%M:%S')}] FPS: {fps:.2f} ({interval_count} frames in {elapsed:.2f}s)", flush=True)
                last_print = now
                interval_count = 0
            continue

        try:
            resR, resC, payload = res
        except Exception:
            continue

        # payload -> numpy -> reshape
        # Convert payload to numpy array and reshape according to resolution
        try:
            img_idx = np.frombuffer(payload, dtype=np.uint8)
            if img_idx.size != resR * resC:
                # 长度不匹配则丢弃
                # If size doesn't match, drop the frame
                continue
            img_idx = img_idx.reshape((resR, resC))
        except Exception:
            continue

        try:
            color_img = cv2.applyColorMap(img_idx, cv2.COLORMAP_MAGMA)
            cv2.imshow("frame", color_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
        except Exception:
            # If display fails, skip this frame and continue
            continue

        # 帧计数并按间隔打印 FPS
        # Count frames and print FPS at intervals
        interval_count += 1
        now = time.time()
        if now - last_print >= 5.0:
            elapsed = now - last_print
            fps = interval_count / elapsed if elapsed > 0 else 0.0
            print(f"[{time.strftime('%H:%M:%S')}] FPS: {fps:.2f} ({interval_count} frames in {elapsed:.2f}s)", flush=True)
            last_print = now
            interval_count = 0

    cv2.destroyAllWindows()


# 启动线程
# Start threads
t_reader = threading.Thread(target=reader_thread, name="serial-reader", daemon=True)
t_relay = threading.Thread(target=relay_thread, name="serial-relay", daemon=True)
t_processor = threading.Thread(target=processor_thread, name="frame-processor", daemon=True)

t_reader.start()
t_relay.start()
t_processor.start()

try:
    while not stop_event.is_set():
        time.sleep(0.1)
except KeyboardInterrupt:
    stop_event.set()
finally:
    stop_event.set()
    t_reader.join(timeout=1.0)
    t_relay.join(timeout=1.0)
    t_processor.join(timeout=1.0)
    try:
        ser.close()
    except Exception:
        pass
    cv2.destroyAllWindows()
