# UDP 图片发送端接口文档

本文档描述发送端如何通过 UDP 将深度图片发送给 `scripts/udp_infer_viewer.py`，由接收端执行与当前 gRPC 服务相同的本地推理算法，并实时显示结果。

## 接收端启动

推荐四宫格显示：

```powershell
python scripts/udp_infer_viewer.py --host 0.0.0.0 --port 50060 --view montage
```

参数说明：

| 参数 | 说明 |
| --- | --- |
| `--host` | UDP 监听地址，通常使用 `0.0.0.0` |
| `--port` | UDP 监听端口，默认建议 `50060` |
| `--view montage` | 四宫格显示：灰度深度、伪彩深度、骨架、轮廓 |

发送端需要把 UDP 包发到运行接收端电脑的 IP 地址和端口，例如 `192.168.1.10:50060`。

## 推荐协议：一帧一个 UDP 包

### 数据格式

每个 UDP datagram 直接放一张完整图片的编码字节。

```text
UDP payload = PNG image bytes
```

当前接收端会把整包内容作为 `image_data` 传给 `RealtimePoseEngine.infer()`，这与 gRPC 服务中的 `InferRequest.image_data` 语义一致。

### 图片要求

| 项目 | 要求 |
| --- | --- |
| 图片格式 | 推荐 PNG |
| 图片内容 | 单通道 `uint8` 深度图 |
| 图片尺寸 | 当前发送端为 `100x100` |
| 发送频率 | 约 10 fps，即每秒 10 张 |
| 传输方式 | UDP |

说明：

- “原始深度 PNG 图片”表示深度数据已经编码成 PNG 文件字节，不是裸的 `100 * 100 = 10000` 字节数组。
- 这种情况下接收端启动时不需要加 `--raw-depth 100x100`。
- `--raw-depth 100x100` 只用于发送裸 `uint8` 深度数组的场景。

### Python 发送示例

```python
import socket
import time
from pathlib import Path


receiver_ip = "192.168.1.10"
receiver_port = 50060
image_dir = Path("frames")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

for image_path in sorted(image_dir.glob("*.png")):
    image_bytes = image_path.read_bytes()
    sock.sendto(image_bytes, (receiver_ip, receiver_port))
    time.sleep(0.1)  # 10 fps
```

### OpenCV 发送示例

如果发送端拿到的是 `100x100` 的 `uint8` 深度矩阵，可以先编码成 PNG 再发送：

```python
import socket
import time

import cv2
import numpy as np


receiver_ip = "192.168.1.10"
receiver_port = 50060

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:
    depth = np.zeros((100, 100), dtype=np.uint8)  # 替换为真实深度图

    ok, buf = cv2.imencode(".png", depth, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        continue

    sock.sendto(buf.tobytes(), (receiver_ip, receiver_port))
    time.sleep(0.1)
```

## 可选协议：分片 UDP

普通 UDP 单包 payload 建议不要超过网络 MTU 太多。`100x100` 的 PNG 通常可以直接单包发送；如果后续图片变大，可以使用接收端支持的分片协议。

### 分片包格式

每个 UDP payload 由固定 12 字节头部加分片数据组成：

```text
0               4               8       10      12
+---------------+---------------+-------+-------+----------------
| magic         | frame_id       | chunk | total | payload ...
+---------------+---------------+-------+-------+----------------
| 4 bytes       | 4 bytes       |2 bytes|2 bytes|
| ASCII "MSXU"  | uint32 BE     |uint16 |uint16 |
```

字段说明：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `magic` | 4 字节 | 固定为 ASCII `MSXU` |
| `frame_id` | `uint32` 大端 | 帧编号，同一张图片的所有分片必须相同 |
| `chunk` | `uint16` 大端 | 当前分片序号，从 `0` 开始 |
| `total` | `uint16` 大端 | 该帧总分片数 |
| `payload` | bytes | 当前分片承载的图片字节 |

接收端会按 `chunk` 从小到大拼接所有分片，然后把拼接结果当作完整 PNG 图片进行推理。

### 分片发送示例

```python
import math
import socket
import struct
import time
from pathlib import Path


receiver_ip = "192.168.1.10"
receiver_port = 50060
chunk_payload_size = 1200

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
header = struct.Struct("!4sIHH")

frame_id = 0
for image_path in sorted(Path("frames").glob("*.png")):
    image_bytes = image_path.read_bytes()
    total = math.ceil(len(image_bytes) / chunk_payload_size)

    for chunk_id in range(total):
        start = chunk_id * chunk_payload_size
        end = start + chunk_payload_size
        payload = image_bytes[start:end]
        packet = header.pack(b"MSXU", frame_id, chunk_id, total) + payload
        sock.sendto(packet, (receiver_ip, receiver_port))

    frame_id = (frame_id + 1) & 0xFFFFFFFF
    time.sleep(0.1)
```

## 接收端显示结果

`--view montage` 四宫格含义如下：

| 位置 | 内容 |
| --- | --- |
| 左上 | 灰度深度图 |
| 右上 | 伪彩深度图 |
| 左下 | 骨架结果 |
| 右下 | 轮廓结果 |

窗口顶部状态栏会显示当前帧号、人数、推理耗时、端到端延迟、显示 FPS 和发送端地址。

## 实时性策略

接收端内部只保留最新一帧。如果发送端 10 fps 持续发送，而推理耗时超过 100 ms，接收端会自动丢弃积压旧帧，优先显示最新推理结果。

## 常见问题

### 发送 PNG 时是否需要 `--raw-depth 100x100`？

不需要。PNG 是已经编码好的图片字节，接收端可以直接解码。

### 什么时候需要 `--raw-depth 100x100`？

只有发送端直接发送裸深度数据时才需要，例如 UDP payload 正好是 `10000` 字节，每个字节代表一个像素。

裸数据接收端命令示例：

```powershell
python scripts/udp_infer_viewer.py --host 0.0.0.0 --port 50060 --view montage --raw-depth 100x100
```

### UDP 丢包怎么办？

UDP 不保证可靠传输。当前实时显示以低延迟为优先，允许偶发丢帧。若必须保证每帧都处理，应改用 TCP、gRPC 或在应用层增加确认重传机制。
