# 四路算法输出视频格式说明

## 1. 帧率与分辨率

| 路别        | 帧率   | 分辨率    | 输出图像格式 |
| ----------- | ------ | --------- | ------------ |
| S1 灰度深度 | 20 FPS | 320 x 320 | PNG          |
| S2 彩色深度 | 20 FPS | 320 x 320 | PNG          |
| S3 人体骨骼 | 20 FPS | 320 x 320 | PNG          |
| S4 人体轮廓 | 20 FPS | 320 x 320 | PNG          |

说明：每个时间点输出四张 PNG 图，四路 PNG 长度可变。

## 2. 具体存储方式（四路合帧字节流）

发送二进制字节流，每条复合帧包含同一时刻的四路 PNG 子帧。

### 2.1 复合帧头结构（固定 32 字节，小端）

| 偏移 | 长度 | 字段             | 类型   | 说明                                    |
| ---- | ---- | ---------------- | ------ | --------------------------------------- |
| 0    | 2    | magic            | uint16 | 固定 0xA55A                             |
| 2    | 1    | version          | uint8  | 协议版本，当前 1                        |
| 3    | 1    | stream_count     | uint8  | 固定 4                                  |
| 4    | 4    | frame_id         | uint32 | 复合帧递增帧号                          |
| 8    | 8    | timestamp_ms     | uint64 | Unix 毫秒时间戳                         |
| 16   | 4    | payload_len      | uint32 | 复合负载总长度                          |
| 20   | 4    | payload_checksum | uint32 | 复合负载校验，sum(payload) & 0xFFFFFFFF |
| 24   | 8    | reserved         | uint64 | 预留，填 0                              |

### 2.2 复合负载结构

复合负载由四个子帧顺序组成，每个子帧为 子帧头 + PNG 数据。

子帧头固定 16 字节，小端：

| 偏移 | 长度 | 字段         | 类型   | 说明                                       |
| ---- | ---- | ------------ | ------ | ------------------------------------------ |
| 0    | 1    | stream_id    | uint8  | 1=S1, 2=S2, 3=S3, 4=S4                     |
| 1    | 1    | codec        | uint8  | 固定 2，表示 PNG                           |
| 2    | 2    | reserved     | uint16 | 预留，填 0                                 |
| 4    | 2    | width        | uint16 | 固定 320                                   |
| 6    | 2    | height       | uint16 | 固定 320                                   |
| 8    | 4    | png_len      | uint32 | 当前子帧 PNG 字节长度                      |
| 12   | 4    | png_checksum | uint32 | 当前 PNG 校验，sum(png_bytes) & 0xFFFFFFFF |

四个子帧顺序固定：S1 -> S2 -> S3 -> S4。

复合负载总长度计算：

payload_len = (16 + S1_png_len) + (16 + S2_png_len) + (16 + S3_png_len) + (16 + S4_png_len)

## 3. 解析方式

接收端按以下流程解析每条复合字节流帧：

1. 先读 32 字节复合帧头，校验 magic、version、stream_count。
2. 按 payload_len 读取完整复合负载，校验 payload_checksum。
3. 依次读取 4 个子帧头，获取 stream_id、png_len、png_checksum。
4. 按 png_len 读取对应 PNG 字节并校验 png_checksum。
5. 使用 PNG 解码库将每路字节还原为图像。
6. 依据 stream_id 放入 S1 到 S4 对应槽位，并使用 frame_id 或 timestamp_ms 做顺序控制。
