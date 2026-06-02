# AI 模型 gRPC 接口对接说明

本文档说明平台如何调用当前的 AI 模型接口。接口以 `ai.proto` 为准，调用方式为 gRPC Unary RPC。当前服务端内部接的是实时 ToF 姿态/轮廓处理链路（基于 Ultralytics 推理），对外仍然只暴露一个 `Infer` 接口。

## 1. 接口概览

- 服务名：`ModelService`
- 方法名：`Infer`
- 输入：一张 PNG 图片的字节流
- 输出：8 张 PNG 图片字节流 + 人数统计 + 处理耗时

## 2. 请求格式

### `InferRequest`

| 字段           | 类型       | 说明                       |
| -------------- | ---------- | -------------------------- |
| `frame_id`   | `string` | 帧编号，用于请求和响应对应 |
| `image_data` | `bytes`  | 输入 PNG 图片的原始字节流  |

### 输入约定

- 输入图片建议为单张 PNG。
- 当前服务逻辑按图像二进制直接解码；平台只需要传入 PNG 文件内容即可。
- 如果图片较大，客户端和服务端都应设置较大的消息尺寸上限。

## 3. 响应格式

### `InferResponse`

| 字段                   | 类型       | 说明                   |
| ---------------------- | ---------- | ---------------------- |
| `frame_id`           | `string` | 原样返回请求中的帧编号 |
| `output_image_S11`   | `bytes`  | 第一组输出图像 1       |
| `output_image_S12`   | `bytes`  | 第一组输出图像 2       |
| `output_image_S13`   | `bytes`  | 第一组输出图像 3       |
| `output_image_S14`   | `bytes`  | 第一组输出图像 4       |
| `output_image_S21`   | `bytes`  | 第二组输出图像 1       |
| `output_image_S22`   | `bytes`  | 第二组输出图像 2       |
| `output_image_S23`   | `bytes`  | 第二组输出图像 3       |
| `output_image_S24`   | `bytes`  | 第二组输出图像 4       |
| `person_count`       | `int32`  | 当前帧统计人数         |
| `processing_time_ms` | `int32`  | 本次推理耗时，毫秒     |

### 返回值约定

- `output_image_S11` 到 `output_image_S14`：对应缓存前一帧与当前帧插帧后的输出。
- `output_image_S21` 到 `output_image_S24`：对应当前接收输入直接推理后的输出。
- 如果是第一帧，前一帧相关字段可能为空字节串。

## 4. 调用方式

### 最常见的客户端调用流程

1. 平台读取一张 PNG 文件的原始字节。
2. 填入 `InferRequest.frame_id` 和 `InferRequest.image_data`。
3. 调用 `ModelServiceStub.Infer(...)`。
4. 收到 `InferResponse` 后，把 `output_image_S11` 到 `output_image_S24` 写成 PNG 文件，或者直接显示到界面上。

### 现成测试脚本

仓库里已经提供了一个最小客户端脚本：`scripts/grpc_client_test.py`。

启动方式：

```bash
python scripts/grpc_client_test.py outputs/tmp_png_size_from_video_frame.png frame_000001 --host 127.0.0.1 --port 50051
```

这个脚本会：

- 读取输入 PNG
- 调用 gRPC 服务端的 `Infer`
- 把返回的 8 张图片保存到 `outputs/client_test/`
- 打印 `processing_time_ms` 和 round-trip 耗时

### Python 示例

```python
import grpc
import ai_pb2
import ai_pb2_grpc


def call_model(image_path: str, frame_id: str, target: str = "127.0.0.1:50051"):
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    options = [
        ("grpc.max_send_message_length", 50 * 1024 * 1024),
        ("grpc.max_receive_message_length", 50 * 1024 * 1024),
    ]

    with grpc.insecure_channel(target, options=options) as channel:
        stub = ai_pb2_grpc.ModelServiceStub(channel)
        request = ai_pb2.InferRequest(
            frame_id=frame_id,
            image_data=image_bytes,
        )
        response = stub.Infer(request, timeout=10.0)

    outputs = {
        "S11": response.output_image_S11,
        "S12": response.output_image_S12,
        "S13": response.output_image_S13,
        "S14": response.output_image_S14,
        "S21": response.output_image_S21,
        "S22": response.output_image_S22,
        "S23": response.output_image_S23,
        "S24": response.output_image_S24,
    }

    for name, data in outputs.items():
        if data:
            with open(f"{frame_id}_{name}.png", "wb") as f:
                f.write(data)

    print("frame_id:", response.frame_id)
    print("person_count:", response.person_count)
    print("processing_time_ms:", response.processing_time_ms)


if __name__ == "__main__":
    call_model("input.png", "frame_000001")
```

### 直接取回返回值并落盘

如果调用方不想复用仓库里的测试脚本，最小可用逻辑如下：

```python
with open("input.png", "rb") as f:
    image_bytes = f.read()

request = ai_pb2.InferRequest(
    frame_id="frame_000001",
    image_data=image_bytes,
)
response = stub.Infer(request, timeout=10.0)

for name in ["S11", "S12", "S13", "S14", "S21", "S22", "S23", "S24"]:
    data = getattr(response, f"output_image_{name}")
    if data:
        with open(f"frame_000001_{name}.png", "wb") as f:
            f.write(data)
```

## 5. 客户端注意事项

- gRPC 消息里包含多张 PNG，建议把发送和接收上限调大。
- `frame_id` 需要由平台侧保证唯一或可追踪。
- 客户端收到返回后，直接将 `bytes` 写成 `.png` 文件即可，无需再次编码。
- 如果平台侧要做 UI 展示，建议按 `S11` 到 `S24` 的字段名保持固定映射。

## 6. 服务端地址约定

默认测试地址可设为：`127.0.0.1:50051`

如果是部署到服务器，平台只需要替换为实际地址即可，例如：

- `10.0.0.12:50051`
- `model.example.com:50051`

## 7. 一句话说明

平台调用时，只需要把 PNG 图片的原始字节通过 `InferRequest.image_data` 发给 `ModelService.Infer`，然后从 `InferResponse` 里取回 8 张 PNG 和统计信息即可。
