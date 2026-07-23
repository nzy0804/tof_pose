# MaixSense 异步 gRPC 客户端对接说明

本文档面向客户端接入方，说明如何使用异步提交接口让采集发送和 AI 结果返回解耦。同步 `Infer` 接口仍保留用于调试、smoke test 和兼容旧客户端；实时生产链路建议使用 `SubmitFrames` + `SubscribeResults`。

## 1. 服务地址

默认 AI 服务地址：

```text
222.71.62.147:50052
```

实际部署时以现场配置为准。服务端需要启动异步接口：

```bash
--async-infer \
--async-device-window 3 \
--async-result-buffer-size 1000 \
--async-prepare-workers 8 \
--async-result-workers 8
```

如果目标是多设备凑大 batch，建议同时启用动态 batch：

```bash
--dynamic-batching \
--dynamic-max-batch-size 80 \
--dynamic-max-wait-ms 300 \
--model-instances 1
```

`model-instances=1` 可以让多个设备进入同一个动态队列，更容易合成大 batch。多实例会把设备分散到多个队列，适合吞吐已经足够但单队列排队过长的情况。

## 2. 接口总览

```proto
service ModelService {
  rpc Infer(InferRequest) returns (InferResponse);

  rpc SubmitFrames(SubmitFramesRequest) returns (SubmitFramesResponse);
  rpc SubscribeResults(ResultSubscribeRequest) returns (stream InferResultEvent);
}
```

推荐流程：

```text
客户端先建立 SubscribeResults 长连接
  |
采集到一批 20 帧
  |
SubmitFrames 提交，服务端只校验和入队
  |
收到 accepted=true 后，客户端继续采集和提交下一批
  |
AI 完成后，服务端通过 SubscribeResults 推送 InferResultEvent
```

客户端不要再等待完整 AI 结果后才发送下一批。正确做法是：**等待 SubmitFrames 的入队 ACK，不等待 AI 结果**。

## 3. SubmitFrames

### 3.1 请求

```proto
message SubmitFramesRequest {
  string device_id = 1;
  string batch_id = 2;
  int64 sequence_id = 3;
  repeated InferImage images = 4;
  int32 batch_size = 5;
}
```

字段说明：

| 字段            | 必填   | 说明                                                                              |
| --------------- | ------ | --------------------------------------------------------------------------------- |
| `device_id`   | 是     | 设备唯一 ID。服务端按该字段隔离跟踪状态和 in-flight 窗口。                        |
| `batch_id`    | 建议填 | 批次 ID，建议客户端生成全局唯一值。为空时服务端按`device_id-sequence_id` 兜底。 |
| `sequence_id` | 是     | 同一设备内严格递增的批次序号，建议从 1 开始每批 +1。                              |
| `images`      | 是     | 一批图片，建议 20 帧。每帧使用`image_data` 或 `object_key`。                  |
| `batch_size`  | 是     | 应等于`images` 数量。                                                           |

`InferImage`：

```proto
message InferImage {
  string frame_id = 1;
  int64 capture_timestamp_ms = 2;
  bytes image_data = 3 [deprecated = true];
  string object_key = 4;
}
```

输入方式：

| 方式           | 说明                                                        |
| -------------- | ----------------------------------------------------------- |
| `object_key` | 推荐生产使用。客户端先上传原始帧到 OSS，再提交 object key。 |
| `image_data` | 可用于调试或内网小流量测试。                                |

### 3.2 响应

```proto
message SubmitFramesResponse {
  string device_id = 1;
  string batch_id = 2;
  int64 sequence_id = 3;
  bool accepted = 4;
  int32 accepted_count = 5;
  int32 device_inflight_batches = 6;
  int32 retry_after_ms = 7;
  string message = 8;
}
```

响应说明：

| 字段                        | 说明                                                          |
| --------------------------- | ------------------------------------------------------------- |
| `accepted`                | `true` 表示已入队，客户端可以继续提交下一批。               |
| `accepted_count`          | 本次接受的帧数。                                              |
| `device_inflight_batches` | 当前设备未完成批次数。                                        |
| `retry_after_ms`          | `accepted=false` 时建议等待时间。                           |
| `message`                 | 状态说明，例如`accepted`、`device_inflight_window_full`。 |

### 3.3 忙碌处理

服务端按设备限制 in-flight 批次数，默认建议 `3`。超过窗口时不会抛异常，而是返回：

```text
accepted=false
message=device_inflight_window_full
retry_after_ms=100
```

客户端收到后应等待 `retry_after_ms`，然后使用同一个 `sequence_id` 重试，不要跳号。

## 4. SubscribeResults

### 4.1 请求

```proto
message ResultSubscribeRequest {
  repeated string device_ids = 1;
  string consumer_id = 2;
}
```

字段说明：

| 字段            | 说明                                   |
| --------------- | -------------------------------------- |
| `device_ids`  | 只订阅指定设备；为空表示订阅全部设备。 |
| `consumer_id` | 客户端标识，用于日志排查。             |

建议客户端启动后先建立该 stream，再开始调用 `SubmitFrames`。当前版本结果缓存主要服务活跃订阅者；如果 stream 断开，断开期间的结果不保证补发。

### 4.2 结果事件

```proto
message InferResultEvent {
  string device_id = 1;
  string batch_id = 2;
  int64 sequence_id = 3;
  repeated InferResult results = 4;
  int32 processing_time_ms = 5;
  string error_message = 6;
}
```

字段说明：

| 字段                   | 说明                             |
| ---------------------- | -------------------------------- |
| `device_id`          | 对应提交设备。                   |
| `batch_id`           | 对应提交批次。                   |
| `sequence_id`        | 对应提交序号。                   |
| `results`            | 每帧 AI 结果，顺序对应提交帧。   |
| `processing_time_ms` | 从服务端接受到结果发布的总耗时。 |
| `error_message`      | 非空表示该批处理失败。           |

`InferResult` 里的关键字段：

| 字段                            | 说明                      |
| ------------------------------- | ------------------------- |
| `frame_id`                    | 结果帧 ID。               |
| `capture_timestamp_ms`        | 原始采集时间戳。          |
| `person_count`                | 人数。                    |
| `person_status`               | 人员状态。                |
| `person_distance`             | 人员距离结果。            |
| `action_level`                | 动作等级。                |
| `pseudo_color_object_key`     | 兼容废弃字段，当前为空。  |
| `skeleton_contour_object_key` | 100x100 骨架/轮廓结果图 OSS key。 |

生产环境结果图通过 `skeleton_contour_object_key` 获取；当前服务端只上传一张 100x100 骨架/轮廓图。`pseudo_color_image`、`pseudo_color_image_format`、`pseudo_color_object_key` 是兼容废弃字段，当前为空。

## 5. 顺序要求

同一设备必须按 `sequence_id` 顺序提交：

```text
device-001 sequence_id=1 -> SubmitFrames -> accepted=true
device-001 sequence_id=2 -> SubmitFrames -> accepted=true
device-001 sequence_id=3 -> SubmitFrames -> accepted=true
```

客户端可以不等待 sequence 1 的 AI 结果就提交 sequence 2，但应等待 sequence 1 的 SubmitFrames ACK 后再提交 sequence 2。

不要这样做：

```text
同一设备同时并发 SubmitFrames(sequence_id=2)
同一设备同时并发 SubmitFrames(sequence_id=3)
```

原因：服务端需要按设备维护上一帧、动作、mask、轮廓等状态。跨设备可以并发；同设备提交顺序必须稳定。

## 6. Python 客户端示例

```python
import threading
import time
import grpc

import ai_pb2
import ai_pb2_grpc



TARGET = "222.71.62.147:50052"
DEVICE_ID = "device-001"
BATCH_SIZE = 20



def subscribe_results(stub):
    request = ai_pb2.ResultSubscribeRequest(
        device_ids=[DEVICE_ID],
        consumer_id="device-001-client",
    )
    for event in stub.SubscribeResults(request):
        if event.error_message:
            print("AI failed", event.device_id, event.sequence_id, event.error_message)
            continue
        print(
            "AI result",
            event.device_id,
            event.sequence_id,
            len(event.results),
            event.processing_time_ms,
        )



def submit_loop(stub):
    sequence_id = 1
    while True:
        images = collect_20_frames()
        request = ai_pb2.SubmitFramesRequest(
            device_id=DEVICE_ID,
            batch_id=f"{DEVICE_ID}-{sequence_id}",
            sequence_id=sequence_id,
            batch_size=len(images),
        )
        request.images.extend(images)

        response = stub.SubmitFrames(request, timeout=3)
        if response.accepted:
            sequence_id += 1
            continue

        if response.retry_after_ms > 0:
            time.sleep(response.retry_after_ms / 1000.0)
        # retry same sequence_id



def collect_20_frames():
    # 生产环境建议填 object_key；调试时可以填 image_data。
    raise NotImplementedError



channel = grpc.insecure_channel(
    TARGET,
    options=[
        ("grpc.max_send_message_length", 160 * 1024 * 1024),
        ("grpc.max_receive_message_length", 160 * 1024 * 1024),
    ],
)
stub = ai_pb2_grpc.ModelServiceStub(channel)

threading.Thread(target=subscribe_results, args=(stub,), daemon=True).start()
submit_loop(stub)
```

## 7. 推荐参数

首版联调建议：

| 参数                       | 建议值      |
| -------------------------- | ----------- |
| 每批帧数                   | `20`      |
| 每设备 in-flight           | `3`       |
| `dynamic_max_batch_size` | `80`      |
| `dynamic_max_wait_ms`    | `200-300` |
| `model_instances`        | `1`       |
| `max_workers`            | `24-32`   |

如果服务端日志里经常看到：

```text
Dynamic infer batch: frames=20 streams=1
```

说明客户端并发供料还不够，或者设备数太少。

如果经常看到：

```text
Dynamic infer batch: frames=60/80 streams>1
```

说明多设备动态混批已经生效。

## 8. 常见错误

| 现象                                                         | 原因                            | 处理                                                            |
| ------------------------------------------------------------ | ------------------------------- | --------------------------------------------------------------- |
| `FAILED_PRECONDITION: async inference is disabled`         | 服务端未加`--async-infer`     | 重启服务并加启动参数。                                          |
| `accepted=false, device_inflight_window_full`              | 该设备未完成批次超过窗口        | 等`retry_after_ms` 后用同一 `sequence_id` 重试。            |
| `INVALID_ARGUMENT: images must not be empty`               | 没有提交图片                    | 检查`images`。                                                |
| `INVALID_ARGUMENT: batch_size does not match images count` | `batch_size` 与图片数量不一致 | 修正`batch_size`。                                            |
| `sequence_id_must_increase_per_device`                     | 同设备序号重复或回退            | 使用单调递增序号，重试 busy 时不要跳号。                        |
| stream 断开后漏结果                                          | 当前 MVP 不保证断线补发         | 客户端应保持 stream 常驻；后续可增加`from_sequence_id` 补拉。 |

## 9. 与同步 Infer 的关系

`Infer` 仍可用于：

- 单批调试；
- smoke test；
- 对比同步/异步结果；
- 老客户端兼容。

生产实时链路推荐使用：

```text
SubmitFrames + SubscribeResults
```

这样服务端可以持续积累待推理数据，动态 batch 才能稳定吃满 GPU。
