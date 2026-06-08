# MaixSense gRPC Batch Session Summary - 2026-06-05

本文档用于下次会话快速恢复上下文，记录本次围绕 MaixSense gRPC 算法服务的关键决策、已完成改动、部署状态、性能观察和待办事项。

## 一句话概览

当前 gRPC 算法服务已经从单帧/固定返回逻辑改成动态 batch：客户端一次传入 N 张图片，服务端为每张输入生成插帧和当前帧两个结果，所以返回 `2 * N` 个 `InferResult`；每个结果包含 2 张 PNG 图片，因此总返回图片 blob 数量是 `4 * N`。例如输入 10 张，返回 20 个 result，共 40 张 PNG 图片。

## 当前关键决策

- 不使用 systemd 启动服务，Linux 服务器上使用 `nohup` 或前台命令运行。
- gRPC 服务端监听端口使用 `50052`。
- 服务端可执行文件部署目录：

```bash
/data/care/care-sense-iot-platform/bin/
```

- 云服务器角色：

```text
服务端服务器: 222.71.62.147
调用方服务器: 8.149.246.201
```

- 推理设备显式传入：

```bash
--device cuda:0
```

- YOLO 的 batch 推理不再逐张调用，而是把本次 batch 展开后的所有输出帧组成列表传给 YOLO：

```text
N 张输入图 -> 2N 张待推理图，包括 interpolated 和 current
seg_model.track(color_imgs, ...)
pose_model.predict(color_imgs, ...)
```

- 骨骼轮廓图的背景应是伪彩图，不是黑色背景。
- 骨骼轮廓图中不应出现文字、距离、ID、置信度等额外信息，只保留骨骼和轮廓叠加。
- `batch_size` 已加入 `InferRequest`，服务端会校验：如果客户端传了非 0 的 `batch_size`，它必须等于 `images` 数量。

## Proto 当前结构

核心文件：

```text
ai.proto
ai_pb2.py
ai_pb2_grpc.py
```

当前协议要点：

```proto
message InferRequest {
  string device_id = 1;
  string batch_id = 2;
  repeated InferImage images = 3;
  int32 batch_size = 4;
}

message InferResult {
  string frame_id = 1;
  int64 capture_timestamp_ms = 2;
  bytes pseudo_color_image = 3;
  bytes skeleton_contour_image = 4;
  int32 person_count = 5;
  int32 processing_time_ms = 6;
  ResultKind result_kind = 7;
  int32 input_index = 8;
  int32 output_index = 9;
}
```

`ResultKind`：

```proto
RESULT_KIND_INTERPOLATED
RESULT_KIND_CURRENT
```

时间字段含义：

- `InferResponse.processing_time_ms`：服务端处理整个 batch 的耗时，包含解码、插帧、YOLO 推理、后处理、渲染、PNG 编码等服务端算法流程。
- `InferResult.processing_time_ms`：单个输出帧后处理分析耗时，不等于完整端到端耗时。
- `InferResult.capture_timestamp_ms`：客户端输入图片的采集时间戳。`current` 使用当前输入图时间戳；`interpolated` 在有上一帧和当前帧时间戳时使用两者中点。

重新生成 pb 的命令：

```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ai.proto
```

## 已完成的代码改动

涉及文件：

```text
ai.proto
ai_pb2.py
scripts/grpc_server.py
scripts/grpc_client_test.py
scripts/serial_to_grpc_test.py
scripts/export_grpc_png_samples.py
scripts/infer_service.py
scripts/udp_infer_viewer.py
src/tof_pose/realtime_service.py
```

主要完成内容：

- 修改 `ai.proto`，支持 batch 输入和多结果输出。
- 根据新 proto 重新生成了 `ai_pb2.py`。
- gRPC 服务端 `Infer` 支持 `repeated InferImage images`。
- 服务端校验空 batch 和 `batch_size` 不匹配。
- 输出从旧逻辑的固定图片集改为：

```text
每张输入图:
  1. interpolated result
  2. current result

每个 result:
  1. pseudo_color_image
  2. skeleton_contour_image
```

- YOLO 分割模型和姿态模型都支持一次接收图片列表进行 batch 推理。
- 插帧逻辑已经接入：每张输入图会和上一张源深度图生成中间帧；如果没有上一帧，则插帧使用当前帧副本，保证输出数量稳定。
- 骨骼轮廓图改为伪彩图背景，只绘制骨骼和轮廓，不绘制文字标签、ID、距离等信息。
- gRPC 客户端测试脚本已支持 batch 输入并保存返回的两类图片。
- 新增 CPU 并行参数：

```bash
--decode-workers
--render-workers
```

- `--decode-workers`：并行解码客户端传来的 PNG 图片。
- `--render-workers`：并行渲染返回图和 PNG 编码。
- 服务端日志格式已加时间。
- 服务端 batch 日志已拆分阶段耗时。

## 当前服务端可输入参数

服务脚本入口：

```bash
python scripts/grpc_server.py --help
```

关键参数：

```bash
--host
--port
--max-workers
--max-msg-mb
--stateless
--pose-only
--no-pose-validate
--model-path
--pose-model-path
--pose-conf
--pose-kpt-conf
--pose-kpt-min-points
--device
--decode-workers
--render-workers
```

推荐本地/服务器启动参数：

```bash
./maixsense-grpc-server \
  --host 0.0.0.0 \
  --port 50052 \
  --max-workers 4 \
  --max-msg-mb 100 \
  --device cuda:0 \
  --decode-workers 4 \
  --render-workers 4
```

说明：

- `--max-workers` 是 gRPC 请求并发 worker 数。
- `--decode-workers` 和 `--render-workers` 是单个 batch 内部 CPU 阶段并行线程数。
- 单张 GPU 上不建议盲目开多个服务进程，否则会重复加载模型并争抢 GPU 显存和算力。

## 日志与性能诊断

日志格式已改为带时间：

```text
2026-06-05 11:28:48 [INFO] ...
```

每次 batch 推理完成后会打印类似：

```text
Infer batch timing: inputs=1 outputs=2 decode_ms=36 interpolate_ms=0 yolo_prepare_ms=25 yolo_infer_ms=2549 seg_yolo_ms=2290 pose_yolo_ms=258 postprocess_ms=4 render_ms=1 png_encode_ms=9 total_ms=2625 decode_workers=4 render_workers=4 device=cuda:0
```

字段含义：

- `decode_ms`：服务端解码输入 PNG 图片耗时。
- `interpolate_ms`：插帧耗时。
- `yolo_prepare_ms`：生成 YOLO 输入伪彩图的耗时。
- `yolo_infer_ms`：YOLO batch 推理总耗时。
- `seg_yolo_ms`：分割模型推理耗时。
- `pose_yolo_ms`：姿态模型推理耗时。
- `postprocess_ms`：后处理耗时。
- `render_ms`：渲染返回图片耗时。
- `png_encode_ms`：PNG 编码耗时。
- `total_ms`：算法侧 batch 总耗时。

性能判断要点：

- A100 上单路 batch 仍然可能慢，不一定是 GPU 没工作，可能瓶颈在 CPU 解码、YOLO 前处理、track 后处理、渲染、PNG 编码或 gRPC 返回大 payload。
- 20 张图的 YOLO forward 可以 batch，但不是整个 pipeline 都并行。
- `track()` 包含跟踪逻辑，可能比纯 `predict()` 有更多 CPU 后处理。
- 当前日志能帮助定位到底是 `seg_yolo_ms`、`pose_yolo_ms`、`png_encode_ms` 还是其他阶段拖慢。

## 本地部署状态

本次已经在本地电脑使用 `maixpose` conda 环境启动了 gRPC 服务。

环境：

```text
E:\Anaconda\envs\maixpose\python.exe
torch 2.10.0+cu126
cuda available: True
device_count: 1
```

本地服务：

```text
host: 0.0.0.0
port: 50052
PID: 27688
device: cuda:0
decode-workers: 4
render-workers: 4
```

本地日志：

```text
E:\Project\Lab Project\MaixSense\logs\grpc_server_local.log
```

验证结果：

```text
输入: 1 张 PNG
返回: 2 个 InferResult
返回图片 blob: 4 张 PNG
processing_time_ms: 2625
roundtrip_ms: 2629
```

本地客户端调用示例：

```powershell
& 'E:\Anaconda\envs\maixpose\python.exe' scripts\grpc_client_test.py `
  4884018e2f21fd70b2aa4ce96ffdb3e0.png `
  --host 127.0.0.1 `
  --port 50052 `
  --timeout 180 `
  --max-msg-mb 100 `
  --device-id local_test `
  --batch-id local_smoke
```

停止本地服务：

```powershell
Stop-Process -Id 27688
```

## Linux 服务器部署方式

构建命令，也就是用户指定的可执行文件来源：

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd '/mnt/e/Project/Lab Project/MaixSense'; source maixsense/bin/activate; bash build_executable_wsl.sh"
```

注意：

- 本次本地验证使用的是 Python 脚本启动，不是重新打包后的可执行文件。
- 当前 Codex 沙箱里曾遇到 WSL `Ubuntu-22.04` 不可用的问题，因此如果要生成 Linux 可执行文件，应在用户自己的 Windows/WSL 环境执行上面的构建命令。

服务器启动命令示例，不使用 systemd：

```bash
cd /data/care/care-sense-iot-platform/bin

nohup ./maixsense-grpc-server \
  --host 0.0.0.0 \
  --port 50052 \
  --max-workers 4 \
  --max-msg-mb 100 \
  --device cuda:0 \
  --decode-workers 4 \
  --render-workers 4 \
  > maixsense-grpc-server.log 2>&1 &
```

查看是否启动：

```bash
ps -ef | grep maixsense-grpc-server
ss -lntp | grep 50052
```

查看日志：

```bash
tail -f /data/care/care-sense-iot-platform/bin/maixsense-grpc-server.log
```

停止服务：

```bash
ps -ef | grep maixsense-grpc-server
kill <PID>
```

如果有父子两个进程，优先停父进程；必要时把同一个服务的相关 PID 都停掉。不要保留多个监听同一端口或同一 GPU 的旧进程。

## 服务器上常见问题记录

问题：`nohup` 后 `Exit 127`

可能原因：

- 当前目录不对，执行的 `./maixsense-grpc-server` 不存在。
- 可执行文件没有执行权限。
- Linux 动态库或打包依赖缺失。

检查：

```bash
pwd
ls -lh ./maixsense-grpc-server
chmod +x ./maixsense-grpc-server
cat maixsense-grpc-server.log
```

问题：GPU 占用率为 0

可能原因：

- 没有客户端请求进来。
- 没显式传 `--device cuda:0`。
- 服务其实没启动或启动了旧版本。
- 模型仍在 CPU 上跑。
- GPU 推理阶段很短，但 CPU 前后处理很长，`nvidia-smi` 刷新时刚好看不到。

现在应通过服务端 batch timing 日志判断，不只看 `nvidia-smi`。

问题：GPU 不高但处理不过来

可能原因：

- PNG 解码、渲染和 PNG 编码占 CPU。
- gRPC 返回 40 张 PNG，payload 大。
- `track()` 的 CPU 后处理成本高。
- gRPC `--max-workers`、客户端发送频率、网络吞吐共同限制。

优先看：

```text
decode_ms
yolo_prepare_ms
seg_yolo_ms
pose_yolo_ms
postprocess_ms
render_ms
png_encode_ms
total_ms
```

## 当前 Git 工作区状态

本次会话后，当前工作区存在未提交修改：

```text
M ai.proto
M ai_pb2.py
M scripts/export_grpc_png_samples.py
M scripts/grpc_client_test.py
M scripts/grpc_server.py
M scripts/infer_service.py
M scripts/serial_to_grpc_test.py
M scripts/udp_infer_viewer.py
M src/tof_pose/realtime_service.py
?? logs/
```

`logs/` 是本地运行产生的日志目录，通常不应提交。

## 已验证项目

已运行并通过：

```bash
python -m py_compile src/tof_pose/realtime_service.py scripts/grpc_server.py
```

本地 gRPC smoke test 已通过：

```text
127.0.0.1:50052
1 input -> 2 results -> 4 PNG blobs
```

## 待办事项

1. 在用户自己的 WSL Ubuntu-22.04 环境重新执行构建命令，生成新的 Linux 可执行文件。
2. 把新的 `dist/maixsense-grpc-server` 上传到服务器目录 `/data/care/care-sense-iot-platform/bin/`。
3. 服务器上停止旧服务，确认没有多个 `maixsense-grpc-server` 残留进程。
4. 用新参数启动服务器：

```bash
--device cuda:0 --decode-workers 4 --render-workers 4 --max-msg-mb 100
```

5. 从调用方服务器 `8.149.246.201` 请求服务端 `222.71.62.147:50052`，验证 batch 输入 N 张时返回 `2N` 个 result 和 `4N` 张 PNG。
6. 观察服务器日志中的 batch timing，确定真实瓶颈。
7. 如果 `seg_yolo_ms` 仍然过高，评估是否可以改用 `predict()` 或减少/关闭跟踪逻辑。
8. 如果 `png_encode_ms` 或网络 payload 过高，评估是否降低 PNG 压缩、改 JPEG/WebP、或减少返回图片数量。
9. 如果 batch 内 CPU 阶段仍慢，尝试 `--decode-workers 8` 和 `--render-workers 8`，但需要观察 CPU 使用率和总体延迟。
10. 最后决定是否提交这些协议和代码改动，并把 `logs/` 加入忽略或清理。

## 下次会话建议入口

下次可以直接让 Codex 读取本文件：

```text
docs/session_grpc_batch_summary_2026_06_05.md
```

然后继续做以下任一任务：

- 重新编译 Linux 可执行文件。
- 上传部署到 `222.71.62.147`。
- 根据 batch timing 日志继续优化性能。
- 检查 proto/client/server 三端兼容性。
- 整理提交或打包发布。
