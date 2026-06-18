# MaixSense 会话上下文总结（2026-06-18）

> 用途：下次新建对话时导入本文件，快速恢复当前 MaixSense gRPC / TensorRT / 轮廓识别调试上下文。  
> 安全提醒：本文不包含服务器密码、OSS AccessKey 或任何密钥值。新会话中不要把密钥写入总结或日志。

## 1. 当前目标

当前主线是调试 MaixSense AI 算法服务中“骨骼正常、轮廓容易漏检”的问题，尤其是：

- 同一批真实原始输入里，部分帧模型原始有轮廓候选，置信度不低。
- 但后处理后真正保留下来的轮廓数量仍为 0。
- 当前日志把这类情况标记为 `raw_candidate_unhandled`。
- 已确认这不是简单的低置信度外形过滤导致。

对外描述和新增日志中尽量使用 `model` / `模型`，不要重新引入旧的 detector-brand 字样。

## 2. 当前代码状态

本地项目路径：

```text
E:\Project\Lab Project\MaixSense
```

当前工作区状态：

```text
 M scripts/grpc_server.py
 M src/tof_pose/realtime_service.py
?? .codex_tmp_oss_samples/
?? scripts/compare_model_backends.py
```

主要代码改动：

- `scripts/grpc_server.py`
  - 增加了若干模型阈值/后处理相关启动参数。
  - 包括模型置信度、姿态关键点阈值、mask 阈值、轮廓新轨迹/已有轨迹置信度等参数。

- `src/tof_pose/realtime_service.py`
  - 增加模型识别相关日志字段：
    - `contour_counts`
    - `contour_conf_max`
    - `contour_conf_avg`
    - `contour_conf_peak`
    - `pose_counts`
    - `pose_conf_max`
    - `pose_conf_avg`
    - `pose_conf_peak`
    - `pose_kpt_gate_points_max`
    - `post_contour_counts`
    - `post_contour_reject_reasons`
    - `final_person_counts`
    - `pose_fallback_counts`
  - 增加了姿态兜底逻辑：当轮廓没有留下来但姿态关键点可靠时，仍可输出骨骼和人数。
  - 已取消低置信度轮廓的两类外形过滤：
    - 长宽比异常过滤。
    - 贴边且过高过滤。
  - 当前仍保留基础可用性检查：
    - mask 面积过小。
    - mask 面积过大。
    - 空框/无效 mask。
    - mask 跳变保护。
    - 轨迹置信度/轨迹存在性相关逻辑。

- `scripts/compare_model_backends.py`
  - 诊断脚本，当前未跟踪。

- `.codex_tmp_oss_samples/`
  - 本地临时样本目录，包含 6 月 4 日样本索引、预览图、诊断输出等。

## 3. 当前 AI 服务运行状态

AI 服务器：

```text
IP: 222.71.62.147
用户: care
```

服务目录：

```text
/data/care/care-sense-iot-platform/bin
```

当前运行版本：

```text
/data/care/care-sense-iot-platform/bin/maixsense-grpc-server-trt-runtime-20260618-contour-relaxed
```

当前 symlink：

```text
/data/care/care-sense-iot-platform/bin/maixsense-grpc-server-trt-current
```

当前日志：

```text
/data/care/care-sense-iot-platform/bin/maixsense-grpc-server-trt-b20-contour-relaxed.log
```

当前 PID 文件：

```text
/data/care/care-sense-iot-platform/bin/maixsense-grpc-server-trt-b20-contour-relaxed.pid
```

最近确认的 PID：

```text
3760823
```

服务端口：

```text
50052
```

启动方式：

- 不使用 systemd。
- 使用 `nohup`。
- 当前为 batch-20 TensorRT engine。
- 当前模型实例数为 6。

当前启动参数核心配置：

```text
--host 0.0.0.0
--port 50052
--max-workers 6
--max-msg-mb 160
--device cuda:0
--decode-workers 6
--render-workers 6
--model-instances 6
--warmup-batch-size 20
--device-binding-ttl-sec 0
--output-format jpeg
--jpeg-quality 60
--cpu-worker-mode process
--cpu-process-start-method fork
--model-path /data/care/trt-export-lowmem-20260612-b20w1/maixsense-seg-lowmem-b20w1.engine
--pose-model-path /data/care/trt-export-lowmem-20260612-b20w1/maixsense-pose-lowmem-b20w1.engine
```

## 4. 当前部署流程已整理成 Skill

已更新本地 Codex skill：

```text
C:\Users\Zy188\.codex\skills\maixsense-trt-oss-deploy
```

核心文件：

```text
C:\Users\Zy188\.codex\skills\maixsense-trt-oss-deploy\SKILL.md
C:\Users\Zy188\.codex\skills\maixsense-trt-oss-deploy\references\server-runbook.md
```

校验结果：

```text
Skill is valid!
```

下次如果需要执行“本地改代码 → 上传服务器 → 服务器编译 → 部署重启 → 删除服务器源码”，应使用：

```text
$maixsense-trt-oss-deploy
```

重要规则：

- 不打印密码、AccessKey、Secret。
- 不删除 runtime、engine、日志、PID、env 文件。
- 只删除 `/data/care/build` 下本次临时源码目录和 tar 包。
- 停服务时避免用过宽的进程匹配，优先 PID 文件和端口监听进程。
- 尽量一次编译完成，避免反复 PyInstaller。

## 5. 关键验证：同一批真实 6 月 4 日原始输入

已从 OSS 确认并下载同一批真实原始输入，共 20 张：

```text
device-003 / 2026-06-04
1780544313977-tof-1780544313977-2556.png
...
1780544316276-tof-1780544316276-2575.png
```

本地临时目录：

```text
.codex_tmp_oss_samples/june4_same_batch_raw_1780544313977_1780544316276
```

这 20 张真实输入的 OSS objectKey：

```text
care-sense-device/frames/device-003/2026/06/04/1780544313977-tof-1780544313977-2556.png
care-sense-device/frames/device-003/2026/06/04/1780544314076-tof-1780544314076-2557.png
care-sense-device/frames/device-003/2026/06/04/1780544314186-tof-1780544314186-2558.png
care-sense-device/frames/device-003/2026/06/04/1780544314302-tof-1780544314302-2559.png
care-sense-device/frames/device-003/2026/06/04/1780544314390-tof-1780544314390-2560.png
care-sense-device/frames/device-003/2026/06/04/1780544314493-tof-1780544314493-2561.png
care-sense-device/frames/device-003/2026/06/04/1780544314577-tof-1780544314577-2562.png
care-sense-device/frames/device-003/2026/06/04/1780544314677-tof-1780544314677-2563.png
care-sense-device/frames/device-003/2026/06/04/1780544314777-tof-1780544314777-2564.png
care-sense-device/frames/device-003/2026/06/04/1780544314877-tof-1780544314877-2565.png
care-sense-device/frames/device-003/2026/06/04/1780544314976-tof-1780544314976-2566.png
care-sense-device/frames/device-003/2026/06/04/1780544315080-tof-1780544315080-2567.png
care-sense-device/frames/device-003/2026/06/04/1780544315401-tof-1780544315401-2568.png
care-sense-device/frames/device-003/2026/06/04/1780544315586-tof-1780544315586-2569.png
care-sense-device/frames/device-003/2026/06/04/1780544315676-tof-1780544315676-2570.png
care-sense-device/frames/device-003/2026/06/04/1780544315887-tof-1780544315887-2571.png
care-sense-device/frames/device-003/2026/06/04/1780544315977-tof-1780544315977-2572.png
care-sense-device/frames/device-003/2026/06/04/1780544316098-tof-1780544316098-2573.png
care-sense-device/frames/device-003/2026/06/04/1780544316176-tof-1780544316176-2574.png
care-sense-device/frames/device-003/2026/06/04/1780544316276-tof-1780544316276-2575.png
```

重测请求：

```text
device_id=device-003-real-june4-retest
batch_id=codex-same-real-june4-relaxed-1781770151245-20
```

返回结果：

```text
input_count=20
result_count=40
current_count=20
interpolated_count=20
current_person_counts=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
interpolated_person_counts=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
current_person_sum=20
interpolated_person_sum=20
empty_skeleton_outputs=0
server_processing_time_ms=964
grpc_elapsed_ms=1238.9
```

服务端日志核心结果：

```text
contour_counts=[0,1,0,0,0,1,0,0,0,0,0,0,1,0,1,1,0,0,1,1]
contour_conf_max=[-,0.423,-,-,-,0.484,-,-,-,-,-,-,0.800,-,0.287,0.622,-,-,0.323,0.596]
contour_conf_avg=0.505
contour_conf_peak=0.800
post_contour_counts=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
post_contour_reject_reasons=[no_candidate,raw_candidate_unhandled,no_candidate,no_candidate,no_candidate,raw_candidate_unhandled,no_candidate,no_candidate,no_candidate,no_candidate,no_candidate,no_candidate,raw_candidate_unhandled,no_candidate,raw_candidate_unhandled,raw_candidate_unhandled,no_candidate,no_candidate,raw_candidate_unhandled,raw_candidate_unhandled]
final_person_counts=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
pose_fallback_counts=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
```

重要结论：

- 这批数据里有 7 帧原始轮廓候选。
- 候选置信度不低，最高到 `0.800`。
- 但后处理后真正保留下来的轮廓数量全是 0。
- 取消低置信度“长宽比/贴边过高”限制后，这批结果没有改善。
- 当前日志里这两个已取消的过滤原因出现次数为 0。
- 因此这批轮廓丢失不是由这两个限制导致。

## 6. 当前关键判断

当前最可能的问题区域在“模型原始结果 → 后处理 records”之间。

当前代码里候选进入最终 `records` 前会经过这些步骤：

1. 读取模型 boxes / masks。
2. 计算 `match_count = min(len(boxes), len(masks), len(track_ids), len(conf_scores))`。
3. 姿态与轮廓轨迹匹配，用于骨骼门控。
4. 轮廓置信度/轨迹规则：
   - 新轨迹阈值。
   - 已有轨迹阈值。
   - track 缺失。
   - track center/area jump。
5. mask 二值化。
6. mask 跳变保护。
7. mask 形状基础检查：
   - 面积过小。
   - 面积过大。
   - 空框。
8. `estimate_person_distance_from_mask()` 提取轮廓。
9. 追加到 `records`。

现在日志显示 `raw_candidate_unhandled`，说明：

- 日志已看到原始候选。
- 但 `contour_reject_reasons` 没有记录具体原因。
- 最终 `records` 仍然为空。

这意味着当前日志粒度仍不够，下一步应继续细化 `raw_candidate_unhandled`。

## 7. 下一步建议

优先级最高：

1. 继续补齐 `raw_candidate_unhandled` 的细分原因。
2. 对每个原始候选逐个记录：
   - `match_count`
   - box 数量
   - mask 数量
   - track_id 是否存在
   - conf 是否通过
   - mask 二值化后的面积
   - mask 跳变保护结果
   - shape 检查结果
   - `estimate_person_distance_from_mask()` 是否返回 contour
   - contour 面积
   - valid depth pixels
3. 如果只是 `estimate_person_distance_from_mask()` 内部没有记录原因，需要给这个函数增加可返回/可记录的原因字段。

可能要重点检查：

- `result.masks.data` 是否有候选但 mask 二值化后几乎为空。
- TensorRT engine 的 mask 输出是否与 pt 版本存在差异。
- `match_count` 是否正常。
- `track_id` 是否在 batch 推理/track 中异常。
- `estimate_person_distance_from_mask()` 中腐蚀/闭运算是否把细小 mask 处理没了。
- mask ROI 是否和 box 对齐。

## 8. 部署/验证注意事项

如需继续改代码并部署：

1. 使用 `$maixsense-trt-oss-deploy` skill。
2. 本地先执行语法检查。
3. 上传源码包到 `/data/care/build`。
4. 服务器使用 `/data/care/tensorrt-export-venv/bin/python -m PyInstaller` 编译。
5. 部署到新的 runtime 目录。
6. 更新 `maixsense-grpc-server-trt-current` symlink。
7. 使用 `nohup` 重启，不能使用 systemd。
8. 验证端口 `50052`。
9. 用同一批 6 月 4 日真实原始输入重测。
10. 删除服务器临时源码目录和 tar 包。

不要删除：

- runtime 目录。
- TensorRT engine 目录。
- 日志文件。
- PID 文件。
- OSS env 文件。

## 9. 后端/平台侧背景

后端服务器：

```text
IP: 8.149.246.201
用户: root
```

之前平台侧已经做过的关键优化：

- 按 `device_id` 独立攒批。
- 同设备满批立即 flush。
- 未满批按超时 flush。
- AI worker pool 限流。
- gRPC AI client 长连接复用。
- 同一个 `device_id` 只允许一个 AI RPC 在飞。

当前这次轮廓问题主要集中在算法侧，不是平台侧积压/调度问题。

## 10. 当前结论一句话

同一批真实 6 月 4 日原始输入重测后确认：人和骨骼稳定输出，但轮廓仍然全部没有进入最终结果；已取消的低置信度外形限制不是原因，下一步应把 `raw_candidate_unhandled` 细分到具体后处理分支。
