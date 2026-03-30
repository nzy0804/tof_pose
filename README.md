# tof_pose

tof_pose 项目当前围绕 3 个可直接运行的流程组织：

- `scripts/tof_pose.py`：从 MaixSense 串口数据流进行实时姿态推理
- `scripts/tof_record.py`：采集并保存 ToF 伪彩视频
- `scripts/tof_video_inference.py`：对已录制视频执行离线姿态推理

## 目录结构

- `src/maixsense/`：可复用的核心业务代码
- `assets/models/`：YOLO 姿态模型权重
- `outputs/videos/`：录制视频与推理结果视频
- `scripts/`：项目运行入口脚本

## 运行示例

```powershell
python scripts/tof_pose.py COM8
python scripts/tof_record.py COM8
python scripts/tof_video_inference.py
```
