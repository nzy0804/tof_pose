# tof_pose

[English](./README.md) | [中文](./README-ZH.md)

`tof_pose` 是一个面向 MaixSense ToF 深度数据流的人体姿态估计项目，支持实时测距、伪彩视频录制、离线推理，以及围绕 YOLO pose 的轻量训练流程。

## 功能特性

- 基于 MaixSense 串口深度数据进行实时姿态推理
- 结合深度信息和人体轮廓估计每个人到相机的距离
- 录制 ToF 伪彩视频，便于后续查看和标注
- 对录制视频执行离线姿态推理
- 提供抽帧、CVAT 标注转换和本地模型微调脚本

## 快速开始

```powershell
python scripts/tof_pose.py COM8
python scripts/tof_record.py COM8
python scripts/tof_video_inference.py
```

## 运行流程

### 实时姿态推理

`scripts/tof_pose.py` 会读取 MaixSense 串口深度帧，将其转换为伪彩图，执行 YOLO 姿态推理，绘制人体骨架，并叠加 `P1 Dist~...` 这类人物距离估计结果。

人物周围的白色轮廓线表示当前用于距离估计的人体主体区域。

### 视频录制

`scripts/tof_record.py` 会将实时深度流录制为伪彩视频，便于后续标注、可视化和离线推理。

### 离线视频推理

`scripts/tof_video_inference.py` 会对已录制的伪彩视频执行姿态推理，并输出带骨架结果的视频。

说明：

- 当前离线流程使用的是伪彩视频，而不是原始深度值。
- 离线推理目前不支持可靠的真实距离估计。
- 距离估计功能当前只在实时 ToF 流程中生效。

## 数据集整理

使用下面的命令从视频抽帧并生成标准训练目录：

```powershell
python scripts/prepare_dataset.py --input outputs/videos --dataset-root dataset --sample-every 5 --touch-labels
```

生成的目录包括：

- `dataset/images/train|val|test`
- `dataset/labels/train|val|test`
- `dataset/meta/manifest.csv`
- `dataset/dataset.yaml`

详细规范与标注检查清单见 [dataset/README.md](./dataset/README.md)。

## 标注结果转换

如果你从 CVAT 导出的是 `COCO Keypoints 1.0` 格式，可用下面的命令转换为 `YOLO pose` 标签：

```powershell
python scripts/convert_coco_to_yolo_pose.py --input-dir "task_1_dataset_2026_03_31_11_47_13_coco keypoints 1.0" --dataset-root dataset --split train --copy-images --overwrite
```

## 模型训练

使用下面的命令基于整理好的数据集训练姿态模型：

```powershell
yolo pose train data=dataset/dataset.yaml model=assets/models/yolo11n-pose.pt epochs=30 imgsz=320 batch=8 device=0 project=runs/tof_pose name=pose_debug_v1
```

参数说明：

- `data=dataset/dataset.yaml`：数据集配置文件
- `model=assets/models/yolo11n-pose.pt`：本地初始 pose 模型
- `imgsz=320`：与当前推理分辨率保持一致
- `batch=8`：适合作为中小显存环境的起步设置
- `device=0`：使用第 1 张 GPU

## 新模型部署

训练完成后，推荐将最佳权重放到：

```text
assets/models/tof_pose_best.pt
```

当前项目已经默认使用这个文件，默认模型路径定义在 [src/tof_pose/paths.py](./src/tof_pose/paths.py) 中。

## 目录结构

- `src/tof_pose/`：核心业务代码
- `assets/models/`：YOLO 姿态模型权重
- `outputs/videos/`：录制视频与推理结果
- `scripts/`：项目运行入口脚本
- `dataset/`：训练数据目录模板与说明
- `runs/`：训练输出目录

## 常用命令

```powershell
python scripts/tof_pose.py COM8
python scripts/tof_record.py COM8
python scripts/tof_video_inference.py
python scripts/prepare_dataset.py --input outputs/videos --dataset-root dataset --sample-every 5 --touch-labels
python scripts/convert_coco_to_yolo_pose.py --input-dir "task_1_dataset_2026_03_31_11_47_13_coco keypoints 1.0" --dataset-root dataset --split train --copy-images --overwrite
yolo pose train data=dataset/dataset.yaml model=assets/models/yolo11n-pose.pt epochs=30 imgsz=320 batch=8 device=0 project=runs/tof_pose name=pose_debug_v1
```
